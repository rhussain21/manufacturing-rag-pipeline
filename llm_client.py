import os
import requests
import json
import time
import logging
import traceback
import platform
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv


def detect_device() -> str:
    """Detect the current device. Returns: 'jetson', 'mac', or 'linux'."""
    # Check for Jetson first (most specific)
    try:
        if os.path.exists('/proc/device-tree/model'):
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip()
                if 'jetson' in model.lower():
                    return 'jetson'
    except Exception:
        pass
    if os.path.exists('/etc/nv_tegra_release'):
        return 'jetson'
    if platform.system() == 'Darwin':
        return 'mac'
    return 'linux'


# Load platform-specific .env file
project_root = Path(__file__).parent
device = detect_device()
platform_env = project_root / f'.env.{device}'
generic_env = project_root / '.env'

if platform_env.exists():
    load_dotenv(platform_env, override=True)
elif generic_env.exists():
    load_dotenv(generic_env, override=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

MAX_RETRIES = 3
BASE_BACKOFF = 2  # seconds


def _classify_error(e: Exception) -> str:
    """Classify an exception into a category for structured logging."""
    msg = str(e).lower()
    etype = type(e).__name__

    if isinstance(e, requests.exceptions.Timeout):
        return 'timeout'
    if isinstance(e, requests.exceptions.ConnectionError):
        return 'connection_error'
    if isinstance(e, requests.exceptions.HTTPError):
        code = getattr(e.response, 'status_code', 0) if hasattr(e, 'response') else 0
        if code == 429:
            return 'rate_limit'
        if code >= 500:
            return 'server_error'
        return f'http_{code}'
    if 'rate' in msg and 'limit' in msg:
        return 'rate_limit'
    if 'quota' in msg:
        return 'quota_exceeded'
    if 'timeout' in msg:
        return 'timeout'
    if 'resource' in msg and 'exhausted' in msg:
        return 'quota_exceeded'
    if 'json' in msg or 'parse' in msg or 'decode' in msg:
        return 'parse_error'
    if 'safety' in msg or 'blocked' in msg:
        return 'safety_filter'
    return etype


class BaseLLMClient(ABC):
    """Common interface every LLM client in this project implements — lets
    calling code (agent nodes, eval/judge scripts, tools/resilient_batch
    jobs) depend on this contract instead of a specific provider's SDK.
    This isn't aspirational: the eval harness already swaps GeminiClient
    (generates real answers) and OpenAIClient (scores them as an
    independent judge) through identical call sites without either one
    knowing about the other.

    Not enforced via a shared __init__ — each provider's constructor takes
    genuinely different arguments (a model name, a base_url, nothing at
    all) — only the methods calling code actually depends on are part of
    the contract.
    """

    model_name: str

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str, temperature: float = 0.7) -> str:
        """Returns the full generated text. Retries internally on
        transient errors (rate limits, timeouts, server errors) per the
        provider's own backoff policy; raises after exhausting retries."""
        raise NotImplementedError

    def generate_stream(self, prompt: str, system_prompt: str, temperature: float = 0.7):
        """Generator of text deltas. Default implementation falls back to
        one non-streamed chunk via generate() — only GeminiClient overrides
        this with real token streaming today. Callers (agents/streaming.py)
        can always call generate_stream() with no hasattr() check, on any
        client that implements this interface."""
        yield self.generate(prompt, system_prompt, temperature)


class OllamaClient(BaseLLMClient):
    def __init__(self, model="llama3:latest", base_url="http://localhost:11434"):
        self.model = model
        self.model_name = model
        self.base_url = base_url
    
    def generate(self, prompt, system_prompt, temperature=0.7):
        full_prompt = f"{system_prompt}\n\n{prompt}"
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "temperature": temperature,
                        "stream": False
                    },
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()

                if "response" in result:
                    return result["response"]
                elif "error" in result:
                    raise RuntimeError(f"Ollama error: {result['error']}")
                else:
                    raise KeyError(f"Unexpected Ollama response format. Got: {json.dumps(result, indent=2)}")

            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError) as e:
                last_error = e
                wait = BASE_BACKOFF ** attempt
                logger.warning(f"Ollama attempt {attempt}/{MAX_RETRIES} failed ({_classify_error(e)}), retrying in {wait}s")
                time.sleep(wait)
            except requests.exceptions.HTTPError as e:
                error_cat = _classify_error(e)
                if error_cat == 'rate_limit' and attempt < MAX_RETRIES:
                    wait = BASE_BACKOFF ** attempt * 2
                    logger.warning(f"Ollama rate limited, waiting {wait}s")
                    time.sleep(wait)
                    last_error = e
                else:
                    raise
            except Exception:
                raise

        raise last_error or RuntimeError(f"Ollama failed after {MAX_RETRIES} attempts")


class GeminiClient(BaseLLMClient):
    def __init__(self, model="gemini-2.5-flash"):
        from google import genai
        from google.genai import types
        self.model_name = model

        # Prefer Vertex AI (uses GCP credits) over AI Studio (api_key billing)
        gcp_project = os.getenv("GCP_PROJECT")
        gcp_location = os.getenv("GCP_LOCATION", "us-central1")

        if gcp_project:
            # Vertex AI uses ADC (project + api_key are mutually exclusive)
            # Try multiple credential paths in case running as different user
            # (e.g. root on Jetson but gcloud auth done as redwan)
            adc_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not adc_path or not os.path.exists(adc_path):
                for path in [
                    "/root/.config/gcloud/application_default_credentials.json",
                    "/home/redwan/.config/gcloud/application_default_credentials.json",
                ]:
                    if os.path.exists(path):
                        adc_path = path
                        break
            if adc_path and os.path.exists(adc_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = adc_path
            self._client = genai.Client(
                vertexai=True,
                project=gcp_project,
                location=gcp_location,
            )
            logger.info(f"Gemini using Vertex AI (project={gcp_project}, location={gcp_location})")
        else:
            # Fallback to AI Studio (api_key)
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Set GCP_PROJECT for Vertex AI or GEMINI_API_KEY for AI Studio"
                )
            self._client = genai.Client(api_key=api_key)
            logger.info("Gemini using AI Studio (api_key)")

        self._types = types

    def generate(self, prompt, system_prompt, temperature=0.7):
        full_prompt = f"{system_prompt}\n\n{prompt}"
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config=self._types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=8192,
                        http_options=self._types.HttpOptions(timeout=60000),  # ms; prevents an indefinite hang on a stalled connection
                    )
                )
                if response.text is None:
                    # Safety filter or empty response
                    candidates = getattr(response, 'candidates', [])
                    reason = 'unknown'
                    if candidates:
                        reason = getattr(candidates[0], 'finish_reason', 'unknown')
                    raise RuntimeError(f"Gemini returned empty response (finish_reason={reason})")
                return response.text

            except Exception as e:
                error_cat = _classify_error(e)
                last_error = e

                if error_cat in ('rate_limit', 'quota_exceeded', 'timeout', 'server_error', 'connection_error'):
                    wait = BASE_BACKOFF ** attempt * (3 if error_cat == 'rate_limit' else 1)
                    logger.warning(f"Gemini attempt {attempt}/{MAX_RETRIES} failed ({error_cat}), retrying in {wait}s")
                    time.sleep(wait)
                elif error_cat == 'safety_filter':
                    # Don't retry safety blocks — they'll keep failing
                    raise
                elif attempt < MAX_RETRIES:
                    wait = BASE_BACKOFF ** attempt
                    logger.warning(f"Gemini attempt {attempt}/{MAX_RETRIES} failed ({error_cat}: {e}), retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise

        raise last_error or RuntimeError(f"Gemini failed after {MAX_RETRIES} attempts")

    def generate_stream(self, prompt, system_prompt, temperature=0.7):
        """Generator of text deltas via the SDK's generate_content_stream.

        Retry policy differs from generate() in exactly one way, on purpose:
        a failure BEFORE any chunk has been yielded retries from scratch
        (nothing has been shown to the user yet, so a clean restart is
        invisible), but a failure AFTER partial text has streamed does NOT
        retry — a restart would duplicate or splice text mid-answer.
        Instead it yields one short trailing note and stops cleanly, so a
        UI consumer (st.write_stream) never sees a mid-stream exception.
        """
        full_prompt = f"{system_prompt}\n\n{prompt}"
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            yielded_any = False
            try:
                stream = self._client.models.generate_content_stream(
                    model=self.model_name,
                    contents=full_prompt,
                    config=self._types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=8192,
                        http_options=self._types.HttpOptions(timeout=60000),  # ms; prevents an indefinite hang on a stalled connection
                    ),
                )
                for chunk in stream:
                    # chunk.text can be None (e.g. a final metadata-only
                    # chunk) — skip those rather than yielding "None".
                    text = getattr(chunk, "text", None)
                    if text:
                        yielded_any = True
                        yield text
                if not yielded_any:
                    # Same classification path as generate()'s empty-response
                    # case: the 'safety' keyword makes _classify_error treat
                    # it as safety_filter (no retry — it'll keep failing).
                    raise RuntimeError("Gemini stream returned empty response (safety filter or empty response)")
                return

            except Exception as e:
                error_cat = _classify_error(e)
                last_error = e

                if yielded_any:
                    # Partial text already shown — never retry (see docstring).
                    logger.warning(
                        f"Gemini stream failed after partial output ({error_cat}: {e}); not retrying"
                    )
                    yield "\n\n_[response cut off due to an error]_"
                    return

                if error_cat in ('rate_limit', 'quota_exceeded', 'timeout', 'server_error', 'connection_error'):
                    wait = BASE_BACKOFF ** attempt * (3 if error_cat == 'rate_limit' else 1)
                    logger.warning(f"Gemini stream attempt {attempt}/{MAX_RETRIES} failed ({error_cat}), retrying in {wait}s")
                    time.sleep(wait)
                elif error_cat == 'safety_filter':
                    # Don't retry safety blocks — they'll keep failing
                    raise
                elif attempt < MAX_RETRIES:
                    wait = BASE_BACKOFF ** attempt
                    logger.warning(f"Gemini stream attempt {attempt}/{MAX_RETRIES} failed ({error_cat}: {e}), retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise

        raise last_error or RuntimeError(f"Gemini stream failed after {MAX_RETRIES} attempts")


class OpenAIClient(BaseLLMClient):
    def __init__(self, model="gpt-4o"):
        from openai import OpenAI
        self.model_name = model
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY in your .env file")
        self._client = OpenAI(api_key=api_key)

    def generate(self, prompt, system_prompt, temperature=0.7):
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                )
                return response.choices[0].message.content

            except Exception as e:
                error_cat = _classify_error(e)
                last_error = e

                if error_cat in ('rate_limit', 'timeout', 'server_error', 'connection_error'):
                    wait = BASE_BACKOFF ** attempt * (3 if error_cat == 'rate_limit' else 1)
                    logger.warning(f"OpenAI attempt {attempt}/{MAX_RETRIES} failed ({error_cat}), retrying in {wait}s")
                    time.sleep(wait)
                elif attempt < MAX_RETRIES:
                    wait = BASE_BACKOFF ** attempt
                    logger.warning(f"OpenAI attempt {attempt}/{MAX_RETRIES} failed ({error_cat}: {e}), retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise

        raise last_error or RuntimeError(f"OpenAI failed after {MAX_RETRIES} attempts")


class ClaudeClient(BaseLLMClient):
    def __init__(self, model="claude-sonnet-4-6"):
        import anthropic
        self.model_name = model
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Set ANTHROPIC_API_KEY in your .env file")
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt, system_prompt, temperature=0.7):
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.messages.create(
                    model=self.model_name,
                    max_tokens=8192,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text

            except Exception as e:
                error_cat = _classify_error(e)
                last_error = e

                if error_cat in ('rate_limit', 'timeout', 'server_error', 'connection_error'):
                    wait = BASE_BACKOFF ** attempt * (3 if error_cat == 'rate_limit' else 1)
                    logger.warning(f"Claude attempt {attempt}/{MAX_RETRIES} failed ({error_cat}), retrying in {wait}s")
                    time.sleep(wait)
                elif attempt < MAX_RETRIES:
                    wait = BASE_BACKOFF ** attempt
                    logger.warning(f"Claude attempt {attempt}/{MAX_RETRIES} failed ({error_cat}: {e}), retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise

        raise last_error or RuntimeError(f"Claude failed after {MAX_RETRIES} attempts")


