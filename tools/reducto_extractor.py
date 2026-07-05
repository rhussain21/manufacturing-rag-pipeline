"""
ReductoExtractor — parse PDFs via the Reducto document-understanding API.

Modular alternative extraction backend to PDFExtractor (PyMuPDF/pdfplumber).
Reducto returns layout-aware, structured output — tables reconstructed as
markdown, figures described in natural language, and headers/footers/page
numbers filtered out at the source rather than leaking into the text as
boilerplate.

Usage (flat text — drop-in compatible with PDFExtractor.extract_text):
    from tools.reducto_extractor import ReductoExtractor

    text, page_count = ReductoExtractor.extract_text("path/to/paper.pdf")

Usage (full parse result — native chunks, blocks, usage/cost):
    result = ReductoExtractor.extract_full("path/to/paper.pdf")
    chunks = result["result"]["chunks"]     # Reducto's semantic chunks
    usage = result["usage"]                 # {"num_pages": .., "credits": ..}

Requires REDUCTO_API_KEY in the environment.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ReductoExtractor:
    """Parse PDFs via Reducto's document-understanding API."""

    MAX_RETRIES = 2
    RETRY_DELAY = 3.0
    TIMEOUT = 120  # seconds; long/complex PDFs can take a while to parse

    # Filtered out of chunk content/embed fields — targets the exact
    # boilerplate (repeated headers/footers/page numbers) that causes
    # near-duplicate false positives in the corpus's SimHash gate.
    FILTER_BLOCKS = ["Header", "Footer", "Page Number"]

    _client = None  # lazy singleton — avoid re-instantiating per call

    # ── Public API ─────────────────────────────────────────────────────

    @classmethod
    def extract_text(cls, pdf_path: str) -> Tuple[str, int]:
        """Extract flat text from a local PDF file via Reducto.

        Drop-in compatible with PDFExtractor.extract_text: same signature,
        same (text, page_count) return shape. Requests chunked parsing and
        concatenates chunk content into one string, so callers that only
        want flat text still get it from a single request.

        Returns:
            (text, page_count) tuple. Empty string and 0 on failure.
        """
        result = cls.extract_full(pdf_path)
        if not result:
            return "", 0

        chunks = result.get("result", {}).get("chunks", [])
        text = "\n\n".join(c["content"] for c in chunks)
        page_count = result.get("usage", {}).get("num_pages", 0)
        return text, page_count

    @classmethod
    def extract_full(cls, pdf_path: str, chunk_mode: str = "variable") -> Optional[Dict[str, Any]]:
        """Upload + parse a local PDF file, returning the full Reducto result.

        Args:
            pdf_path: Path to the PDF file on disk.
            chunk_mode: Reducto chunking strategy — "variable" (semantic,
                default), "page", "section", "block", "page_sections", or
                "disabled" (single chunk).

        Returns:
            Full parse response as a plain dict (job_id, duration, result,
            usage, studio_link — see Reducto's parse.run response schema).
            `result["result"]["chunks"]` is always populated, even for large
            documents where the API returns a `type: "url"` pointer instead
            of inline chunks — that indirection is resolved here so callers
            never need to special-case document size.
            None on failure (missing dependency, auth error, or repeated
            request failure).
        """
        client = cls._get_client()
        if client is None:
            return None

        for attempt in range(1, cls.MAX_RETRIES + 1):
            try:
                upload = client.upload(file=Path(pdf_path))
                result = client.parse.run(
                    input=upload.file_id,
                    formatting={"table_output_format": "md"},
                    retrieval={
                        "chunking": {"chunk_mode": chunk_mode},
                        "filter_blocks": cls.FILTER_BLOCKS,
                    },
                    settings={"timeout": cls.TIMEOUT},
                )
                result_dict = result.model_dump()

                if result_dict["result"].get("type") == "url":
                    import requests
                    resp = requests.get(result_dict["result"]["url"], timeout=cls.TIMEOUT)
                    resp.raise_for_status()
                    result_dict["result"]["chunks"] = resp.json()["chunks"]

                return result_dict

            except Exception as e:
                logger.warning(f"Reducto parse error (attempt {attempt}): {e}")
                if attempt < cls.MAX_RETRIES:
                    time.sleep(cls.RETRY_DELAY)

        logger.error(f"Reducto extraction failed after {cls.MAX_RETRIES} attempts: {pdf_path}")
        return None

    # ── Client ───────────────────────────────────────────────────────

    @classmethod
    def _get_client(cls):
        if cls._client is not None:
            return cls._client
        try:
            from reducto import Reducto
        except ImportError:
            logger.error("reducto package not installed — pip install reducto")
            return None
        try:
            cls._client = Reducto()  # reads REDUCTO_API_KEY from env
        except Exception as e:
            logger.error(f"Failed to initialize Reducto client: {e}")
            return None
        return cls._client
