"""
Data Quality Filter — statistical quality gates for the corpus.

Complements ContentQualityGate (content_screener.py), which blocks obvious
garbage (404s, login walls, empty pages) before the LLM screen.  This module
applies corpus-level statistical thresholds derived from Notebook 01 findings:

    - 40% of ingested documents produced zero signals
    - Extreme-length docs (>500K chars) are almost always zero-signal manuals
    - Short docs (<200 chars) are table-of-contents stubs or scrape failures
    - Non-English content leaks in through PDF and RSS sources
    - High boilerplate ratio correlates with zero-signal outcomes
    - Near-duplicate documents inflate vector index without adding coverage

Pipeline position (run after extract, before vectorize):
    download → extract → [ContentQualityGate] → [LLM screen] → **DataQualityFilter** → vectorize → signal extraction

Usage:
    from etl.data_quality import DataQualityFilter

    dqf = DataQualityFilter()
    result = dqf.screen(text, metadata)
    if not result["pass"]:
        print(result["reason"])   # human-readable rejection reason
"""

import hashlib
import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Domain seed keywords (OT/ICS/Manufacturing relevance) ─────────────────────
# Derived from Notebook 01 term coverage analysis (99.67% coverage at these terms)
DOMAIN_KEYWORDS: List[str] = [
    "plc", "scada", "hmi", "dcs", "opc", "modbus", "profibus", "ethernet/ip",
    "iec 61131", "iec 62443", "iiot", "ot", "ics", "automation", "controller",
    "inverter", "servo", "fieldbus", "historian", "edge computing", "digital twin",
    "predictive maintenance", "condition monitoring", "robotics", "cnc", "fanuc",
    "siemens", "rockwell", "allen-bradley", "beckhoff", "schneider", "abb",
    "manufacturing", "industrial", "factory", "production", "machine learning",
]

# ── Thresholds (grounded in Notebook 01 findings) ─────────────────────────────
MIN_CHARS = 200          # below this: likely stub or scrape failure
MAX_CHARS = 500_000      # above this: flag as oversized (likely full manual)
MIN_TOKEN_DIVERSITY = 0.05   # unique_tokens / total_tokens; below = repetitive boilerplate
MAX_BOILERPLATE_RATIO = 0.40 # ratio of lines that are pure headers/numbers/whitespace
MIN_DOMAIN_SCORE = 0.02      # fraction of domain keywords present; below = off-topic
NEAR_DUPLICATE_THRESHOLD = 2  # SimHash bit-distance; at or below = near-duplicate


class DataQualityFilter:
    """Statistical quality gates for ingested documents.

    Each gate returns (passed: bool, reason: str).  screen() runs all gates
    and returns an aggregated verdict dict.
    """

    def __init__(
        self,
        min_chars: int = MIN_CHARS,
        max_chars: int = MAX_CHARS,
        min_token_diversity: float = MIN_TOKEN_DIVERSITY,
        max_boilerplate_ratio: float = MAX_BOILERPLATE_RATIO,
        min_domain_score: float = MIN_DOMAIN_SCORE,
        check_language: bool = True,
    ):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_token_diversity = min_token_diversity
        self.max_boilerplate_ratio = max_boilerplate_ratio
        self.min_domain_score = min_domain_score
        self.check_language = check_language

    # ── Individual gates ───────────────────────────────────────────────

    def gate_length(self, text: str) -> Tuple[bool, str]:
        """Reject documents that are too short or flag those that are too long."""
        n = len(text)
        if n < self.min_chars:
            return False, f"too_short: {n} chars (min {self.min_chars})"
        if n > self.max_chars:
            # Flag but don't hard-reject — oversized docs may still be valuable
            return True, f"oversized_flagged: {n} chars (max {self.max_chars})"
        return True, "ok"

    def gate_token_diversity(self, text: str) -> Tuple[bool, str]:
        """Reject repetitive boilerplate by measuring unique/total token ratio."""
        tokens = text.lower().split()
        if not tokens:
            return False, "empty_tokens"
        ratio = len(set(tokens)) / len(tokens)
        if ratio < self.min_token_diversity:
            return False, f"low_diversity: {ratio:.3f} (min {self.min_token_diversity})"
        return True, f"diversity_ok: {ratio:.3f}"

    def gate_boilerplate(self, text: str) -> Tuple[bool, str]:
        """Reject documents where most lines are headers, numbers, or whitespace.

        Blank lines are excluded entirely (from both the boilerplate count and
        the total) — they're meaningful structural separators in markdown
        output (paragraph/section breaks), not evidence of boilerplate the way
        repeated blank space is in a raw flat-text page dump.
        """
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            return False, "no_lines"

        boilerplate_count = 0
        for line in lines:
            stripped = line.strip()
            # Pure numbers/punctuation, short all-caps headers, page markers
            if (
                re.fullmatch(r'[\d\s\.\-–—/\\]+', stripped)
                or (len(stripped) < 40 and stripped.isupper())
                or re.fullmatch(r'(page\s*\d+|\d+\s*of\s*\d+)', stripped, re.I)
            ):
                boilerplate_count += 1

        ratio = boilerplate_count / len(lines)
        if ratio > self.max_boilerplate_ratio:
            return False, f"high_boilerplate: {ratio:.2%} of lines (max {self.max_boilerplate_ratio:.0%})"
        return True, f"boilerplate_ok: {ratio:.2%}"

    def gate_language(self, text: str) -> Tuple[bool, str]:
        """Reject non-English documents. Requires langdetect."""
        if not self.check_language:
            return True, "language_check_disabled"
        try:
            from langdetect import detect, LangDetectException
            sample = text[:3000]  # detect on first 3K chars — fast and sufficient
            lang = detect(sample)
            if lang != "en":
                return False, f"non_english: detected '{lang}'"
            return True, f"language_ok: {lang}"
        except ImportError:
            logger.warning("langdetect not installed — skipping language gate")
            return True, "language_check_skipped"
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return True, "language_check_failed"

    def gate_domain_relevance(self, text: str) -> Tuple[bool, str]:
        """Reject off-topic documents by scoring against OT/ICS keyword seed set."""
        text_lower = text.lower()
        hits = sum(1 for kw in DOMAIN_KEYWORDS if kw in text_lower)
        score = hits / len(DOMAIN_KEYWORDS)
        if score < self.min_domain_score:
            return False, f"off_topic: domain_score={score:.3f} (min {self.min_domain_score})"
        return True, f"domain_ok: score={score:.3f} ({hits}/{len(DOMAIN_KEYWORDS)} keywords)"

    # ── Corpus-level near-duplicate detection ──────────────────────────

    @staticmethod
    def simhash(text: str, bits: int = 64) -> int:
        """Compute a simple SimHash fingerprint for near-duplicate detection."""
        tokens = re.findall(r'\w+', text.lower())
        if not tokens:
            return 0

        vector = [0] * bits
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            for i in range(bits):
                if h & (1 << i):
                    vector[i] += 1
                else:
                    vector[i] -= 1

        fingerprint = 0
        for i in range(bits):
            if vector[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint

    @staticmethod
    def hamming_distance(h1: int, h2: int) -> int:
        """Count differing bits between two SimHash fingerprints."""
        return bin(h1 ^ h2).count("1")

    def is_near_duplicate(self, text: str, known_hashes: List[int]) -> Tuple[bool, int]:
        """Check if text is a near-duplicate of any document in known_hashes.

        Returns (is_duplicate: bool, fingerprint: int).
        """
        fp = self.simhash(text)
        for existing in known_hashes:
            if self.hamming_distance(fp, existing) <= NEAR_DUPLICATE_THRESHOLD:
                return True, fp
        return False, fp

    # ── Main screen method ─────────────────────────────────────────────

    def screen(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        known_hashes: Optional[List[int]] = None,
    ) -> Dict:
        """Run all quality gates and return a verdict dict.

        Args:
            text: Raw document text.
            metadata: Optional dict with title, source_name, content_type, etc.
            known_hashes: Optional list of SimHash fingerprints for dedup check.

        Returns:
            {
                "pass": bool,
                "reason": str,          # first failing gate, or "all_gates_passed"
                "flags": [str],         # non-fatal warnings (e.g. oversized)
                "gates": {gate: str},   # per-gate results
                "simhash": int,         # fingerprint for caller to store
            }
        """
        metadata = metadata or {}
        known_hashes = known_hashes or []
        gates = {}
        flags = []

        # Gate 1: length
        passed, msg = self.gate_length(text)
        gates["length"] = msg
        if "oversized_flagged" in msg:
            flags.append(msg)
        elif not passed:
            return self._reject("length", msg, gates, flags)

        # Gate 2: token diversity
        passed, msg = self.gate_token_diversity(text)
        gates["token_diversity"] = msg
        if not passed:
            return self._reject("token_diversity", msg, gates, flags)

        # Gate 3: boilerplate ratio
        passed, msg = self.gate_boilerplate(text)
        gates["boilerplate"] = msg
        if not passed:
            return self._reject("boilerplate", msg, gates, flags)

        # Gate 4: language
        passed, msg = self.gate_language(text)
        gates["language"] = msg
        if not passed:
            return self._reject("language", msg, gates, flags)

        # Gate 5: domain relevance
        passed, msg = self.gate_domain_relevance(text)
        gates["domain_relevance"] = msg
        if not passed:
            return self._reject("domain_relevance", msg, gates, flags)

        # Gate 6: near-duplicate (optional — only runs if known_hashes provided)
        is_dup, fp = self.is_near_duplicate(text, known_hashes)
        gates["near_duplicate"] = f"duplicate_found" if is_dup else "unique"
        if is_dup:
            return self._reject("near_duplicate", "near_duplicate: simhash match found", gates, flags, simhash=fp)

        _, fp = False, self.simhash(text)

        return {
            "pass": True,
            "reason": "all_gates_passed",
            "flags": flags,
            "gates": gates,
            "simhash": fp,
        }

    @staticmethod
    def _reject(gate: str, reason: str, gates: Dict, flags: List, simhash: int = 0) -> Dict:
        return {
            "pass": False,
            "reason": reason,
            "failed_gate": gate,
            "flags": flags,
            "gates": gates,
            "simhash": simhash,
        }

    # ── Batch audit (notebook / CLI use) ──────────────────────────────

    def audit_corpus(self, records: List[Dict]) -> List[Dict]:
        """Run screen() over a list of {'text': ..., 'metadata': ...} dicts.

        Useful for running a quality audit from a notebook without touching
        the pipeline.  Returns the input records annotated with quality results.
        """
        known_hashes: List[int] = []
        results = []
        for rec in records:
            text = rec.get("text") or rec.get("transcript", "")
            meta = rec.get("metadata", {})
            result = self.screen(text, meta, known_hashes)
            if result.get("simhash"):
                known_hashes.append(result["simhash"])
            results.append({**rec, "quality": result})
        return results
