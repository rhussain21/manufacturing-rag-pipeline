"""
PDFExtractor — download and extract text from PDF files.

PDFExtractor itself is the legacy backend (PyMuPDF primary, pdfplumber
fallback). The pipeline's primary extraction entry point is the
module-level extract_pdf_text_smart(), which prefers Reducto
(tools/reducto_extractor.py) and only falls back to PDFExtractor when
Reducto fails, is out of credits, or the document exceeds
REDUCTO_MAX_PAGES (cost control, checked for free via local PyMuPDF).

General-purpose PDF handling for the ingestion pipeline. Supports:
  - Downloading PDFs from any URL (with retries, streaming, content-type checks)
  - Extracting text via PyMuPDF (primary) with pdfplumber fallback
  - NUL-byte stripping (prevents PostgreSQL insert failures)
  - arXiv URL helpers for routing decisions

Usage (primary entry point — Reducto with legacy fallback):
    from tools.pdf_extractor import extract_pdf_text_smart

    text, page_count, method = extract_pdf_text_smart("path/to/paper.pdf")

Usage (legacy extractor directly, e.g. for comparison):
    from tools.pdf_extractor import PDFExtractor

    text, page_count = PDFExtractor.extract_text("path/to/paper.pdf")

Usage (download + extract, Reducto-primary):
    extractor = PDFExtractor(pdf_dir="media/pdf")
    result = extractor.download_and_extract(
        pdf_url="https://example.com/paper.pdf",
        title="Some Paper Title",
    )
    if result:
        print(result["text"][:200])
        print(result["pdf_path"])
        print(result["extraction_method"])  # "reducto", "pymupdf", or "pdfplumber"
"""

import logging
import os
import re
import time
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# Reducto is the primary PDF extractor (see extract_pdf_text_smart below).
# Documents longer than this fall back to the legacy extractor directly,
# skipping Reducto entirely, as a cost-control ceiling — page count is
# checked for free via local PyMuPDF before any paid call is made.
REDUCTO_MAX_PAGES = 1500


class PDFExtractor:
    """Download PDFs and extract full text."""

    USER_AGENT = "Mozilla/5.0 (compatible; IndustrySignalsBot/1.0)"
    MAX_RETRIES = 2
    RETRY_DELAY = 3.0  # seconds between retries
    TIMEOUT = 60  # PDF downloads can be large

    def __init__(self, pdf_dir: str = "media/pdf"):
        self.pdf_dir = pdf_dir
        os.makedirs(self.pdf_dir, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────

    def download_and_extract(
        self,
        pdf_url: str,
        title: str = "Untitled",
        source_url: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Download a PDF and extract full text.

        Args:
            pdf_url: Direct URL to the PDF file.
            title: Paper/document title for filename generation.
            source_url: Original page URL (e.g. arXiv /abs/ page). Stored in result.

        Returns:
            Dict with keys: text, pdf_path, source_url, pdf_url, page_count,
            char_count, extraction_method ("reducto", "pymupdf", or
            "pdfplumber") — or None on failure.
        """
        if not pdf_url:
            logger.error(f"No PDF URL provided for: {title}")
            return None

        # Download PDF
        pdf_path = self._download_pdf(pdf_url, title)
        if not pdf_path:
            return None

        # Extract text — Reducto primary, legacy fallback (see function docstring)
        text, page_count, extraction_method = extract_pdf_text_smart(pdf_path)
        if not text or len(text.strip()) < 200:
            logger.warning(
                f"Insufficient text extracted from PDF "
                f"({len(text) if text else 0} chars): {title}"
            )
            return None

        logger.info(
            f"PDF extracted via {extraction_method}: {page_count} pages, "
            f"{len(text)} chars — {title[:60]}"
        )

        return {
            "text": text,
            "pdf_path": pdf_path,
            "source_url": source_url or pdf_url,
            "pdf_url": pdf_url,
            "page_count": page_count,
            "extraction_method": extraction_method,
            "char_count": len(text),
            "pdf_pub_date": self.extract_pub_date(pdf_path),
        }

    @staticmethod
    def extract_pub_date(pdf_path: str) -> str:
        """Extract publication date from PDF document metadata.

        Tries PyMuPDF doc.metadata fields (creationDate, modDate) and
        parses the PDF date format: D:YYYYMMDDHHmmSS

        Returns ISO date string (YYYY-MM-DD) or empty string.
        """
        try:
            import fitz
            doc = fitz.open(pdf_path)
            meta = doc.metadata or {}
            doc.close()
            for key in ('creationDate', 'modDate'):
                raw = meta.get(key, '')
                if raw and raw.startswith('D:') and len(raw) >= 10:
                    digits = raw[2:10]  # YYYYMMdd
                    if digits.isdigit():
                        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
        except Exception:
            pass
        return ''

    @staticmethod
    def extract_text(pdf_path: str) -> Tuple[str, int]:
        """Extract text from a local PDF file.

        Tries PyMuPDF first (faster, better layout preservation),
        falls back to pdfplumber. Strips NUL bytes.

        Args:
            pdf_path: Path to the PDF file on disk.

        Returns:
            (text, page_count) tuple. Empty string and 0 on failure.
        """
        text, pages, _method = PDFExtractor.extract_text_with_method(pdf_path)
        return text, pages

    @staticmethod
    def extract_text_with_method(pdf_path: str) -> Tuple[str, int, str]:
        """Same as extract_text, but also reports which backend produced the
        result — used for the 'transcription_model' provenance column.

        Returns:
            (text, page_count, method) — method is "pymupdf", "pdfplumber",
            or "failed".
        """
        text, pages = PDFExtractor._extract_with_pymupdf(pdf_path)
        if text and len(text.strip()) > 200:
            return text, pages, "pymupdf"

        logger.info(f"PyMuPDF extraction insufficient, trying pdfplumber: {pdf_path}")
        text, pages = PDFExtractor._extract_with_pdfplumber(pdf_path)
        if text and len(text.strip()) > 200:
            return text, pages, "pdfplumber"
        return text, pages, "failed"

    # ── arXiv URL helpers ─────────────────────────────────────────────

    @staticmethod
    def is_arxiv_url(url: str) -> bool:
        """Check if a URL points to arXiv."""
        if not url:
            return False
        host = urlparse(url).hostname or ""
        return "arxiv.org" in host

    @staticmethod
    def construct_arxiv_pdf_url(abs_url: str) -> str:
        """Construct PDF URL from an arXiv /abs/ URL.

        http://arxiv.org/abs/2401.12345v1 → http://arxiv.org/pdf/2401.12345v1
        """
        if not abs_url:
            return ""
        return re.sub(r'/abs/', '/pdf/', abs_url)

    # ── PDF download ───────────────────────────────────────────────────

    def _download_pdf(self, pdf_url: str, title: str) -> Optional[str]:
        """Download PDF to disk with retries. Returns filepath or None."""
        # Build filename from title
        clean_title = re.sub(r'[^\w\s-]', '', title)
        clean_title = re.sub(r'\s+', '_', clean_title).strip('_')
        if len(clean_title) > 180:
            clean_title = clean_title[:180]
        filename = f"{clean_title}.pdf"
        filepath = os.path.join(self.pdf_dir, filename)

        # Skip if already downloaded
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            logger.info(f"PDF already on disk: {filename}")
            return filepath

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                logger.debug(f"PDF download attempt {attempt}/{self.MAX_RETRIES}: {pdf_url}")
                resp = requests.get(
                    pdf_url,
                    timeout=self.TIMEOUT,
                    headers={"User-Agent": self.USER_AGENT},
                    stream=True,
                )
                resp.raise_for_status()

                # Verify we actually got a PDF
                content_type = resp.headers.get("Content-Type", "")
                if "pdf" not in content_type and "octet-stream" not in content_type:
                    logger.warning(
                        f"Server returned non-PDF content-type: {content_type} "
                        f"for {pdf_url}"
                    )
                    # Still try — some servers serve PDFs with wrong content-type

                # Stream to disk
                with open(filepath, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)

                file_size = os.path.getsize(filepath)
                if file_size < 1000:
                    logger.warning(f"PDF suspiciously small ({file_size} bytes), retrying")
                    os.remove(filepath)
                    continue

                logger.info(f"PDF downloaded: {filename} ({file_size / 1024:.0f} KB)")
                return filepath

            except requests.RequestException as e:
                logger.warning(f"PDF download error (attempt {attempt}): {e}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY)

        logger.error(f"PDF download failed after {self.MAX_RETRIES} attempts: {pdf_url}")
        return None

    # ── Text extraction backends ──────────────────────────────────────

    @staticmethod
    def _extract_with_pymupdf(pdf_path: str) -> Tuple[str, int]:
        """Extract text using PyMuPDF (fitz). Returns (text, page_count)."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.debug("PyMuPDF (fitz) not installed")
            return "", 0

        try:
            doc = fitz.open(pdf_path)
            pages = []
            for page in doc:
                page_text = page.get_text("text")
                if page_text.strip():
                    pages.append(page_text)
            doc.close()

            text = "\n\n".join(pages)
            # Light cleanup: collapse excessive whitespace, remove form feeds
            text = re.sub(r'\f', '\n', text)
            text = re.sub(r'\n{4,}', '\n\n\n', text)
            # Strip NUL bytes that some PDFs contain (breaks PostgreSQL)
            text = text.replace('\x00', '')

            return text, len(pages)

        except Exception as e:
            logger.warning(f"PyMuPDF extraction error: {e}")
            return "", 0

    @staticmethod
    def _extract_with_pdfplumber(pdf_path: str) -> Tuple[str, int]:
        """Extract text using pdfplumber (fallback). Returns (text, page_count)."""
        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber not installed — cannot extract PDF text")
            return "", 0

        try:
            pages = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        pages.append(page_text)

            text = "\n\n".join(pages)
            # Strip NUL bytes
            text = text.replace('\x00', '')
            return text, len(pages)

        except Exception as e:
            logger.warning(f"pdfplumber extraction error: {e}")
            return "", 0


# ── Primary extraction entry point (Reducto, with legacy fallback) ───────────

def _get_page_count(pdf_path: str) -> Optional[int]:
    """Free, local page count via PyMuPDF — no Reducto call involved.
    Returns None if the file can't be opened."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        pages = doc.page_count
        doc.close()
        return pages
    except Exception as e:
        logger.warning(f"Could not read page count for {pdf_path}: {e}")
        return None


def extract_pdf_text_smart(pdf_path: str) -> Tuple[str, int, str]:
    """Extract PDF text, preferring Reducto with automatic legacy fallback.

    Routing:
      1. Page count is checked for free via local PyMuPDF first.
      2. If it exceeds REDUCTO_MAX_PAGES, Reducto is skipped entirely
         (cost control) and the legacy extractor is used directly.
      3. Otherwise Reducto is tried first. If it fails for any reason
         (out of credits, API/network error, etc.), falls back to the
         legacy extractor rather than losing the document.

    Returns:
        (text, page_count, method) — method is "reducto", "pymupdf",
        "pdfplumber", or "failed".
    """
    from tools.reducto_extractor import ReductoExtractor

    page_count = _get_page_count(pdf_path)

    if page_count is not None and page_count > REDUCTO_MAX_PAGES:
        logger.info(
            f"PDF exceeds {REDUCTO_MAX_PAGES}-page Reducto threshold "
            f"({page_count} pages) — using legacy extractor: {pdf_path}"
        )
        return PDFExtractor.extract_text_with_method(pdf_path)

    text, reducto_pages = ReductoExtractor.extract_text(pdf_path)
    if text and len(text.strip()) > 200:
        return text, reducto_pages, "reducto"

    logger.warning(f"Reducto extraction failed or empty, falling back to legacy: {pdf_path}")
    return PDFExtractor.extract_text_with_method(pdf_path)
