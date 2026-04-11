"""
Text extraction from PDF documents.

Handles PDF extraction via `pdfplumber` with shared text cleaning for
downstream tokenization. Plain-text files need no dedicated extractor
because `Path.read_text` with tolerant encoding suffices.
"""

from loguru  import logger
from pathlib import Path
from re      import sub


def clean_text(raw: str) -> str:
    """
    Normalize extracted text for downstream tokenization.

    Strips page number markers, collapses runs of whitespace and newlines to
    single spaces, and removes non-ASCII artifacts that commonly appear in
    PDF extraction output.

    Args:
        raw: Unprocessed text from any extraction source.

    Returns:
        Cleaned text, or empty string if input is empty.
    """
    if not raw:
        return ""

    text = sub(r"(?m)^\s*\d+\s*$", "", raw)
    text = sub(r"[^\x00-\x7f]", " ", text)
    text = sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf(path: Path, label: str | None = None) -> str:
    """
    Extract raw text from a PDF file via `pdfplumber`.

    Iterates all pages and joins their text content with newlines. Handles
    multi-column layouts and embedded headers/footers through `pdfplumber`'s
    default text extraction, which reads in visual order.

    Args:
        path: Filesystem path to the PDF file.

    Returns:
        Concatenated page text, or empty string if no text is found.
    """
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        logger.debug(f"Extracting {label or path.name}: {len(pdf.pages)} pages")
        return "\n".join(
            text for page in pdf.pages
            if (text := page.extract_text())
        )
