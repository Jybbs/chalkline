"""
Text extraction from PDF documents.

Handles PDF extraction via `pdfplumber` with shared text cleaning
for downstream tokenization. Plain-text files need no dedicated
extractor because `Path.read_text` with tolerant encoding suffices.
"""

import pdfplumber

from pathlib import Path
from re      import sub


def clean_text(raw: str) -> str:
    """
    Normalize extracted text for downstream tokenization.

    Strips page number markers, collapses runs of whitespace and
    newlines to single spaces, and removes non-ASCII artifacts that
    commonly appear in PDF extraction output.

    Args:
        raw: Unprocessed text from any extraction source.

    Returns:
        Cleaned text, or empty string if input is empty.
    """
    if not raw:
        return ""

    return sub(
        r"\s+", " ",
        sub(r"[^\x00-\x7f]", " ", sub(r"(?m)^\s*\d+\s*$", "", raw))
    ).strip()


def extract_pdf(path: Path) -> str:
    """
    Extract raw text from a PDF file via `pdfplumber`.

    Iterates all pages and joins their text content with newlines.
    Handles multi-column layouts and embedded headers/footers through
    `pdfplumber`'s default text extraction, which reads in visual
    order.

    Args:
        path: Filesystem path to the PDF file.

    Returns:
        Concatenated page text, or empty string if no text is found.
    """
    with pdfplumber.open(path) as pdf:
        return "\n".join(
            text for page in pdf.pages
            if (text := page.extract_text())
        )
