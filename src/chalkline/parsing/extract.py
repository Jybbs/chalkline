"""
Text extraction from PDF and plain-text documents.

Handles PDF extraction via `pdfplumber` and plain-text file reading,
with shared text cleaning for downstream tokenization. Each extractor
returns clean text suitable for skill extraction in CL-06.
"""

import re

from pathlib import Path

import pdfplumber


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

    text = re.sub(r"(?m)^\s*\d+\s*$", "", raw)
    text = re.sub(r"[^\x00-\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            if text := page.extract_text():
                pages.append(text)
    return "\n".join(pages)


def extract_text(path: Path) -> str:
    """
    Read a plain-text file with tolerant encoding.

    Uses UTF-8 with replacement to handle Windows-1252 artifacts that
    commonly appear in job posting exports and older resume formats.

    Args:
        path: Filesystem path to the text file.

    Returns:
        File contents as a string.
    """
    return path.read_text(encoding="utf-8", errors="replace")
