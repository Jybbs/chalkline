"""
Tests for document text extraction and cleaning.

Validates PDF extraction via `pdfplumber` and text normalization for
downstream tokenization.
"""

from pathlib import Path
from pytest  import mark

from chalkline.parsing.extract import clean_text, extract_pdf


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "parsing"


class TestCleanText:
    """
    Validate text normalization for downstream tokenization.
    """

    def test_collapses_whitespace(self):
        """
        Multiple spaces and newlines collapse to single spaces.
        """
        assert clean_text("hello   world\n\nfoo") == "hello world foo"

    @mark.parametrize("text", ["", "   \n\n  ", "\n  1  \n  2  \n"])
    def test_empty_output(self, text: str):
        """
        Degenerate inputs (empty, whitespace-only, page-numbers-only)
        normalize to an empty string.
        """
        assert clean_text(text) == ""

    def test_strips_non_ascii(self):
        """
        Non-ASCII artifacts are replaced with spaces and collapsed.
        """
        assert "\\x" not in clean_text("hello\x80world")

    def test_strips_page_numbers(self):
        """
        Standalone page number lines are removed.
        """
        result = clean_text("Some text\n  42  \nMore text")
        assert "42" not in result
        assert "Some text" in result


class TestExtractPdf:
    """
    Validate PDF text extraction via `pdfplumber`.
    """

    def test_empty_pdf(self):
        """
        A PDF with no extractable text returns an empty string.
        """
        assert extract_pdf(FIXTURES / "empty.pdf") == ""

    def test_known_text(self):
        """
        A sample resume PDF returns its known text content with no binary
        artifacts.
        """
        result = extract_pdf(FIXTURES / "sample.pdf")
        assert "Walt Amper" in result
        assert "Journeyman Electrician" in result.replace("\n", " ")
        assert "OSHA 30" in result
        assert "\x00" not in result

    def test_multi_page_pdf(self):
        """
        Text from multiple pages is joined with newlines.
        """
        result = extract_pdf(FIXTURES / "multi_page.pdf")
        assert "Page 1" in result
        assert "Page 2" in result

