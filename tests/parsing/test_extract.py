"""
Tests for document text extraction and cleaning.

Validates PDF extraction via `pdfplumber`, plain-text reading with
tolerant encoding, and text normalization for downstream tokenization.
"""

from pathlib import Path

from chalkline.parsing.extract import clean_text, extract_pdf, extract_text

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


class TestCleanText:
    """
    Validate text normalization for downstream tokenization.
    """

    def test_collapses_whitespace(self):
        """
        Multiple spaces and newlines collapse to single spaces.
        """
        assert clean_text("hello   world\n\nfoo") == "hello world foo"

    def test_empty_input_returns_empty(self):
        """
        An empty input returns an empty string without raising.
        """
        assert clean_text("") == ""

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

    def test_known_text_from_pdf(self):
        """
        A sample resume PDF returns its known text content with
        no binary artifacts.
        """
        result = extract_pdf(FIXTURES / "sample.pdf")
        assert "Walt Amper" in result
        assert "Journeyman Electrician" in result.replace("\n", " ")
        assert "OSHA 30" in result
        assert "\x00" not in result

    def test_empty_pdf(self):
        """
        A PDF with no extractable text returns an empty string.
        """
        assert extract_pdf(FIXTURES / "empty.pdf") == ""

    def test_multi_page_pdf(self):
        """
        Text from multiple pages is joined with newlines.
        """
        result = extract_pdf(FIXTURES / "multi_page.pdf")
        assert "Page 1" in result
        assert "Page 2" in result


class TestExtractText:
    """
    Validate plain-text file reading with tolerant encoding.
    """

    def test_contents_preserved(self, tmp_path: Path):
        """
        A plain-text file returns its contents faithfully.
        """
        path = tmp_path / "sample.txt"
        path.write_text("Hello, world!", encoding="utf-8")
        assert extract_text(path) == "Hello, world!"

    def test_empty_file(self, tmp_path: Path):
        """
        An empty file returns an empty string without raising.
        """
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")
        assert extract_text(path) == ""

    def test_windows_encoding_tolerant(self, tmp_path: Path):
        """
        Windows-1252 bytes are replaced rather than raising.
        """
        path = tmp_path / "windows.txt"
        path.write_bytes(b"smart \x93quotes\x94")
        result = extract_text(path)
        assert "smart" in result
        assert "quotes" in result
