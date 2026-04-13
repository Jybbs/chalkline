"""
Tests for HTML card builders and layout utilities.
"""

from pytest import mark
from typing import Callable

from chalkline.collection.schemas import Posting
from chalkline.display.loaders    import Layout


class TestAnnotate:
    """
    Validate glossary annotation, dedup, and tooltip structure.
    """

    def test_first_occurrence_wrapped(self, layout: Layout):
        """
        A known glossary term's first occurrence is wrapped in a
        tooltip span.
        """
        assert "cl-term" in layout.annotate("<p>apprenticeship program</p>")

    @mark.parametrize("html", [
        "<p>BLS and Bureau of Labor Statistics</p>",
        "<p>apprenticeship and apprenticeship</p>"
    ], ids=["alias_dedup", "duplicate_skipped"])
    def test_single_tooltip(self, layout: Layout, html: str):
        """
        Alias/canonical pairs and repeated terms both produce
        exactly one tooltip, preventing cluttered annotations.
        """
        assert layout.annotate(html).count("cl-term") == 1


class TestPostingCard:
    """
    Validate metadata rendering in posting cards.
    """

    def test_card_renders_metadata(
        self,
        layout  : Layout,
        posting : Callable[..., Posting]
    ):
        """
        Card includes the title, company, and formatted date when all
        fields are present.
        """
        card = layout.posting_card(posting(
            company = "Cianbro",
            title   = "Electrician"
        ))
        assert "Electrician" in card.text
        assert "Cianbro" in card.text
        assert "Jan 01, 2026" in card.text

    def test_null_location_fallback(
        self,
        layout  : Layout,
        posting : Callable[..., Posting]
    ):
        """
        Missing location falls back to "Maine" in the metadata line.
        """
        assert "Maine" in layout.posting_card(posting(date_posted=None)).text


class TestStatRowColumns:
    """
    Validate ceiling-division grid column computation.
    """

    @mark.parametrize(("n_tiles", "rows", "expected_cols"), [
        (1, 1, 1),
        (5, 1, 5),
        (5, 2, 3),
        (6, 2, 3),
        (7, 2, 4)
    ])
    def test_column_count(
        self,
        expected_cols : int,
        layout        : Layout,
        n_tiles       : int,
        rows          : int
    ):
        """
        `ceil(n_tiles / rows)` via `-(-n // rows)` sets the CSS
        grid-template-columns repeat count.
        """
        pairs = [(f"L{i}", f"V{i}") for i in range(n_tiles)]
        html  = str(layout._stat_row(pairs, rows))
        assert f"repeat({expected_cols},1fr)" in html


