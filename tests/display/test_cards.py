"""
Tests for HTML card builders and layout utilities.
"""

from typing import Callable

from pytest import mark

from chalkline.collection.schemas import Posting
from chalkline.display.loaders    import Layout
from chalkline.display.theme      import Theme
from chalkline.pathways.clusters  import Clusters


class TestAnnotate:
    """
    Validate glossary annotation, dedup, and tooltip structure.
    """

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

    def test_first_occurrence_wrapped(self, layout: Layout):
        """
        A known glossary term's first occurrence is wrapped in a
        tooltip span.
        """
        assert "cl-term" in layout.annotate("<p>apprenticeship program</p>")


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


class TestVerdictThresholds:
    """
    Validate confidence-to-verdict mapping in the sidebar identity
    card via `Layout.you_are_here`.
    """

    @mark.parametrize(("confidence", "expected"), [
        (0,   "Exploratory"),
        (39,  "Exploratory"),
        (40,  "Multiple good fits"),
        (69,  "Multiple good fits"),
        (70,  "Strong match"),
        (100, "Strong match")
    ])
    def test_verdict_label(
        self,
        clusters   : Clusters,
        confidence : int,
        expected   : str,
        layout     : Layout,
        theme      : Theme
    ):
        """
        Bisect thresholds (40, 70) partition confidence into three
        verdict buckets rendered in the sidebar card.
        """
        profile = next(iter(clusters.values()))
        html    = layout.you_are_here(
            confidence    = confidence,
            n_advancement = 1,
            n_lateral     = 1,
            profile       = profile,
            theme         = theme
        ).text
        assert expected in html
