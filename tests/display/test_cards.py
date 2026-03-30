"""
Tests for HTML card builders.
"""

from datetime import date

from chalkline.collection.schemas import Posting
from chalkline.display.loaders    import Layout


class TestEmployerCard:
    """
    Validate conditional link rendering in employer cards.
    """

    def test_career_url_included(self, layout: Layout):
        """
        Non-empty career URL renders both posting and career links.
        """
        html = layout.employer_card(
            career_url  = "https://example.com/careers",
            member_type = "General",
            name        = "Acme Corp",
            posting_url = "https://example.com/posting"
        ).text
        assert "View Posting" in html
        assert "Career Page" in html

    def test_career_url_omitted(self, layout: Layout):
        """
        Empty career URL renders only the posting link.
        """
        html = layout.employer_card(
            career_url  = "",
            member_type = "General",
            name        = "Acme Corp",
            posting_url = "https://example.com/posting"
        ).text
        assert "View Posting" in html
        assert "Career Page" not in html


class TestPostingCard:
    """
    Validate description truncation and metadata rendering in
    posting cards.
    """

    def test_long_truncates_at_word(self, layout: Layout):
        """
        Description over 200 characters truncates at the last word
        boundary before the limit and appends an ellipsis.
        """
        description = "alpha " * 40
        html = layout.posting_card(Posting(
            company     = "Test Co",
            date_posted = None,
            description = description,
            source_url  = "https://example.com",
            title       = "Worker"
        )).text
        assert "..." in html
        assert "al..." not in html

    def test_null_location_fallback(self, layout: Layout):
        """
        Missing location falls back to "Maine" in the metadata line.
        """
        html = layout.posting_card(Posting(
            company     = "Test Co",
            date_posted = None,
            description = "x" * 50,
            source_url  = "https://example.com",
            title       = "Worker"
        )).text
        assert "Maine" in html

    def test_short_preserves(self, layout: Layout):
        """
        Description under 200 characters passes through without
        truncation or ellipsis.
        """
        description = (
            "A short but valid description for testing"
            " purposes, well over fifty."
        )
        html = layout.posting_card(Posting(
            company     = "Test Co",
            date_posted = date(2026, 1, 15),
            description = description,
            source_url  = "https://example.com",
            title       = "Worker"
        )).text
        assert description in html
        assert "..." not in html
