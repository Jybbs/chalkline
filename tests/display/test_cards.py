"""
Tests for HTML card builders.
"""

from chalkline.collection.schemas import Posting
from chalkline.display.loaders    import Layout


class TestPostingCard:
    """
    Validate metadata rendering in posting cards.
    """

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
