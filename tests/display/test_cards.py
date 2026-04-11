"""
Tests for HTML card builders.
"""

from typing import Callable

from chalkline.collection.schemas import Posting
from chalkline.display.loaders    import Layout


class TestPostingCard:
    """
    Validate metadata rendering in posting cards.
    """

    def test_null_location_fallback(
        self,
        layout  : Layout,
        posting : Callable[..., Posting]
    ):
        """
        Missing location falls back to "Maine" in the metadata line.
        """
        assert "Maine" in layout.posting_card(posting(date_posted=None)).text
