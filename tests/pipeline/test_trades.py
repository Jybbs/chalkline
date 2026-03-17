"""
Tests for trade index prefix matching.
"""

from chalkline.pipeline.schemas import ApprenticeshipContext
from chalkline.pipeline.trades  import TradeIndex


class TestTradeIndex:
    """
    Validate prefix-based matching between skill strings and
    reference records.
    """

    def test_inflection_match(self):
        """
        4-char prefix overlap catches inflectional variants,
        matching "welding" against a trade titled "Welder".
        """
        trades = TradeIndex(
            apprenticeships = [ApprenticeshipContext(
                min_hours   = 8000,
                prefixes    = {"weld"},
                rapids_code = "001",
                title       = "Welder"
            )],
            programs = []
        )
        apps, _ = trades.match(["welding"])
        assert len(apps) == 1

    def test_no_match(self):
        """
        Unrelated terms produce no matches.
        """
        trades = TradeIndex(
            apprenticeships = [ApprenticeshipContext(
                min_hours   = 8000,
                prefixes    = {"elec"},
                rapids_code = "001",
                title       = "Electrician"
            )],
            programs = []
        )
        apps, _ = trades.match(["concrete"])
        assert apps == []

    def test_short_words_excluded(self):
        """
        Words shorter than 4 characters are excluded from prefix
        matching to avoid false positives on articles and
        prepositions.
        """
        trades = TradeIndex(
            apprenticeships = [ApprenticeshipContext(
                min_hours   = 8000,
                prefixes    = {"code", "spec"},
                rapids_code = "001",
                title       = "NEC Code Specialist"
            )],
            programs = []
        )
        apps, _ = trades.match(["the NEC code"])
        assert len(apps) == 1

        apps, _ = trades.match(["on"])
        assert apps == []
