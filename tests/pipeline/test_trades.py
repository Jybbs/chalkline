"""
Tests for trade index prefix matching.
"""

from chalkline.pipeline.trades import _prefix_set


class TestPrefixSet:
    """
    Validate 4-character prefix extraction for apprenticeship and
    program matching.
    """

    def test_inflection(self):
        """
        4-char prefix catches inflectional variants that the trade
        index relies on for apprenticeship and program matching.
        """
        assert _prefix_set("welding") & _prefix_set("Welder")
        assert _prefix_set("electrical wiring") & _prefix_set("Electrician")
        assert not _prefix_set("scaffolding") & _prefix_set("concrete")

    def test_short_words(self):
        """
        Words shorter than 4 characters are excluded from the prefix
        set to avoid false positives on articles and prepositions.
        """
        assert _prefix_set("the NEC code") == {"code"}
        assert _prefix_set("on") == set()
