"""
Tests for shared enrichment context and prefix matching.
"""

from chalkline.pipeline.enrichment import prefix_set


class TestPrefixSet:
    """
    Validate 4-character prefix extraction for apprenticeship and
    program matching.
    """

    def test_inflection(self):
        """
        4-char prefix catches inflectional variants that the enrichment
        pipeline relies on for apprenticeship and program matching.
        """
        assert prefix_set("welding") & prefix_set("Welder")
        assert prefix_set("electrical wiring") & prefix_set("Electrician")
        assert not prefix_set("scaffolding") & prefix_set("concrete")

    def test_short_words(self):
        """
        Words shorter than 4 characters are excluded from the prefix
        set to avoid false positives on articles and prepositions.
        """
        assert prefix_set("the NEC code") == {"code"}
        assert prefix_set("on") == set()
