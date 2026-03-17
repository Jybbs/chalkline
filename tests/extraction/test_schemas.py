"""
Tests for extraction schemas.

Validates `ConfidenceTier` ordering and `OnetSkillType` concrete type
membership.
"""

from chalkline.extraction.schemas import ConfidenceTier, OnetSkillType


class TestOnetSchemas:
    """
    Validate lexicon data schemas for O*NET entries.
    """

    def test_concrete_types_count(self):
        """
        Exactly four element types feed the normalization index.
        """
        assert sum(m.is_concrete for m in OnetSkillType) == 4

    def test_confidence_tier_ordering(self):
        """
        Tier values sort so that higher-confidence tiers compare
        lower, matching the `_match` priority logic.
        """
        assert (
            ConfidenceTier.ABBREVIATION.value
            < ConfidenceTier.MULTI_WORD.value
            < ConfidenceTier.SINGLE_WORD.value
        )
