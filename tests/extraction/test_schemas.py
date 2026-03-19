"""
Tests for extraction schemas.

Validates `OnetSkillType` concrete type membership.
"""

from chalkline.extraction.schemas import OnetSkillType


class TestOnetSchemas:
    """
    Validate lexicon data schemas for O*NET entries.
    """

    def test_concrete_types_count(self):
        """
        Exactly four element types feed the normalization index.
        """
        assert sum(m.is_concrete for m in OnetSkillType) == 4
