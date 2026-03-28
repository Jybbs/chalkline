"""
Tests for pathways schemas.

Validates `OnetSkillType` enum membership counts.
"""

from chalkline.pathways.schemas import SkillType


class TestSkillType:
    """
    Validate lexicon data schemas for O*NET entries.
    """

    def test_element_type_count(self):
        """
        Seven element types span the O*NET skill taxonomy.
        """
        assert len(SkillType) == 7
