"""
Tests for extraction schemas.

Validates `ConfidenceTier` membership, `CorpusStatistics` extra-field
rejection, `OnetSkillType` enum members, `OnetSkill` field constraints,
and `OnetOccupation` model structure.
"""

from pytest import mark, raises

from chalkline.extraction.schemas import ConfidenceTier, CorpusStatistics
from chalkline.extraction.schemas import OnetOccupation, OnetSkill, OnetSkillType


SAMPLE_SKILL = {
    "name" : "Autodesk AutoCAD",
    "type" : "technology"
}

SAMPLE_OCCUPATION = {
    "job_zone" : 3,
    "sector"   : "Building Construction",
    "skills"   : [SAMPLE_SKILL],
    "soc_code" : "47-2111.00",
    "title"    : "Electricians"
}


class TestOnetSchemas:
    """
    Validate lexicon data schemas for O*NET entries.
    """

    def test_concrete_types_count(self):
        """
        Exactly six element types feed the normalization index.
        """
        assert sum(m.is_concrete for m in OnetSkillType) == 6

    def test_confidence_tier_members(self):
        """
        The three confidence tiers are defined for Aho-Corasick matching.
        """
        assert len(ConfidenceTier) == 3

    def test_confidence_tier_ordering(self):
        """
        Tier values sort so that higher-confidence tiers compare lower,
        matching the `_match` priority logic.
        """
        assert (
            ConfidenceTier.ABBREVIATION.value
            < ConfidenceTier.MULTI_WORD.value
            < ConfidenceTier.SINGLE_WORD.value
        )

    def test_occupation_extra(self):
        """
        Unknown fields raise `ValidationError` per `extra="forbid"`.
        """
        with raises(Exception, match="Extra inputs"):
            OnetOccupation(**SAMPLE_OCCUPATION, unknown="value")

    @mark.parametrize("job_zone", [0, 6])
    def test_occupation_job_zone(self, job_zone):
        """
        Job zone values outside 1-5 are rejected.
        """
        with raises(Exception):
            OnetOccupation(**{**SAMPLE_OCCUPATION, "job_zone": job_zone})

    def test_occupation_missing(self):
        """
        Omitting required fields raises `ValidationError`.
        """
        with raises(Exception):
            OnetOccupation(job_zone=3, title="Test")

    def test_occupation_valid(self):
        """
        A complete occupation validates without error.
        """
        occupation = OnetOccupation(**SAMPLE_OCCUPATION)
        assert occupation.soc_code == "47-2111.00"
        assert len(occupation.skills) == 1

    def test_skill_empty_name(self):
        """
        Empty skill names violate the `NonEmptyStr` constraint.
        """
        with raises(Exception):
            OnetSkill(name="", type="technology")

    def test_skill_extra(self):
        """
        Unknown fields raise `ValidationError` per `extra="forbid"`.
        """
        with raises(Exception, match="Extra inputs"):
            OnetSkill(**SAMPLE_SKILL, unknown="value")

    def test_skill_type_coercion(self):
        """
        Raw strings coerce to enum members through Pydantic.
        """
        assert OnetSkill(name="Test Skill", type="task").type is OnetSkillType.TASK

    def test_skill_type_members(self):
        """
        All nine O*NET element types are defined.
        """
        assert len(OnetSkillType) == 9

    def test_skill_valid_with_ratings(self):
        """
        KSA-typed skills carry populated importance and level.
        """
        s = OnetSkill(
            importance = 3.62,
            level      = 3.88,
            name       = "Critical Thinking",
            type       = "skill"
        )
        assert (s.importance, s.level) == (3.62, 3.88)

    def test_skill_valid_without_ratings(self):
        """
        Concrete-typed skills default importance and level to `None`.
        """
        assert ((s := OnetSkill(**SAMPLE_SKILL)).importance, s.level) == (None, None)

    def test_statistics_extra(self):
        """
        Unknown fields raise `ValidationError` per `extra="forbid"`.
        """
        with raises(Exception, match="Extra inputs"):
            CorpusStatistics(
                matrix_sparsity         = 0.9,
                mean_skills_per_posting = 5.0,
                skill_frequency         = {"welding": 3},
                unknown                 = "value",
                vocabulary_size         = 10
            )
