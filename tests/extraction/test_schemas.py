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
        Exactly four element types feed the normalization index.
        """
        assert sum(m.is_concrete for m in OnetSkillType) == 4

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

    @mark.parametrize(("cls", "kwargs"), [
        (OnetOccupation, SAMPLE_OCCUPATION),
        (OnetSkill, SAMPLE_SKILL),
        (CorpusStatistics, {
            "matrix_sparsity"         : 0.9,
            "mean_skills_per_posting" : 5.0,
            "skill_frequency"         : {"welding" : 3},
            "vocabulary_size"         : 10
        })
    ])
    def test_extra_fields(self, cls: type, kwargs: dict):
        """
        Unknown fields raise `ValidationError` per `extra="forbid"`.
        """
        with raises(Exception, match="Extra inputs"):
            cls(**kwargs, unknown="value")

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

    def test_skill_empty_name(self):
        """
        Empty skill names violate the `NonEmptyStr` constraint.
        """
        with raises(Exception):
            OnetSkill(name="", type="technology")

