"""
Schemas for lexicon validation, skill extraction, and corpus
statistics.

Defines confidence tiers for Aho-Corasick matching, the certification
model for CareerOneStop entries, corpus-level statistics computed after
vectorization, and the O*NET element type taxonomy, skill entry structure,
and occupation profile model that `onet.json` is validated against at load
time.
"""

from enum     import StrEnum
from pydantic import BaseModel, Field

from chalkline import NonEmptyStr


class Certification(BaseModel, extra="forbid"):
    """
    A CareerOneStop certification linked to stakeholder SOC codes.

    Each certification contributes its name and acronym (when present
    and non-ambiguous) as Aho-Corasick patterns. Descriptions are
    decomposed into matchable sub-phrases via POS-based NP/VP
    chunking at curation time.
    """

    name      : NonEmptyStr
    soc_codes : list[str]

    acronym      : str | None       = None
    organization : str | None       = None
    phrases      : list[str] | None = None
    type         : str | None       = None


class ConfidenceTier(StrEnum):
    """
    Match confidence for Aho-Corasick surface form hits.

    Multi-word matches are the most specific, abbreviation matches
    carry moderate confidence, and single-word matches are the most
    ambiguous. When two matches conflict at the same position, the
    higher-confidence tier wins.
    """

    ABBREVIATION = "abbreviation"
    MULTI_WORD   = "multi_word"
    SINGLE_WORD  = "single_word"


class CorpusStatistics(BaseModel, extra="forbid"):
    """
    Aggregate statistics computed after IDF-weighted vectorization.

    Captures vocabulary coverage, matrix density, and per-posting
    skill counts for downstream diagnostics and threshold tuning.
    """

    matrix_sparsity         : float
    mean_skills_per_posting : float
    skill_frequency         : dict[str, int]
    vocabulary_size         : int


class OnetSkillType(StrEnum):
    """
    O*NET element types across the 21 stakeholder SOC codes.

    Concrete types (`DWA`, `TASK`, `TECHNOLOGY`, `TOOL`) feed the
    normalization index. Abstract KSA types (`ABILITY`, `KNOWLEDGE`,
    `SKILL`) are excluded from normalization but remain available for
    occupation-level Jaccard matching.
    """

    ABILITY    = "ability"
    DWA        = "dwa"
    KNOWLEDGE  = "knowledge"
    SKILL      = "skill"
    TASK       = "task"
    TECHNOLOGY = "technology"
    TOOL       = "tool"

    @property
    def is_concrete(self) -> bool:
        """
        Whether this type feeds the normalization index.

        Concrete types carry matchable text that appears in posting
        descriptions. Abstract KSA types are excluded from normalization
        but remain available for Jaccard matching.

        Returns:
            `True` if this type is concrete, `False` for KSA types.
        """
        return self not in {
            OnetSkillType.ABILITY,
            OnetSkillType.KNOWLEDGE,
            OnetSkillType.SKILL
        }


class OnetSkill(BaseModel, extra="forbid"):
    """
    A single skill entry within an O*NET occupation.

    Concrete element types carry `None` for `importance` and `level`,
    whereas abstract KSA types populate both fields with numeric
    ratings. Decomposable types (Tasks, DWAs) carry pre-computed
    sub-phrases from POS-based chunking performed at curation time.
    """

    name : NonEmptyStr
    type : OnetSkillType

    importance : float | None     = None
    level      : float | None     = None
    phrases    : list[str] | None = None


class OnetOccupation(BaseModel, extra="forbid"):
    """
    An O*NET occupation with its full skill profile.

    Each of the 21 stakeholder SOC codes maps to one occupation
    containing skills across all 8 element types.
    """

    job_zone : int = Field(ge=1, le=5)
    sector   : NonEmptyStr
    skills   : list[OnetSkill]
    soc_code : NonEmptyStr
    title    : NonEmptyStr
