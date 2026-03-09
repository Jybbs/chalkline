"""
O*NET occupation and skill schemas for lexicon validation.

Defines the element type taxonomy, skill entry structure, and occupation
profile model that `onet.json` is validated against at load time.
"""

from enum     import StrEnum
from pydantic import BaseModel, Field

from chalkline import NonEmptyStr


class OnetSkillType(StrEnum):
    """
    O*NET element types across the 21 stakeholder SOC codes.

    Concrete types (`ALTERNATE_TITLE`, `DWA`, `TASK`, `TECHNOLOGY`,
    `TOOL`) feed the normalization index. Abstract KSA types (`ABILITY`,
    `KNOWLEDGE`, `SKILL`) are excluded from normalization but remain
    available for occupation-level Jaccard matching.
    """

    ABILITY         = "ability"
    ALTERNATE_TITLE = "alternate_title"
    DWA             = "dwa"
    KNOWLEDGE       = "knowledge"
    SKILL           = "skill"
    TASK            = "task"
    TECHNOLOGY      = "technology"
    TOOL            = "tool"

    @property
    def is_concrete(self) -> bool:
        """
        Whether this type feeds the normalization index.

        Concrete types carry matchable text that appears in posting
        descriptions. Abstract KSA types are excluded from normalization but
        remain available for Jaccard matching.
        """
        return self not in {
            OnetSkillType.ABILITY,
            OnetSkillType.KNOWLEDGE,
            OnetSkillType.SKILL
        }

    @property
    def is_decomposable(self) -> bool:
        """
        Whether this type requires POS-based chunking.

        Tasks and DWAs are sentence-length entries that must be decomposed
        into sub-phrases for Aho-Corasick matching.
        """
        return self in {
            OnetSkillType.DWA,
            OnetSkillType.TASK
        }


class OnetSkill(BaseModel, extra="forbid"):
    """
    A single skill entry within an O*NET occupation.

    Concrete element types carry `None` for `importance` and `level`,
    whereas abstract KSA types populate both fields with numeric ratings.
    """

    name : NonEmptyStr
    type : OnetSkillType

    importance : float | None = None
    level      : float | None = None


class OnetOccupation(BaseModel, extra="forbid"):
    """
    An O*NET occupation with its full skill profile.

    Each of the 21 stakeholder SOC codes maps to one occupation containing
    skills across all 8 element types.
    """

    job_zone : int = Field(ge=1, le=5)
    sector   : NonEmptyStr
    skills   : list[OnetSkill]
    soc_code : NonEmptyStr
    title    : NonEmptyStr
