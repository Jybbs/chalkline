"""
Schemas for lexicon validation and skill extraction.

Defines the certification model for CareerOneStop entries, the O*NET element
type taxonomy, skill entry structure, and occupation profile model that
`onet.json` is validated against at load time, and the pattern bundle that
groups surface forms for Aho-Corasick matching.
"""

from enum     import StrEnum
from pydantic import BaseModel, Field
from typing   import NamedTuple


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
        descriptions. Abstract KSA types are excluded from normalization but
        remain available for Jaccard matching.

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
    whereas abstract KSA types populate both fields with numeric ratings.
    Decomposable types (Tasks, DWAs) carry pre-computed sub-phrases from
    POS-based chunking performed at curation time.
    """

    name : str
    type : OnetSkillType

    importance : float | None     = None
    level      : float | None     = None
    phrases    : list[str] | None = None


class OnetOccupation(BaseModel, extra="forbid"):
    """
    An O*NET occupation with its full skill profile.

    Each of the 21 stakeholder SOC codes maps to one occupation containing
    skills across all 8 element types.
    """

    job_zone : int = Field(ge=1, le=5)
    sector   : str
    skills   : list[OnetSkill]
    soc_code : str
    title    : str

    @property
    def embedding_text(self) -> str:
        """
        Canonical text representation for sentence encoding.

        Concatenates the occupation title with its Task and DWA element
        names, producing the input string for the sentence transformer
        during SOC vector construction.

        Returns:
            `"{title}: {task1}, {task2}, ..."` format.
        """
        return f"{self.title}: {', '.join(s.name for s in self.task_elements)}"

    @property
    def task_elements(self) -> list[OnetSkill]:
        """
        Task and DWA elements from this occupation's skill profile.

        These are the concrete work-activity elements that describe what
        workers actually do, as opposed to KSA abstractions or
        technology/tool listings.

        Returns:
            Skills where type is `TASK` or `DWA`.
        """
        return [
            s for s in self.skills
            if s.type in {OnetSkillType.TASK, OnetSkillType.DWA}
        ]


class PatternBundle(NamedTuple):
    """
    Complete output of surface form generation.

    Groups the parallel pattern and canonical name lists with the character
    set derived from all patterns, used as a preprocessing filter during
    extraction.
    """

    canonicals : list[str]
    chars      : set[str]
    patterns   : list[str]

    def span_of(self, end: int, idx: int) -> tuple[int, int]:
        """
        Convert `iter_long` output to a start/end span.

        Args:
            end: End position (inclusive) returned by `iter_long`.
            idx: Pattern index returned by `iter_long`.

        Returns:
            A (start, exclusive end) pair for boundary checking.
        """
        return end - len(self.patterns[idx]) + 1, end + 1
