"""
Pydantic models and enums for the career pathway domain.

Pure data contracts with field declarations, validators, and self-derived
properties. Behavioral containers (`Cluster`, `Clusters`, `Task`) live in
`clusters.py`. No query logic, no builders, no computational imports.
"""

from enum     import StrEnum
from pydantic import BaseModel, Field, field_validator, model_validator


class CareerEdge(BaseModel, extra="forbid"):
    """
    A single edge in the reach view with credential metadata.

    Each edge connects the matched cluster to a neighboring cluster,
    carrying the target cluster ID, the cosine similarity weight, and the
    credentials that bridge the specific transition, filtered by the
    destination_percentile and source_percentile dual-threshold rule.
    """

    cluster_id : int   = Field(ge=0)
    weight     : float = Field(ge=-1, le=1)

    credentials: list[Credential] = Field(default_factory=list)


class Credential(BaseModel, extra="forbid"):
    """
    Unified credential record for the career pathway graph.

    Represents an apprenticeship, certification, or educational program with
    pre-computed `embedding_text` and `label` from curation. The pipeline
    encodes `embedding_text` with the sentence transformer and attaches the
    vector to each instance. The graph stores credentials on edges via
    `model_dump`, which excludes the runtime-only vector. Type-specific
    display fields live in `metadata`.
    """

    embedding_text : str
    kind           : str
    label          : str

    metadata : dict = Field(default_factory=dict)
    vector   : list[float] | None = Field(default=None, exclude=True)

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, v: dict, info) -> dict:
        """
        Enforce required metadata keys per credential kind.
        """
        match info.data.get("kind"):
            case "apprenticeship" : required = {"min_hours", "rapids_code"}
            case "program"        : required = {"credential", "institution", "url"}
            case _                : required = set()
        if missing := required - v.keys():
            raise ValueError(f"Missing keys for {info.data['kind']}: {missing}")
        return v

    @property
    def description(self) -> str:
        """
        Parenthesized display string combining type label and optional
        term hours.
        """
        if self.hours:
            return f"({self.type_label}, {self.hours:,} hours)"
        return f"({self.type_label})"

    @property
    def hours(self) -> int | None:
        """
        Minimum apprenticeship term hours, if applicable.
        """
        return self.metadata.get("min_hours")

    @property
    def type_label(self) -> str:
        """
        Display name for the credential type, falling back to the titlecased
        kind when metadata carries no specific label.
        """
        return self.metadata.get("credential", self.kind.title())


class LaborRecord(BaseModel, extra="ignore"):
    """
    BLS and O*NET labor market data for one occupation.

    Flattens the nested `outlook`, `projections`, and `wages` objects
    from `labor.json` into a single model. Uses `extra="ignore"` so the
    source can carry additional fields beyond what the display layer
    needs without failing validation.
    """

    soc_title: str

    annual_median  : float | None = Field(default=None, ge=0)
    bright_outlook : bool         = False
    employment     : int          = Field(default=0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _flatten_nested(cls, data):
        """
        Hoist fields from nested `outlook`, `projections`, and `wages`
        objects to the top level.
        """
        if isinstance(data, dict):
            for key in ("outlook", "projections", "wages"):
                if (nested := data.pop(key, None)) and isinstance(nested, dict):
                    data.update(nested)
        return data


class Occupation(BaseModel, extra="forbid"):
    """
    An O*NET occupation with its full skill profile.

    Each of the 21 stakeholder SOC codes maps to one occupation containing
    skills across all 8 element types.
    """

    job_zone : int = Field(ge=1, le=5)
    sector   : str
    skills   : list[Skill]
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
    def task_elements(self) -> list[Skill]:
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
            if s.type in {SkillType.TASK, SkillType.DWA}
        ]


class Reach(BaseModel, extra="forbid"):
    """
    Local reach exploration view from the matched cluster.

    Shows advancement paths (edges to higher Job Zone clusters) and lateral
    pivots (edges to same Job Zone clusters), each with per-edge credential
    metadata identifying the training that bridges each specific transition.
    """

    advancement : list[CareerEdge] = Field(default_factory=list)
    lateral     : list[CareerEdge] = Field(default_factory=list)


class Skill(BaseModel, extra="forbid"):
    """
    A single skill entry within an O*NET occupation.

    Concrete element types carry `None` for `importance` and `level`,
    whereas abstract KSA types populate both fields with numeric ratings.
    Decomposable types (Tasks, DWAs) carry pre-computed sub-phrases from
    POS-based chunking performed at curation time.
    """

    name : str
    type : SkillType

    importance : float | None     = None
    level      : float | None     = None
    phrases    : list[str] | None = None


class SkillType(StrEnum):
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
