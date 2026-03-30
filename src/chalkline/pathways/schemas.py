"""
Pydantic models and enums for the career pathway domain.

Pure data contracts with field declarations, validators, and self-derived
properties. Behavioral containers (`Cluster`, `Clusters`, `Task`) live in
`clusters.py`. No query logic, no builders, no computational imports.
"""

from collections.abc import Iterator
from dataclasses     import dataclass
from enum            import StrEnum
from pydantic        import BaseModel, Field, field_validator, model_validator


class CareerEdge(BaseModel, extra="forbid"):
    """
    A single edge in the reach view with credential metadata.

    Each edge connects the matched cluster to a neighboring cluster,
    carrying the target cluster ID, the cosine similarity weight, and the
    credentials that bridge the specific transition, filtered by the
    destination_percentile and source_percentile dual-threshold rule.
    """

    cluster_id : int   = Field(ge=0)
    soc_title  : str
    weight     : float = Field(ge=-1, le=1)

    credentials : list[Credential] = Field(default_factory=list)


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
    def hours(self) -> int | None:
        """
        Minimum apprenticeship term hours, if applicable.
        """
        return self.metadata.get("min_hours")

    @property
    def key(self) -> str | tuple[str, str]:
        """
        Deduplication key based on credential kind.

        Apprenticeships deduplicate by RAPIDS code, programs by
        (institution, label), and all others by label.
        """
        match self.kind:
            case "apprenticeship" : return self.metadata["rapids_code"]
            case "program"        : return (self.metadata["institution"], self.label)
            case _                : return self.label

    @property
    def type_label(self) -> str:
        """
        Display name for the credential type, falling back to the titlecased
        kind when metadata carries no specific label.
        """
        return self.metadata.get("credential", self.kind.title())


class LaborRecord(BaseModel, extra="ignore"):
    """
    Unified BLS and O*NET labor market data for one occupation.

    Flattens the nested `outlook`, `projections`, and `wages` objects from
    `labor.json` into a single model. Uses `extra="ignore"` because the
    source contains fields beyond what the display layer needs.
    """

    soc_title: str

    annual_10       : float        = Field(default=0,    ge=0)
    annual_25       : float        = Field(default=0,    ge=0)
    annual_75       : float        = Field(default=0,    ge=0)
    annual_90       : float        = Field(default=0,    ge=0)
    annual_median   : float | None = Field(default=None, ge=0)
    bright_outlook  : bool         = False
    change_percent  : float | None = None
    education       : str          = ""
    employment      : int          = Field(default=0,    ge=0)
    openings        : float | None = None
    outlook_reasons : list[str]    = Field(default_factory=list)

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

    @property
    def percentiles(self) -> list[float]:
        """
        Wage values at the 10th, 25th, median, 75th, and 90th percentiles.
        Empty when median is unavailable.
        """
        if not self.annual_median:
            return []

        return [
            self.annual_10,
            self.annual_25,
            self.annual_median,
            self.annual_75,
            self.annual_90
        ]


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


@dataclass
class Occupations:
    """
    Indexed collection of O*NET occupations with derived views.

    Wraps a list of `Occupation` records and provides a cached skill-to-type
    lookup used by the display layer for grouping skills by O*NET category.
    """

    items : list[Occupation]

    def __getitem__(self, index: int) -> Occupation:
        """
        Look up an occupation by index.
        """
        return self.items[index]

    def __iter__(self) -> Iterator[Occupation]:
        """
        Iterate occupation records.
        """
        return iter(self.items)

    def __len__(self) -> int:
        """
        Number of occupations.
        """
        return len(self.items)

    @property
    def skill_type_index(self) -> dict[str, SkillType]:
        """
        Skill name to O*NET element type across all occupations.
        """
        return {s.name: s.type for occ in self.items for s in occ.skills}


class Reach(BaseModel, extra="forbid"):
    """
    Local reach exploration view from the matched cluster.

    Shows advancement paths (edges to higher Job Zone clusters) and lateral
    pivots (edges to same Job Zone clusters), each with per-edge credential
    metadata identifying the training that bridges each specific transition.
    """

    advancement : list[CareerEdge] = Field(default_factory=list)
    lateral     : list[CareerEdge] = Field(default_factory=list)

    @property
    def all_credentials(self) -> list[Credential]:
        """
        Flattened credentials from all edges.
        """
        return [c for edge in self.all_edges for c in edge.credentials]

    @property
    def all_edges(self) -> list[CareerEdge]:
        """
        Combined advancement and lateral edges.
        """
        return self.advancement + self.lateral

    @property
    def credentials_by_kind(self) -> dict[str, list[Credential]]:
        """
        Deduplicated credentials grouped by kind.

        Uses `Credential.key` for deduplication, which is RAPIDS code for
        apprenticeships, (institution, label) for programs, and label for
        all others.
        """
        by_kind: dict[str, dict] = {}
        for c in self.all_credentials:
            by_kind.setdefault(c.kind, {})[c.key] = c
        return {
            k: sorted(v.values(), key=lambda c: c.label)
            for k, v in sorted(by_kind.items())
        }


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
