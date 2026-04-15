"""
Pydantic models and enums for the career pathway domain.

Pure data contracts with field declarations, validators, and self-derived
properties. Behavioral containers (`Cluster`, `Clusters`, `Task`) live in
`clusters.py`. No query logic, no builders, no computational imports.
"""

from enum      import StrEnum
from functools import cached_property
from pydantic  import BaseModel, Field, model_validator
from typing    import Self


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

    metadata : dict               = Field(default_factory=dict)
    vector   : list[float] | None = Field(default=None, exclude=True)

    @property
    def card_detail(self) -> str:
        """
        Dot-joined detail line for credential cards, combining hours and
        institution when both are present.

        Unlike `detail_label` (which picks one or the other), this joins all
        available metadata with a centered dot so the card shows the fullest
        possible detail line.
        """
        return " \u00b7 ".join(filter(None, (
            f"{self.hours:,} hours" if self.hours else None,
            self.metadata.get("institution")
        )))

    @property
    def detail_label(self) -> str:
        """
        Concise detail line for card and recipe displays.

        Shows hours when available (e.g. "4,000 hours"), falls back to
        institution name, then to the titlecased kind.
        """
        if self.hours:
            return f"{self.hours:,} hours"
        return self.metadata.get("institution", self.kind.title())

    @property
    def hours(self) -> int | None:
        """
        Minimum apprenticeship term hours, if applicable.
        """
        return self.metadata.get("min_hours")

    @cached_property
    def stems(self) -> set[str]:
        """
        Stemmed content words from `embedding_text` for BM25 scoring against
        task stems, filtering stop words via Zipf threshold. Mirrors
        `Cluster.task_stems` so credentials index into `Clusters.bm25_idf`
        without a separate IDF table.
        """
        from nltk.stem import SnowballStemmer
        from re        import findall
        from wordfreq  import zipf_frequency

        stemmer = SnowballStemmer("english")
        return {
            stemmer.stem(w)
            for w in findall(r"[a-zA-Z]{3,}", self.embedding_text.lower())
            if zipf_frequency(w, "en") < 6.0
        }

    @property
    def type_label(self) -> str:
        """
        Display name for the credential type, falling back to the titlecased
        kind when metadata carries no specific label.
        """
        return self.metadata.get("credential", self.kind.title())

    @property
    def url(self) -> str:
        """
        External URL for program credentials, empty for other kinds.
        """
        return self.metadata.get("url", "")

    @model_validator(mode="after")
    def _validate_metadata(self) -> Self:
        """
        Enforce required metadata keys per credential kind.
        """
        match self.kind:
            case "apprenticeship" : required = {"min_hours", "rapids_code"}
            case "program"        : required = {"credential", "institution", "url"}
            case _                : return self
        if missing := required - self.metadata.keys():
            raise ValueError(f"Missing keys for {self.kind}: {missing}")
        return self


class LaborRecord(BaseModel, extra="ignore"):
    """
    BLS and O*NET labor market data for one occupation.

    Flattens the nested `outlook`, `projections`, and `wages` objects from
    `labor.json` into a single model. Uses `extra="ignore"` so the source
    can carry additional fields beyond what the display layer needs without
    failing validation.
    """

    annual_25      : float | None = Field(default=None, ge=0)
    annual_75      : float | None = Field(default=None, ge=0)
    annual_median  : float | None = Field(default=None, ge=0)
    bright_outlook : bool         = False
    employment     : int | None   = Field(default=0, ge=0)
    soc_title      : str          = ""

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

    Each SOC code in the curated reference set maps to one occupation
    containing skills across all 8 element types.
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

    @property
    def edges(self) -> list[CareerEdge]:
        """
        All reach edges, advancement followed by lateral.
        """
        return self.advancement + self.lateral


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
    O*NET element types across the curated SOC reference set.

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
