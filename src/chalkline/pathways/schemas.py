"""
Schemas for the career pathway domain.

Defines O*NET occupation and credential reference models, the unified
`Cluster` and `Clusters` dataclasses, and the career graph edge and
reach models that together describe the fitted career landscape.
"""

import numpy as np

from dataclasses import dataclass, field
from enum        import StrEnum
from functools   import cached_property
from pydantic    import BaseModel, Field
from typing      import NamedTuple

from chalkline.collection.schemas import Posting


class LaborOutlook(BaseModel, extra="ignore"):
    """
    O*NET Bright Outlook designation for an occupation.
    """

    bright_outlook  : bool       = False
    outlook_reasons : list[str]  = Field(default_factory=list)


class LaborProjections(BaseModel, extra="ignore"):
    """
    BLS 10-year employment projections for an occupation.
    """

    change_percent : float | None = None
    education      : str          = ""
    openings       : float | None = None


class LaborWages(BaseModel, extra="ignore"):
    """
    BLS OEWS annual wage percentiles for an occupation.
    """

    annual_10     : float        = 0
    annual_25     : float        = 0
    annual_75     : float        = 0
    annual_90     : float        = 0
    annual_median : float | None = None
    employment    : int          = 0


class LaborRecord(BaseModel, extra="ignore"):
    """
    Unified BLS and O*NET labor market data for one occupation.

    Uses `extra="ignore"` because `labor.json` contains fields
    beyond what the display layer needs (hourly wages, projected
    employment, related occupations, training).
    """

    outlook     : LaborOutlook | None     = None
    projections : LaborProjections | None = None
    soc_title   : str
    wages       : LaborWages | None       = None


class CareerEdge(BaseModel, extra="forbid"):
    """
    A single edge in the reach view with credential metadata.

    Each edge connects the matched cluster to a neighboring cluster,
    carrying the target cluster ID, the cosine similarity weight, and the
    credentials that bridge the specific transition, filtered by the
    destination_percentile and source_percentile dual-threshold rule.
    """

    cluster_id  : int              = Field(ge=0)
    credentials : list[Credential] = Field(default_factory=list)
    weight      : float


@dataclass
class Cluster:
    """
    Unified per-cluster representation combining profile metadata,
    membership indices, resolved postings, and optional O*NET task
    embeddings for gap analysis.
    """

    cluster_id   : int
    job_zone     : int
    members      : np.ndarray
    modal_title  : str
    postings     : list[Posting]
    sector       : str
    size         : int
    soc_title    : str

    tasks : list[Task] = field(default_factory=list)

    @property
    def display_label(self) -> str:
        """
        Human-readable cluster identifier for dropdown labels.
        """
        return f"Cluster {self.cluster_id}: {self.soc_title} (JZ {self.job_zone})"

    @property
    def profile_dict(self) -> dict[str, str | int]:
        """
        Display-ready profile summary for tree views.
        """
        return {
            "Sector"      : self.sector,
            "Job Zone"    : self.job_zone,
            "Size"        : self.size,
            "Modal Title" : self.modal_title
        }

    @property
    def short_title(self) -> str:
        """
        Truncated SOC title for chart labels (30 chars max).
        """
        return self.soc_title[:30]

    @property
    def treemap_label(self) -> str:
        """
        Label for cluster in treemap visualization.
        """
        return f"{self.short_title} ({self.size})"


@dataclass
class Clusters:
    """
    Indexed collection of clusters with eagerly-derived matrices.

    Constructed once after profiling is complete. Provides the per-cluster
    dict for individual lookups and pre-stacked centroid and embedding
    vector matrices for vectorized operations in graph construction, resume
    matching, and visualization.
    """

    centroids : np.ndarray
    items     : dict[int, Cluster]
    vectors   : np.ndarray

    cluster_ids : list[int] = field(init=False)

    def __post_init__(self):
        self.cluster_ids = sorted(self.items)

    def __getitem__(self, cluster_id: int) -> Cluster:
        """
        Look up a cluster by ID.
        """
        return self.items[cluster_id]

    def __iter__(self):
        """
        Iterate sorted cluster IDs.
        """
        return iter(self.cluster_ids)

    def __len__(self) -> int:
        """
        Number of clusters.
        """
        return len(self.items)

    def pairs(self):
        """
        Iterate (cluster_id, cluster) tuples in sorted ID order.
        """
        return ((cid, self.items[cid]) for cid in self.cluster_ids)

    def by_sector(self, sector: str) -> list[tuple[int, Cluster]]:
        """
        All clusters in a given sector, sorted by job zone then
        title.

        Args:
            sector: Sector name to filter by.
        """
        return sorted(
            (
                (cid, self.items[cid]) for cid in self.cluster_ids
                if self.items[cid].sector == sector
            ),
            key=lambda x: (x[1].job_zone, x[1].soc_title)
        )

    def values(self):
        """
        Iterate cluster objects in sorted ID order.
        """
        return (self.items[cid] for cid in self.cluster_ids)


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

    Wraps a list of `Occupation` records and provides a
    cached skill-to-type lookup used by the display layer for
    grouping skills by O*NET category.
    """

    items : list[Occupation]

    @cached_property
    def skill_type_index(self) -> dict[str, SkillType]:
        """
        Skill name to O*NET element type across all occupations.
        """
        return {s.name: s.type for occ in self.items for s in occ.skills}

    def __getitem__(self, index: int) -> Occupation:
        """
        Look up an occupation by index.
        """
        return self.items[index]

    def __iter__(self):
        """
        Iterate occupation records.
        """
        return iter(self.items)

    def __len__(self) -> int:
        """
        Number of occupations.
        """
        return len(self.items)


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

    def credentials_by_kind(self, kind: str) -> dict:
        """
        Deduplicated credentials of a given kind, keyed for
        deduplication by the appropriate metadata field.

        For apprenticeships, deduplicates by RAPIDS code.
        For programs, deduplicates by (institution, label).
        For others, deduplicates by label.

        Args:
            kind: Credential kind string ("apprenticeship",
                  "program", "certification").

        Returns:
            Dict of unique credentials (values are Credential
            objects).
        """
        match kind:
            case "apprenticeship" : key = lambda c: c.metadata["rapids_code"]
            case "program"        : key = lambda c: (c.metadata["institution"], c.label)
            case _                : key = lambda c: c.label

        return {key(c): c for c in self.all_credentials if c.kind == kind}


class Task(NamedTuple):
    """
    A single O*NET Task or DWA element with its sentence embedding.

    Produced during SOC task encoding and attached to `Cluster.tasks` for
    per-task cosine gap analysis during resume matching.
    """

    name   : str
    vector : np.ndarray
