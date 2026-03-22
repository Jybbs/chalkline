"""
Schemas for the career pathway domain.

Defines O*NET occupation and credential reference models, the unified
`Cluster` and `Clusters` dataclasses, and the career graph edge and
neighborhood models that together describe the fitted career landscape.
"""

import numpy as np

from dataclasses import dataclass, field
from enum        import StrEnum
from pydantic    import BaseModel, Field
from typing      import NamedTuple

from chalkline.collection.schemas import Posting


class CareerEdge(BaseModel, extra="forbid"):
    """
    A single edge in the neighborhood view with credential metadata.

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
        return (
            f"Cluster {self.cluster_id}: {self.soc_title} "
            f"(JZ {self.job_zone})"
        )


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


class Neighborhood(BaseModel, extra="forbid"):
    """
    Local neighborhood exploration view from the matched cluster.

    Shows advancement paths (edges to higher Job Zone clusters) and lateral
    pivots (edges to same Job Zone clusters), each with per-edge credential
    metadata identifying the training that bridges each specific transition.
    """

    advancement : list[CareerEdge] = Field(default_factory=list)
    lateral     : list[CareerEdge] = Field(default_factory=list)

    @property
    def all_edges(self) -> list[CareerEdge]:
        """
        Combined advancement and lateral edges.
        """
        return self.advancement + self.lateral


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


class Task(NamedTuple):
    """
    A single O*NET Task or DWA element with its sentence embedding.

    Produced during SOC task encoding and attached to `Cluster.tasks` for
    per-task cosine gap analysis during resume matching.
    """

    name   : str
    vector : np.ndarray
