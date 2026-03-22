"""
Schemas for the career pathway domain.

Defines O*NET occupation and credential reference models, Ward-linkage HAC
cluster structures, and the career graph edge and neighborhood models that
together describe the fitted career landscape.
"""

import numpy as np

from dataclasses           import dataclass, field
from enum                  import StrEnum
from pydantic              import BaseModel, Field
from sklearn.preprocessing import normalize
from typing                import NamedTuple


class CareerEdge(BaseModel, extra="forbid"):
    """
    A single edge in the neighborhood view with credential metadata.

    Each edge connects the matched cluster to a neighboring cluster,
    carrying the target cluster's profile, the cosine similarity weight,
    and the credentials that bridge the specific transition, filtered by
    the destination_percentile and source_percentile dual-threshold rule.
    """

    credentials : list[Credential] = Field(default_factory=list)
    profile     : ClusterProfile
    weight      : float


@dataclass
class ClusterAssignments:
    """
    Ward-linkage HAC results with eagerly-derived cluster structure.

    Accepts the raw label array from `AgglomerativeClustering.fit_predict`
    and derives the sorted unique IDs and per-cluster member indices once at
    construction, so downstream consumers access structural properties
    rather than reimplementing boolean masking.
    """

    labels : np.ndarray

    cluster_ids : list[int] = field(init=False)
    members     : dict[int, np.ndarray] = field(init=False)

    def __post_init__(self):
        self.cluster_ids = np.unique(self.labels).tolist()
        self.members = {
            cid: np.where(self.labels == cid)[0]
            for cid in self.cluster_ids
        }

    def centroids(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Mean SVD coordinates per cluster for resume distance computation.

        Args:
            coordinates: (n_postings, n_components) SVD-reduced space.
        """
        return np.stack([
            coordinates[self.members[cid]].mean(axis=0)
            for cid in self.cluster_ids
        ])

    def cluster_vectors(self, raw_vectors: np.ndarray) -> np.ndarray:
        """
        Mean posting embedding per cluster in the full embedding space,
        L2-normalized for cosine similarity against occupations and
        credentials.

        Args:
            raw_vectors: (n_postings, embedding_dim) unnormalized.
        """
        return normalize(np.stack([
            raw_vectors[self.members[cid]].mean(axis=0)
            for cid in self.cluster_ids
        ]))


class ClusterProfile(BaseModel, extra="forbid"):
    """
    Domain characteristics of a single career cluster.

    Aggregates the Job Zone, sector, posting count, and representative
    titles for a cluster identified by Ward-linkage HAC on sentence
    embeddings. Consumed by the pathway graph for node construction and by
    the career display for cluster presentation.
    """

    cluster_id  : int = Field(ge=0)
    job_zone    : int = Field(ge=1, le=5)
    modal_title : str
    sector      : str
    size        : int = Field(ge=1)
    soc_title   : str

    @property
    def display_label(self) -> str:
        """
        Human-readable cluster identifier for dropdown labels.
        """
        return (
            f"Cluster {self.cluster_id}: {self.soc_title} "
            f"(JZ {self.job_zone})"
        )


class ClusterTasks(NamedTuple):
    """
    Per-cluster O*NET Task+DWA names with aligned embedding vectors.

    Produced by the `soc_tasks` Hamilton node and consumed by
    `ResumeMatcher` for per-task cosine gap analysis against resume
    embeddings.
    """

    labels  : list[str]
    vectors : np.ndarray


class Credential(BaseModel, extra="forbid"):
    """
    Unified credential record for the career pathway graph.

    Represents an apprenticeship, certification, or educational program
    with pre-computed `embedding_text` and `label` from curation. The
    pipeline encodes `embedding_text` with the sentence transformer and
    attaches the vector to each instance. The graph stores credentials
    on edges via `model_dump`, which excludes the runtime-only vector.
    Type-specific display fields live in `metadata`.
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
