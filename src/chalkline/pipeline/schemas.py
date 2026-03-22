"""
Pipeline configuration and shared data schemas.

Centralizes pipeline hyperparameters, the unified credential model, and
structural data types consumed across matching, graph construction, and
report generation.
"""

import numpy as np

from dataclasses                     import dataclass, field
from functools                       import cached_property
from hamilton.caching.fingerprinting import hash_value, hash_primitive
from pathlib                         import Path
from pydantic                        import BaseModel, Field
from sentence_transformers           import SentenceTransformer
from sklearn.preprocessing           import normalize
from typing                          import NamedTuple

from chalkline.collection.schemas import Posting


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

    cluster_ids : list[int]             = field(init=False)
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
    the career report for cluster display.
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


@dataclass
class Corpus:
    """
    Filtered posting corpus with eagerly-derived key ordering.

    Keeps `Posting` objects intact so downstream consumers access
    `.description` and `.title` through the posting rather than through
    parallel dicts. The sorted key list is computed once at construction for
    deterministic encoding and manifest ordering.
    """

    postings    : dict[str, Posting]
    posting_ids : list[str] = field(init=False)

    def __post_init__(self):
        self.posting_ids = sorted(self.postings)

    @property
    def descriptions(self) -> list[str]:
        """
        Posting descriptions in deterministic sorted-key order for sentence
        encoding.
        """
        return [self.postings[pid].description for pid in self.posting_ids]


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

    metadata : dict              = Field(default_factory=dict)
    vector   : list[float] | None = Field(default=None, exclude=True)


@dataclass
class Encoder:
    """
    Sentence-transformer wrapper for the Hamilton pipeline.

    Defaults to L2-normalized output and enabled progress bars so
    call sites stay clean. The `name` field records the HuggingFace
    model identifier for Hamilton cache fingerprinting.
    """

    name  : str
    model : SentenceTransformer = field(init=False)

    def __post_init__(self):
        self.model = SentenceTransformer(self.name)

    def encode(self, texts: list[str], unit: bool = True) -> np.ndarray:
        """
        Encode texts with the sentence transformer.

        Args:
            texts : Strings to encode.
            unit  : L2-normalize output (default True).
        """
        return self.model.encode(
            texts,
            normalize_embeddings = unit,
            show_progress_bar    = True
        )


@hash_value.register(Encoder)
def _hash_encoder(obj, *args, **kwargs) -> str:
    """
    Fingerprint an encoder by its model name for Hamilton caching.
    """
    return hash_primitive(obj.name)


class PipelineConfig(BaseModel, extra="forbid"):
    """
    End-to-end configuration for the Chalkline pipeline.

    Required path fields locate the corpus, lexicons, and output
    directories. Optional fields control hyperparameters with defaults tuned
    for the 922-posting Maine construction corpus.
    """

    lexicon_dir  : Path
    output_dir   : Path
    postings_dir : Path

    cluster_count          : int = 20
    component_count        : int = 10
    destination_percentile : int = 5
    embedding_model        : str = "all-mpnet-base-v2"
    lateral_neighbors      : int = 2
    random_seed            : int = 42
    soc_neighbors          : int = 3
    source_percentile      : int = 75
    upward_neighbors       : int = 2

    @cached_property
    def hamilton_cache_dir(self) -> Path:
        """
        Hamilton disk cache directory for fitted node results.
        """
        return Path(".cache") / "hamilton"

    @cached_property
    def pipeline_dir(self) -> Path:
        """
        Default directory for serialized pipeline artifacts.
        """
        return self.output_dir / "pipeline"
