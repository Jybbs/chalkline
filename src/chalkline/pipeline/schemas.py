"""
Pipeline configuration and shared reference data schemas.

Centralizes pipeline hyperparameters alongside shared data models for
apprenticeship and educational program reference data consumed across
matching, graph construction, and report generation modules.
"""

import numpy as np

from dataclasses           import dataclass, field
from functools             import cached_property
from pathlib               import Path
from pydantic              import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from typing                import Literal, NamedTuple

from chalkline.collection.schemas import Posting


class ApprenticeshipContext(BaseModel, extra="forbid"):
    """
    AGC-registered apprenticeship program from curated reference data.

    Each record represents a RAPIDS-coded trade with pre-computed
    `min_hours` and 4-character prefix sets for runtime matching. Used by
    the pathway graph for credential enrichment and by the career report for
    trade display.
    """

    min_hours   : int
    prefixes    : set[str]
    rapids_code : str
    title       : str

    credential_kind : Literal["apprenticeship"] = "apprenticeship"

    @property
    def embedding_text(self) -> str:
        """
        Text representation for sentence encoding.
        """
        return self.title


@dataclass(slots=True)
class ClusterAssignments:
    """
    Ward-linkage HAC results with eagerly-derived cluster structure.

    Accepts the raw label array from `AgglomerativeClustering.fit_predict`
    and derives the sorted unique IDs and per-cluster member indices once at
    construction, so downstream consumers access structural properties
    rather than reimplementing boolean masking.
    """

    labels : np.ndarray

    cluster_ids : np.ndarray            = field(init=False)
    members     : dict[int, np.ndarray] = field(init=False)

    def __post_init__(self):
        self.cluster_ids = np.array(sorted(set(self.labels)))
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
        return np.stack([
            normalize(
                raw_vectors[self.members[cid]].mean(axis=0, keepdims=True)
            )[0]
            for cid in self.cluster_ids
        ])


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


@dataclass(slots=True)
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


class ClusterTasks(NamedTuple):
    """
    Per-cluster O*NET Task+DWA names with aligned embedding vectors.

    Produced by the `soc_tasks` Hamilton node and consumed by
    `ResumeMatcher` for per-task cosine gap analysis against resume
    embeddings.
    """

    labels  : list[str]
    vectors : np.ndarray


class Credentials(NamedTuple):
    """
    Bundled credential records with their sentence-transformer embeddings.

    Produced by the `credentials` Hamilton node and consumed by the graph
    constructor, which unpacks `records` for edge metadata and `vectors` for
    cosine similarity against cluster centroids.
    """

    records : list
    vectors : np.ndarray


@dataclass(slots=True)
class Encoder:
    """
    L2-normalizing sentence-transformer wrapper.

    Encodes text into embedding vectors, L2-normalized by default for cosine
    similarity. Pass `unit=False` for unnormalized output when downstream
    nodes handle normalization separately.
    """

    model: SentenceTransformer

    def encode(self, texts: list[str], unit: bool = True) -> np.ndarray:
        """
        Encode texts with the sentence transformer.

        Args:
            texts : Strings to encode.
            unit  : L2-normalize output for cosine similarity (default True).
        """
        vectors = self.model.encode(texts, show_progress_bar=False)
        return normalize(vectors) if unit else vectors


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


class ProgramRecommendation(BaseModel, extra="forbid"):
    """
    Normalized educational program recommendation.

    Unifies community college programs and university programs into a
    consistent schema with pre-computed 4-character prefix sets for runtime
    matching via `TradeIndex`.
    """

    credential  : str
    institution : str
    prefixes    : set[str]
    program     : str
    url         : str

    credential_kind : Literal["program"] = "program"

    @property
    def embedding_text(self) -> str:
        """
        Text representation for sentence encoding.

        Concatenates credential level, program name, and institution so the
        sentence transformer captures the full program identity.
        """
        return f"{self.credential} {self.program} {self.institution}"
