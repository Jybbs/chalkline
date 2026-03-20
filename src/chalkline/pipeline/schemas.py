"""
Pipeline configuration and shared reference data schemas.

Centralizes pipeline hyperparameters alongside shared data models for
apprenticeship and educational program reference data consumed across
matching, graph construction, and report generation modules.
"""

from functools import cached_property
from pathlib   import Path
from pydantic  import BaseModel, Field


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
    max_gaps               : int = 10
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


class PipelineManifest(BaseModel, extra="forbid"):
    """
    Provenance metadata for serialized pipeline artifacts.

    Tracks which corpus, embedding model, and configuration produced the
    fitted artifacts so that stale caches can be detected on reload.
    """

    component_count : int
    corpus_size     : int
    embedding_model : str
    posting_ids     : list[str]
    timestamp       : str


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
