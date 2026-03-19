"""
Pipeline configuration and shared reference data schemas.

Centralizes pipeline hyperparameters alongside shared data models for
apprenticeship and educational program reference data consumed across
matching, pathways, and report generation modules.
"""

from functools import cached_property
from pathlib   import Path
from pydantic  import BaseModel, Field

from chalkline import UnitInterval


class ApprenticeshipContext(BaseModel, extra="forbid"):
    """
    AGC-registered apprenticeship program from curated reference data.

    Each record represents a RAPIDS-coded trade with pre-computed
    `min_hours` and 4-character prefix sets for runtime matching. Used by
    the pathway graph for node enrichment and by the resume matcher for
    gap-to-trade linking.
    """

    min_hours   : int
    prefixes    : set[str]
    rapids_code : str
    title       : str


class ClusterProfile(BaseModel, extra="forbid"):
    """
    Domain characteristics of a single career cluster.

    Aggregates the Job Zone, sector, posting count, and union skill set for
    a cluster identified by hierarchical agglomerative clustering. Consumed
    by the pathway graph for node construction and by the career report for
    cluster display.
    """

    cluster_id : int      = Field(ge=0)
    job_zone   : int      = Field(ge=1, le=5)
    sector     : str
    size       : int      = Field(ge=1)
    skills     : set[str]
    terms      : list[str] = Field(default_factory=list)

    apprenticeship : ApprenticeshipContext | None = None
    programs       : list[ProgramRecommendation]  = Field(default_factory=list)

    @cached_property
    def rank(self) -> tuple[int, int]:
        """
        Sort key for edge direction: (Job Zone, cluster ID).

        Lower rank is the edge source in the career DAG, ensuring edges flow
        from entry-level to advanced roles.
        """
        return (self.job_zone, self.cluster_id)


class PipelineManifest(BaseModel, extra="forbid"):
    """
    Provenance metadata for serialized pipeline artifacts.

    Tracks which corpus and configuration produced the fitted artifacts so
    that stale caches can be detected on reload. The `geometry_params` field
    stores the output of `Chalkline.geometry_pipeline.get_params(deep=True)`
    for ground-truth reproducibility without manually mirroring config
    values.
    """

    corpus_size     : int
    geometry_params : dict
    posting_ids     : list[str]
    timestamp       : str


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

    distance_metric    : str          = "euclidean"
    max_components     : int          = 20
    max_graph_density  : float        = 0.05
    min_cooccurrence   : float | str  = "auto"
    random_seed        : int          = 42
    top_k_gaps         : int          = 10
    variance_threshold : UnitInterval = 0.85

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
