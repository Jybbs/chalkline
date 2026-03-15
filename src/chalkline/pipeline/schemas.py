"""
Pipeline configuration and shared reference data schemas.

Centralizes pipeline hyperparameters alongside shared data models for
apprenticeship and educational program reference data consumed across
matching, pathways, and report generation modules.
"""

from enum      import StrEnum
from functools import cached_property
from pathlib   import Path
from pydantic  import BaseModel, Field, field_validator

from chalkline import UnitInterval


class ApprenticeshipContext(BaseModel, extra="forbid"):
    """
    AGC-registered apprenticeship program from the stakeholder
    reference data.

    Each record represents a RAPIDS-coded trade with its required
    term hours. Used by the pathway graph for node enrichment and
    by the resume matcher for gap-to-trade linking.
    """

    rapids_code : str
    term_hours  : str
    title       : str


class ClusterProfile(BaseModel, extra="forbid"):
    """
    Domain characteristics of a single career cluster.

    Aggregates the Job Zone, sector, posting count, and union skill
    set for a cluster identified by hierarchical agglomerative
    clustering. Consumed by the pathway graph for node construction
    and by the career report for cluster display.
    """

    cluster_id : int       = Field(ge=0)
    job_zone   : int       = Field(ge=1, le=5)
    sector     : str
    size       : int       = Field(ge=1)
    skills     : list[str]
    terms      : list[str] = Field(default_factory=list)

    apprenticeship : ApprenticeshipContext | None       = None
    programs       : list[ProgramRecommendation]        = Field(default_factory=list)

    @cached_property
    def rank(self) -> tuple[int, int]:
        """
        Sort key for edge direction: (Job Zone, cluster ID).

        Lower rank is the edge source in the career DAG, ensuring
        edges flow from entry-level to advanced roles.
        """
        return (self.job_zone, self.cluster_id)

    @field_validator("skills", mode="before")
    @classmethod
    def sort_skills(cls, v: set[str] | list[str]) -> list[str]:
        """
        Accept a set or list and store as a sorted list for
        stable node attribute output.
        """
        return sorted(v)


class DistanceMetric(StrEnum):
    """
    Distance functions for resume matching and clustering comparison.

    `EUCLIDEAN` is the default because `StandardScaler(with_mean=False)`
    is always applied after PCA, making Euclidean on scaled coordinates
    equivalent to standardized Euclidean on raw PCA output. `COSINE`
    and `STANDARDIZED_EUCLIDEAN` exist for comparison experiments in
    CL-08.
    """

    COSINE                 = "cosine"
    EUCLIDEAN              = "euclidean"
    STANDARDIZED_EUCLIDEAN = "standardized_euclidean"


class PipelineConfig(BaseModel, extra="forbid"):
    """
    End-to-end configuration for the Chalkline pipeline.

    Required path fields locate the corpus, lexicons, and output directories.
    Optional fields control hyperparameters with defaults tuned for the
    922-posting Maine construction corpus.
    """

    lexicon_dir  : Path
    output_dir   : Path
    postings_dir : Path

    distance_metric    : DistanceMetric = DistanceMetric.EUCLIDEAN
    max_components     : int            = 20
    min_cooccurrence   : float | str    = "auto"
    random_seed        : int            = 42
    reference_dir      : Path           = Path("data/stakeholder/reference")
    top_k_gaps         : int            = 10
    variance_threshold : UnitInterval   = 0.85


class ProgramRecommendation(BaseModel, extra="forbid"):
    """
    Normalized educational program recommendation.

    Unifies community college programs (where the institution field is
    `college` and credential is `credential`) and university programs
    (where institution is `campus` and credential is `degree`) into a
    consistent schema.
    """

    credential  : str
    institution : str
    program     : str
    url         : str
