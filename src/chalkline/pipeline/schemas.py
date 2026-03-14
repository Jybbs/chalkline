"""
Pipeline configuration and shared reference data schemas.

Centralizes pipeline hyperparameters alongside shared data models for
apprenticeship and educational program reference data consumed across
matching, pathways, and report generation modules.
"""

from enum     import StrEnum
from pathlib  import Path
from pydantic import BaseModel

from chalkline import UnitInterval


class ApprenticeshipContext(BaseModel, extra="forbid"):
    """
    AGC-registered apprenticeship program linked to a skill gap.

    Maps a gap skill to a RAPIDS-coded trade with term hours,
    providing a concrete training timeline for skill acquisition.
    """

    rapids_code : str
    term_hours  : str
    trade       : str


class DistanceMetric(StrEnum):
    """
    Distance functions for resume matching and clustering comparison.

    `EUCLIDEAN` is the default because `StandardScaler(with_mean=False)` is
    always applied after PCA, making Euclidean on scaled coordinates
    equivalent to standardized Euclidean on raw PCA output. `COSINE` and
    `STANDARDIZED_EUCLIDEAN` exist for comparison experiments in CL-08.
    """

    COSINE                 = "cosine"
    EUCLIDEAN              = "euclidean"
    STANDARDIZED_EUCLIDEAN = "standardized_euclidean"


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

    Unifies community college programs (where the institution field
    is `college` and credential is `credential`) and university
    programs (where institution is `campus` and credential is
    `degree`) into a consistent schema.
    """

    credential  : str
    institution : str
    program     : str
    url         : str
