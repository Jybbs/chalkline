"""
Pipeline configuration with validated parameters.

Centralizes directory paths, thresholds, and hyperparameters that
propagate through every pipeline step. The `extra="forbid"` policy
surfaces typos and stale fields immediately.
"""

from enum     import StrEnum
from pathlib  import Path
from pydantic import BaseModel

from chalkline import UnitInterval


class DistanceMetric(StrEnum):
    """
    Distance functions for resume matching and clustering comparison.

    `EUCLIDEAN` is the default because `StandardScaler(with_mean=False)`
    is always applied after PCA, making Euclidean on scaled coordinates
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
    directories. Optional fields control hyperparameters with defaults
    tuned for the 922-posting Maine construction corpus.
    """

    lexicon_dir  : Path
    output_dir   : Path
    postings_dir : Path

    distance_metric      : DistanceMetric = DistanceMetric.EUCLIDEAN
    max_components       : int            = 20
    min_cooccurrence_pct : float          = 0.05
    random_seed          : int            = 42
    reference_dir        : Path           = Path("data/stakeholder/reference")
    top_k_gaps           : int            = 10
    variance_threshold   : UnitInterval   = 0.85
