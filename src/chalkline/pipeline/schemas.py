"""
Pipeline configuration.

Centralizes pipeline hyperparameters as a Pydantic model with defaults tuned
for the 922-posting Maine construction corpus.
"""

from functools import cached_property
from pathlib   import Path
from pydantic  import BaseModel


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
    embedding_model        : str = "Alibaba-NLP/gte-base-en-v1.5"
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
