"""
Pipeline configuration.

Pure Pydantic config model with defaults tuned for the 922-posting Maine
construction corpus. No query logic, no builders, no computational imports.
"""

from pathlib  import Path
from pydantic import BaseModel, Field


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

    cluster_count          : int = Field(default=20,  ge=2)
    component_count        : int = Field(default=10,  ge=1)
    destination_percentile : int = Field(default=5,   ge=0, le=100)
    embedding_model        : str = "Alibaba-NLP/gte-base-en-v1.5"
    lateral_neighbors      : int = Field(default=2,   ge=1)
    random_seed            : int = Field(default=42,  ge=0)
    soc_neighbors          : int = Field(default=3,   ge=1)
    source_percentile      : int = Field(default=75,  ge=0, le=100)
    upward_neighbors       : int = Field(default=2,   ge=1)

    @property
    def hamilton_cache_dir(self) -> Path:
        """
        Hamilton disk cache directory for fitted node results.
        """
        return Path(".cache") / "hamilton"

    @property
    def pipeline_dir(self) -> Path:
        """
        Default directory for serialized pipeline artifacts.
        """
        return self.output_dir / "pipeline"
