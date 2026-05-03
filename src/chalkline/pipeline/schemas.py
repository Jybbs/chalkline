"""
Pipeline configuration and cache row schema.

Holds the Pydantic config model with defaults tuned for the 922-posting
Maine construction corpus, alongside the NamedTuple row shape for Hamilton's
on-disk metadata store. No query logic, no builders, no computational
imports.
"""

from pathlib  import Path
from pydantic import BaseModel, Field
from typing   import NamedTuple


class CacheRow(NamedTuple):
    """
    One row from Hamilton's `metadata_store.db`.
    """

    node    : str
    code    : str
    data    : str
    created : str


class PipelineConfig(BaseModel, extra="forbid"):
    """
    End-to-end configuration for the Chalkline pipeline.

    Required path fields locate the corpus and lexicons. Optional fields
    control hyperparameters with defaults tuned for the 922-posting Maine
    construction corpus.
    """

    lexicon_dir  : Path
    postings_dir : Path

    cluster_count          : int   = Field(default=20,   ge=2)
    component_count        : int   = Field(default=15,   ge=1)
    consensus_seeds        : int   = Field(default=50,   ge=1, le=200)
    destination_percentile : int   = Field(default=20,   ge=0, le=100)
    embedding_model        : str   = "Alibaba-NLP/gte-base-en-v1.5"
    lateral_neighbors      : int   = Field(default=2,    ge=1)
    random_seed            : int   = Field(default=42,   ge=0)
    soc_neighbors          : int   = Field(default=3,    ge=1)
    soc_softmax_tau        : float = Field(default=0.02, gt=0.0)
    soc_wage_round         : int   = Field(default=10,   ge=1)
    soc_wage_topk          : int   = Field(default=3,    ge=1)
    upward_neighbors       : int   = Field(default=2,    ge=1)

    @property
    def hamilton_cache_dir(self) -> Path:
        """
        Hamilton disk cache directory for fitted node results.
        """
        return Path(".cache") / "hamilton"
