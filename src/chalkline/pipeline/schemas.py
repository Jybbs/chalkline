"""
Pipeline configuration and sentence-transformer encoding.

Centralizes pipeline hyperparameters and the sentence-transformer wrapper
with Hamilton cache fingerprinting.
"""

import numpy as np

from dataclasses                     import dataclass, field
from functools                       import cached_property
from hamilton.caching.fingerprinting import hash_value, hash_primitive
from pathlib                         import Path
from pydantic                        import BaseModel
from sentence_transformers           import SentenceTransformer


@dataclass
class Encoder:
    """
    Sentence-transformer wrapper for the Hamilton pipeline.

    Defaults to L2-normalized output and enabled progress bars so call sites
    stay clean. The `name` field records the HuggingFace model identifier
    for Hamilton cache fingerprinting.
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
