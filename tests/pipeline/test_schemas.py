"""
Tests for pipeline configuration, cluster structure, and corpus ordering.

Validates `PipelineConfig` defaults, centroid and cluster vector shapes,
and `Corpus` description alignment with sorted posting keys.
"""

import numpy as np

from pathlib import Path
from pytest  import mark

from chalkline.collection.schemas import Corpus, Posting
from chalkline.pipeline.schemas   import PipelineConfig


class TestClusterStructure:
    """
    Validate centroid and cluster vector shapes and normalization.
    """

    def test_centroids_shape(self, centroids, cluster_ids, coordinates):
        """
        One centroid row per cluster in the SVD-reduced space.
        """
        assert centroids.shape == (len(cluster_ids), coordinates.shape[1])

    def test_cluster_vectors_unit(self, cluster_vectors):
        """
        Cluster vectors are L2-normalized for cosine similarity.
        """
        np.testing.assert_allclose(
            np.linalg.norm(cluster_vectors, axis=1),
            1.0,
            atol=1e-6
        )


class TestCorpus:
    """
    Validate corpus key ordering and description alignment.
    """

    def test_descriptions_aligned(self):
        """
        Descriptions follow the same sorted-key order as `posting_ids`.
        """
        postings = {
            f"b_{i}": Posting(
                company     = "Co",
                date_posted = None,
                description = f"{'x' * 50} {i}",
                id          = f"b_{i}",
                source_url  = "https://example.com",
                title       = "Worker"
            )
            for i in range(3)
        }
        corpus = Corpus(postings)
        assert len(corpus.descriptions) == 3
        assert corpus.descriptions[0] == postings[corpus.posting_ids[0]].description


class TestPipelineConfig:
    """
    Validate default hyperparameters.
    """

    model = "Alibaba-NLP/gte-base-en-v1.5"

    @mark.parametrize("expected, field", [
        (20,    "cluster_count"),
        (10,    "component_count"),
        (5,     "destination_percentile"),
        (model, "embedding_model"),
        (2,     "lateral_neighbors"),
        (42,    "random_seed"),
        (3,     "soc_neighbors"),
        (75,    "source_percentile"),
        (2,     "upward_neighbors")
    ])
    def test_defaults(self, expected, field: str, tmp_path: Path):
        """
        Each hyperparameter has the expected default value.
        """
        config = PipelineConfig(
            lexicon_dir  = tmp_path,
            output_dir   = tmp_path,
            postings_dir = tmp_path
        )
        assert getattr(config, field) == expected
