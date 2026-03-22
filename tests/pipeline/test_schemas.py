"""
Tests for pipeline configuration, cluster structure, and corpus ordering.

Validates `PipelineConfig` defaults, `ClusterAssignments` per-cluster
aggregation shapes and normalization, and `Corpus` description alignment
with sorted posting keys.
"""

import numpy as np

from pathlib import Path
from pytest  import mark

from chalkline.collection.schemas import Corpus, Posting
from chalkline.pipeline.schemas   import PipelineConfig


class TestClusterAssignments:
    """
    Validate cluster structure derivation and per-cluster aggregation.
    """

    def test_centroids_shape(self, assignments, coordinates):
        """
        One centroid row per cluster in the SVD-reduced space.
        """
        assert assignments.centroids(coordinates).shape == (
            len(assignments.cluster_ids),
            coordinates.shape[1]
        )

    def test_cluster_vectors_unit(self, assignments, raw_vectors):
        """
        Cluster vectors are L2-normalized for cosine similarity.
        """
        np.testing.assert_allclose(
            np.linalg.norm(assignments.cluster_vectors(raw_vectors), axis=1),
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
    Validate `PipelineConfig` defaults.
    """

    @mark.parametrize("field, expected", [
        ("cluster_count",          20),
        ("component_count",        10),
        ("destination_percentile", 5),
        ("embedding_model",        "all-mpnet-base-v2"),
        ("lateral_neighbors",      2),
        ("random_seed",            42),
        ("soc_neighbors",          3),
        ("source_percentile",      75),
        ("upward_neighbors",       2)
    ])
    def test_defaults(self, expected, field: str, tmp_path: Path):
        """
        Optional fields receive their documented defaults.
        """
        config = PipelineConfig(
            lexicon_dir  = tmp_path,
            output_dir   = tmp_path,
            postings_dir = tmp_path
        )
        assert getattr(config, field) == expected
