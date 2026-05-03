"""
Tests for Hamilton DAG node functions in `steps`.

Validates that individual pipeline nodes produce correct shapes and types
when fed synthetic fixture data, catching transform contract violations
that would silently corrupt downstream results.
"""

import numpy as np

from pathlib import Path
from pytest  import raises

from chalkline.pipeline         import steps
from chalkline.pipeline.schemas import PipelineConfig


class TestAssignments:
    """
    Validate consensus clustering node.
    """

    def test_deterministic(
        self,
        pipeline_config : PipelineConfig,
        raw_vectors     : np.ndarray
    ):
        """
        Identical inputs produce identical labels because the internal seed
        loop is `range(consensus_seeds)` rather than a sampled subset.
        """
        first  = steps.assignments(pipeline_config, raw_vectors)
        second = steps.assignments(pipeline_config, raw_vectors)

        assert np.array_equal(first, second)

    def test_shape_and_range(
        self,
        pipeline_config : PipelineConfig,
        raw_vectors     : np.ndarray
    ):
        """
        One label per posting, every label inside `[0, cluster_count)`.
        """
        labels = steps.assignments(pipeline_config, raw_vectors)

        assert labels.shape == (raw_vectors.shape[0],)
        assert labels.min() >= 0
        assert labels.max() < pipeline_config.cluster_count


class TestCorpus:
    """
    Validate corpus loading node.
    """

    def test_empty_raises(self, tmp_path: Path):
        """
        An empty postings directory raises `FileNotFoundError`.
        """
        (tmp_path / "corpus.json").write_text("[]")
        with raises(FileNotFoundError):
            steps.corpus(PipelineConfig(
                lexicon_dir  = tmp_path,
                postings_dir = tmp_path
            ))
