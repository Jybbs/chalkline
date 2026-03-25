"""
Tests for Hamilton DAG node functions in `steps`.

Validates that individual pipeline nodes produce correct shapes and types
when fed synthetic fixture data, catching transform contract violations that
would silently corrupt downstream results.
"""

from pathlib import Path
from pytest  import raises

from chalkline.pipeline         import steps
from chalkline.pipeline.schemas import PipelineConfig


class TestClusters:
    """
    Validate unified cluster construction.
    """

    def test_cluster_count(self, cluster_ids, clusters):
        """
        One cluster per assignment.
        """
        assert len(clusters) == len(cluster_ids)

    def test_cluster_has_titles(self, clusters):
        """
        Every cluster carries both an O*NET occupation title and a
        modal posting title.
        """
        for cluster in clusters.values():
            assert cluster.soc_title
            assert cluster.modal_title

    def test_cluster_jz_range(self, clusters):
        """
        Job Zones are within the O*NET 1-5 range.
        """
        for cluster in clusters.values():
            assert 1 <= cluster.job_zone <= 5


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
                output_dir   = tmp_path,
                postings_dir = tmp_path
            ))
