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


class TestProfiles:
    """
    Validate cluster profile construction.
    """

    def test_profile_count(self, profiles, cluster_ids):
        """
        One profile per cluster.
        """
        assert len(profiles) == len(cluster_ids)

    def test_profile_jz_range(self, profiles):
        """
        Job Zones are within the O*NET 1-5 range.
        """
        for profile in profiles.values():
            assert 1 <= profile.job_zone <= 5

    def test_profile_has_titles(self, profiles):
        """
        Every profile carries both an O*NET occupation title and a modal
        posting title.
        """
        for profile in profiles.values():
            assert profile.soc_title
            assert profile.modal_title
