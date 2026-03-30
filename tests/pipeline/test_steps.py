"""
Tests for Hamilton DAG node functions in `steps`.

Validates that individual pipeline nodes produce correct shapes and types
when fed synthetic fixture data, catching transform contract violations
that would silently corrupt downstream results.
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
