"""
Tests for corpus ordering and pipeline configuration.

Validates `Corpus` description alignment with sorted posting keys and
`PipelineConfig` constraint boundaries.
"""

from pathlib import Path

from pydantic import ValidationError
from pytest   import mark, raises

from chalkline.collection.schemas import Corpus, Posting
from chalkline.pipeline.schemas   import PipelineConfig


def _posting(key: str) -> Posting:
    """
    Minimal posting with a given ID for corpus construction.
    """
    return Posting(
        company     = "Co",
        date_posted = None,
        description = f"{'x' * 50} {key}",
        id          = key,
        source_url  = "https://example.com",
        title       = "Worker"
    )


class TestCorpus:
    """
    Validate corpus key ordering, description alignment, and positional
    access.
    """

    def test_at_retrieves_by_position(self):
        """
        `at` returns postings at the given sorted-key positions,
        preserving the order of the requested indices.
        """
        postings = {k: _posting(k) for k in ["c", "a", "b"]}
        corpus   = Corpus(postings)
        result   = corpus.at([2, 0])
        assert result[0].id == "c"
        assert result[1].id == "a"

    def test_descriptions_aligned(self):
        """
        Descriptions follow the same sorted-key order as `posting_ids`.
        """
        postings = {f"b_{i}": _posting(f"b_{i}") for i in range(3)}
        corpus   = Corpus(postings)
        assert len(corpus.descriptions) == 3
        assert corpus.descriptions[0] == postings[corpus.posting_ids[0]].description

    def test_empty_corpus(self):
        """
        An empty posting dict produces empty IDs and descriptions
        without raising.
        """
        corpus = Corpus({})
        assert corpus.posting_ids  == []
        assert corpus.descriptions == []


class TestPipelineConfig:
    """
    Validate hyperparameter constraints.
    """

    @mark.parametrize(("field", "value"), [
        ("cluster_count",          1),
        ("component_count",        0),
        ("destination_percentile", -1),
        ("destination_percentile", 101),
        ("source_percentile",      -1),
        ("source_percentile",      101),
        ("lateral_neighbors",      0),
        ("upward_neighbors",       0)
    ])
    def test_out_of_range_rejected(self, field: str, value: int):
        """
        Hyperparameters outside their valid ranges are rejected by
        Pydantic field constraints.
        """
        with raises(ValidationError):
            PipelineConfig(
                lexicon_dir  = Path("."),
                postings_dir = Path("."),
                **{field: value}
            )

    def test_hamilton_cache_dir(self):
        """
        The derived cache directory is under `.cache/hamilton`.
        """
        config = PipelineConfig(lexicon_dir=Path("."), postings_dir=Path("."))
        assert config.hamilton_cache_dir == Path(".cache/hamilton")
