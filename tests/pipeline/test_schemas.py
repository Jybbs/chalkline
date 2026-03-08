"""
Tests for pipeline configuration schemas.

Validates `PipelineConfig` field constraints, defaults, and the
`extra="forbid"` policy that catches typos early.
"""

from pathlib import Path
from pytest  import raises

from chalkline.pipeline.schemas import DistanceMetric, PipelineConfig


class TestDistanceMetric:
    """
    Validate `DistanceMetric` enum values and serialization.
    """

    def test_member_count(self):
        """
        Exactly three distance metrics are defined.
        """
        assert len(DistanceMetric) == 3

    def test_member_values(self):
        """
        Each member serializes to its expected lowercase string.
        """
        assert DistanceMetric.COSINE                 == "cosine"
        assert DistanceMetric.EUCLIDEAN              == "euclidean"
        assert DistanceMetric.STANDARDIZED_EUCLIDEAN == "standardized_euclidean"

    def test_string_coercion_in_config(self, tmp_path: Path):
        """
        A raw string coerces to the correct enum member when
        passed through `PipelineConfig`.
        """
        config = PipelineConfig(
            distance_metric = "cosine",
            lexicon_dir     = tmp_path,
            output_dir      = tmp_path,
            postings_dir    = tmp_path
        )
        assert config.distance_metric is DistanceMetric.COSINE


class TestPipelineConfig:
    """
    Validate `PipelineConfig` constraints and defaults.
    """

    def test_defaults_applied(self, tmp_path: Path):
        """
        Optional fields receive their documented defaults.
        """
        config = PipelineConfig(
            lexicon_dir  = tmp_path / "lexicons",
            output_dir   = tmp_path / "output",
            postings_dir = tmp_path / "postings"
        )
        assert config.distance_metric == DistanceMetric.EUCLIDEAN
        assert config.max_components == 20
        assert config.min_cooccurrence_pct == 0.05
        assert config.random_seed == 42
        assert config.reference_dir == Path("data/stakeholder/reference")
        assert config.top_k_gaps == 10
        assert config.variance_threshold == 0.85

    def test_extra_fields_rejected(self, tmp_path: Path):
        """
        Unknown fields raise `ValidationError` per
        `extra="forbid"`.
        """
        with raises(Exception, match="Extra inputs"):
            PipelineConfig(
                lexicon_dir  = tmp_path,
                output_dir   = tmp_path,
                postings_dir = tmp_path,
                stale_field  = True
            )

    def test_missing_required_fields(self):
        """
        Omitting required path fields raises `ValidationError`.
        """
        with raises(Exception):
            PipelineConfig()

    def test_random_seed_propagates(self, tmp_path: Path):
        """
        `random_seed` is present and accessible for stochastic
        steps.
        """
        config = PipelineConfig(
            lexicon_dir  = tmp_path,
            output_dir   = tmp_path,
            postings_dir = tmp_path,
            random_seed  = 123
        )
        assert config.random_seed == 123

    def test_variance_threshold_rejects_zero(self, tmp_path: Path):
        """
        `variance_threshold` must be strictly greater than zero.
        """
        with raises(Exception):
            PipelineConfig(
                lexicon_dir        = tmp_path,
                output_dir         = tmp_path,
                postings_dir       = tmp_path,
                variance_threshold = 0.0
            )

    def test_variance_threshold_accepts_one(self, tmp_path: Path):
        """
        `variance_threshold` of exactly 1.0 is valid, meaning
        keep all variance.
        """
        config = PipelineConfig(
            lexicon_dir        = tmp_path,
            output_dir         = tmp_path,
            postings_dir       = tmp_path,
            variance_threshold = 1.0
        )
        assert config.variance_threshold == 1.0

    def test_variance_threshold_rejects_above_one(
        self,
        tmp_path : Path
    ):
        """
        `variance_threshold` above 1.0 violates the `UnitInterval`
        upper bound.
        """
        with raises(Exception):
            PipelineConfig(
                lexicon_dir        = tmp_path,
                output_dir         = tmp_path,
                postings_dir       = tmp_path,
                variance_threshold = 1.5
            )
