"""
Tests for pipeline configuration schemas.

Validates `PipelineConfig` field constraints, defaults, and the
`extra="forbid"` policy that catches typos early.
"""

from pathlib import Path
from pytest  import mark, raises

from chalkline.pipeline.schemas import DistanceMetric, PipelineConfig


class TestPipelineConfig:
    """
    Validate `PipelineConfig` constraints, defaults, and `DistanceMetric`
    enum behavior.
    """

    @staticmethod
    def _config(tmp_path: Path, **overrides) -> PipelineConfig:
        """
        Build a `PipelineConfig` with `tmp_path` for all required
        directories.
        """
        return PipelineConfig(
            lexicon_dir  = tmp_path,
            output_dir   = tmp_path,
            postings_dir = tmp_path,
            **overrides
        )

    @mark.parametrize("field, expected", [
        ("distance_metric",      DistanceMetric.EUCLIDEAN),
        ("max_components",       20),
        ("min_cooccurrence_pct", 0.05),
        ("random_seed",          42),
        ("reference_dir",        Path("data/stakeholder/reference")),
        ("top_k_gaps",           10),
        ("variance_threshold",   0.85)
    ])
    def test_defaults(self, expected, field: str, tmp_path: Path):
        """
        Optional fields receive their documented defaults.
        """
        assert getattr(self._config(tmp_path), field) == expected

    def test_distance_metric_coercion(self, tmp_path: Path):
        """
        A raw string coerces to the correct enum member when passed through
        `PipelineConfig`.
        """
        assert self._config(
            tmp_path, distance_metric="cosine"
        ).distance_metric is DistanceMetric.COSINE

    def test_distance_metric_members(self):
        """
        All expected distance metrics are defined.
        """
        assert set(DistanceMetric) == {"cosine", "euclidean", "standardized_euclidean"}

    def test_extra_fields(self, tmp_path: Path):
        """
        Unknown fields raise `ValidationError` per `extra="forbid"`.
        """
        with raises(Exception, match="Extra inputs"):
            self._config(tmp_path, stale_field=True)

    def test_invalid_distance_metric(self, tmp_path: Path):
        """
        Unrecognized metric strings fail at construction, not downstream in
        sklearn.
        """
        with raises(Exception):
            self._config(tmp_path, distance_metric="manhattan")

    def test_missing_fields(self):
        """
        Omitting required path fields raises `ValidationError`.
        """
        with raises(Exception):
            PipelineConfig()

    def test_override(self, tmp_path: Path):
        """
        Non-default values are accepted for optional fields.
        """
        config = self._config(tmp_path, max_components=50, random_seed=99)
        assert config.max_components == 50
        assert config.random_seed == 99

    @mark.parametrize("threshold", [0.0, 1.5])
    def test_variance_threshold_boundary(
        self,
        threshold : float,
        tmp_path  : Path
    ):
        """
        Values at or beyond `UnitInterval` bounds are rejected.
        """
        with raises(Exception):
            self._config(tmp_path, variance_threshold=threshold)

    def test_variance_threshold_one(self, tmp_path: Path):
        """
        `variance_threshold` of exactly 1.0 is valid, meaning keep all
        variance.
        """
        assert self._config(tmp_path, variance_threshold=1.0).variance_threshold == 1.0
