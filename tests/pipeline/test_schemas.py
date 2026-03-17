"""
Tests for pipeline configuration and shared reference data schemas.

Validates `PipelineConfig` field constraints, defaults, the `extra="forbid"`
policy, and shared `ApprenticeshipContext` and `ProgramRecommendation` data
models.
"""

from pathlib import Path
from pytest  import mark, raises

from chalkline.pipeline.schemas import ApprenticeshipContext
from chalkline.pipeline.schemas import PipelineConfig, ProgramRecommendation


class TestPipelineConfig:
    """
    Validate `PipelineConfig` constraints and defaults.
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

    def test_cooccurrence_float(self, tmp_path: Path):
        """
        `min_cooccurrence` accepts a float fraction, not only the
        default `"auto"` string.
        """
        config = self._config(tmp_path, min_cooccurrence=0.05)
        assert config.min_cooccurrence == 0.05

    @mark.parametrize("field, expected", [
        ("distance_metric",      "euclidean"),
        ("max_components",       20),
        ("min_cooccurrence",     "auto"),
        ("random_seed",          42),
        ("top_k_gaps",           10),
        ("variance_threshold",   0.85)
    ])
    def test_defaults(self, expected, field: str, tmp_path: Path):
        """
        Optional fields receive their documented defaults.
        """
        assert getattr(self._config(tmp_path), field) == expected

    def test_extra_fields(self, tmp_path: Path):
        """
        Unknown fields raise `ValidationError` per `extra="forbid"`.
        """
        with raises(Exception, match="Extra inputs"):
            self._config(tmp_path, stale_field=True)

    def test_missing_fields(self):
        """
        Omitting required path fields raises `ValidationError`.
        """
        with raises(Exception):
            PipelineConfig()

    @mark.parametrize("threshold", [0.0, 1.5])
    def test_variance_threshold_boundary(self, threshold: float, tmp_path: Path):
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

    def test_apprenticeship_extra(self):
        """
        Unknown fields are rejected per extra="forbid".
        """
        with raises(Exception, match="Extra inputs"):
            ApprenticeshipContext(
                min_hours   = 8000,
                prefixes    = {"elec"},
                rapids_code = "90046",
                title       = "Electrician",
                unknown     = True
            )

    def test_program_extra(self):
        """
        Unknown fields are rejected per extra="forbid".
        """
        with raises(Exception, match="Extra inputs"):
            ProgramRecommendation(
                credential  = "AAS",
                institution = "CMCC",
                prefixes    = {"test"},
                program     = "Test",
                unknown     = True,
                url         = "https://example.com"
            )
