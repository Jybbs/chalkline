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

    @mark.parametrize("field, expected", [
        ("cluster_count",          20),
        ("component_count",        10),
        ("destination_percentile", 5),
        ("embedding_model",        "all-mpnet-base-v2"),
        ("lateral_neighbors",      2),
        ("max_gaps",               10),
        ("random_seed",            42),
        ("soc_neighbors",          3),
        ("source_percentile",      75),
        ("upward_neighbors",       2)
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
            PipelineConfig.model_validate({})

    def test_apprenticeship_extra(self):
        """
        Unknown fields are rejected per extra="forbid".
        """
        with raises(Exception, match="Extra inputs"):
            ApprenticeshipContext.model_validate({
                "min_hours"   : 8000,
                "prefixes"    : {"elec"},
                "rapids_code" : "90046",
                "title"       : "Electrician",
                "unknown"     : True
            })

    def test_program_extra(self):
        """
        Unknown fields are rejected per extra="forbid".
        """
        with raises(Exception, match="Extra inputs"):
            ProgramRecommendation.model_validate({
                "credential"  : "AAS",
                "institution" : "CMCC",
                "prefixes"    : {"test"},
                "program"     : "Test",
                "unknown"     : True,
                "url"         : "https://example.com"
            })
