"""
Tests for crawl manifest generation.

Validates that all stakeholder URLs appear in the manifest, inactive
counts match expectations, and tracking parameters are stripped.
"""

from json    import dumps, loads
from pathlib import Path
from pytest  import fixture, skip

from chalkline.collection.manifest import generate


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@fixture
def reference_dir(tmp_path: Path) -> Path:
    """
    Temporary reference directory with a copy of `career_urls.json`.

    Uses the real stakeholder data when available, falling back to
    a synthetic entry for CI environments without the data.
    """
    real_path = (
        Path(__file__).parents[2]
        / "data/stakeholder/reference/career_urls.json"
    )

    (ref_dir := tmp_path / "reference").mkdir()
    (ref_dir / "career_urls.json").write_text(
        real_path.read_text(encoding="utf-8")
        if real_path.exists()
        else dumps([{
            "category" : "STATIC_HTML",
            "company"  : "Test Corp",
            "source"   : "dot_prequal",
            "url"      : "https://testcorp.com/careers/"
        }]),
        encoding="utf-8"
    )
    return ref_dir


# -----------------------------------------------------------------------------
# Manifest Tests
# -----------------------------------------------------------------------------


class TestManifest:
    """
    Validate manifest generation and URL cleaning.
    """

    def test_all_urls_classified(self, reference_dir: Path, tmp_path: Path):
        """
        Every URL from `career_urls.json` appears in the manifest.
        """
        assert len(generate(
            output_dir    = tmp_path / "postings",
            reference_dir = reference_dir
        )) == len(loads(
            (reference_dir / "career_urls.json").read_bytes()
        ))

    def test_generate_creates_file(self, reference_dir: Path, tmp_path: Path):
        """
        Manifest generation writes `manifest.json` to the output
        dir.
        """
        generate(
            output_dir    = (output_dir := tmp_path / "postings"),
            reference_dir = reference_dir
        )
        assert (output_dir / "manifest.json").exists()

    def test_inactive_count(
        self,
        reference_dir : Path,
        tmp_path      : Path
    ):
        """
        Exactly 9 URLs are marked inactive with the full dataset.

        Skips when the real `career_urls.json` is not available.
        """
        if not (
            Path(__file__).parents[2]
            / "data/stakeholder/reference/career_urls.json"
        ).exists():
            skip("Full career_urls.json not available")

        assert sum(
            not e.active for e in generate(
                output_dir    = tmp_path / "postings",
                reference_dir = reference_dir
            )
        ) == 9

    def test_no_query_params(self, reference_dir: Path, tmp_path: Path):
        """
        No manifest URL contains query parameters.
        """
        assert all(
            "?" not in entry.url
            for entry in generate(
                output_dir    = tmp_path / "postings",
                reference_dir = reference_dir
            )
        )
