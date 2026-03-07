"""
Tests for storage, manifest generation, and corpus statistics.

Validates save/load round-tripping, deduplication, manifest URL
classification, tracking parameter removal, and statistics
reporting.
"""

from datetime import date
from json     import dumps, loads
from pathlib  import Path
from pytest   import fixture, skip

from chalkline.collection.manifest import generate, _strip_tracking_params
from chalkline.collection.models   import Posting, SourceType, make_posting_id
from chalkline.collection.stats    import CorpusStats
from chalkline.collection.storage  import deduplicate, load, save


SAMPLE_DESCRIPTION = (
    "Seeking an experienced electrician for commercial construction "
    "projects. Must have valid journeyman license and OSHA 10 "
    "certification. Responsibilities include conduit bending, "
    "blueprint reading, and NEC code compliance."
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@fixture
def sample_posting() -> Posting:
    """
    A minimal valid posting for testing.
    """
    return Posting(
        company        = "Cianbro",
        date_collected = date(2026, 3, 5),
        date_posted    = date(2026, 3, 1),
        description    = SAMPLE_DESCRIPTION,
        id             = make_posting_id(
            company     = "Cianbro",
            date_posted = date(2026, 3, 1),
            title       = "Electrician"
        ),
        source_type    = SourceType.DIRECT_SCRAPE,
        source_url     = "https://www.cianbro.com/careers-list",
        title          = "Electrician"
    )


@fixture
def reference_dir(tmp_path: Path) -> Path:
    """
    Temporary reference directory with a copy of `career_urls.json`.

    Uses the real stakeholder data when available, falling back to a
    single synthetic entry for CI environments without the data.
    """
    ref_dir  = tmp_path / "reference"
    ref_dir.mkdir()
    src_path = (
        Path(__file__).parent.parent.parent
        / "data" / "stakeholder" / "reference" / "career_urls.json"
    )
    if src_path.exists():
        (ref_dir / "career_urls.json").write_text(
            src_path.read_text(encoding="utf-8"),
            encoding="utf-8"
        )
    else:
        (ref_dir / "career_urls.json").write_text(
            dumps([
                {
                    "company" : "Test Corp",
                    "source"  : "dot_prequal",
                    "url"     : "https://testcorp.com/careers/"
                }
            ]),
            encoding="utf-8"
        )
    return ref_dir


# -----------------------------------------------------------------------------
# Storage Tests
# -----------------------------------------------------------------------------


class TestStorage:
    """
    Validate save, load, and incremental collection behavior.
    """

    def test_incremental_save(self, sample_posting: Posting, tmp_path: Path):
        """
        Saving new postings merges with existing ones on disk.
        """
        save(
            postings     = [sample_posting],
            postings_dir = tmp_path
        )

        second = Posting(
            company        = "Reed & Reed",
            date_collected = date(2026, 3, 5),
            date_posted    = None,
            description    = SAMPLE_DESCRIPTION,
            id             = make_posting_id(
                company     = "Reed & Reed",
                date_posted = None,
                title       = "Laborer"
            ),
            source_type    = SourceType.DIRECT_SCRAPE,
            source_url     = "https://reed-reed.com/jobs/",
            title          = "Laborer"
        )
        save(
            postings     = [second],
            postings_dir = tmp_path
        )

        assert len(load(tmp_path)) == 2

    def test_load_empty_directory(self, tmp_path: Path):
        """
        Loading from a non-existent corpus returns an empty list.
        """
        assert load(tmp_path / "nonexistent") == []

    def test_save_and_load_roundtrip(self, sample_posting: Posting, tmp_path: Path):
        """
        Save followed by load produces identical postings.
        """
        save(
            postings     = [sample_posting],
            postings_dir = tmp_path
        )
        loaded = load(tmp_path)
        assert len(loaded) == 1
        assert loaded[0] == sample_posting


# -----------------------------------------------------------------------------
# Deduplication Tests
# -----------------------------------------------------------------------------


class TestDeduplication:
    """
    Validate that deduplication retains the most recently collected
    version.
    """

    def test_duplicate_keeps_latest(self, sample_posting: Posting):
        """
        When two postings share an ID, the later `date_collected`
        wins.
        """
        older = sample_posting.model_copy(update={
            "date_collected" : date(2026, 3, 1)
        })
        newer = sample_posting.model_copy(update={
            "date_collected" : date(2026, 3, 5),
            "description"    : (
                "Updated description with new content that is long "
                "enough to pass the 50-character minimum validation."
            )
        })

        result = deduplicate([older, newer])
        assert len(result) == 1
        assert result[0].date_collected == date(2026, 3, 5)

    def test_no_duplicates_preserved(self, sample_posting: Posting):
        """
        Non-duplicate postings are all retained.
        """
        other = Posting(
            company        = "Reed & Reed",
            date_collected = date.today(),
            date_posted    = None,
            description    = SAMPLE_DESCRIPTION,
            id             = "reed-reed_laborer_undated",
            source_type    = SourceType.DIRECT_SCRAPE,
            source_url     = "https://reed-reed.com/jobs/",
            title          = "Laborer"
        )
        assert len(deduplicate([sample_posting, other])) == 2


# -----------------------------------------------------------------------------
# Manifest Tests
# -----------------------------------------------------------------------------


class TestManifest:
    """
    Validate manifest generation, URL classification, and cleaning.
    """

    def test_all_urls_classified(self, reference_dir: Path, tmp_path: Path):
        """
        Every URL from `career_urls.json` appears in the manifest.
        """
        src = loads(
            (reference_dir / "career_urls.json").read_text(
                encoding="utf-8"
            )
        )
        entries = generate(
            output_dir    = tmp_path / "postings",
            reference_dir = reference_dir
        )
        assert len(entries) == len(src)

    def test_generate_creates_file(self, reference_dir: Path, tmp_path: Path):
        """
        Manifest generation writes `manifest.json` to the output
        dir.
        """
        output_dir = tmp_path / "postings"
        generate(
            output_dir    = output_dir,
            reference_dir = reference_dir
        )
        assert (output_dir / "manifest.json").exists()

    def test_inactive_count(self, reference_dir: Path, tmp_path: Path):
        """
        Exactly 9 URLs are marked inactive with the full dataset.

        Skips when the real `career_urls.json` is not available.
        """
        src_path = (
            Path(__file__).parent.parent.parent
            / "data" / "stakeholder" / "reference"
            / "career_urls.json"
        )
        if not src_path.exists():
            skip("Full career_urls.json not available")

        entries = generate(
            output_dir    = tmp_path / "postings",
            reference_dir = reference_dir
        )
        assert len([e for e in entries if not e.active]) == 9

    def test_no_tracking_params(self, reference_dir: Path, tmp_path: Path):
        """
        No manifest URL contains Google Ads tracking parameters.
        """
        entries = generate(
            output_dir    = tmp_path / "postings",
            reference_dir = reference_dir
        )
        tracking = {
            "gad_campaignid",
            "gad_source",
            "gbraid",
            "gclid",
            "srsltid"
        }
        for entry in entries:
            for param in tracking:
                assert param not in entry.url


class TestStripTrackingParams:
    """
    Validate tracking parameter removal from URLs.
    """

    def test_preserves_clean_url(self):
        """
        URLs without tracking parameters pass through unchanged.
        """
        url = "https://example.com/careers/"
        assert _strip_tracking_params(url) == url

    def test_removes_gclid(self):
        """
        Google click ID is stripped while preserving other
        parameters.
        """
        result = _strip_tracking_params(
            "https://example.com/jobs?gclid=abc123&page=1"
        )
        assert "gclid" not in result
        assert "page=1" in result

    def test_removes_multiple_params(self):
        """
        All five tracking parameter types are stripped
        simultaneously.
        """
        result = _strip_tracking_params(
            "https://example.com/careers/"
            "?gad_source=1&gad_campaignid=123"
            "&gbraid=abc&gclid=def&srsltid=ghi"
        )
        assert "?" not in result or "gad" not in result


# -----------------------------------------------------------------------------
# Corpus Statistics Tests
# -----------------------------------------------------------------------------


class TestCorpusStats:
    """
    Validate corpus statistics reporting.
    """

    def test_empty_corpus(self):
        """
        The reporter handles an empty corpus without error.
        """
        stats = CorpusStats([])
        assert stats.total == 0
        assert stats.company_counts == {}
        assert stats.date_range is None
        assert stats.source_type_counts == {}
        assert "Total postings: 0" in stats.report()

    def test_populated_corpus(self, sample_posting: Posting):
        """
        Statistics reflect the postings in the corpus.
        """
        stats = CorpusStats([sample_posting])
        assert stats.total == 1
        assert stats.company_counts == {"Cianbro": 1}
        assert stats.date_range is not None
        assert "DIRECT_SCRAPE" in stats.source_type_counts
