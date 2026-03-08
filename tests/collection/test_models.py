"""
Tests for collection domain models and enums.

Validates enum membership and serialization for `ScrapeCategory` and
`SourceType`, Pydantic model constraints for `Posting`, composite
key generation via `Posting.make_id`, `ScrapeCategory` property
mappings, and `ManifestEntry` active derivation.
"""

from datetime import date
from enum     import StrEnum
from pytest   import mark, raises

from chalkline.collection.models import ManifestEntry, Posting
from chalkline.collection.models import ScrapeCategory, SourceType
from tests.conftest              import SAMPLE_DESCRIPTION


# -----------------------------------------------------------------------------
# Enum Tests
# -----------------------------------------------------------------------------


@mark.parametrize("enum_cls", [ScrapeCategory, SourceType])
def test_enum_is_str_enum(enum_cls):
    """
    Collection enums are `StrEnum` subclasses.
    """
    assert issubclass(enum_cls, StrEnum)


@mark.parametrize("enum_cls, expected", [
    (ScrapeCategory, {
        "APPLICATION_ONLY",
        "ENGAGEDTAS",
        "PDF_ONLY",
        "STATIC_HTML",
        "WORKABLE",
        "WORKDAY"
    }),
    (SourceType, {
        "DIRECT_SCRAPE",
        "WORKABLE_API",
        "WORKDAY_API"
    })
])
def test_expected_enum_members(enum_cls, expected):
    """
    Each enum contains exactly its expected member values.
    """
    assert {m.value for m in enum_cls} == expected


# -----------------------------------------------------------------------------
# Posting Model Tests
# -----------------------------------------------------------------------------


class TestPosting:
    """
    Validate the `Posting` Pydantic model constraints.
    """

    def test_auto_id_generation(self):
        """
        Omitting `id` auto-computes it from company, date, and title.
        """
        assert Posting(
            company        = "Cianbro",
            date_collected = date.today(),
            date_posted    = date(2026, 3, 1),
            description    = SAMPLE_DESCRIPTION,
            source_type    = SourceType.DIRECT_SCRAPE,
            source_url     = "https://example.com",
            title          = "Electrician"
        ).id == Posting.make_id(
            company     = "Cianbro",
            date_posted = date(2026, 3, 1),
            title       = "Electrician"
        )

    def test_extra_fields_rejected(self, sample_posting: Posting):
        """
        Unknown fields raise `ValidationError` with `extra="forbid"`.
        """
        with raises(Exception, match="Extra inputs"):
            Posting.model_validate(
                sample_posting.model_dump() | {"salary": 50000}
            )

    def test_minimum_description_length(self):
        """
        Descriptions shorter than 50 characters are rejected.
        """
        with raises(Exception, match="at least 50 characters"):
            Posting(
                company        = "Test",
                date_collected = date.today(),
                date_posted    = None,
                description    = "Too short",
                source_type    = SourceType.DIRECT_SCRAPE,
                source_url     = "https://example.com",
                title          = "Worker"
            )

    def test_valid_posting_roundtrip(self, sample_posting: Posting):
        """
        A valid posting serializes and deserializes without loss.
        """
        assert Posting.model_validate(
            sample_posting.model_dump(mode="json")
        ) == sample_posting

    def test_valid_source_types(self):
        """
        All `SourceType` members are accepted by the model.
        """
        assert all(
            Posting(
                company        = "Test",
                date_collected = date.today(),
                date_posted    = None,
                description    = SAMPLE_DESCRIPTION,
                id             = f"test-{source.value}",
                source_type    = source,
                source_url     = "https://example.com",
                title          = "Worker"
            ).source_type == source
            for source in SourceType
        )


# -----------------------------------------------------------------------------
# Composite ID Tests
# -----------------------------------------------------------------------------


class TestPostingMakeId:
    """
    Validate composite key generation for deduplication.
    """

    def test_deterministic(self):
        """
        Identical inputs produce the same composite key.
        """
        assert Posting.make_id(**(args := {
            "company"     : "Cianbro",
            "date_posted" : date(2026, 3, 1),
            "title"       : "Electrician"
        })) == Posting.make_id(**args)

    def test_none_date(self):
        """
        A `None` date produces an "undated" segment in the key.
        """
        assert "undated" in Posting.make_id(
            company     = "Cianbro",
            date_posted = None,
            title       = "Electrician"
        )

    def test_special_characters_slugified(self):
        """
        Spaces, ampersands, and dots are replaced with hyphens.
        """
        assert not any(c in Posting.make_id(
            company     = "R.J. Grondin & Sons",
            date_posted = date(2026, 1, 1),
            title       = "Heavy Equip. Operator"
        ) for c in " &.")


# -----------------------------------------------------------------------------
# ScrapeCategory Property Tests
# -----------------------------------------------------------------------------


class TestScrapeCategoryProperties:
    """
    Validate `scrapeable` and `source_type` property mappings.
    """

    @mark.parametrize("category", [
        ScrapeCategory.ENGAGEDTAS,
        ScrapeCategory.STATIC_HTML,
        ScrapeCategory.WORKABLE,
        ScrapeCategory.WORKDAY
    ])
    def test_scrapeable_categories(self, category: ScrapeCategory):
        """
        Active scraping categories report as scrapeable.
        """
        assert category.scrapeable

    @mark.parametrize("category", [
        ScrapeCategory.APPLICATION_ONLY,
        ScrapeCategory.PDF_ONLY
    ])
    def test_non_scrapeable_categories(self, category: ScrapeCategory):
        """
        Application-only and PDF-only categories are not scrapeable.
        """
        assert not category.scrapeable

    @mark.parametrize("category, expected", [
        (ScrapeCategory.WORKABLE, SourceType.WORKABLE_API),
        (ScrapeCategory.WORKDAY,  SourceType.WORKDAY_API)
    ])
    def test_api_source_types(self, category: ScrapeCategory, expected: SourceType):
        """
        ATS categories map to their specific API source types.
        """
        assert category.source_type == expected

    @mark.parametrize("category", [
        ScrapeCategory.ENGAGEDTAS,
        ScrapeCategory.STATIC_HTML
    ])
    def test_html_source_type(self, category: ScrapeCategory):
        """
        HTML-based categories map to `DIRECT_SCRAPE`.
        """
        assert category.source_type == SourceType.DIRECT_SCRAPE


# -----------------------------------------------------------------------------
# ManifestEntry Tests
# -----------------------------------------------------------------------------


class TestManifestEntry:
    """
    Validate `ManifestEntry` model constraints and derived fields.
    """

    def test_active_derived_from_scrapeable(self):
        """
        The `active` field is set from `category.scrapeable`
        regardless of input.
        """
        entry = ManifestEntry(
            category = ScrapeCategory.STATIC_HTML,
            company  = "Cianbro",
            source   = "dot_prequal",
            url      = "https://cianbro.com/careers-list"
        )
        assert entry.active is True

    def test_inactive_for_application_only(self):
        """
        Application-only entries are always inactive.
        """
        entry = ManifestEntry(
            category = ScrapeCategory.APPLICATION_ONLY,
            company  = "Test Corp",
            source   = "dot_prequal",
            url      = "https://example.com/apply"
        )
        assert entry.active is False

    def test_extra_fields_rejected(self):
        """
        Unknown fields raise `ValidationError` with `extra="forbid"`.
        """
        with raises(Exception, match="Extra inputs"):
            ManifestEntry(
                category = ScrapeCategory.STATIC_HTML,
                company  = "Test",
                source   = "test",
                url      = "https://example.com",
                notes    = "should not be allowed"
            )


# -----------------------------------------------------------------------------
# Date Parsing Tests
# -----------------------------------------------------------------------------


class TestParseIsoDate:
    """
    Validate ISO date parsing used by the `Posting` model and
    ATS scrapers.
    """

    def test_plain_date(self):
        """
        A standard ISO date string parses correctly.
        """
        assert Posting.parse_iso_date("2026-03-01") == date(2026, 3, 1)

    def test_timestamp_truncated(self):
        """
        Full ISO timestamps are truncated to the date portion.
        """
        assert Posting.parse_iso_date(
            "2026-03-01T14:30:00Z"
        ) == date(2026, 3, 1)

    def test_none_returns_none(self):
        """
        `None` input returns `None` without error.
        """
        assert Posting.parse_iso_date(None) is None

    def test_empty_string_returns_none(self):
        """
        An empty string returns `None` without error.
        """
        assert Posting.parse_iso_date("") is None

    def test_malformed_returns_none(self):
        """
        Unparseable strings return `None` without raising.
        """
        assert Posting.parse_iso_date("not-a-date") is None
