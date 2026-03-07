"""
Tests for collection domain models and enums.

Validates enum membership and serialization for `ScrapeCategory` and
`SourceType`, Pydantic model constraints for `Posting`, and composite
key generation via `make_posting_id`.
"""

from datetime import date
from pytest   import fixture, raises

from chalkline.collection.models import Posting, ScrapeCategory, SourceType
from chalkline.collection.models import make_posting_id


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


# -----------------------------------------------------------------------------
# Enum Tests
# -----------------------------------------------------------------------------


class TestScrapeCategory:
    """
    Validate `ScrapeCategory` enum membership and string
    serialization.
    """

    def test_all_members_are_strings(self):
        """
        Every `ScrapeCategory` member serializes as its string value.
        """
        for member in ScrapeCategory:
            assert isinstance(member.value, str)
            assert member == member.value

    def test_expected_members(self):
        """
        The enum contains exactly the expected scraping approaches.
        """
        expected = {
            "APPLICATION_ONLY",
            "CIANBRO",
            "ENGAGEDTAS",
            "PDF_ONLY",
            "STATIC_HTML",
            "WORKABLE",
            "WORKDAY"
        }
        assert {m.value for m in ScrapeCategory} == expected


class TestSourceType:
    """
    Validate `SourceType` enum membership and string serialization.
    """

    def test_all_members_are_strings(self):
        """
        Every `SourceType` member serializes as its string value.
        """
        for member in SourceType:
            assert isinstance(member.value, str)
            assert member == member.value

    def test_expected_members(self):
        """
        The enum contains exactly the expected acquisition methods.
        """
        expected = {
            "AGC_EXPORT",
            "AGGREGATOR",
            "ATS_SCRAPE",
            "DIRECT_SCRAPE",
            "WORKABLE_API",
            "WORKDAY_API"
        }
        assert {m.value for m in SourceType} == expected


# -----------------------------------------------------------------------------
# Posting Model Tests
# -----------------------------------------------------------------------------


class TestPosting:
    """
    Validate the `Posting` Pydantic model constraints.
    """

    def test_extra_fields_rejected(self, sample_posting: Posting):
        """
        Unknown fields raise `ValidationError` with `extra="forbid"`.
        """
        data = sample_posting.model_dump()
        data["salary"] = 50000
        with raises(Exception, match="Extra inputs"):
            Posting.model_validate(data)

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
                id             = "test-id",
                source_type    = SourceType.DIRECT_SCRAPE,
                source_url     = "https://example.com",
                title          = "Worker"
            )

    def test_valid_posting_roundtrip(self, sample_posting: Posting):
        """
        A valid posting serializes and deserializes without loss.
        """
        roundtripped = Posting.model_validate(
            sample_posting.model_dump(mode="json")
        )
        assert roundtripped == sample_posting

    def test_valid_source_types(self):
        """
        All `SourceType` members are accepted by the model.
        """
        for source in SourceType:
            posting = Posting(
                company        = "Test",
                date_collected = date.today(),
                date_posted    = None,
                description    = SAMPLE_DESCRIPTION,
                id             = f"test-{source.value}",
                source_type    = source,
                source_url     = "https://example.com",
                title          = "Worker"
            )
            assert posting.source_type == source


# -----------------------------------------------------------------------------
# Composite ID Tests
# -----------------------------------------------------------------------------


class TestMakePostingId:
    """
    Validate composite key generation for deduplication.
    """

    def test_deterministic(self):
        """
        Identical inputs produce the same composite key.
        """
        id1 = make_posting_id(
            company     = "Cianbro",
            date_posted = date(2026, 3, 1),
            title       = "Electrician"
        )
        id2 = make_posting_id(
            company     = "Cianbro",
            date_posted = date(2026, 3, 1),
            title       = "Electrician"
        )
        assert id1 == id2

    def test_none_date(self):
        """
        A `None` date produces an "undated" segment in the key.
        """
        assert "undated" in make_posting_id(
            company     = "Cianbro",
            date_posted = None,
            title       = "Electrician"
        )

    def test_special_characters_slugified(self):
        """
        Spaces, ampersands, and dots are replaced with hyphens.
        """
        result = make_posting_id(
            company     = "R.J. Grondin & Sons",
            date_posted = date(2026, 1, 1),
            title       = "Heavy Equip. Operator"
        )
        assert " " not in result
        assert "&" not in result
        assert "." not in result
