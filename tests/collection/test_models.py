"""
Tests for collection domain models and enums.

Validates enum membership and serialization for `ScrapeCategory` and
`SourceType`, Pydantic model constraints for `Posting`, and composite
key generation via `make_posting_id`.
"""

from datetime import date
from pytest   import mark, raises

from chalkline.collection.models import make_posting_id
from chalkline.collection.models import Posting, ScrapeCategory, SourceType
from tests.conftest              import SAMPLE_DESCRIPTION


# -----------------------------------------------------------------------------
# Enum Tests
# -----------------------------------------------------------------------------


@mark.parametrize("enum_cls", [ScrapeCategory, SourceType])
def test_all_enum_members_are_strings(enum_cls):
    """
    Every member of each `StrEnum` serializes as its string value.
    """
    for member in enum_cls:
        assert isinstance(member.value, str)
        assert member == member.value


class TestScrapeCategory:
    """
    Validate `ScrapeCategory` enum membership.
    """

    def test_expected_members(self):
        """
        The enum contains exactly the expected scraping approaches.
        """
        expected = {
            "APPLICATION_ONLY",
            "ENGAGEDTAS",
            "PDF_ONLY",
            "STATIC_HTML",
            "WORKABLE",
            "WORKDAY"
        }
        assert {m.value for m in ScrapeCategory} == expected


class TestSourceType:
    """
    Validate `SourceType` enum membership.
    """

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
        ).id == make_posting_id(
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


class TestMakePostingId:
    """
    Validate composite key generation for deduplication.
    """

    def test_deterministic(self):
        """
        Identical inputs produce the same composite key.
        """
        args = {
            "company"     : "Cianbro",
            "date_posted" : date(2026, 3, 1),
            "title"       : "Electrician"
        }
        assert make_posting_id(**args) == make_posting_id(**args)

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
        assert not any(c in result for c in " &.")
