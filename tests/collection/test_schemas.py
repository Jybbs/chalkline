"""
Tests for collection domain schemas.

Validates Pydantic model constraints for `Posting` and composite
key generation via `Posting.make_id`.
"""

from datetime import date
from pytest   import mark, raises

from chalkline.collection.schemas import Posting


class TestPosting:
    """
    Validate `Posting` model constraints and composite key
    generation.
    """

    def test_auto_id_generation(self, sample_posting: Posting):
        """
        Omitting `id` auto-computes it from company, date, and
        title.
        """
        assert sample_posting.id == Posting.make_id(
            company     = sample_posting.company,
            date_posted = sample_posting.date_posted,
            title       = sample_posting.title
        )

    def test_auto_id_with_string_date(self):
        """
        String `date_posted` values are coerced during auto-ID
        generation.
        """
        posting = Posting.model_validate({
            "company"     : "Cianbro",
            "date_posted" : "2026-03-01",
            "description" : "x" * 50,
            "source_url"  : "https://example.com",
            "title"       : "Electrician"
        })
        assert "2026-03-01" in posting.id

    def test_auto_id_with_timestamp_date(self):
        """
        Trailing timestamps are truncated by the `CoercedDate`
        field validator before Pydantic parses.
        """
        posting = Posting.model_validate({
            "company"     : "Cianbro",
            "date_posted" : "2026-03-01T14:30:00Z",
            "description" : "x" * 50,
            "source_url"  : "https://example.com",
            "title"       : "Electrician"
        })
        assert posting.date_posted == date(2026, 3, 1)

    def test_composite_key_format(self):
        """
        The composite key follows `company_title_date` format.
        """
        assert Posting.make_id(
            company     = "Cianbro",
            date_posted = date(2026, 3, 1),
            title       = "Electrician"
        ) == "cianbro_electrician_2026-03-01"

    @mark.parametrize("field", ["company", "source_url", "title"])
    def test_empty_string_rejected(
        self,
        sample_posting : Posting,
        field          : str
    ):
        """
        Empty strings on `NonEmptyStr` fields raise
        `ValidationError`.
        """
        with raises(Exception, match="at least 1 character"):
            Posting.model_validate(
                sample_posting.model_dump() | {field: ""}
            )

    def test_explicit_id_preserved(self):
        """
        An explicitly provided `id` is not overwritten by
        auto-generation.
        """
        posting = Posting(
            company     = "Cianbro",
            date_posted = date(2026, 3, 1),
            description = "x" * 50,
            id          = "custom-id",
            source_url  = "https://example.com",
            title       = "Worker"
        )
        assert posting.id == "custom-id"

    def test_extra_fields_rejected(self, sample_posting: Posting):
        """
        Unknown fields raise `ValidationError` per
        `extra="forbid"`.
        """
        with raises(Exception, match="Extra inputs"):
            Posting.model_validate(
                sample_posting.model_dump() | {"salary": 50000}
            )

    def test_make_id_none_date(self):
        """
        A `None` date produces an "undated" segment in the key.
        """
        assert "undated" in Posting.make_id(
            company     = "Cianbro",
            date_posted = None,
            title       = "Electrician"
        )

    def test_make_id_special_characters(self):
        """
        Spaces, ampersands, and dots are replaced with hyphens.
        """
        assert set(" &.").isdisjoint(Posting.make_id(
            company     = "R.J. Grondin & Sons",
            date_posted = date(2026, 1, 1),
            title       = "Heavy Equip. Operator"
        ))

    def test_minimum_description_length(self):
        """
        Descriptions shorter than 50 characters are rejected.
        """
        with raises(Exception, match="at least 50 characters"):
            Posting(
                company     = "Test",
                date_posted = None,
                description = "Too short",
                source_url  = "https://example.com",
                title       = "Worker"
            )

    def test_valid_posting_roundtrip(self, sample_posting: Posting):
        """
        A valid posting serializes and deserializes without loss.
        """
        assert Posting.model_validate(
            sample_posting.model_dump(mode="json")
        ) == sample_posting
