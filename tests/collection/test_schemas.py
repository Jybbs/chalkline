"""
Tests for collection domain schemas.

Validates Pydantic model constraints for `Posting` and composite key
generation via `Posting.make_id`.
"""

from datetime import date
from pytest   import mark, raises

from chalkline.collection.schemas import Posting


class TestPosting:
    """
    Validate `Posting` model constraints and composite key generation.
    """

    def test_composite_key_format(self):
        """
        The composite key follows `company_title_date` format.
        """
        assert (
            Posting.make_id("Cianbro", date(2026, 3, 1), "Electrician")
            == "cianbro_electrician_2026-03-01"
        )

    @mark.parametrize("date_str", ["2026-03-01 14:30:00", "2026-03-01T14:30:00Z"])
    def test_date_coercion(self, date_str: str, sample_posting: Posting):
        """
        Timestamp `date_posted` values in both space-separated and ISO
        formats are truncated to date by the `[:10]` slice.
        """
        posting = Posting.model_validate(
            sample_posting.model_dump() | {"date_posted" : date_str}
        )
        assert posting.date_posted == date(2026, 3, 1)
        assert posting.id is not None
        assert "2026-03-01" in posting.id

    def test_explicit_id(self):
        """
        An explicitly provided `id` is not overwritten by auto-generation.
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

    def test_extra_fields(self, sample_posting: Posting):
        """
        Unknown fields raise `ValidationError` per `extra="forbid"`.
        """
        with raises(Exception, match="Extra inputs"):
            Posting.model_validate(sample_posting.model_dump() | {"salary": 50000})

    def test_make_id_none_date(self):
        """
        A `None` date produces an "undated" segment in the key.
        """
        assert "undated" in Posting.make_id("Cianbro", None, "Electrician")

    def test_make_id_special_characters(self):
        """
        Spaces, ampersands, and dots are replaced with hyphens.
        """
        assert set(" &.").isdisjoint(Posting.make_id(
            "R.J. Grondin & Sons", date(2026, 1, 1), "Heavy Equip. Operator"
        ))

    def test_make_id_stopword_collision(self):
        """
        Companies differing only by a stopword produce identical composite
        keys, documenting the collision boundary so that changes to stopword
        handling are caught.
        """
        assert (
            Posting.make_id(
                "Reed and Sons", date(2026, 1, 1), "Laborer"
            )
            == Posting.make_id(
                "Reed Sons", date(2026, 1, 1), "Laborer"
            )
        )

    def test_make_id_strips_stopwords(self):
        """
        Stopwords "and", "of", "the" are removed from slugs.
        """
        assert Posting.make_id(
            "The Company of Maine", date(2026, 1, 1), "Operator"
        ).startswith("company-maine_")

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

