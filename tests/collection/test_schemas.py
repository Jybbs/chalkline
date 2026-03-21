"""
Tests for collection domain schemas.

Validates Pydantic model constraints for `Posting` and composite key
generation via the `id` field.
"""

from datetime import date
from pytest   import mark, raises

from chalkline.collection.schemas import Posting


class TestPosting:
    """
    Validate `Posting` model constraints and composite key generation.
    """

    def test_composite_key_format(self, posting):
        """
        The composite key follows `company_title_date` format.
        """
        assert (
            posting("Cianbro", date(2026, 3, 1), "Electrician").id
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
        assert "2026-03-01" in posting.id

    def test_id_roundtrip(self, sample_posting: Posting):
        """
        Serialized `id` survives a model_dump / model_validate round
        trip without recomputation.
        """
        data = sample_posting.model_dump()
        assert data["id"]
        assert Posting.model_validate(data).id == sample_posting.id

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

    def test_special_character_slugification(self, posting):
        """
        Spaces, ampersands, and dots are replaced with hyphens.
        """
        assert set(" &.").isdisjoint(
            posting("R.J. Grondin & Sons", date(2026, 1, 1), "Heavy Equip. Operator").id
        )

    def test_stopword_collision(self, posting):
        """
        Companies differing only by a stopword produce identical
        composite keys, documenting the collision boundary so that
        changes to stopword handling are caught.
        """
        assert (
            posting("Reed and Sons", date(2026, 1, 1), "Laborer").id
            == posting("Reed Sons", date(2026, 1, 1), "Laborer").id
        )

    def test_stopword_removal(self, posting):
        """
        Stopwords "and", "of", "the" are removed from slugs.
        """
        assert posting(
            "The Company of Maine", date(2026, 1, 1), "Operator"
        ).id.startswith("company-maine_")

    def test_undated_key(self, posting):
        """
        A `None` date produces an "undated" segment in the key.
        """
        assert "undated" in posting(date_posted=None).id
