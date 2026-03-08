"""
Tests for corpus storage and deduplication.

Validates save/load round-tripping, incremental collection, and
deduplication behavior that retains the most recently collected
version of each posting.
"""

from datetime import date
from pathlib  import Path

from chalkline.collection.models  import Posting
from chalkline.collection.storage import deduplicate, load, save


# -----------------------------------------------------------------------------
# Storage Tests
# -----------------------------------------------------------------------------


class TestStorage:
    """
    Validate save, load, and incremental collection behavior.
    """

    def test_incremental_save(
        self,
        sample_posting : Posting,
        second_posting : Posting,
        tmp_path       : Path
    ):
        """
        Saving new postings merges with existing ones on disk.
        """
        save([sample_posting], tmp_path)
        save([second_posting], tmp_path)
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
        save([sample_posting], tmp_path)
        assert load(tmp_path) == [sample_posting]


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

        assert len(result := deduplicate([older, newer])) == 1
        assert result[0].date_collected == date(2026, 3, 5)

    def test_no_duplicates_preserved(
        self,
        sample_posting : Posting,
        second_posting : Posting
    ):
        """
        Non-duplicate postings are all retained.
        """
        assert len(deduplicate([sample_posting, second_posting])) == 2
