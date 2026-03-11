"""
Tests for corpus storage and deduplication.

Validates save/load round-tripping, incremental collection, and
deduplication that retains the most recently collected version.
"""

from datetime import date
from pathlib  import Path

from chalkline.collection.schemas import Posting
from chalkline.collection.storage import deduplicate, load, save


class TestStorage:
    """
    Validate storage operations and deduplication behavior.
    """

    def test_deduplicate_empty(self):
        """
        Deduplicating an empty list returns an empty list.
        """
        assert deduplicate([]) == []

    def test_duplicate_keeps_latest(self, sample_posting: Posting):
        """
        When two postings share an ID, the later `date_collected` wins.
        """
        assert len(result := deduplicate([
            sample_posting.model_copy(
                update={"date_collected": date(2026, 3, 1)}
            ),
            sample_posting
        ])) == 1
        assert result[0].date_collected == date(2026, 3, 5)

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

    def test_no_duplicates(
        self,
        sample_posting : Posting,
        second_posting : Posting
    ):
        """
        Non-duplicate postings are all retained.
        """
        assert len(deduplicate([sample_posting, second_posting])) == 2

    def test_save_creates_parents(
        self,
        sample_posting : Posting,
        tmp_path       : Path
    ):
        """
        Saving to a non-existent nested directory creates it.
        """
        save([sample_posting], (nested := tmp_path / "a" / "b"))
        assert load(nested) == [sample_posting]

    def test_save_deduplicates(
        self,
        sample_posting : Posting,
        tmp_path       : Path
    ):
        """
        Saving the same posting twice retains only one copy.
        """
        save([sample_posting], tmp_path)
        save([sample_posting], tmp_path)
        assert len(load(tmp_path)) == 1

    def test_save_load_roundtrip(
        self,
        sample_posting : Posting,
        tmp_path       : Path
    ):
        """
        Save followed by load produces identical postings.
        """
        save([sample_posting], tmp_path)
        assert load(tmp_path) == [sample_posting]
