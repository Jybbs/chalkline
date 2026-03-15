"""
Tests for corpus storage and deduplication.

Validates save/load round-tripping, incremental collection, and
deduplication that retains the most recently collected version.
"""

from datetime import date
from pathlib  import Path
from pytest   import fixture

from chalkline.collection.schemas import Posting
from chalkline.collection.storage import CorpusStorage


@fixture
def storage(tmp_path: Path) -> CorpusStorage:
    """
    Fresh corpus storage backed by a temporary directory.
    """
    return CorpusStorage(tmp_path)


class TestCorpusStorage:
    """
    Validate storage operations and deduplication behavior.
    """

    def test_deduplicate_empty(self, storage: CorpusStorage):
        """
        Deduplicating an empty list returns an empty list.
        """
        assert storage.deduplicate([]) == []

    def test_duplicate_keeps_latest(
        self,
        sample_posting : Posting,
        storage        : CorpusStorage
    ):
        """
        When two postings share an ID, the later `date_collected` wins.
        """
        assert len(result := storage.deduplicate([
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
        storage        : CorpusStorage
    ):
        """
        Saving new postings merges with existing ones on disk.
        """
        storage.save([sample_posting])
        storage.save([second_posting])
        assert len(storage.load()) == 2

    def test_load_empty_directory(self, tmp_path: Path):
        """
        Loading from a non-existent corpus returns an empty list.
        """
        assert CorpusStorage(tmp_path / "nonexistent").load() == []

    def test_save_deduplicates(
        self,
        sample_posting : Posting,
        storage        : CorpusStorage
    ):
        """
        Saving the same posting twice retains only one copy.
        """
        storage.save([sample_posting])
        storage.save([sample_posting])
        assert len(storage.load()) == 1

    def test_save_load_roundtrip(
        self,
        sample_posting : Posting,
        storage        : CorpusStorage
    ):
        """
        Save followed by load produces identical postings.
        """
        storage.save([sample_posting])
        assert storage.load() == [sample_posting]
