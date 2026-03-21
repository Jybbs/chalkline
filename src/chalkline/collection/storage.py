"""
Persistent storage for the job posting corpus.

Serializes `Posting` records as JSON and supports incremental collection by
appending without overwriting. Deduplication retains the most recently
collected version when duplicate IDs appear.
"""

from loguru   import logger
from pathlib  import Path
from pydantic import TypeAdapter

from chalkline.collection.schemas import Posting


class CorpusStorage:
    """
    JSON-backed persistence for the posting corpus.

    Owns the `corpus.json` lifecycle within a single directory, supporting
    incremental collection where each `save` merges new postings with the
    existing corpus and deduplicates by composite posting ID.
    """

    Postings = TypeAdapter(list[Posting])

    def __init__(self, postings_dir: Path):
        """
        Bind storage to a corpus directory.

        Args:
            postings_dir: Directory containing `corpus.json`.
        """
        self.corpus_path  = postings_dir / "corpus.json"
        self.postings_dir = postings_dir

    def deduplicate(self, postings: list[Posting]) -> list[Posting]:
        """
        Retain the most recently collected version of each posting.

        When duplicate `id` values exist, the posting with the latest
        `date_collected` wins.

        Args:
            postings: The list of postings to deduplicate.

        Returns:
            A deduplicated list with the latest version of each.
        """
        return list({
            p.id: p for p in
            sorted(postings, key=lambda p: p.date_collected)
        }.values())

    def load(self) -> list[Posting]:
        """
        Load all postings from the corpus file.

        Returns an empty list when `corpus.json` does not exist, allowing
        the collector to bootstrap from an empty directory.

        Returns:
            The deserialized list of postings.
        """
        try:
            result = self.Postings.validate_json(self.corpus_path.read_bytes())
            logger.info(f"Loaded {len(result)} postings from {self.corpus_path}")
            return result

        except FileNotFoundError:
            return []

    def save(self, postings: list[Posting]):
        """
        Persist postings to disk with deduplication.

        Merges `postings` with any existing corpus on disk, deduplicates by
        composite `id`, and writes the result.

        Args:
            postings: The new postings to save.
        """
        self.postings_dir.mkdir(parents=True, exist_ok=True)

        self.corpus_path.write_bytes(
            self.Postings.dump_json(
                merged := self.deduplicate(self.load() + postings),
                indent  = 2
            )
        )
        logger.info(f"Saved {len(merged)} postings to {self.corpus_path}")
