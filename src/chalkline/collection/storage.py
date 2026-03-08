"""
Persistent storage for the job posting corpus.

Serializes `Posting` records as JSON and supports incremental
collection by appending without overwriting. Deduplication retains
the most recently collected version when duplicate IDs appear.
"""

from logging import getLogger
from pathlib import Path

from chalkline.collection.models import POSTINGS, Posting

logger = getLogger(__name__)


def deduplicate(postings: list[Posting]) -> list[Posting]:
    """
    Retain the most recently collected version of each posting.

    When duplicate `id` values exist, the posting with the latest
    `date_collected` wins.

    Args:
        postings: The list of postings to deduplicate.

    Returns:
        A deduplicated list with the latest version of each.
    """
    by_id = {
        p.id: p for p in
        sorted(postings, key=lambda p: p.date_collected)
    }
    return list(by_id.values())


def load(postings_dir: Path) -> list[Posting]:
    """
    Load all postings from the corpus file.

    Returns an empty list when `corpus.json` does not exist,
    allowing the collector to bootstrap from an empty directory.

    Args:
        postings_dir: The directory containing `corpus.json`.

    Returns:
        The deserialized list of postings.
    """
    try:
        return POSTINGS.validate_json((postings_dir / "corpus.json").read_bytes())
    except FileNotFoundError:
        return []


def save(postings: list[Posting], postings_dir: Path):
    """
    Persist postings to disk with deduplication.

    Merges `postings` with any existing corpus on disk,
    deduplicates by composite `id`, and writes the result.

    Args:
        postings     : The new postings to save.
        postings_dir : The directory to write `corpus.json` into.
    """
    postings_dir.mkdir(parents=True, exist_ok=True)

    (corpus_path := postings_dir / "corpus.json").write_bytes(
        POSTINGS.dump_json(
            merged := deduplicate(load(postings_dir) + postings),
            indent=2
        )
    )
    logger.info(f"Saved {len(merged)} postings to {corpus_path}")
