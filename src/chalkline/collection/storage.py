"""
Persistent storage for the job posting corpus.

Serializes `Posting` records as JSON in `data/postings/` and supports
incremental collection by appending new postings without overwriting
existing ones. Deduplication retains the most recently collected
version when duplicate composite IDs appear.
"""

from logging  import getLogger
from operator import attrgetter
from pathlib  import Path

from chalkline.collection.models import POSTINGS, Posting

logger = getLogger(__name__)


def deduplicate(postings: list[Posting]) -> list[Posting]:
    """
    Retain the most recently collected version of each posting.

    When duplicate `id` values exist, the posting with the latest
    `date_collected` wins. Stable sort order is preserved for
    non-duplicate entries.

    Args:
        postings: The list of postings to deduplicate.

    Returns:
        A deduplicated list with the latest version of each posting.
    """
    return list({
        p.id: p for p in sorted(postings, key=attrgetter("date_collected"))
    }.values())


def load(postings_dir: Path) -> list[Posting]:
    """
    Load all postings from the corpus file.

    Returns an empty list when the corpus file does not exist,
    allowing the collector to bootstrap from an empty directory.

    Args:
        postings_dir: The directory containing `corpus.json`.

    Returns:
        The deserialized list of postings.
    """
    if not (corpus_path := postings_dir / "corpus.json").exists():
        return []

    return POSTINGS.validate_json(corpus_path.read_bytes())


def save(postings: list[Posting], postings_dir: Path):
    """
    Persist postings to disk with deduplication.

    Merges `postings` with any existing corpus on disk, deduplicates
    by composite `id`, and writes the merged result. Creates the
    output directory if it does not exist.

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
