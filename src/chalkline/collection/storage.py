"""
Persistent storage for the job posting corpus.

Serializes `Posting` records as JSON in `data/postings/` and supports
incremental collection by appending new postings without overwriting
existing ones. Deduplication retains the most recently collected
version when duplicate composite IDs appear.
"""

from json     import dumps, loads
from logging  import getLogger
from operator import attrgetter
from pathlib  import Path

from chalkline.collection.models import Posting

logger = getLogger(__name__)


CORPUS_FILE = "corpus.json"


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
    if not (corpus_path := postings_dir / CORPUS_FILE).exists():
        return []

    return [
        Posting.model_validate(record)
        for record in loads(
            corpus_path.read_text(encoding="utf-8")
        )
    ]


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

    merged = deduplicate(load(postings_dir) + postings)

    (corpus_path := postings_dir / CORPUS_FILE).write_text(
        data     = dumps([p.model_dump(mode="json") for p in merged], indent=2),
        encoding = "utf-8"
    )
    logger.info(f"Saved {len(merged)} postings to {corpus_path}")
