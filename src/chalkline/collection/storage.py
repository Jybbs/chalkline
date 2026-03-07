"""
Persistent storage for the job posting corpus.

Serializes `Posting` records as JSON in `data/postings/` and supports
incremental collection by appending new postings without overwriting
existing ones. Deduplication retains the most recently collected
version when duplicate composite IDs appear.
"""

from json    import dumps, loads
from logging import getLogger
from pathlib import Path

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
    by_id: dict[str, Posting] = {}
    for posting in postings:
        if (existing := by_id.get(posting.id)) is not None:
            if posting.date_collected >= existing.date_collected:
                by_id[posting.id] = posting
        else:
            by_id[posting.id] = posting
    return list(by_id.values())


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
    corpus_path = postings_dir / CORPUS_FILE
    if not corpus_path.exists():
        return []

    raw = loads(corpus_path.read_text(encoding="utf-8"))
    return [Posting.model_validate(record) for record in raw]


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

    existing = load(postings_dir)
    merged   = deduplicate(existing + postings)

    (postings_dir / CORPUS_FILE).write_text(
        dumps(
            [p.model_dump(mode="json") for p in merged],
            indent=2
        ),
        encoding="utf-8"
    )
    logger.info(
        f"Saved {len(merged)} postings to "
        f"{postings_dir / CORPUS_FILE} "
        f"({len(postings)} new, {len(existing)} existing)"
    )
