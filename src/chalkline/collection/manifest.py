"""
Crawl manifest generation from stakeholder career URLs.

Reads `career_urls.json`, strips query parameters, and writes the
manifest to `data/postings/manifest.json`. Each record in the source
JSON carries its own `category` field, so classification is
data-driven rather than inferred from URL structure.
"""

from json      import loads
from logging   import getLogger
from pathlib   import Path
from w3lib.url import url_query_cleaner

from chalkline.collection.models import MANIFEST, ManifestEntry, ScrapeCategory

logger = getLogger(__name__)


def generate(output_dir: Path, reference_dir: Path) -> list[ManifestEntry]:
    """
    Build and persist the crawl manifest from stakeholder data.

    Reads `career_urls.json`, strips query parameters, and writes
    the result to `manifest.json`. Returns the list of manifest
    entries for downstream consumption.

    Args:
        output_dir    : Directory to write `manifest.json` into.
        reference_dir : Directory containing `career_urls.json`.

    Returns:
        The complete list of manifest entries.
    """
    entries = []

    for record in loads(
        (reference_dir / "career_urls.json").read_bytes()
    ):
        url = url_query_cleaner(record["url"])
        entries.append(ManifestEntry(
            category = ScrapeCategory(record["category"]),
            company  = record["company"],
            source   = record["source"],
            url      = url
        ))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_bytes(
        MANIFEST.dump_json(entries, indent=2)
    )

    logger.info(
        f"Manifest: {len(entries)} URLs "
        f"({(active := sum(e.active for e in entries))} active, "
        f"{len(entries) - active} inactive)"
    )
    return entries
