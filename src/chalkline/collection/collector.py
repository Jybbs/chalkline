"""
Orchestrate corpus collection across all manifest URLs.

Reads the crawl manifest, dispatches each active URL to its
appropriate extractor, and stores collected postings through the
storage layer. Logs each attempt with its result to support
debugging and coverage tracking.
"""

from collections import Counter
from logging     import basicConfig, getLogger, INFO
from pathlib     import Path

from chalkline.collection.manifest           import generate
from chalkline.collection.models             import ManifestEntry, Posting
from chalkline.collection.scrapers.base      import HttpClient
from chalkline.collection.scrapers.heuristic import HeuristicScraper
from chalkline.collection.scrapers.workable  import WorkableScraper
from chalkline.collection.scrapers.workday   import WorkdayScraper
from chalkline.collection.storage            import save

logger = getLogger(__name__)


class Collector:
    """
    Run the full collection pipeline across all manifest URLs.

    Generates the crawl manifest from stakeholder reference data,
    dispatches each active URL to the appropriate extractor, and
    persists results to disk. Inactive URLs are skipped with a log
    message.
    """

    def __init__(self, postings_dir: Path, reference_dir: Path):
        """
        Initialize the collector with output and reference paths.

        Builds a dispatch table mapping each `ScrapeCategory` to the
        extractor that handles it, sharing a single `HttpClient`
        across all extractors.

        Args:
            postings_dir  : Directory for storing collected postings.
            reference_dir : Directory containing `career_urls.json`.
        """
        http = HttpClient()
        self._dispatch = {
            category: extractor
            for extractor in (
                HeuristicScraper(http=http),
                WorkableScraper(http=http),
                WorkdayScraper(http=http)
            )
            for category in extractor.categories
        }
        self._postings  = postings_dir
        self._reference = reference_dir

    def _scrape_entry(self, entry: ManifestEntry) -> list[Posting]:
        """
        Dispatch a manifest entry to its extractor by category.

        Looks up the extractor registered for the entry's
        `ScrapeCategory` and delegates extraction. Returns an empty
        list for categories without a registered extractor.

        Args:
            entry: The manifest entry to scrape.

        Returns:
            A list of postings extracted from the entry's URL.
        """
        if (extractor := self._dispatch.get(entry.category)):
            return extractor.extract(
                company    = entry.company,
                source_url = entry.url
            )
        return []

    def run(self) -> list[Posting]:
        """
        Execute the full collection pipeline.

        Generates the manifest, scrapes all active URLs, stores
        results, and logs a corpus statistics report.

        Returns:
            All postings collected in this run.
        """
        active = [
            e for e in generate(
                output_dir    = self._postings,
                reference_dir = self._reference
            ) if e.active
        ]

        all_postings = []
        logger.info(f"Scraping {len(active)} active URLs")

        for entry in active:
            try:
                all_postings.extend(
                    postings := self._scrape_entry(entry)
                )
                logger.info(
                    f"{entry.company}: {len(postings)} postings "
                    f"from {entry.url}"
                )
            except Exception as error:
                logger.error(
                    f"{entry.company}: failed to scrape "
                    f"{entry.url} ({error})"
                )

        if all_postings:
            save(all_postings, self._postings)

        _log_stats(all_postings)
        return all_postings


def _log_stats(postings: list[Posting]):
    """
    Log a human-readable summary of collected postings.

    Includes total count, company breakdown, date range, and source
    type distribution.
    """
    companies = Counter(p.company for p in postings)
    sources   = Counter(p.source_type.value for p in postings)
    dates     = [p.date_posted for p in postings if p.date_posted]

    lines = [
        "Corpus Statistics",
        f"  Total postings: {len(postings)}",
        f"  Companies: {len(companies)}",
        f"  Date range: "
        f"{(min(dates).isoformat(), max(dates).isoformat()) if dates else 'no dates available'}"
    ]
    for label, counter, limit in (
        ("Source types",  sources,   None),
        ("Top companies", companies, 10)
    ):
        if counter:
            lines.append(f"  {label}:")
            lines.extend(
                f"    {k}: {v}" for k, v in counter.most_common(limit)
            )

    logger.info("\n" + "\n".join(lines))


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------


if __name__ == "__main__":

    basicConfig(
        format = "%(asctime)s %(levelname)s %(name)s: %(message)s",
        level  = INFO
    )

    Collector(
        postings_dir  = Path("data/postings"),
        reference_dir = Path("data/stakeholder/reference")
    ).run()
