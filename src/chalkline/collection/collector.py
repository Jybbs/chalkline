"""
Orchestrate corpus collection across all manifest URLs.

Reads the crawl manifest, dispatches each active URL to its
appropriate scraper, and stores collected postings through the
storage layer. Logs each attempt with its result to support
debugging and coverage tracking.
"""

from logging import basicConfig, getLogger, INFO
from pathlib import Path

from chalkline.collection.manifest           import generate
from chalkline.collection.models             import ManifestEntry, Posting
from chalkline.collection.models             import ScrapeCategory
from chalkline.collection.scrapers.base      import BaseScraper
from chalkline.collection.scrapers.cianbro   import CianbroScraper
from chalkline.collection.scrapers.heuristic import HeuristicScraper
from chalkline.collection.scrapers.workable  import WorkableScraper
from chalkline.collection.scrapers.workday   import WorkdayScraper
from chalkline.collection.stats              import CorpusStats
from chalkline.collection.storage            import save

logger = getLogger(__name__)


class Collector:
    """
    Run the full collection pipeline across all manifest URLs.

    Generates the crawl manifest from stakeholder reference data,
    dispatches each active URL to the appropriate scraper, and
    persists results to disk. Inactive URLs are skipped with a log
    message.
    """

    def __init__(self, postings_dir: Path, reference_dir: Path):
        """
        Initialize the collector with output and reference paths.

        Args:
            postings_dir  : Directory for storing collected postings.
            reference_dir : Directory containing `career_urls.json`.
        """
        self._cianbro   = CianbroScraper()
        self._heuristic = HeuristicScraper()
        self._postings  = postings_dir
        self._reference = reference_dir
        self._scraper   = BaseScraper()
        self._workable  = WorkableScraper()
        self._workday   = WorkdayScraper()

    def _scrape_entry(self, entry: ManifestEntry) -> list[Posting]:
        """
        Dispatch a manifest entry to its scraper by category.

        Uses `match`/`case` on the `ScrapeCategory` enum to route
        each URL to the correct extraction strategy.

        Args:
            entry: The manifest entry to scrape.

        Returns:
            A list of postings extracted from the entry's URL.
        """
        match entry.category:
            case ScrapeCategory.CIANBRO:
                return self._scrape_html(entry, cianbro=True)
            case ScrapeCategory.ENGAGEDTAS:
                return self._scrape_html(entry)
            case ScrapeCategory.STATIC_HTML:
                return self._scrape_html(entry)
            case ScrapeCategory.WORKABLE:
                return self._workable.extract(
                    company    = entry.company,
                    scraper    = self._scraper,
                    source_url = entry.url
                )
            case ScrapeCategory.WORKDAY:
                return self._workday.extract(
                    company    = entry.company,
                    scraper    = self._scraper,
                    source_url = entry.url
                )
            case _:
                return []

    def _scrape_html(
        self,
        entry   : ManifestEntry,
        cianbro : bool = False
    ) -> list[Posting]:
        """
        Fetch HTML and route to the Cianbro or heuristic parser.

        Returns an empty list when the fetch fails, allowing the
        collector to continue with the remaining manifest entries.

        Args:
            entry   : The manifest entry to fetch and parse.
            cianbro : When `True`, use the Cianbro-specific parser.

        Returns:
            A list of postings extracted from the fetched HTML.
        """
        response = self._scraper.fetch(entry.url)
        if response is None:
            return []

        if cianbro:
            return self._cianbro.extract(
                html       = response.text,
                source_url = entry.url
            )

        return self._heuristic.extract(
            company    = entry.company,
            html       = response.text,
            source_url = entry.url
        )

    def run(self) -> list[Posting]:
        """
        Execute the full collection pipeline.

        Generates the manifest, scrapes all active URLs, stores
        results, and logs a corpus statistics report.

        Returns:
            All postings collected in this run.
        """
        entries = generate(
            output_dir    = self._postings,
            reference_dir = self._reference
        )

        all_postings: list[Posting] = []
        active = [e for e in entries if e.active]
        logger.info(f"Scraping {len(active)} active URLs")

        for entry in active:
            try:
                postings = self._scrape_entry(entry)
                all_postings.extend(postings)
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
            save(
                postings     = all_postings,
                postings_dir = self._postings
            )

        logger.info(f"\n{CorpusStats(all_postings).report()}")
        return all_postings


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
