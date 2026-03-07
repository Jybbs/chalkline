"""
Workable API scraper.

Extracts postings from Workable's public JSON API, which exposes job
listings at a documented `/api/v1/widget/` endpoint, avoiding HTML
parsing entirely. Used for Consigli's postings.
"""

from datetime import date
from logging  import getLogger
from re       import compile

from chalkline.collection.models        import parse_iso_date
from chalkline.collection.models        import Posting, ScrapeCategory, SourceType
from chalkline.collection.scrapers.base import BaseScraper

logger = getLogger(__name__)


class WorkableScraper(BaseScraper):
    """
    Query Workable's public JSON API for job listings.

    Derives the API endpoint from the career page URL's company slug,
    then fetches structured job data without HTML parsing. Each job
    entry includes title, description, location fields, and a
    publication date.
    """

    _API_BASE    = "https://apply.workable.com/api/v1/widget"
    _URL_PATTERN = compile(r"workable\.com/([^/?]+)")

    categories  = frozenset({ScrapeCategory.WORKABLE})
    source_type = SourceType.WORKABLE_API

    def extract(self, company: str, source_url: str) -> list[Posting]:
        """
        Fetch job listings from Workable's widget API.

        Constructs the API URL from the career page slug, fetches
        the JSON payload, and converts each job entry into a
        `Posting`.

        Args:
            company    : The employer name for the postings.
            source_url : The Workable career page URL.

        Returns:
            A list of `Posting` records from the API response.
        """
        if not (match := self._URL_PATTERN.search(source_url)):
            logger.error(
                f"Cannot extract Workable slug from: {source_url}"
            )
            return []

        if not isinstance(data := self._http.fetch_json(
            f"{self._API_BASE}/{match.group(1)}"
        ), dict) or "jobs" not in data:
            logger.warning(
                f"No Workable jobs found for: {match.group(1)}"
            )
            return []

        today = date.today()
        return [
            Posting(
                company        = company,
                date_collected = today,
                date_posted    = parse_iso_date(job.get("published_on")),
                description    = description,
                location       = (
                    ", ".join(filter(None, (
                        job.get("city"),
                        job.get("state"),
                        job.get("country")
                    ))) or None
                ),
                source_type    = self.source_type,
                source_url     = source_url,
                title          = title
            )
            for job in data["jobs"]
            if (title := job.get("title", ""))
            and len(description := job.get("description", "")) >= 50
        ]
