"""
Workable API scraper.

Extracts postings from Workable's public JSON API, which exposes job
listings at a documented `/api/v1/widget/` endpoint, avoiding HTML
parsing entirely. Used for Consigli's postings.
"""

from datetime import date
from logging  import getLogger
from re       import search

from chalkline.collection.models        import Posting, SourceType, make_posting_id
from chalkline.collection.scrapers.base import BaseScraper

logger = getLogger(__name__)


class WorkableScraper:
    """
    Query Workable's public JSON API for job listings.

    Derives the API endpoint from the career page URL's company slug,
    then fetches structured job data without HTML parsing. Each job
    entry includes title, description, location fields, and a
    publication date.
    """

    def _extract_slug(self, url: str) -> str | None:
        """
        Derive the company slug from a Workable career page URL.

        The slug appears as the first path segment after
        `workable.com/` and identifies the employer's API namespace.

        Args:
            url: The Workable career page URL.

        Returns:
            The company slug, or `None` if the URL does not match.
        """
        if (match := search(r"workable\.com/([^/?]+)", url)):
            return match.group(1)
        return None

    def _parse_date(self, date_str: str | None) -> date | None:
        """
        Parse a Workable ISO date string, tolerating time suffixes.

        Workable returns dates as full ISO timestamps, so truncating
        to the first 10 characters extracts the date portion.

        Args:
            date_str: The raw date string from the API response.

        Returns:
            The parsed date, or `None` if parsing fails.
        """
        if not date_str:
            return None
        try:
            return date.fromisoformat(date_str[:10])
        except (ValueError, TypeError):
            return None

    def extract(
        self,
        company    : str,
        scraper    : BaseScraper,
        source_url : str
    ) -> list[Posting]:
        """
        Fetch job listings from Workable's widget API.

        Constructs the API URL from the career page slug, fetches
        the JSON payload, and converts each job entry into a
        `Posting`.

        Args:
            company    : The employer name for the postings.
            scraper    : The HTTP client for making requests.
            source_url : The Workable career page URL.

        Returns:
            A list of `Posting` records from the API response.
        """
        slug = self._extract_slug(source_url)
        if not slug:
            logger.error(
                f"Cannot extract Workable slug from: {source_url}"
            )
            return []

        data = scraper.fetch_json(
            f"https://apply.workable.com/api/v1/widget/{slug}"
        )
        if not data or "jobs" not in data:
            logger.warning(f"No Workable jobs found for: {slug}")
            return []

        postings = []
        today    = date.today()

        for job in data["jobs"]:
            title       = job.get("title", "")
            description = job.get("description", "")
            if not title or len(description) < 50:
                continue

            location_parts = [
                job.get("city", ""),
                job.get("state", ""),
                job.get("country", "")
            ]
            location = (
                ", ".join(p for p in location_parts if p) or None
            )

            postings.append(Posting(
                company        = company,
                date_collected = today,
                date_posted    = self._parse_date(
                    job.get("published_on")
                ),
                description    = description,
                id             = make_posting_id(
                    company     = company,
                    date_posted = self._parse_date(
                        job.get("published_on")
                    ),
                    title       = title
                ),
                location       = location,
                source_type    = SourceType.WORKABLE_API,
                source_url     = source_url,
                title          = title
            ))

        return postings
