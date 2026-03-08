"""
Workable API scraper.

Extracts postings from Workable's public JSON API, which exposes job
listings at a documented `/api/v1/widget/` endpoint. Descriptions
are stripped of HTML markup before storage. Used for Consigli's
postings.
"""

from logging  import getLogger
from pydantic import BaseModel
from re       import search

from chalkline.collection.models        import ManifestEntry, Posting, ScrapeCategory
from chalkline.collection.scrapers.base import BaseScraper

logger = getLogger(__name__)


# -----------------------------------------------------------------------------
# Response Models
# -----------------------------------------------------------------------------


class WorkableJob(BaseModel, extra="ignore"):
    """
    A single job entry from Workable's widget API response.

    Workable's fields are already snake_case, so no aliases are
    needed. The `location` property combines the separate city,
    state, and country fields into a single string.
    """

    city         : str | None = None
    country      : str | None = None
    description  : str        = ""
    published_on : str | None = None
    state        : str | None = None
    title        : str        = ""

    @property
    def location(self) -> str | None:
        """
        Combine city, state, and country into a location string.
        """
        return (
            ", ".join(filter(None, (self.city, self.state, self.country)))
            or None
        )


class WorkableResponse(BaseModel, extra="ignore"):
    """
    Top-level response from Workable's widget API.
    """

    jobs: list[WorkableJob] = []


# -----------------------------------------------------------------------------
# Scraper
# -----------------------------------------------------------------------------

class WorkableScraper(BaseScraper):
    """
    Query Workable's public JSON API for job listings.

    Derives the API endpoint from the career page URL's company slug,
    then fetches structured job data. HTML markup in descriptions is
    stripped to produce clean text for downstream skill extraction.
    """

    categories = frozenset({ScrapeCategory.WORKABLE})

    def extract(self, entry: ManifestEntry) -> list[Posting]:
        """
        Fetch job listings from Workable's widget API.

        Constructs the API URL from the career page slug, validates
        the JSON payload against `WorkableResponse`, and converts
        each job entry into a `Posting`.

        Args:
            entry: The manifest entry to scrape.

        Returns:
            A list of `Posting` records from the API response.
        """
        if not (match := search(r"workable\.com/([^/?]+)", entry.url)):
            logger.error(
                f"Cannot extract Workable slug from: {entry.url}"
            )
            return []

        response = self._http.request(
            f"https://apply.workable.com/api/v1/widget/{match.group(1)}"
        )
        if not isinstance(data := response.json() if response else None, dict):
            return []

        result = WorkableResponse.model_validate(data)
        if not result.jobs:
            logger.warning(
                f"No Workable jobs found for: {match.group(1)}"
            )
            return []

        return [
            Posting(
                company        = entry.company,
                date_posted    = Posting.parse_iso_date(job.published_on),
                description    = description,
                location       = job.location,
                source_type    = entry.category.source_type,
                source_url     = entry.url,
                title          = job.title
            )
            for job in result.jobs
            if job.title
            if len(description := self.strip_html(job.description)) >= 50
        ]
