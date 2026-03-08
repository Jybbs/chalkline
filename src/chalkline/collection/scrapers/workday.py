"""
Workday API scraper.

Extracts postings from Workday's semi-public search endpoint, which
career sites expose at `/wday/cxs/` as a JSON API accepting POST
requests with search criteria. Used for Maine DOT postings.
"""

from itertools import count
from logging   import getLogger
from pydantic  import BaseModel, Field
from re        import search

from chalkline.collection.models        import ManifestEntry, Posting, ScrapeCategory
from chalkline.collection.scrapers.base import BaseScraper

logger = getLogger(__name__)


# -----------------------------------------------------------------------------
# Response Models
# -----------------------------------------------------------------------------


class WorkdayJob(BaseModel, extra="ignore"):
    """
    A single job entry from Workday's search API response.

    Workday uses camelCase field names, so aliases map them to
    snake_case attributes.
    """

    bullet_fields  : list[str]  = Field(alias="bulletFields", default_factory=list)
    external_path  : str | None = Field(alias="externalPath", default=None)
    locations_text : str | None = Field(alias="locationsText", default=None)
    posted_on      : str | None = Field(alias="postedOn", default=None)
    title          : str | None = None


class WorkdayResponse(BaseModel, extra="ignore"):
    """
    Search response from Workday's `/wday/cxs/` endpoint.
    """

    job_postings: list[WorkdayJob] = Field(
        alias           = "jobPostings",
        default_factory = list
    )


# -----------------------------------------------------------------------------
# Scraper
# -----------------------------------------------------------------------------

class WorkdayScraper(BaseScraper):
    """
    Query Workday's semi-public CXS search endpoint for job listings.

    Workday career sites expose a `/wday/cxs/` JSON endpoint that
    accepts POST requests with pagination parameters. Each result
    includes a summary and an `externalPath` for fetching the full
    job description from a detail endpoint.
    """

    categories = frozenset({ScrapeCategory.WORKDAY})

    def _fetch_detail(self, base_url: str, external_path: str) -> str | None:
        """
        Fetch the full job description from a Workday detail endpoint.

        The search API returns only summaries, so each posting's
        `externalPath` must be fetched separately to get the
        complete description for skill extraction.

        Args:
            base_url      : The Workday domain from the career page
                URL, already extracted by `extract`.
            external_path : The relative path from the search
                result's `externalPath` field.

        Returns:
            The full job description text, or `None` on failure.
        """
        response = self._http.request(
            f"{base_url}/wday/cxs{external_path}"
        )
        data = response.json() if response else None
        if (
            isinstance(data, dict)
            and isinstance(info := data.get("jobPostingInfo"), dict)
            and (html := info.get("jobDescription"))
        ):
            return self.strip_html(html) or None
        return None

    def extract(self, entry: ManifestEntry) -> list[Posting]:
        """
        Paginate through Workday's search API and collect postings.

        Issues POST requests with incrementing offsets until the API
        returns fewer results than the page size. Validates each page
        against `WorkdayResponse` and fetches the full description
        for each posting via `_fetch_detail`.

        Args:
            entry: The manifest entry to scrape.

        Returns:
            A list of `Posting` records from the API response.
        """
        if not (api_match := search(
            r"(https?://[^/]+\.myworkdayjobs\.com)/([^/?]+)",
            entry.url
        )):
            logger.error(
                f"Cannot build Workday API URL from: {entry.url}"
            )
            return []

        base_url  = api_match.group(1)
        api_url   = f"{base_url}/wday/cxs/{api_match.group(2)}/jobs"
        page_size = 20
        postings  = []

        for offset in count(step=page_size):
            response = self._http.request(api_url, method="POST", json={
                "appliedFacets" : {},
                "limit"         : page_size,
                "offset"        : offset,
                "searchText"    : ""
            })

            if not isinstance(
                data := response.json() if response else None, dict
            ):
                break

            result = WorkdayResponse.model_validate(data)
            if not result.job_postings:
                break

            postings.extend(
                Posting(
                    company        = entry.company,
                    date_posted    = Posting.parse_iso_date(job.posted_on),
                    description    = description,
                    location       = job.locations_text,
                    source_type    = entry.category.source_type,
                    source_url     = entry.url,
                    title          = title
                )
                for job in result.job_postings
                if (title := job.title)
                if len(description := "\n".join(filter(None, [
                    title,
                    *job.bullet_fields,
                    self._fetch_detail(base_url, external)
                    if (external := job.external_path) else None
                ]))) >= 50
            )

            if len(result.job_postings) < page_size:
                break

        return postings
