"""
Workday API scraper.

Extracts postings from Workday's semi-public search endpoint, which
career sites expose at `/wday/cxs/` as a JSON API accepting POST
requests with search criteria. Used for Maine DOT postings.
"""

from datetime  import date
from itertools import count
from logging   import getLogger
from re        import compile

from chalkline.collection.models        import parse_iso_date
from chalkline.collection.models        import Posting, ScrapeCategory, SourceType
from chalkline.collection.scrapers.base import BaseScraper

logger = getLogger(__name__)


class WorkdayScraper(BaseScraper):
    """
    Query Workday's semi-public CXS search endpoint for job listings.

    Workday career sites expose a `/wday/cxs/` JSON endpoint that
    accepts POST requests with pagination parameters. Each result
    includes a summary and an `externalPath` for fetching the full
    job description from a detail endpoint.
    """

    _PAGE_SIZE   = 20
    _URL_PATTERN = compile(
        r"(https?://[^/]+\.myworkdayjobs\.com)/([^/?]+)"
    )

    categories  = frozenset({ScrapeCategory.WORKDAY})
    source_type = SourceType.WORKDAY_API

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
            The full job description HTML, or `None` on failure.
        """
        return (
            data.get("jobPostingInfo", {}).get("jobDescription", "")
            if isinstance(data := self._http.fetch_json(
                f"{base_url}/wday/cxs{external_path}"
            ), dict)
            else None
        )

    def extract(self, company: str, source_url: str) -> list[Posting]:
        """
        Paginate through Workday's search API and collect postings.

        Issues POST requests with incrementing offsets until the API
        returns fewer results than the page size. Fetches the full
        description for each posting via `_fetch_detail`.

        Args:
            company    : The employer name for the postings.
            source_url : The Workday career page URL.

        Returns:
            A list of `Posting` records from the API response.
        """
        if not (api_match := self._URL_PATTERN.search(source_url)):
            logger.error(
                f"Cannot build Workday API URL from: {source_url}"
            )
            return []

        base_url = api_match.group(1)
        api_url  = f"{base_url}/wday/cxs/{api_match.group(2)}/jobs"
        postings = []
        today    = date.today()

        for offset in count(0, self._PAGE_SIZE):
            data = self._http.post_json(api_url, json={
                "appliedFacets" : {},
                "limit"         : self._PAGE_SIZE,
                "offset"        : offset,
                "searchText"    : ""
            })

            if not isinstance(data, dict) or not (
                job_postings := data.get("jobPostings", [])
            ):
                break

            postings.extend(
                Posting(
                    company        = company,
                    date_collected = today,
                    date_posted    = parse_iso_date(job.get("postedOn")),
                    description    = description,
                    location       = job.get("locationsText"),
                    source_type    = self.source_type,
                    source_url     = source_url,
                    title          = title
                )
                for job in job_postings
                if (title := job.get("title"))
                if len(description := "\n".join(filter(None, [
                    title,
                    *job.get("bulletFields", []),
                    *(
                        [detail]
                        if (external := job.get("externalPath"))
                        and (detail := self._fetch_detail(
                            base_url, external
                        ))
                        else []
                    )
                ]))) >= 50
            )

            if len(job_postings) < self._PAGE_SIZE:
                break

        return postings
