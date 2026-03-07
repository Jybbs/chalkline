"""
Workday API scraper.

Extracts postings from Workday's semi-public search endpoint, which
career sites expose at `/wday/cxs/` as a JSON API accepting POST
requests with search criteria. Used for Maine DOT postings.
"""

from datetime import date
from logging  import getLogger
from re       import match, search

from chalkline.collection.models        import Posting, SourceType, make_posting_id
from chalkline.collection.scrapers.base import BaseScraper

logger = getLogger(__name__)


class WorkdayScraper:
    """
    Query Workday's semi-public CXS search endpoint for job listings.

    Workday career sites expose a `/wday/cxs/` JSON endpoint that
    accepts POST requests with pagination parameters. Each result
    includes a summary and an `externalPath` for fetching the full
    job description from a detail endpoint.
    """

    def _build_api_url(self, url: str) -> str | None:
        """
        Derive the Workday CXS API endpoint from a career page URL.

        Extracts the base domain and tenant slug from the
        `myworkdayjobs.com` URL to construct the `/wday/cxs/` path.

        Args:
            url: The Workday career page URL.

        Returns:
            The CXS API endpoint URL, or `None` if the URL does
            not match the expected pattern.
        """
        if (result := search(
            r"(https?://[^/]+\.myworkdayjobs\.com)/([^/?]+)", url
        )):
            return (
                f"{result.group(1)}/wday/cxs/"
                f"{result.group(2)}/jobs"
            )
        return None

    def _fetch_detail(
        self,
        external_path : str,
        scraper       : BaseScraper,
        source_url    : str
    ) -> str | None:
        """
        Fetch the full job description from a Workday detail endpoint.

        The search API returns only summaries, so each posting's
        `externalPath` must be fetched separately to get the
        complete description for skill extraction.

        Args:
            external_path : The relative path from the search
                result's `externalPath` field.
            scraper       : The HTTP client for making requests.
            source_url    : The original career page URL for
                deriving the base domain.

        Returns:
            The full job description HTML, or `None` on failure.
        """
        base = match(
            r"(https?://[^/]+\.myworkdayjobs\.com)", source_url
        )
        if not base:
            return None

        data = scraper.fetch_json(
            f"{base.group(1)}/wday/cxs{external_path}"
        )
        if not data:
            return None

        return data.get("jobPostingInfo", {}).get(
            "jobDescription", ""
        )

    def _parse_date(self, date_str: str | None) -> date | None:
        """
        Parse a Workday ISO date string, tolerating time suffixes.

        Workday returns dates as full ISO timestamps, so truncating
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
        Paginate through Workday's search API and collect postings.

        Issues POST requests with incrementing offsets until the API
        returns fewer results than the page size. Fetches the full
        description for each posting via `_fetch_detail`.

        Args:
            company    : The employer name for the postings.
            scraper    : The HTTP client for making requests.
            source_url : The Workday career page URL.

        Returns:
            A list of `Posting` records from the API response.
        """
        api_url = self._build_api_url(source_url)
        if not api_url:
            logger.error(
                f"Cannot build Workday API URL from: {source_url}"
            )
            return []

        postings  = []
        today     = date.today()
        offset    = 0
        page_size = 20

        while True:
            payload = {
                "appliedFacets" : {},
                "limit"         : page_size,
                "offset"        : offset,
                "searchText"    : ""
            }

            try:
                scraper._enforce_delay(api_url)
                response = scraper._client.post(api_url, json=payload)
                response.raise_for_status()
                data = response.json()
            except Exception as error:
                logger.error(f"Workday API error: {error}")
                break

            job_postings = data.get("jobPostings", [])
            if not job_postings:
                break

            for job in job_postings:
                title         = job.get("title", "")
                bullet_fields = job.get("bulletFields", [])
                location      = job.get("locationsText", None)
                posted_on     = job.get("postedOn", None)

                desc_parts = [title]
                desc_parts.extend(bullet_fields)
                if (external := job.get("externalPath")):
                    detail = self._fetch_detail(
                        external_path = external,
                        scraper       = scraper,
                        source_url    = source_url
                    )
                    if detail:
                        desc_parts.append(detail)

                description = "\n".join(
                    p for p in desc_parts if p
                )
                if not title or len(description) < 50:
                    continue

                postings.append(Posting(
                    company        = company,
                    date_collected = today,
                    date_posted    = self._parse_date(posted_on),
                    description    = description,
                    id             = make_posting_id(
                        company     = company,
                        date_posted = self._parse_date(posted_on),
                        title       = title
                    ),
                    location       = location,
                    source_type    = SourceType.WORKDAY_API,
                    source_url     = source_url,
                    title          = title
                ))

            if len(job_postings) < page_size:
                break
            offset += page_size

        return postings
