"""
Shared infrastructure for career page crawling.

Provides `HttpClient` for HTTP transport with retries, timeouts,
`robots.txt` compliance, and rate limiting, and `BaseScraper` as
the abstract base for site-specific extractors.
"""

from abc                import ABC, abstractmethod
from httpx              import Client, HTTPStatusError, RequestError, Response
from logging            import getLogger
from time               import monotonic, sleep
from urllib.parse       import urlparse
from urllib.robotparser import RobotFileParser

from chalkline.collection.models import Posting, ScrapeCategory, SourceType

logger = getLogger(__name__)


class HttpClient:
    """
    HTTP client with rate limiting, retries, and `robots.txt` respect.

    Tracks the last request timestamp per domain to enforce a minimum
    crawl delay between requests. Checks `robots.txt` before each
    fetch and skips disallowed URLs.
    """

    _USER_AGENT = "ChalklineBot/0.1 (+https://github.com/Jybbs/chalkline)"

    def __init__(
        self,
        crawl_delay : float = 2.0,
        retries     : int   = 3,
        timeout     : float = 30.0
    ):
        """
        Configure the HTTP client and rate-limiting parameters.

        Args:
            crawl_delay : Minimum seconds between requests to the
                same domain.
            retries     : Maximum number of retry attempts per URL.
            timeout     : HTTP request timeout in seconds.
        """
        self._client = Client(
            follow_redirects = True,
            headers          = {"User-Agent": self._USER_AGENT},
            timeout          = timeout
        )
        self._crawl_delay  = crawl_delay
        self._domain_times : dict[str, float] = {}
        self._retries      = retries
        self._robots_cache : dict[str, RobotFileParser] = {}

    def _check_robots(self, url: str) -> bool:
        """
        Consult `robots.txt` for the URL's origin.

        Caches the parsed `robots.txt` per origin so repeated checks
        against the same domain avoid redundant fetches. When
        `robots.txt` is unreachable, assumes access is allowed.

        Args:
            url: The full URL to check against `robots.txt`.

        Returns:
            `True` if the URL is allowed, `False` if disallowed.
        """
        origin = f"{(p := urlparse(url)).scheme}://{p.hostname}"

        if origin not in self._robots_cache:
            rp = RobotFileParser(f"{origin}/robots.txt")
            try:
                rp.read()
            except Exception:
                rp.parse([])
            self._robots_cache[origin] = rp

        return self._robots_cache[origin].can_fetch(self._USER_AGENT, url)

    def _enforce_delay(self, url: str):
        """
        Sleep until the minimum crawl delay has elapsed for this domain.

        Prevents aggressive request rates that could trigger rate
        limiting or IP bans on employer career sites.

        Args:
            url: The URL whose domain to throttle against.
        """
        domain = urlparse(url).hostname or ""
        if (last := self._domain_times.get(domain)) is not None:
            if (remaining := self._crawl_delay - (monotonic() - last)) > 0:
                sleep(remaining)
        self._domain_times[domain] = monotonic()

    def _parse_json(self, response: Response | None) -> dict | list | None:
        """
        Extract JSON from a response, returning `None` on failure.
        """
        if not response:
            return None
        try:
            return response.json()
        except Exception as error:
            logger.error(f"JSON decode failed for {response.url}: {error}")
            return None

    def _request(self, method: str, url: str, **kwargs) -> Response | None:
        """
        Execute an HTTP request with retries, rate limiting, and
        `robots.txt` compliance.

        Shared implementation behind `fetch` and `post`, dispatching
        to the appropriate `httpx.Client` method via `method` name.

        Args:
            method   : The HTTP method name, namely "get" or "post".
            url      : The URL to request.
            **kwargs : Forwarded to the `httpx.Client` method.

        Returns:
            The HTTP response, or `None` on failure.
        """
        if not self._check_robots(url):
            logger.warning(f"Blocked by robots.txt: {url}")
            return None

        self._enforce_delay(url)

        for attempt in range(1, self._retries + 1):
            try:
                response = getattr(self._client, method)(url, **kwargs)
                response.raise_for_status()
                return response
            except (HTTPStatusError, RequestError) as error:
                detail = (
                    f"HTTP {error.response.status_code}"
                    if isinstance(error, HTTPStatusError)
                    else f"Request error ({error})"
                )
                logger.warning(
                    f"{detail} on attempt "
                    f"{attempt}/{self._retries}: {url}"
                )

            if attempt < self._retries:
                sleep(self._crawl_delay)

        logger.error(f"All {self._retries} attempts failed: {url}")
        return None

    def fetch(self, url: str) -> Response | None:
        """
        GET a URL with retries, rate limiting, and `robots.txt` check.

        Returns `None` when the URL is disallowed by `robots.txt` or
        all retry attempts fail, logging the reason in either case.

        Args:
            url: The URL to fetch.

        Returns:
            The HTTP response, or `None` on failure.
        """
        return self._request("get", url)

    def fetch_json(self, url: str) -> dict | list | None:
        """
        GET a URL and parse the response body as JSON.

        Convenience wrapper for ATS API endpoints that return structured
        data. Returns `None` on fetch failure or JSON decode error.

        Args:
            url: The URL to fetch and parse.

        Returns:
            The parsed JSON payload, or `None` on failure.
        """
        return self._parse_json(self.fetch(url))

    def post(self, url: str, json: dict | list | None = None) -> Response | None:
        """
        POST to a URL with rate limiting and `robots.txt` check.

        Mirrors `fetch` semantics for POST endpoints, used by ATS
        scrapers that require POST requests for paginated search.

        Args:
            url  : The URL to post to.
            json : JSON-serializable payload for the request body.

        Returns:
            The HTTP response, or `None` on failure.
        """
        return self._request("post", url, json=json)

    def post_json(
        self,
        url  : str,
        json : dict | list | None = None
    ) -> dict | list | None:
        """
        POST to a URL and parse the response body as JSON.

        Convenience wrapper for ATS API endpoints that require POST
        requests and return structured data.

        Args:
            url  : The URL to post to.
            json : JSON-serializable payload for the request body.

        Returns:
            The parsed JSON payload, or `None` on failure.
        """
        return self._parse_json(self.post(url, json=json))


class BaseScraper(ABC):
    """
    Shared constructor and contract for site-specific extractors.

    Subclasses declare `categories` and `source_type` as class
    attributes and implement `extract` to handle their specific
    scraping approach.
    """

    categories  : frozenset[ScrapeCategory]
    source_type : SourceType

    def __init__(self, http: HttpClient):
        """
        Bind the shared HTTP client for fetching pages and APIs.

        Args:
            http: The HTTP client for making requests.
        """
        self._http = http

    def __init_subclass__(cls, **kwargs):
        """
        Validate that concrete subclasses declare `categories` and
        `source_type` at class definition time rather than failing
        when the collector builds its dispatch table.
        """
        super().__init_subclass__(**kwargs)

        if getattr(cls, "__abstractmethods__", frozenset()):
            return

        if missing := [a for a in ("categories", "source_type") if not hasattr(cls, a)]:
            raise TypeError(
                f"{cls.__name__} must define: {', '.join(missing)}"
            )

    @abstractmethod
    def extract(self, company: str, source_url: str) -> list[Posting]:
        ...
