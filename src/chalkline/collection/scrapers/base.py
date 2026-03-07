"""
Base scraper with shared HTTP infrastructure.

Provides configurable retries, timeouts, User-Agent identification,
`robots.txt` compliance, and per-domain crawl delay enforcement.
Concrete scraper implementations use `BaseScraper` as their HTTP
client for fetching pages and API endpoints.
"""

from httpx              import Client, HTTPStatusError, RequestError, Response
from logging            import getLogger
from time               import monotonic, sleep
from urllib.parse       import urlparse
from urllib.robotparser import RobotFileParser

logger = getLogger(__name__)


USER_AGENT = "ChalklineBot/0.1 (+https://github.com/Jybbs/chalkline)"


DEFAULT_CRAWL_DELAY = 2.0
DEFAULT_RETRIES     = 3
DEFAULT_TIMEOUT     = 30.0


class BaseScraper:
    """
    HTTP client with rate limiting, retries, and `robots.txt` respect.

    Tracks the last request timestamp per domain to enforce a minimum
    crawl delay between requests. Checks `robots.txt` before each
    fetch and skips disallowed URLs.
    """

    def __del__(self):
        """
        Release the underlying `httpx` connection pool on cleanup.

        Using `__del__` rather than a context manager because the
        scraper's lifetime spans the entire collection run and is not
        scoped to a single `with` block.
        """
        self._client.close()

    def __init__(
        self,
        crawl_delay : float = DEFAULT_CRAWL_DELAY,
        retries     : int   = DEFAULT_RETRIES,
        timeout     : float = DEFAULT_TIMEOUT
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
            headers          = {"User-Agent": USER_AGENT},
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
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.hostname}"

        if origin not in self._robots_cache:
            rp = RobotFileParser()
            rp.set_url(f"{origin}/robots.txt")
            try:
                rp.read()
            except Exception:
                rp.allow_all = True
            self._robots_cache[origin] = rp

        return self._robots_cache[origin].can_fetch(USER_AGENT, url)

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
            elapsed = monotonic() - last
            if elapsed < self._crawl_delay:
                sleep(self._crawl_delay - elapsed)
        self._domain_times[domain] = monotonic()

    def fetch(self, url: str) -> Response | None:
        """
        Fetch a URL with retries, rate limiting, and `robots.txt` check.

        Returns `None` when the URL is disallowed by `robots.txt` or
        all retry attempts fail, logging the reason in either case.

        Args:
            url: The URL to fetch.

        Returns:
            The HTTP response, or `None` on failure.
        """
        if not self._check_robots(url):
            logger.warning(f"Blocked by robots.txt: {url}")
            return None

        self._enforce_delay(url)

        for attempt in range(1, self._retries + 1):
            try:
                response = self._client.get(url)
                response.raise_for_status()
                return response
            except HTTPStatusError as error:
                logger.warning(
                    f"HTTP {error.response.status_code} on attempt "
                    f"{attempt}/{self._retries}: {url}"
                )
            except RequestError as error:
                logger.warning(
                    f"Request error on attempt "
                    f"{attempt}/{self._retries}: {url} ({error})"
                )

            if attempt < self._retries:
                sleep(self._crawl_delay)

        logger.error(f"All {self._retries} attempts failed: {url}")
        return None

    def fetch_json(self, url: str) -> dict | list | None:
        """
        Fetch a URL and parse the response body as JSON.

        Convenience wrapper for ATS API endpoints that return structured
        data. Returns `None` on fetch failure or JSON decode error.

        Args:
            url: The URL to fetch and parse.

        Returns:
            The parsed JSON payload, or `None` on failure.
        """
        response = self.fetch(url)
        if response is None:
            return None

        try:
            return response.json()
        except Exception as error:
            logger.error(f"JSON decode failed for {url}: {error}")
            return None
