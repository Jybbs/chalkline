"""
Shared infrastructure for career page crawling.

Provides `HttpClient` for HTTP transport with retries, timeouts,
`robots.txt` compliance, and rate limiting, and `BaseScraper` as
the abstract base for site-specific extractors.
"""

from abc                import ABC, abstractmethod
from httpx              import Client, HTTPStatusError, RequestError, Response
from logging            import WARNING, getLogger
from tenacity           import Retrying, before_sleep_log
from tenacity           import retry_if_exception_type, stop_after_attempt, wait_fixed
from time               import monotonic, sleep
from urllib.parse       import urlparse
from urllib.robotparser import RobotFileParser
from w3lib.html         import replace_tags

from chalkline.collection.models import ManifestEntry, Posting, ScrapeCategory

logger = getLogger(__name__)


class HttpClient:
    """
    HTTP client with rate limiting, retries, and `robots.txt` respect.

    Per-domain crawl delay is enforced automatically via an httpx
    request hook. Retries are handled declaratively by tenacity.
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
        domain_times: dict[str, float] = {}

        def throttle(request):
            domain = request.url.host or ""
            if (
                (last := domain_times.get(domain)) is not None
                and (wait := crawl_delay - (monotonic() - last)) > 0
            ):
                sleep(wait)
            domain_times[domain] = monotonic()

        self._client = Client(
            event_hooks      = {"request": [throttle]},
            follow_redirects = True,
            headers          = {"User-Agent": self._USER_AGENT},
            timeout          = timeout
        )
        self._retrier = Retrying(
            before_sleep = before_sleep_log(logger, WARNING),
            reraise      = True,
            retry        = retry_if_exception_type((HTTPStatusError, RequestError)),
            stop         = stop_after_attempt(retries),
            wait         = wait_fixed(crawl_delay)
        )
        self._robots: dict[str, RobotFileParser] = {}

    def _robots_allowed(self, url: str) -> bool:
        """
        Check `robots.txt` for the URL's origin, caching per origin.

        Fetches `robots.txt` through our own httpx client rather than
        `RobotFileParser.read()` so the request inherits the shared
        User-Agent header and per-domain rate limiting.
        """
        origin = f"{(p := urlparse(url)).scheme}://{p.hostname}"

        if origin not in self._robots:
            rp = RobotFileParser()
            try:
                response = self._client.get(f"{origin}/robots.txt")
                rp.parse(response.text.splitlines())
            except Exception:
                rp.parse([])
            self._robots[origin] = rp

        return self._robots[origin].can_fetch(self._USER_AGENT, url)

    def request(
        self, 
        url    : str, 
        method : str = "GET", 
        **kwargs
    ) -> Response | None:
        """
        Execute an HTTP request with `robots.txt` compliance,
        per-domain rate limiting, and tenacity-managed retries.

        Args:
            url      : The URL to request.
            method   : The HTTP method, defaulting to "GET".
            **kwargs : Forwarded to `httpx.Client.request`.

        Returns:
            The HTTP response, or `None` on failure.
        """
        if not self._robots_allowed(url):
            logger.warning(f"Blocked by robots.txt: {url}")
            return None

        try:
            for attempt in self._retrier:
                with attempt:
                    response = self._client.request(method, url, **kwargs)
                    response.raise_for_status()
                    return response
        except (HTTPStatusError, RequestError):
            logger.error(f"All retries failed: {url}")
            return None


class BaseScraper(ABC):
    """
    Shared constructor and contract for site-specific extractors.

    Subclasses declare `categories` as a class attribute and implement
    `extract` to handle their specific scraping approach.
    """

    categories : frozenset[ScrapeCategory]

    def __init__(self, http: HttpClient):
        """
        Bind the shared HTTP client for fetching pages and APIs.

        Args:
            http: The HTTP client for making requests.
        """
        self._http = http

    def __init_subclass__(cls, **kwargs):
        """
        Validate that concrete subclasses declare `categories` at
        class definition time rather than failing when the collector
        builds its dispatch table.
        """
        super().__init_subclass__(**kwargs)

        if getattr(cls, "__abstractmethods__", None):
            return

        if not hasattr(cls, "categories"):
            raise TypeError(f"{cls.__name__} must define: categories")

    @abstractmethod
    def extract(self, entry: ManifestEntry) -> list[Posting]:
        """
        Scrape the entry's URL and return postings for its company.
        """
        ...

    @staticmethod
    def strip_html(html: str) -> str:
        """
        Replace HTML tags with spaces and normalize whitespace.
        """
        return " ".join(replace_tags(html, token=" ").split())
