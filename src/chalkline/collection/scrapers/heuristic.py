"""
Heuristic HTML scraper for arbitrary career pages.

Searches for common job posting structures in HTML, including
headings followed by description text, list items with job titles,
and structured `<article>` or `<div>` blocks. EngagedTAS pages are
handled by this scraper because their single-listing HTML structure
matches generic extraction patterns.
"""

from bs4       import BeautifulSoup, Tag
from itertools import takewhile
from re        import compile, IGNORECASE, search

from chalkline.collection.models        import ManifestEntry, Posting, ScrapeCategory
from chalkline.collection.scrapers.base import BaseScraper

ACCORDION_PATTERN    = compile(r"accordion|collapse|toggle", IGNORECASE)
CONTENT_AREA_PATTERN = compile(r"content|main", IGNORECASE)
JOB_PATTERN          = compile(
    r"job|career|position|listing|opening|posting", IGNORECASE
)


class HeuristicScraper(BaseScraper):
    """
    Extract postings from arbitrary career pages using heuristics.

    Tries multiple CSS-like strategies to locate job listing blocks,
    from semantic `<article>` elements down to heading-based
    segmentation as a last resort. When a page yields zero postings,
    logs the URL without raising an error because the employer may
    have no current openings.
    """

    categories = frozenset({
        ScrapeCategory.ENGAGEDTAS,
        ScrapeCategory.STATIC_HTML
    })

    def _extract_description(self, block: Tag) -> str | None:
        """
        Concatenate visible text from content elements in a block.

        Pulls from paragraphs, list items, spans, divs, definition
        descriptions, and table cells to capture the full posting body
        for skill extraction.

        Args:
            block: The HTML element containing posting content.

        Returns:
            The concatenated text, or `None` if no content is found.
        """
        return ("\n".join(
            text
            for el in block.find_all(("p", "li", "span", "div", "dd", "td"))
            if (text := el.get_text(strip=True))
        )) or None

    def _extract_location(self, block: Tag) -> str | None:
        """
        Search for location patterns including Maine-specific formats.

        Tries explicit "Location:" labels first via the combined
        alternation regex, then falls back to city-state patterns
        like "Portland, ME" or "Bangor, Maine".

        Args:
            block: The HTML element to search for location text.

        Returns:
            The extracted location string, or `None` if not found.
        """
        match = search(
            r"location\s*[:\-]\s*(.+?)(?:\n|$)"
            r"|([\w\s]+,\s*(?:ME|Maine))",
            block.get_text(),
            IGNORECASE
        )
        return (match.group(1) or match.group(2)).strip() if match else None

    def _extract_title(self, block: Tag) -> str | None:
        """
        Find the most prominent text element as the job title.

        Searches heading tags h1 through h4, then anchor and
        emphasis elements, returning the first non-empty match.

        Args:
            block: The HTML element to search for a title.

        Returns:
            The extracted title text, or `None` if no heading is
            found.
        """
        return next(
            (text
             for tag in ("h1", "h2", "h3", "h4", "a", "strong", "b")
             if (heading := block.find(tag))
             and (text := heading.get_text(strip=True))),
            None
        )

    def _find_posting_blocks(self, soup: BeautifulSoup) -> list[Tag]:
        """
        Locate elements that likely contain individual job postings.

        Tries article tags, job-related div classes, job-related li
        classes, then accordion patterns, falling back to
        heading-based segmentation when none match.

        Args:
            soup: The parsed HTML document to search.

        Returns:
            A list of HTML elements, each containing one posting.
        """
        return soup.find_all("article") or next(
            (blocks for tag, pattern in (
                ("div", JOB_PATTERN),
                ("li",  JOB_PATTERN),
                ("div", ACCORDION_PATTERN)
            ) if (blocks := soup.find_all(tag, class_=pattern))),
            self._segment_by_headings(soup)
        )

    def _segment_by_headings(self, soup: BeautifulSoup) -> list[Tag]:
        """
        Group content by heading elements as a last-resort strategy.

        Restricts to the main content area to avoid capturing nav
        and footer headings as false posting blocks.

        Args:
            soup: The parsed HTML document to segment.

        Returns:
            A list of synthetic div elements, one per heading.
        """
        main = (
            soup.find("main")
            or soup.find("div", class_=CONTENT_AREA_PATTERN)
            or soup.body
            or soup
        )

        blocks = []

        for heading in main.find_all(("h2", "h3", "h4")):
            block = soup.new_tag("div")
            block.append(heading.__copy__())
            for sibling in takewhile(
                lambda s: s.name not in ("h2", "h3", "h4"),
                heading.find_next_siblings()
            ):
                block.append(sibling.__copy__())
            blocks.append(block)

        return blocks

    def extract(self, entry: ManifestEntry) -> list[Posting]:
        """
        Fetch and parse a career page into posting records.

        Fetches the HTML via the shared HTTP client, then searches
        for posting blocks. Skips blocks without a title, noise
        headings, or descriptions shorter than 50 characters.

        Args:
            entry: The manifest entry to scrape.

        Returns:
            A list of `Posting` records extracted from the page.
        """
        if not (response := self._http.request(entry.url)):
            return []

        return [
            Posting(
                company        = entry.company,
                date_posted    = None,
                description    = description,
                location       = self._extract_location(block),
                source_type    = entry.category.source_type,
                source_url     = entry.url,
                title          = title
            )
            for block in self._find_posting_blocks(
                BeautifulSoup(response.text, "html.parser")
            )
            if (title := self._extract_title(block))
            and title.lower() not in {
                "about", "apply", "benefits", "contact",
                "culture", "home", "menu", "values"
            }
            and len(title) >= 3
            if (description := self._extract_description(block))
            and len(description) >= 50
        ]
