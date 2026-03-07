"""
Heuristic HTML scraper for arbitrary career pages.

Searches for common job posting structures in HTML, including
headings followed by description text, list items with job titles,
and structured `<article>` or `<div>` blocks. EngagedTAS pages are
handled by this scraper because their single-listing HTML structure
matches generic extraction patterns.
"""

from bs4      import BeautifulSoup, Tag
from datetime import date
from re       import compile, IGNORECASE, search

from chalkline.collection.models import Posting, SourceType, make_posting_id


_ACCORDION_PATTERN = compile(
    r"accordion|collapse|toggle", IGNORECASE
)


_CONTENT_AREA_PATTERN = compile(r"content|main", IGNORECASE)


_JOB_DIV_PATTERN = compile(
    r"job|career|position|listing|opening|posting", IGNORECASE
)


_JOB_LI_PATTERN = compile(
    r"job|career|position|listing|opening|posting", IGNORECASE
)


_LOCATION_PATTERNS = [
    r"(?:Location|location|LOCATION)\s*[:\-]\s*(.+?)(?:\n|$)",
    r"([\w\s]+,\s*(?:ME|Maine))"
]


_NOISE_TITLES = {
    "about",
    "apply",
    "benefits",
    "contact",
    "culture",
    "home",
    "menu",
    "values"
}


class HeuristicScraper:
    """
    Extract postings from arbitrary career pages using heuristics.

    Tries multiple CSS-like strategies to locate job listing blocks,
    from semantic `<article>` elements down to heading-based
    segmentation as a last resort. When a page yields zero postings,
    logs the URL without raising an error because the employer may
    have no current openings.
    """

    def _extract_description(self, block: Tag) -> str | None:
        """
        Concatenate visible text from content elements in a block.

        Casts a wider net than the Cianbro scraper by also pulling
        from `<dd>` and `<td>` elements, because arbitrary career
        pages use definition lists and tables for job details.

        Args:
            block: The HTML element containing posting content.

        Returns:
            The concatenated text, or `None` if no content is found.
        """
        texts = []
        for element in block.find_all(
            ["p", "li", "span", "div", "dd", "td"]
        ):
            if (text := element.get_text(strip=True)):
                texts.append(text)
        return "\n".join(texts) if texts else None

    def _extract_location(self, block: Tag) -> str | None:
        """
        Search for location patterns including Maine-specific formats.

        Tries explicit "Location:" labels first, then falls back to
        city-state patterns like "Portland, ME" or "Bangor, Maine".

        Args:
            block: The HTML element to search for location text.

        Returns:
            The extracted location string, or `None` if not found.
        """
        text = block.get_text()
        for pattern in _LOCATION_PATTERNS:
            if (match := search(pattern, text)):
                return match.group(1).strip()
        return None

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
        for tag in ["h1", "h2", "h3", "h4", "a", "strong", "b"]:
            if (heading := block.find(tag)):
                if (text := heading.get_text(strip=True)):
                    return text
        return None

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
        for selector in [
            "article",
            ("div", {"class": _JOB_DIV_PATTERN}),
            ("li", {"class": _JOB_LI_PATTERN})
        ]:
            if isinstance(selector, str):
                blocks = soup.find_all(selector)
            else:
                blocks = soup.find_all(selector[0], selector[1])
            if blocks:
                return blocks

        blocks = soup.find_all(
            "div", class_=_ACCORDION_PATTERN
        )
        if blocks:
            return blocks

        return self._segment_by_headings(soup)

    def _is_noise(self, title: str) -> bool:
        """
        Filter out navigation items and boilerplate headings.

        Career pages often have headings like "About" or "Benefits"
        that would produce spurious postings if not filtered.

        Args:
            title: The candidate job title to check.

        Returns:
            `True` if the title is a known noise heading.
        """
        return title.lower().strip() in _NOISE_TITLES or len(title) < 3

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
            or soup.find("div", class_=_CONTENT_AREA_PATTERN)
            or soup.body
            or soup
        )

        blocks = []

        for heading in main.find_all(["h2", "h3", "h4"]):
            block = BeautifulSoup(
                "<div></div>", "html.parser"
            ).div
            block.append(heading.__copy__())
            for sibling in heading.find_next_siblings():
                if sibling.name in ["h2", "h3", "h4"]:
                    break
                block.append(sibling.__copy__())
            blocks.append(block)

        return blocks

    def extract(
        self,
        company    : str,
        html       : str,
        source_url : str
    ) -> list[Posting]:
        """
        Parse arbitrary career page HTML into posting records.

        Skips blocks without a title, noise headings, or
        descriptions shorter than 50 characters.

        Args:
            company    : The employer name for the postings.
            html       : The raw HTML string to parse.
            source_url : The URL the HTML was fetched from.

        Returns:
            A list of `Posting` records extracted from the page.
        """
        soup     = BeautifulSoup(html, "html.parser")
        postings = []
        today    = date.today()

        for block in self._find_posting_blocks(soup):
            title = self._extract_title(block)
            if not title or self._is_noise(title):
                continue

            description = self._extract_description(block)
            if not description or len(description) < 50:
                continue

            postings.append(Posting(
                company        = company,
                date_collected = today,
                date_posted    = None,
                description    = description,
                id             = make_posting_id(
                    company     = company,
                    date_posted = None,
                    title       = title
                ),
                location       = self._extract_location(block),
                source_type    = SourceType.DIRECT_SCRAPE,
                source_url     = source_url,
                title          = title
            ))

        return postings
