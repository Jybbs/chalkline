"""
Cianbro career page scraper.

Extracts postings from Cianbro's `/careers-list` page, which renders
40+ postings as static HTML blocks, making it the single richest
source in the manifest.
"""

from bs4      import BeautifulSoup, Tag
from datetime import date
from re       import compile, IGNORECASE, search

from chalkline.collection.models import Posting, SourceType, make_posting_id


_ACCORDION_PATTERN = compile(
    r"accordion|collapse|toggle", IGNORECASE
)


_JOB_CONTAINER_PATTERN = compile(
    r"job|career|position|listing|opening", IGNORECASE
)


class CianbroScraper:
    """
    Parse Cianbro's static HTML career page into `Posting` records.

    Cianbro renders 40+ postings as static HTML blocks, making it the
    single richest source in the manifest. Each posting appears as a
    distinct element with a title, location, and description.
    """

    def _extract_description(self, block: Tag) -> str | None:
        """
        Concatenate visible text from content elements in a block.

        Joins paragraph, list item, span, and div text so the
        resulting description captures the full posting body for
        skill extraction.

        Args:
            block: The HTML element containing posting content.

        Returns:
            The concatenated text, or `None` if no content is found.
        """
        texts = []
        for element in block.find_all(["p", "li", "span", "div"]):
            if (text := element.get_text(strip=True)):
                texts.append(text)
        return "\n".join(texts) if texts else None

    def _extract_location(self, block: Tag) -> str | None:
        """
        Search for "Location:" patterns in the block's raw text.

        Cianbro listings embed location as inline text rather than
        structured metadata, so regex matching is the only option.

        Args:
            block: The HTML element to search for location text.

        Returns:
            The extracted location string, or `None` if not found.
        """
        if (match := search(
            r"(?:Location|location)\s*[:\-]\s*(.+?)(?:\n|$)",
            block.get_text()
        )):
            return match.group(1).strip()
        return None

    def _extract_title(self, block: Tag) -> str | None:
        """
        Find the first heading or emphasis element as the job title.

        Searches h2 through h4, then anchor and strong tags, matching
        Cianbro's observed page structures.

        Args:
            block: The HTML element to search for a title.

        Returns:
            The extracted title text, or `None` if no heading is
            found.
        """
        for tag in ["h2", "h3", "h4", "a", "strong"]:
            if (heading := block.find(tag)):
                if (text := heading.get_text(strip=True)):
                    return text
        return None

    def _find_job_blocks(self, soup: BeautifulSoup) -> list[Tag]:
        """
        Locate distinct job listing containers in Cianbro's HTML.

        Tries article elements first, then job-related class names,
        then accordion patterns, falling back to heading-based
        segmentation when no explicit containers exist.

        Args:
            soup: The parsed HTML document to search.

        Returns:
            A list of HTML elements, each containing one posting.
        """
        blocks = soup.find_all("article")
        if blocks:
            return blocks

        blocks = soup.find_all(
            "div", class_=_JOB_CONTAINER_PATTERN
        )
        if blocks:
            return blocks

        blocks = soup.find_all(
            "div", class_=_ACCORDION_PATTERN
        )
        if blocks:
            return blocks

        return self._segment_by_headings(soup)

    def _segment_by_headings(self, soup: BeautifulSoup) -> list[Tag]:
        """
        Group content by heading elements as a fallback strategy.

        Collects all siblings between consecutive h2/h3 headings
        into synthetic div containers, producing one block per
        heading.

        Args:
            soup: The parsed HTML document to segment.

        Returns:
            A list of synthetic div elements, one per heading.
        """
        blocks = []

        for heading in soup.find_all(["h2", "h3"]):
            block = BeautifulSoup(
                "<div></div>", "html.parser"
            ).div
            block.append(heading.__copy__())
            for sibling in heading.find_next_siblings():
                if sibling.name in ["h2", "h3"]:
                    break
                block.append(sibling.__copy__())
            blocks.append(block)

        return blocks

    def extract(self, html: str, source_url: str) -> list[Posting]:
        """
        Parse Cianbro's career page HTML into posting records.

        Skips blocks without a title or with descriptions shorter
        than 50 characters, because those are navigation chrome or
        stubs.

        Args:
            html       : The raw HTML string to parse.
            source_url : The URL the HTML was fetched from.

        Returns:
            A list of `Posting` records extracted from the page.
        """
        soup     = BeautifulSoup(html, "html.parser")
        postings = []
        today    = date.today()

        for block in self._find_job_blocks(soup):
            title = self._extract_title(block)
            if not title:
                continue

            description = self._extract_description(block)
            if not description or len(description) < 50:
                continue

            postings.append(Posting(
                company        = "Cianbro",
                date_collected = today,
                date_posted    = None,
                description    = description,
                id             = make_posting_id(
                    company     = "Cianbro",
                    date_posted = None,
                    title       = title
                ),
                location       = self._extract_location(block),
                source_type    = SourceType.DIRECT_SCRAPE,
                source_url     = source_url,
                title          = title
            ))

        return postings
