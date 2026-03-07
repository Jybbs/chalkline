"""
Domain models and enums for corpus collection.

Defines the `Posting` and `ManifestEntry` data models, the
`ScrapeCategory` and `SourceType` enums that drive dispatch and
provenance tracking, and the composite key builder for deduplication.
"""

from datetime import date
from enum     import StrEnum
from pydantic import BaseModel, Field, model_validator
from re       import sub
from typing   import Annotated, Self

NonEmptyStr = Annotated[str, Field(min_length=1)]


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------


class ScrapeCategory(StrEnum):
    """
    Classification of a career page URL by its scraping approach.

    Drives `match`/`case` dispatch in the collector to route each
    manifest entry to the appropriate scraper implementation.
    """

    APPLICATION_ONLY = "APPLICATION_ONLY"
    CIANBRO          = "CIANBRO"
    ENGAGEDTAS       = "ENGAGEDTAS"
    PDF_ONLY         = "PDF_ONLY"
    STATIC_HTML      = "STATIC_HTML"
    WORKABLE         = "WORKABLE"
    WORKDAY          = "WORKDAY"


class SourceType(StrEnum):
    """
    Acquisition method for a collected posting.

    Distinguishes how a posting entered the corpus so that downstream
    analysis can weight or filter by provenance.
    """

    AGC_EXPORT    = "AGC_EXPORT"
    AGGREGATOR    = "AGGREGATOR"
    ATS_SCRAPE    = "ATS_SCRAPE"
    DIRECT_SCRAPE = "DIRECT_SCRAPE"
    WORKABLE_API  = "WORKABLE_API"
    WORKDAY_API   = "WORKDAY_API"


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """
    Convert text to a URL-safe slug for composite key construction.

    Lowercases, replaces non-alphanumeric runs with hyphens, and strips
    leading/trailing hyphens so the result is safe for composite IDs.

    Args:
        text: The raw string to slugify.

    Returns:
        A lowercase hyphen-separated slug.
    """
    return sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class ManifestEntry(BaseModel, extra="forbid"):
    """
    A single URL from the crawl manifest with its scraping classification.

    Generated from `career_urls.json` by classifying each URL's
    scraping approach and marking application-only and PDF-only pages
    as inactive.
    """

    active   : bool
    category : ScrapeCategory
    company  : NonEmptyStr
    source   : NonEmptyStr
    url      : NonEmptyStr


class Posting(BaseModel, extra="forbid"):
    """
    Canonical schema for a collected job posting.

    The `id` field is a composite key derived from company slug, title
    slug, and date, enabling deterministic deduplication across sources.
    Salary, experience level, and credential fields are intentionally
    omitted because they are unreliable in scraped postings.
    """

    company        : NonEmptyStr
    date_collected : date
    date_posted    : date | None
    description    : NonEmptyStr
    id             : NonEmptyStr
    source_type    : SourceType
    source_url     : NonEmptyStr
    title          : NonEmptyStr

    location: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_description_length(self) -> Self:
        """
        Reject descriptions too short for meaningful skill extraction.

        The 50-character floor filters out stub postings and placeholder
        entries that would contribute noise to the TF-IDF matrix.

        Raises:
            ValueError: When the description is shorter than 50
                characters.
        """
        if len(self.description) < 50:
            raise ValueError(
                f"Description must be at least 50 characters, "
                f"got {len(self.description)}"
            )
        return self


def make_posting_id(
    company     : str,
    date_posted : date | None,
    title       : str
) -> str:
    """
    Build a deterministic composite key for deduplication.

    Uses company slug, title slug, and date to produce the same `id`
    when the same posting is collected from different sources.

    Args:
        company     : Employer name to slugify.
        date_posted : Posting date, or `None` for undated postings.
        title       : Job title to slugify.

    Returns:
        A composite key in the format `company_title_date`.
    """
    date_str = date_posted.isoformat() if date_posted else "undated"
    return f"{_slugify(company)}_{_slugify(title)}_{date_str}"
