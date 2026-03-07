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
from typing   import Annotated, Any

NonEmptyStr = Annotated[str, Field(min_length=1)]


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------


class ScrapeCategory(StrEnum):
    """
    Classification of a career page URL by its scraping approach.

    Drives dispatch in the collector to route each manifest entry
    to the appropriate scraper implementation.
    """

    APPLICATION_ONLY = "APPLICATION_ONLY"
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
    When omitted at construction time, `id` is auto-computed from the
    sibling fields. Salary, experience level, and credential fields are
    intentionally omitted because they are unreliable in scraped
    postings.
    """

    company        : NonEmptyStr
    date_collected : date
    date_posted    : date | None
    description    : Annotated[str, Field(min_length=50)]
    source_type    : SourceType
    source_url     : NonEmptyStr
    title          : NonEmptyStr

    id       : NonEmptyStr        = ""
    location : NonEmptyStr | None = None

    @model_validator(mode="before")
    @classmethod
    def _auto_id(cls, data: Any) -> Any:
        """
        Derive `id` from `company`, `date_posted`, and `title` when
        absent, so callers need not invoke `make_posting_id` manually.
        """
        if isinstance(data, dict) and not data.get("id"):
            data["id"] = make_posting_id(
                company     = data.get("company", ""),
                date_posted = (
                    date.fromisoformat(dp)
                    if isinstance(dp := data.get("date_posted"), str)
                    else dp
                ),
                title = data.get("title", "")
            )
        return data


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
    return (
        f"{_slugify(company)}_{_slugify(title)}_"
        f"{date_posted.isoformat() if date_posted else 'undated'}"
    )


def parse_iso_date(date_str: str | None) -> date | None:
    """
    Parse an ISO date string, tolerating trailing time suffixes.

    ATS APIs (*Workable, Workday*) return dates as full ISO timestamps,
    so truncating to the first 10 characters extracts the date portion.

    Args:
        date_str: The raw date string from the API response.

    Returns:
        The parsed date, or `None` if absent or unparseable.
    """
    try:
        return date.fromisoformat(date_str[:10]) if date_str else None
    except ValueError:
        return None
