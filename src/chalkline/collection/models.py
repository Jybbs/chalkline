"""
Domain models and enums for corpus collection.

Defines the `Posting` and `ManifestEntry` domain models, the
`ScrapeCategory` and `SourceType` enums that drive dispatch and
provenance tracking, and the composite key builder for deduplication.
"""

from datetime import date
from enum     import StrEnum
from pydantic import BaseModel, Field, TypeAdapter, model_validator
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

    @property
    def scrapeable(self) -> bool:
        """
        Whether this category represents an extractable job listing page.
        """
        return self not in {
            ScrapeCategory.APPLICATION_ONLY, ScrapeCategory.PDF_ONLY
        }

    @property
    def source_type(self) -> "SourceType":
        """
        The acquisition method associated with this scraping approach.
        """
        match self:
            case ScrapeCategory.WORKABLE: return SourceType.WORKABLE_API
            case ScrapeCategory.WORKDAY:  return SourceType.WORKDAY_API
            case _:                       return SourceType.DIRECT_SCRAPE


class SourceType(StrEnum):
    """
    Acquisition method for a collected posting.

    Distinguishes how a posting entered the corpus so that downstream
    analysis can weight or filter by provenance.
    """

    DIRECT_SCRAPE = "DIRECT_SCRAPE"
    WORKABLE_API  = "WORKABLE_API"
    WORKDAY_API   = "WORKDAY_API"


# -----------------------------------------------------------------------------
# Domain Models
# -----------------------------------------------------------------------------


class ManifestEntry(BaseModel, extra="forbid"):
    """
    A single URL from the crawl manifest with its scraping classification.

    Generated from `career_urls.json` by classifying each URL's
    scraping approach and marking application-only and PDF-only pages
    as inactive. The `active` field is always derived from the
    category rather than supplied by callers.
    """

    category : ScrapeCategory
    company  : NonEmptyStr
    source   : NonEmptyStr
    url      : NonEmptyStr

    active : bool = False

    @model_validator(mode="after")
    def _derive_active(self) -> "ManifestEntry":
        """
        Set `active` from `category.scrapeable` so callers and JSON
        input need not supply it.
        """
        self.active = self.category.scrapeable
        return self


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

    company     : NonEmptyStr
    date_posted : date | None
    description : Annotated[str, Field(min_length=50)]
    source_type : SourceType
    source_url  : NonEmptyStr
    title       : NonEmptyStr

    date_collected : date                = Field(default_factory=date.today)
    id             : NonEmptyStr         = ""
    location       : NonEmptyStr | None  = None

    @model_validator(mode="before")
    @classmethod
    def _auto_id(cls, data: Any) -> Any:
        """
        Derive `id` from `company`, `date_posted`, and `title` when
        absent, so callers need not call `make_id` manually.
        """
        if isinstance(data, dict) and not data.get("id"):
            data["id"] = cls.make_id(
                company     = data.get("company", ""),
                date_posted = (
                    cls.parse_iso_date(dp)
                    if isinstance(dp := data.get("date_posted"), str)
                    else dp
                ),
                title = data.get("title", "")
            )
        return data

    @staticmethod
    def make_id(
        company     : str,
        date_posted : date | None,
        title       : str
    ) -> str:
        """
        Build a deterministic composite key for deduplication.

        Uses company slug, title slug, and date to produce the same
        `id` when the same posting is collected from different sources.

        Args:
            company     : Employer name to slugify.
            date_posted : Posting date, or `None` for undated postings.
            title       : Job title to slugify.

        Returns:
            A composite key in the format `company_title_date`.
        """
        slug = lambda text: sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        return (
            f"{slug(company)}_{slug(title)}_"
            f"{date_posted.isoformat() if date_posted else 'undated'}"
        )

    @staticmethod
    def parse_iso_date(date_str: str | None) -> date | None:
        """
        Parse an ISO date string, tolerating trailing time suffixes.

        ATS APIs (*Workable, Workday*) return dates as full ISO
        timestamps, so truncating to the first 10 characters extracts
        the date portion.

        Args:
            date_str: The raw date string from the API response.

        Returns:
            The parsed date, or `None` if absent or unparseable.
        """
        try:
            return date.fromisoformat(date_str[:10]) if date_str else None
        except ValueError:
            return None


MANIFEST = TypeAdapter(list[ManifestEntry])
POSTINGS = TypeAdapter(list[Posting])
