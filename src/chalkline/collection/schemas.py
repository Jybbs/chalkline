"""
Schemas for corpus collection.

Defines the `Posting` schema and composite key builder for deduplication.
"""

from datetime import date
from pydantic import BaseModel, BeforeValidator, Field, model_validator
from slugify  import slugify
from typing   import Annotated, Self


class Posting(BaseModel, extra="forbid"):
    """
    Canonical schema for a collected job posting.

    The `id` field is a composite key derived from company slug, title slug,
    and date, enabling deterministic deduplication. When omitted at
    construction time, `id` is auto-computed from the sibling fields.
    """

    company     : str
    date_posted : Annotated[
                      date | None,
                      BeforeValidator(lambda v: v[:10] if isinstance(v, str) else v)
                  ]
    description : Annotated[str, Field(min_length=50)]
    source_url  : str
    title       : str

    date_collected : date = Field(default_factory=date.today)
    id             : str | None = None
    location       : str | None = None

    @model_validator(mode="after")
    def _auto_id(self) -> Self:
        """
        Derive `id` from `company`, `date_posted`, and `title` when absent.

        Returns:
            The model instance with `id` populated.
        """
        self.id = self.id or self.make_id(self.company, self.date_posted, self.title)
        return self

    @staticmethod
    def make_id(
        company     : str,
        date_posted : date | None,
        title       : str
    ) -> str:
        """
        Build a deterministic composite key for deduplication.

        Uses company slug, title slug, and date to produce the same `id`
        when the same posting appears across sources.

        Args:
            company     : Employer name to slugify.
            date_posted : Posting date, or `None` for undated.
            title       : Job title to slugify.

        Returns:
            A composite key in the format `company_title_date`.
        """
        slug = lambda v: slugify(v, stopwords=["and", "of", "the"])
        return (
            f"{slug(company)}_{slug(title)}_"
            f"{date_posted.isoformat() if date_posted else 'undated'}"
        )
