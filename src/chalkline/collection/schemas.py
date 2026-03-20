"""
Schemas for corpus collection.

Defines the `Posting` schema with deterministic composite key deduplication.
"""

from datetime import date
from pydantic import BaseModel, BeforeValidator, Field, model_validator
from slugify  import slugify
from typing   import Annotated, Self


class Posting(BaseModel, extra="forbid"):
    """
    Canonical schema for a collected job posting.

    The `id` field is a deterministic composite key derived from company
    slug, title slug, and date. It is computed once at construction and
    serialized to JSON for deduplication without recomputation on load.
    """

    company     : str
    date_posted : Annotated[
                      date | None,
                      BeforeValidator(lambda v: v[:10] if isinstance(v, str) else v)
                  ]
    description : Annotated[str, Field(min_length=50)]
    source_url  : str
    title       : str

    date_collected : date       = Field(default_factory=date.today)
    id             : str       = ""
    location       : str | None = None

    @model_validator(mode="after")
    def _compute_id(self) -> Self:
        """
        Compute the composite key from identity fields when `id` is not
        already populated from serialized data.
        """
        if not self.id:
            slug = lambda v: slugify(v, stopwords=["and", "of", "the"])
            self.id = (
                f"{slug(self.company)}_{slug(self.title)}_"
                f"{self.date_posted.isoformat() if self.date_posted else 'undated'}"
            )
        return self
