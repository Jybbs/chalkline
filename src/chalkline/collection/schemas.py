"""
Schemas for corpus collection.

Defines the `Posting` schema with deterministic composite key deduplication
and the `Corpus` container that indexes postings for deterministic encoding.
"""

from dataclasses import dataclass, field
from datetime    import date
from pydantic    import BaseModel, Field, model_validator
from slugify     import slugify
from typing      import Annotated, Self


class Posting(BaseModel, extra="forbid"):
    """
    Canonical schema for a collected job posting.

    The `id` field is a deterministic composite key derived from company
    slug, title slug, and date. It is computed once at construction and
    serialized to JSON for deduplication without recomputation on load.
    """

    company     : str
    date_posted : date | None
    description : Annotated[str, Field(min_length=50)]
    source_url  : str
    title       : str

    date_collected : date       = Field(default_factory=date.today)
    id             : str        = ""
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


@dataclass
class Corpus:
    """
    Filtered posting corpus with eagerly-derived key ordering.

    Keeps `Posting` objects intact so downstream consumers access
    `.description` and `.title` through the posting rather than through
    parallel dicts. The sorted key list is computed once at construction for
    deterministic encoding and manifest ordering.
    """

    postings    : dict[str, Posting]
    posting_ids : list[str] = field(init=False)

    def __post_init__(self):
        self.posting_ids = sorted(self.postings)

    def at(self, indices) -> list[Posting]:
        """
        Retrieve postings by their positional indices into the
        sorted key list.

        Args:
            indices: Integer positions into `posting_ids`.

        Returns:
            Postings in the order of the provided indices.
        """
        return [self.postings[self.posting_ids[i]] for i in indices]

    @property
    def descriptions(self) -> list[str]:
        """
        Posting descriptions in deterministic sorted-key order for sentence
        encoding.
        """
        return [self.postings[pid].description for pid in self.posting_ids]
