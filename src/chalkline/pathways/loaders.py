"""
Lexicon and reference data loading for the embedding pipeline.

Deserializes and validates JSON files from disk into typed domain objects.
`LexiconLoader` handles O*NET occupation data with slugified filenames.
`LaborLoader` handles BLS labor market records keyed by SOC title for O(1)
lookup. `StakeholderReference` lazily loads AGC Maine stakeholder JSON files
on first attribute access.
"""

from collections.abc import Iterable, ValuesView
from dataclasses     import dataclass
from difflib         import get_close_matches
from json            import loads
from loguru          import logger
from numpy           import argmax, ndarray
from operator        import attrgetter
from pathlib         import Path
from pydantic        import TypeAdapter
from slugify         import slugify
from typing          import Any, NamedTuple

from chalkline.collection.schemas import Posting
from chalkline.pathways.schemas   import LaborRecord, Occupation, Occupations


class LaborLoader:
    """
    BLS and O*NET labor market data keyed by SOC title.

    Deserializes `labor.json` via Pydantic `TypeAdapter` and builds a
    title-keyed dict for O(1) lookup. Used by the display layer for wage
    distributions, employment projections, and Bright Outlook designations.
    """

    def __init__(self, path: Path):
        """
        Args:
            path: Path to `labor.json`.
        """
        self.items: dict[str, LaborRecord] = {
            r.soc_title: r
            for r in TypeAdapter(list[LaborRecord]).validate_json(path.read_bytes())
        }

    @property
    def median_annual_wage(self) -> float:
        """
        Median of annual median wages across occupations, or 0 if no wage
        data exists.
        """
        from statistics import median
        wages = [r.annual_median for r in self.values() if r.annual_median]
        return median(wages) if wages else 0

    @property
    def total_bright_outlook(self) -> int:
        """
        Count of occupations with Bright Outlook designation.
        """
        return sum(r.bright_outlook for r in self.values())

    @property
    def total_employment(self) -> int:
        """
        Aggregate employment across all occupations.
        """
        return sum(r.employment for r in self.values())

    def get(self, soc_title: str) -> LaborRecord | None:
        """
        Look up labor data by SOC title.
        """
        return self.items.get(soc_title)

    def values(self) -> ValuesView[LaborRecord]:
        """
        Iterate all labor records.
        """
        return self.items.values()

    def wage_pairs(self, soc_titles: Iterable[str]) -> list[WagePair]:
        """
        Sorted (title, wage) pairs for titles with available wage data.

        Args:
            soc_titles: Occupation titles to look up.
        """
        return sorted(
            (
                WagePair(title, rec.annual_median)
                for title in soc_titles
                if (rec := self.get(title)) and rec.annual_median
            ),
            key=attrgetter("wage")
        )


class LexiconLoader:
    """
    Load and validate lexicon files from a directory.

    Each attribute holds the validated contents of one lexicon file, falling
    back to an empty list if the file is missing. File names are derived
    from the label via `slugify` to match the canonical layout in
    `data/lexicons/`.
    """

    def __init__(self, lexicon_dir: Path):
        """
        Args:
            lexicon_dir: Must contain `onet.json`.
        """
        self.lexicon_dir = lexicon_dir
        self.occupations = Occupations(items=self._load(list[Occupation], "O*NET"))

    def _load(self, schema: type, label: str) -> list:
        """
        Validate a JSON lexicon file, returning an empty list if missing.

        Derives the filename from `label` via `slugify` so that callers
        specify only the human-readable lexicon name.

        Args:
            schema : Element type for the `TypeAdapter`.
            label  : Slugified to derive the filename.

        Returns:
            Validated list of lexicon entries.
        """
        path = self.lexicon_dir / f"{slugify(label, separator='')}.json"
        try:
            return TypeAdapter(schema).validate_json(path.read_bytes())

        except FileNotFoundError:
            logger.warning(f"{label} lexicon not found at {path}")
            return []

    def nearest_occupation(self, similarity_row: ndarray) -> Occupation:
        """
        O*NET occupation most similar to a cluster's embedding.

        Args:
            similarity_row: Cosine similarities against all occupations.

        Returns:
            The occupation with highest cosine similarity.
        """
        return self.occupations[int(argmax(similarity_row))]


@dataclass
class StakeholderReference:
    """
    Lazy-loading container for AGC Maine stakeholder reference data with
    dot-notation access.

    Each JSON file in the reference directory becomes an attribute on first
    access, cached thereafter. Missing files produce empty lists, matching
    the fallback behavior of the original dict comprehension.
    """

    reference_dir: Path

    def __getattr__(self, name: str) -> Any:
        path  = self.reference_dir / f"{name}.json"
        value = loads(path.read_text()) if path.exists() else []
        setattr(self, name, value)
        return value

    def filter_boards(self, keywords: set[str]) -> dict[str, list[dict]]:
        """
        Filter job boards by keyword presence in focus and best-for fields,
        grouped by region.

        Args:
            keywords: Sector-derived terms to match against.
        """
        def matches(board: dict) -> bool:
            text = f"{board.get('focus', '')} {board.get('best_for', '')}".lower()
            return any(kw in text for kw in keywords)

        return {
            region: [b for b in boards if matches(b)]
            for region, boards in self.job_boards.items()
        }

    def match_employers(self, postings: list[Posting]) -> list[dict]:
        """
        Fuzzy-match posting companies against AGC members and return
        display-ready employer rows.

        Args:
            postings: Posting objects from the target cluster.
        """
        names      = [m["name"].lower() for m in self.agc_members]
        urls       = {e["company"].lower(): e["url"] for e in self.career_urls}
        by_company = {p.company.lower(): p.source_url for p in postings}

        return list({
            member["name"]: {
                "career_url"  : urls.get(member["name"].lower(), ""),
                "member_type" : member["type"],
                "name"        : member["name"],
                "posting_url" : by_company[company]
            }
            for company in sorted(by_company)
            if (hits := get_close_matches(company, names))
            and (member := self.agc_members[names.index(hits[0])])
        }.values())


class WagePair(NamedTuple):
    """
    SOC title paired with annual median wage for display.
    """

    title : str
    wage  : float
