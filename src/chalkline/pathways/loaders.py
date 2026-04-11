"""
Lexicon and reference data loading for the embedding pipeline.

Deserializes and validates JSON files from disk into typed domain objects.
`LexiconLoader` handles O*NET occupation data. `LaborLoader` handles BLS
labor market records keyed by SOC title for O(1) lookup.
`StakeholderReference` lazily loads AGC Maine stakeholder JSON files on
first attribute access.
"""

from dataclasses     import dataclass
from difflib         import get_close_matches
from json            import loads
from loguru          import logger
from numpy           import argmax, ndarray
from pathlib         import Path
from pydantic        import TypeAdapter
from statistics      import median
from typing          import Any

from chalkline.collection.schemas import Posting
from chalkline.pathways.schemas   import LaborRecord, Occupation


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
        wages = [r.annual_median for r in self.items.values() if r.annual_median]
        return median(wages) if wages else 0

    @property
    def total_bright_outlook(self) -> int:
        """
        Count of occupations with Bright Outlook designation.
        """
        return sum(r.bright_outlook for r in self.items.values())

    @property
    def total_employment(self) -> int:
        """
        Aggregate employment across all occupations.
        """
        return sum(r.employment for r in self.items.values())

    def wage(self, soc_title: str) -> float | None:
        """
        Annual median wage for a SOC title, or None if unavailable.
        """
        return record.annual_median if (record := self.items.get(soc_title)) else None


class LexiconLoader:
    """
    Load and validate the O*NET occupation lexicon from a directory.

    `occupations` holds the validated `list[Occupation]`, falling back to
    an empty list if `onet.json` is missing.
    """

    def __init__(self, lexicon_dir: Path):
        """
        Args:
            lexicon_dir: Must contain `onet.json`.
        """
        path = lexicon_dir / "onet.json"
        try:
            self.occupations = TypeAdapter(
                list[Occupation]
            ).validate_json(path.read_bytes())
        except FileNotFoundError:
            logger.warning(f"O*NET lexicon not found at {path}")
            self.occupations = []

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

    def filter_boards(self, keywords: set[str], limit: int = 4) -> list[dict]:
        """
        Job boards whose focus or best-for fields mention any of the given
        keywords, flattened across regions and capped at `limit`.

        Args:
            keywords : Sector-derived terms to match against.
            limit    : Maximum total boards to return.
        """
        def matches(board: dict) -> bool:
            text = f"{board.get('focus', '')} {board.get('best_for', '')}".lower()
            return any(kw in text for kw in keywords)

        return [
            b
            for boards in self.job_boards.values()
            for b in boards
            if matches(b)
        ][:limit]

    def match_employers(self, postings: list[Posting]) -> list[dict]:
        """
        Fuzzy-match posting companies against AGC members and return
        display-ready employer rows.

        Args:
            postings: Posting objects from the target cluster.
        """
        members_by_lower = {m["name"].lower(): m for m in self.agc_members}
        by_company       = {p.company.lower(): p.source_url for p in postings}

        return list({
            member["name"]: {
                "member_type" : member["type"],
                "name"        : member["name"],
                "posting_url" : by_company[company]
            }
            for company in sorted(by_company)
            if (hits   := get_close_matches(company, members_by_lower))
            and (member := members_by_lower[hits[0]])
        }.values())


