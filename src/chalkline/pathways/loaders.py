"""
Lexicon and reference data loading for the embedding pipeline.

Deserializes and validates JSON files from disk into typed domain
objects. `LexiconLoader` handles O*NET occupation data with slugified
filenames. `LaborLoader` handles BLS labor market records keyed by
SOC title for O(1) lookup. `StakeholderReference` lazily loads AGC
Maine stakeholder JSON files on first attribute access.
"""

from dataclasses import dataclass
from json        import loads
from loguru      import logger
from numpy       import argmax, ndarray
from pathlib     import Path
from pydantic    import TypeAdapter
from slugify     import slugify

from chalkline.pathways.schemas import LaborRecord, Occupation, Occupations


class LaborLoader:
    """
    BLS and O*NET labor market data keyed by SOC title.

    Deserializes `labor.json` via Pydantic `TypeAdapter` and builds
    a title-keyed dict for O(1) lookup. Used by the display layer
    for wage distributions, employment projections, and Bright
    Outlook designations.
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

    def get(self, soc_title: str) -> LaborRecord | None:
        """
        Look up labor data by SOC title.
        """
        return self.items.get(soc_title)

    def values(self):
        """
        Iterate all labor records.
        """
        return self.items.values()


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
        return self.occupations[argmax(similarity_row)]


@dataclass
class StakeholderReference:
    """
    Lazy-loading container for AGC Maine stakeholder reference
    data with dot-notation access.

    Each JSON file in the reference directory becomes an attribute
    on first access, cached thereafter. Missing files produce
    empty lists, matching the fallback behavior of the original
    dict comprehension.
    """

    reference_dir: Path

    def __getattr__(self, name: str) -> list:
        path  = self.reference_dir / f"{name}.json"
        value = loads(path.read_text()) if path.exists() else []
        setattr(self, name, value)
        return value
