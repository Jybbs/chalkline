"""
Lexicon file loading for the embedding pipeline.

Deserializes and validates JSON lexicon files from a directory, returning
empty collections for missing files so that downstream encoding and SOC
assignment can proceed with whichever lexicons are available. File names are
derived from labels via `slugify`.
"""

from loguru   import logger
from numpy    import argmax, ndarray
from pathlib  import Path
from pydantic import TypeAdapter
from slugify  import slugify

from chalkline.extraction.schemas import Certification, OnetOccupation


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
            lexicon_dir: Must contain `onet.json` and `certifications.json`.
        """
        self.lexicon_dir    = lexicon_dir
        self.certifications = self._load(list[Certification],  "Certifications")
        self.occupations    = self._load(list[OnetOccupation], "O*NET")

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

    def nearest_occupation(self, similarity_row: ndarray) -> OnetOccupation:
        """
        O*NET occupation most similar to a cluster's embedding.

        Args:
            similarity_row: Cosine similarities against all occupations.

        Returns:
            The occupation with highest cosine similarity.
        """
        return self.occupations[argmax(similarity_row)]
