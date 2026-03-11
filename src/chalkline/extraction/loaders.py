"""
Lexicon file loading for certifications, OSHA, O*NET, and supplement
data.

Deserializes and validates JSON lexicon files, returning empty collections
on missing files so that downstream normalization can proceed with whichever
lexicons are available.
"""

from logging  import getLogger
from pathlib  import Path
from pydantic import TypeAdapter

from chalkline                    import NonEmptyStr
from chalkline.extraction.schemas import Certification, OnetOccupation


Certifications  = TypeAdapter(list[Certification])
Occupations     = TypeAdapter(list[OnetOccupation])
OshaTerms       = TypeAdapter(list[NonEmptyStr])
SupplementTerms = TypeAdapter(list[NonEmptyStr])
logger          = getLogger(__name__)


def _load(adapter: TypeAdapter, label: str, path: Path) -> list:
    """
    Validate a JSON lexicon file, returning an empty list on missing files.

    Args:
        adapter : Pydantic `TypeAdapter` for the target schema.
        label   : Human-readable lexicon name for the warning message.
        path    : Path to the JSON file.

    Returns:
        Validated list of lexicon entries.
    """
    try:
        return adapter.validate_json(path.read_bytes())
    except FileNotFoundError:
        logger.warning(f"{label} lexicon not found at {path}")
        return []


def load_certifications(path: Path) -> list[Certification]:
    """
    Load and validate the certifications lexicon file.
    """
    return _load(Certifications, "Certifications", path)


def load_onet(path: Path) -> list[OnetOccupation]:
    """
    Load and validate the O*NET lexicon file.
    """
    return _load(Occupations, "O*NET", path)


def load_osha(path: Path) -> list[str]:
    """
    Load and validate the OSHA lexicon file.
    """
    return _load(OshaTerms, "OSHA", path)


def load_supplement(path: Path) -> list[str]:
    """
    Load and validate the supplement lexicon file.
    """
    return _load(SupplementTerms, "Supplement", path)
