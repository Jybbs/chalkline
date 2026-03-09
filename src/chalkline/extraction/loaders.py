"""
Lexicon file loading for OSHA and O*NET data.

Deserializes and validates JSON lexicon files, returning empty collections
on missing files so that downstream normalization can proceed with whichever
lexicon is available.
"""

from logging  import getLogger
from pathlib  import Path
from pydantic import TypeAdapter

from chalkline                    import NonEmptyStr
from chalkline.extraction.schemas import OnetOccupation


Occupations = TypeAdapter(list[OnetOccupation])
OshaTerms   = TypeAdapter(list[NonEmptyStr])
logger      = getLogger(__name__)


def load_onet(path: Path) -> list[OnetOccupation]:
    """
    Load and validate the O*NET lexicon file.

    Returns an empty list when the file is missing, logging a warning so
    that normalization can continue with OSHA alone.

    Args:
        path: Path to `onet.json`.

    Returns:
        Validated list of O*NET occupations.
    """
    try:
        return Occupations.validate_json(path.read_bytes())
    except FileNotFoundError:
        logger.warning(f"O*NET lexicon not found at {path}")
        return []


def load_osha(path: Path) -> list[str]:
    """
    Load and validate the OSHA lexicon file.

    Returns an empty list when the file is missing, logging a warning so
    that normalization can continue with O*NET alone.

    Args:
        path: Path to `osha.json`.

    Returns:
        Validated list of OSHA topic strings.
    """
    try:
        return OshaTerms.validate_json(path.read_bytes())
    except FileNotFoundError:
        logger.warning(f"OSHA lexicon not found at {path}")
        return []
