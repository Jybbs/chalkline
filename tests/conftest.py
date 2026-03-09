"""
Shared test fixtures for the Chalkline test suite.
"""

from json    import loads
from pathlib import Path
from pytest  import fixture

from chalkline.collection.schemas     import Posting
from chalkline.extraction.lexicons    import LexiconRegistry
from chalkline.extraction.loaders     import Occupations
from chalkline.extraction.occupations import OccupationIndex
from chalkline.extraction.schemas     import OnetOccupation


FIXTURES = Path(__file__).resolve().parent / "fixtures"


@fixture
def lexicon_dir(tmp_path: Path) -> Path:
    """
    Write synthetic lexicon files to a temporary directory.
    """
    extraction = FIXTURES / "extraction"
    (tmp_path / "osha.json").write_text(
        (extraction / "osha_terms.json").read_text()
    )
    (tmp_path / "onet.json").write_text(
        (extraction / "onet_occupations.json").read_text()
    )
    return tmp_path


@fixture
def occupation_index(occupations: list[OnetOccupation]) -> OccupationIndex:
    """
    Build an occupation index from synthetic fixture data.
    """
    return OccupationIndex(occupations)


@fixture
def occupations() -> list[OnetOccupation]:
    """
    Parse synthetic O*NET data into validated occupation records.
    """
    return Occupations.validate_json(
        (FIXTURES / "extraction/onet_occupations.json").read_bytes()
    )


@fixture
def registry(occupations: list[OnetOccupation]) -> LexiconRegistry:
    """
    Build a registry from synthetic fixture data.
    """
    return LexiconRegistry(
        occupations,
        loads((FIXTURES / "extraction/osha_terms.json").read_text())
    )


@fixture
def sample_posting() -> Posting:
    """
    A minimal valid posting for testing.
    """
    return _postings()[0]


@fixture
def second_posting() -> Posting:
    """
    A distinct-company posting for multi-posting tests.
    """
    return _postings()[1]


@fixture(params=["47-2111.00", "47-2111"])
def soc(request) -> str:
    """
    Electrician SOC code in both suffixed and bare formats.
    """
    return request.param


def _postings() -> list[Posting]:
    """
    Load posting fixtures from JSON.
    """
    return [
        Posting(**p)
        for p in loads((FIXTURES / "collection/postings.json").read_text())
    ]
