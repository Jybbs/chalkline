"""
Shared test fixtures for the Chalkline test suite.
"""

from json    import loads
from pathlib import Path
from pytest  import fixture

from chalkline.collection.schemas     import Posting
from chalkline.extraction.lexicons    import LexiconRegistry
from chalkline.extraction.loaders     import load_certifications, load_onet
from chalkline.extraction.loaders     import load_osha, load_supplement
from chalkline.extraction.occupations import OccupationIndex
from chalkline.extraction.schemas     import Certification, OnetOccupation
from chalkline.extraction.skills      import SkillExtractor
from chalkline.extraction.vectorize   import SkillVectorizer


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _postings() -> list[Posting]:
    """
    Load posting fixtures from JSON.
    """
    return [
        Posting(**p)
        for p in loads((FIXTURES / "collection/postings.json").read_text())
    ]


@fixture
def certifications() -> list[Certification]:
    """
    Load synthetic certifications from fixture data.
    """
    return load_certifications(FIXTURES / "extraction/certifications.json")


@fixture
def extractor(registry: LexiconRegistry) -> SkillExtractor:
    """
    Build a skill extractor from synthetic fixture data.
    """
    return SkillExtractor(registry)


@fixture
def lexicon_dir(tmp_path: Path) -> Path:
    """
    Write synthetic lexicon files to a temporary directory.
    """
    for src, dst in (
        ("certifications.json",   "certifications.json"),
        ("onet_occupations.json", "onet.json"),
        ("osha_terms.json",       "osha.json"),
        ("supplement_terms.json", "supplement.json")
    ):
        (tmp_path / dst).write_text(
            (FIXTURES / "extraction" / src).read_text()
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
    return load_onet(FIXTURES / "extraction/onet_occupations.json")


@fixture
def osha_terms() -> list[str]:
    """
    Load synthetic OSHA terms from fixture data.
    """
    return load_osha(FIXTURES / "extraction/osha_terms.json")


@fixture
def registry(
    certifications   : list[Certification],
    occupations      : list[OnetOccupation],
    osha_terms       : list[str],
    supplement_terms : list[str]
) -> LexiconRegistry:
    """
    Build a registry from synthetic fixture data.
    """
    return LexiconRegistry(
        certifications   = certifications,
        occupations      = occupations,
        osha_terms       = osha_terms,
        supplement_terms = supplement_terms
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


@fixture
def skill_vectorizer(extractor: SkillExtractor) -> SkillVectorizer:
    """
    Build a vectorizer from synthetic extraction results.
    """
    return SkillVectorizer(extractor.extract({
        "posting-a" : "Fall protection and welding are required. "
                      "Experience with Autodesk AutoCAD preferred.",
        "posting-b" : "Electrical safety training and scaffolding "
                      "inspection. Must know welding techniques."
    }))


@fixture(params=["47-2111", "47-2111.00"])
def soc(request) -> str:
    """
    Electrician SOC code in both bare and suffixed formats.
    """
    return request.param


@fixture
def supplement_terms() -> list[str]:
    """
    Load synthetic supplement terms from fixture data.
    """
    return load_supplement(FIXTURES / "extraction/supplement_terms.json")
