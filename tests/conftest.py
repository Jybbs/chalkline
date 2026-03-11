"""
Shared test fixtures for the Chalkline test suite.

Fixtures form a pipeline chain where each step's output is
independently tappable by any test module:

    corpus → extracted_skills → skill_vectorizer → pca_reducer

Lexicon fixtures feed the extractor via the registry:

    certifications ─┐
    occupations ────┤
    osha_terms ─────┼→ registry → extractor
    supplement_terms┘
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
from chalkline.reduction.pca          import PcaReducer


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _postings() -> list[Posting]:
    """
    Load posting fixtures from JSON.
    """
    return [
        Posting(**p)
        for p in loads((FIXTURES / "collection/postings.json").read_text())
    ]


# ---------------------------------------------------------------------
# Lexicon loading
# ---------------------------------------------------------------------

@fixture
def certifications() -> list[Certification]:
    """
    Load synthetic certifications from fixture data.
    """
    return load_certifications(FIXTURES / "extraction/certifications.json")


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
def supplement_terms() -> list[str]:
    """
    Load synthetic supplement terms from fixture data.
    """
    return load_supplement(FIXTURES / "extraction/supplement_terms.json")


# ---------------------------------------------------------------------
# Extraction pipeline
# ---------------------------------------------------------------------

@fixture
def corpus() -> dict[str, str]:
    """
    Ten synthetic posting texts covering the fixture vocabulary.

    Each posting targets a different skill cluster so downstream
    matrices have enough rank for `TruncatedSVD` and enough
    variation for meaningful clustering and PMI.
    """
    return {
        "posting-01" : "Fall protection and welding are required. "
                       "Experience with Autodesk AutoCAD preferred.",
        "posting-02" : "Electrical safety training and scaffolding "
                       "inspection. Must know welding techniques.",
        "posting-03" : "Concrete finishing and excavation work. "
                       "Rebar installation and building foundations.",
        "posting-04" : "Operate construction equipment. Welding "
                       "inspection and fall protection training.",
        "posting-05" : "Electrical wiring and scaffolding. Load "
                       "charts and rigging hardware experience.",
        "posting-06" : "Asbestos removal and electrical systems "
                       "maintenance. Autodesk AutoCAD drafting.",
        "posting-07" : "Backhoe operation and spread concrete for "
                       "building foundations. Excavation required.",
        "posting-08" : "Certified welding inspector with rigging "
                       "qualification. Weld quality assessment.",
        "posting-09" : "Electrician with laptop computers for "
                       "electrical wiring and electrical systems.",
        "posting-10" : "Equipment operators for concrete finishing "
                       "and scaffolding. Fall protection certified."
    }


@fixture
def extracted_skills(
    corpus    : dict[str, str],
    extractor : SkillExtractor
) -> dict[str, list[str]]:
    """
    Canonical skill lists extracted from the synthetic corpus.

    Mapping from document identifier to sorted, deduplicated
    skill names. Tappable by tests that need skill lists without
    vectorization overhead.
    """
    return extractor.extract(corpus)


@fixture
def extractor(registry: LexiconRegistry) -> SkillExtractor:
    """
    Build a skill extractor from synthetic fixture data.
    """
    return SkillExtractor(registry)


@fixture
def occupation_index(occupations: list[OnetOccupation]) -> OccupationIndex:
    """
    Build an occupation index from synthetic fixture data.
    """
    return OccupationIndex(occupations)


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


# ---------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------

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


@fixture(params=["47-2111", "47-2111.00"])
def soc(request) -> str:
    """
    Electrician SOC code in both bare and suffixed formats.
    """
    return request.param


# ---------------------------------------------------------------------
# Vectorization and reduction
# ---------------------------------------------------------------------

@fixture
def pca_reducer(skill_vectorizer: SkillVectorizer) -> PcaReducer:
    """
    Build a PCA reducer from the shared skill vectorizer.
    """
    return PcaReducer(
        document_ids       = skill_vectorizer.document_ids,
        feature_names      = skill_vectorizer.feature_names,
        max_components     = 4,
        random_seed        = 42,
        tfidf_matrix       = skill_vectorizer.tfidf_matrix,
        variance_threshold = 0.85
    )


@fixture
def skill_vectorizer(extracted_skills: dict[str, list[str]]) -> SkillVectorizer:
    """
    Build a vectorizer from extracted skill lists.
    """
    return SkillVectorizer(extracted_skills)
