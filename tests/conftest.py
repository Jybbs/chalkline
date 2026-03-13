"""
Shared test fixtures for the Chalkline test suite.

Fixtures form a pipeline chain where each step's output is
independently tappable by any test module:

    corpus → extracted_skills → vectorizer → pca_reducer
           ↘                    ↓                   ↓
            sector_labels    network             clustering

Lexicon fixtures feed the extractor via the registry:

    certifications ─┐
    occupations ────┤
    osha_terms ─────┼→ registry → extractor
    supplement_terms┘
"""

from json    import loads
from pathlib import Path
from pytest  import fixture, FixtureRequest

from chalkline.association.apriori       import AprioriComparison
from chalkline.association.cooccurrence  import CooccurrenceNetwork
from chalkline.clustering.comparison     import ClusterComparison
from chalkline.clustering.hierarchical   import compute_sector_labels
from chalkline.clustering.hierarchical   import HierarchicalClusterer
from chalkline.clustering.schemas        import ClusterLabel
from chalkline.collection.schemas        import Posting
from chalkline.extraction.lexicons       import LexiconRegistry
from chalkline.extraction.loaders        import load_certifications, load_onet
from chalkline.extraction.loaders        import load_osha, load_supplement
from chalkline.extraction.occupations    import OccupationIndex
from chalkline.extraction.schemas        import Certification, OnetOccupation
from chalkline.extraction.skills         import SkillExtractor
from chalkline.extraction.vectorize      import SkillVectorizer
from chalkline.reduction.pca             import PcaReducer


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _postings() -> list[Posting]:
    """
    Load posting fixtures from JSON.
    """
    return [
        Posting(**raw)
        for raw in loads((FIXTURES / "collection/postings.json").read_text())
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
    Twenty synthetic posting texts covering the fixture vocabulary.

    Each posting targets a different skill cluster so downstream
    matrices have enough rank for `TruncatedSVD` and enough
    variation for meaningful clustering, PMI, and DBSCAN density
    estimation.
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
                       "and scaffolding. Fall protection certified.",
        "posting-11" : "Blueprint reading and electrical wiring for "
                       "commercial projects. Autodesk AutoCAD drafting.",
        "posting-12" : "Electrical safety and electrical systems "
                       "troubleshooting. Laptop computers for "
                       "diagnostics and welding certification.",
        "posting-13" : "Electrical wiring and scaffolding erection. "
                       "Autodesk AutoCAD layouts and fall protection.",
        "posting-14" : "Backhoe and excavation for site preparation. "
                       "Operate construction equipment and building "
                       "foundations.",
        "posting-15" : "Concrete finishing and rebar placement. "
                       "Spread concrete for foundations and "
                       "excavation grading.",
        "posting-16" : "Operate construction equipment for building "
                       "foundations. Concrete finishing and spread "
                       "concrete required.",
        "posting-17" : "Welding inspection and weld quality control. "
                       "Rigging hardware and fall protection on site.",
        "posting-18" : "Asbestos abatement and electrical safety "
                       "compliance. Scaffolding and fall protection.",
        "posting-19" : "Load charts and rigging hardware for crane "
                       "operations. Welding and scaffolding required.",
        "posting-20" : "Concrete finishing and fall protection. "
                       "Scaffolding and operate construction "
                       "equipment on highway projects."
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


@fixture(params = [
    "47-2111",
    "47-2111.00"
])
def soc(request: FixtureRequest) -> str:
    """
    Electrician SOC code in both bare and suffixed formats.
    """
    return request.param


# ---------------------------------------------------------------------
# Vectorization and reduction
# ---------------------------------------------------------------------

@fixture
def pca_reducer(vectorizer: SkillVectorizer) -> PcaReducer:
    """
    Build a PCA reducer from the shared skill vectorizer.
    """
    return PcaReducer(
        document_ids       = vectorizer.document_ids,
        feature_names      = vectorizer.feature_names,
        max_components     = 4,
        random_seed        = 42,
        tfidf_matrix       = vectorizer.tfidf_matrix,
        variance_threshold = 0.85
    )


@fixture
def vectorizer(extracted_skills: dict[str, list[str]]) -> SkillVectorizer:
    """
    Build a vectorizer from extracted skill lists.
    """
    return SkillVectorizer(extracted_skills)


# ---------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------

@fixture
def apriori(vectorizer: SkillVectorizer) -> AprioriComparison:
    """
    Build an Apriori comparison from the shared skill vectorizer.
    """
    return AprioriComparison(
        binary_matrix = vectorizer.binary_matrix,
        feature_names = vectorizer.feature_names
    )


@fixture
def network(vectorizer: SkillVectorizer) -> CooccurrenceNetwork:
    """
    Build a co-occurrence network from the shared skill vectorizer.
    """
    return CooccurrenceNetwork(
        binary_matrix    = vectorizer.binary_matrix,
        feature_names    = vectorizer.feature_names,
        min_cooccurrence = 0.05,
        random_seed      = 42
    )


# ---------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------

@fixture
def comparison(pca_reducer: PcaReducer) -> ClusterComparison:
    """
    Build a comparison runner without sector labels.
    """
    return ClusterComparison(
        coordinates = pca_reducer.coordinates,
        random_seed = 42
    )


@fixture
def comparison_with_sectors(
    pca_reducer   : PcaReducer,
    sector_labels : list[str]
) -> ClusterComparison:
    """
    Build a comparison runner with sector labels for ARI.
    """
    return ClusterComparison(
        coordinates   = pca_reducer.coordinates,
        random_seed   = 42,
        sector_labels = sector_labels
    )


@fixture
def cluster_labels(
    clusterer  : HierarchicalClusterer,
    vectorizer : SkillVectorizer
) -> list[ClusterLabel]:
    """
    Cluster labels from TF-IDF centroid terms, shared across label tests.
    """
    return clusterer.labels(
        feature_names = vectorizer.feature_names,
        tfidf_matrix  = vectorizer.tfidf_matrix
    )


@fixture
def clusterer(pca_reducer: PcaReducer) -> HierarchicalClusterer:
    """
    Build a hierarchical clusterer from the shared PCA reducer.
    """
    return HierarchicalClusterer(
        coordinates  = pca_reducer.coordinates,
        document_ids = pca_reducer.document_ids
    )


@fixture
def sector_labels(
    extracted_skills : dict[str, list[str]],
    occupation_index : OccupationIndex,
    pca_reducer      : PcaReducer
) -> list[str]:
    """
    Sector labels aligned with PCA reducer document order.
    """
    return compute_sector_labels(
        document_ids     = pca_reducer.document_ids,
        extracted_skills = extracted_skills,
        occupation_index = occupation_index
    )
