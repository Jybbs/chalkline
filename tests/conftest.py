"""
Shared test fixtures for the Chalkline test suite.

Fixtures form an embedding pipeline chain where each step's output is
independently tappable by any test module:

    corpus → raw_vectors → unit_vectors → coordinates → assignments
                                  ↓                          ↓
                           soc_vectors → job_zone_map → profiles → graph
                                                                     ↓
                           credential_vectors ───────────────────→ matcher
"""

import numpy as np

from json                  import loads
from pathlib               import Path
from pytest                import fixture, FixtureRequest
from sklearn.cluster       import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from chalkline.collection.schemas import Posting
from chalkline.extraction.loaders import LexiconLoader
from chalkline.extraction.schemas import Certification, OnetOccupation
from chalkline.pipeline.graph     import CareerPathwayGraph
from chalkline.pipeline.schemas   import ApprenticeshipContext, ClusterProfile
from chalkline.pipeline.schemas   import PipelineConfig, ProgramRecommendation
from chalkline.pipeline.trades    import TradeIndex


FIXTURES        = Path(__file__).resolve().parent / "fixtures"
CLUSTER_COUNT   = 4
COMPONENT_COUNT = 4
CORPUS_SIZE     = 20
EMBEDDING_DIM   = 16


class MockEncoder:
    """
    Deterministic fake encoder producing fixed-size embeddings for tests.

    Replaces SentenceTransformer so tests never load the 400MB model.
    """

    def encode(self, texts, show_progress_bar=False):
        """
        Return deterministic (len(texts), EMBEDDING_DIM) embeddings seeded
        by input length for reproducibility.
        """
        rng = np.random.RandomState(len(texts))
        return rng.randn(len(texts), EMBEDDING_DIM).astype(np.float32)


def _postings() -> list[Posting]:
    """
    Load posting fixtures from JSON.
    """
    return [
        Posting(**raw)
        for raw in loads((FIXTURES / "collection" / "postings.json").read_text())
    ]


# ── Collection fixtures ─────────────────────────────────────────────


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


# ── Lexicon fixtures ─────────────────────────────────────────────────


@fixture
def certifications(lexicon_loader: LexiconLoader) -> list[Certification]:
    """
    Synthetic certification records from the loader.
    """
    return lexicon_loader.certifications


@fixture
def lexicon_dir(tmp_path: Path) -> Path:
    """
    Write synthetic lexicon files to a temporary directory.
    """
    for src, dst in (
        ("certifications.json",   "certifications.json"),
        ("onet_occupations.json", "onet.json")
    ):
        (tmp_path / dst).write_text((FIXTURES / "extraction" / src).read_text())
    return tmp_path


@fixture
def lexicon_loader(lexicon_dir: Path) -> LexiconLoader:
    """
    Load all synthetic lexicon files via `LexiconLoader`.
    """
    return LexiconLoader(lexicon_dir)


@fixture
def occupations(lexicon_loader: LexiconLoader) -> list[OnetOccupation]:
    """
    Synthetic O*NET occupation records from the loader.
    """
    return lexicon_loader.occupations


@fixture(params=["47-2111", "47-2111.00"])
def soc(request: FixtureRequest) -> str:
    """
    Electrician SOC code in both bare and suffixed formats.
    """
    return request.param


# ── Embedding pipeline fixtures ──────────────────────────────────────


@fixture
def assignments(coordinates: np.ndarray) -> np.ndarray:
    """
    Ward-linkage HAC assignments at k=CLUSTER_COUNT.
    """
    return AgglomerativeClustering(
        linkage    = "ward",
        n_clusters = CLUSTER_COUNT
    ).fit_predict(coordinates)


@fixture
def centroids(
    assignments : np.ndarray,
    cluster_ids : list[int],
    coordinates : np.ndarray
) -> np.ndarray:
    """
    Mean SVD coordinates per cluster (CLUSTER_COUNT, COMPONENT_COUNT).
    """
    return np.stack([
        coordinates[assignments == cluster_id].mean(axis=0)
        for cluster_id in cluster_ids
    ])


@fixture
def cluster_ids(assignments: np.ndarray) -> list[int]:
    """
    Sorted unique cluster IDs from assignments.
    """
    return sorted(set(assignments))


@fixture
def cluster_vectors(
    assignments : np.ndarray,
    cluster_ids : list[int],
    raw_vectors : np.ndarray
) -> np.ndarray:
    """
    Mean posting embedding per cluster, L2-normalized (CLUSTER_COUNT,
    EMBEDDING_DIM).
    """
    return np.stack([
        normalize(
            raw_vectors[assignments == cluster_id].mean(axis=0, keepdims=True)
        )[0]
        for cluster_id in cluster_ids
    ])


@fixture
def config(tmp_path: Path) -> PipelineConfig:
    """
    Minimal pipeline config for tests.
    """
    return PipelineConfig(
        cluster_count   = CLUSTER_COUNT,
        component_count = COMPONENT_COUNT,
        lexicon_dir     = tmp_path,
        output_dir      = tmp_path,
        postings_dir    = tmp_path
    )


@fixture
def coordinates(
    svd_model    : TruncatedSVD,
    unit_vectors : np.ndarray
) -> np.ndarray:
    """
    SVD-reduced coordinates (CORPUS_SIZE, COMPONENT_COUNT).
    """
    return svd_model.transform(unit_vectors)


@fixture
def credential_vectors() -> np.ndarray:
    """
    Synthetic credential embeddings (10 credentials, EMBEDDING_DIM).
    """
    rng = np.random.RandomState(77)
    return normalize(rng.randn(10, EMBEDDING_DIM).astype(np.float32))


@fixture
def job_zone_map(cluster_ids: list[int]) -> dict[int, int]:
    """
    Deterministic Job Zone assignment for synthetic clusters.

    Spreads clusters across JZ 2-4 to exercise stepwise k-NN lateral and
    upward edge logic. Production uses top-3 median cosine against O*NET,
    but tests use fixed values to avoid coupling to fixture occupation
    count.
    """
    job_zone_values = [2, 2, 3, 4]
    return {
        cluster_id: job_zone_values[idx % len(job_zone_values)]
        for idx, cluster_id in enumerate(cluster_ids)
    }


@fixture
def mock_encoder() -> MockEncoder:
    """
    Deterministic mock replacing SentenceTransformer for tests.
    """
    return MockEncoder()


@fixture
def pathway_graph(
    centroids          : np.ndarray,
    cluster_vectors    : np.ndarray,
    config             : PipelineConfig,
    credential_vectors : np.ndarray,
    job_zone_map       : dict[int, int],
    profiles           : dict[int, ClusterProfile]
) -> CareerPathwayGraph:
    """
    Career pathway graph built from synthetic embeddings.
    """
    credential_count = len(credential_vectors)
    return CareerPathwayGraph(
        cluster_centroids  = centroids,
        cluster_vectors    = cluster_vectors,
        config             = config,
        credential_labels  = [f"Credential {i}" for i in range(credential_count)],
        credential_types   = ["certification"] * credential_count,
        credential_vectors = credential_vectors,
        job_zone_map       = job_zone_map,
        profiles           = profiles
    )


@fixture
def profiles(
    assignments  : np.ndarray,
    cluster_ids  : list[int],
    job_zone_map : dict[int, int]
) -> dict[int, ClusterProfile]:
    """
    Minimal cluster profiles for graph and matcher tests.
    """
    return {
        cluster_id: ClusterProfile(
            cluster_id  = cluster_id,
            job_zone    = job_zone_map[cluster_id],
            modal_title = f"Title {cluster_id}",
            sector      = f"Sector {cluster_id % 2}",
            size        = int((assignments == cluster_id).sum()),
            soc_title   = f"Occupation {cluster_id}"
        )
        for cluster_id in cluster_ids
    }


@fixture
def raw_vectors() -> np.ndarray:
    """
    Synthetic posting embeddings (CORPUS_SIZE, EMBEDDING_DIM).
    """
    rng = np.random.RandomState(42)
    return rng.randn(CORPUS_SIZE, EMBEDDING_DIM).astype(np.float32)


@fixture
def soc_vectors() -> np.ndarray:
    """
    Synthetic occupation embeddings (5 occupations, EMBEDDING_DIM).
    """
    rng = np.random.RandomState(99)
    return normalize(rng.randn(5, EMBEDDING_DIM).astype(np.float32))


@fixture
def svd_model(unit_vectors: np.ndarray) -> TruncatedSVD:
    """
    Fitted TruncatedSVD for resume projection tests.
    """
    fitted = TruncatedSVD(n_components=COMPONENT_COUNT, random_state=42)
    fitted.fit(unit_vectors)
    return fitted


@fixture
def unit_vectors(raw_vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalized posting embeddings.
    """
    return normalize(raw_vectors)


# ── Trade fixtures ───────────────────────────────────────────────────


@fixture
def apprenticeships() -> list[ApprenticeshipContext]:
    """
    Synthetic apprenticeship reference data for matching tests.
    """
    return [
        ApprenticeshipContext(**raw)
        for raw in loads((FIXTURES / "matching" / "apprenticeships.json").read_text())
    ]


@fixture
def programs() -> list[ProgramRecommendation]:
    """
    Load synthetic educational program fixtures.
    """
    return [
        ProgramRecommendation(**raw)
        for raw in loads((FIXTURES / "matching" / "programs.json").read_text())
    ]


@fixture
def trades(
    apprenticeships : list[ApprenticeshipContext],
    programs        : list[ProgramRecommendation]
) -> TradeIndex:
    """
    Shared trade index with prefix lookups for matching.
    """
    return TradeIndex(
        apprenticeships = apprenticeships,
        programs        = programs
    )
