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

from pathlib               import Path
from pydantic              import TypeAdapter
from pytest                import fixture, FixtureRequest
from sklearn.cluster       import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from typing                import Any

from chalkline.collection.schemas  import Posting
from chalkline.extraction.loaders  import LexiconLoader
from chalkline.extraction.schemas  import Certification, OnetOccupation
from chalkline.matching.matcher    import ResumeMatcher
from chalkline.pipeline.graph      import CareerPathwayGraph
from chalkline.pipeline.schemas    import ApprenticeshipContext, ClusterProfile
from chalkline.pipeline.schemas    import Credentials, PipelineConfig
from chalkline.pipeline.schemas    import ProgramRecommendation
from chalkline.pipeline.trades     import TradeIndex


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
        return np.random.RandomState(len(texts)).randn(
            len(texts), EMBEDDING_DIM
        ).astype(np.float32)


_load = lambda schema, path: TypeAdapter(schema).validate_json(
    (FIXTURES / path).read_bytes()
)


def _embeddings(rows: int, seed: int, unit: bool = False) -> np.ndarray:
    """
    Deterministic random embeddings for test fixtures.
    """
    vectors = np.random.RandomState(seed).randn(rows, EMBEDDING_DIM).astype(np.float32)
    return normalize(vectors) if unit else vectors


def _postings() -> list[Posting]:
    """
    Load posting fixtures from JSON.
    """
    return _load(list[Posting], "collection/postings.json")


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
        normalize(raw_vectors[assignments == cluster_id].mean(axis=0, keepdims=True))[0]
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
    svd          : TruncatedSVD,
    unit_vectors : np.ndarray
) -> np.ndarray:
    """
    SVD-reduced coordinates (CORPUS_SIZE, COMPONENT_COUNT).
    """
    return svd.transform(unit_vectors)


@fixture
def credential_records(
    apprenticeships : list[ApprenticeshipContext],
    programs        : list[ProgramRecommendation]
) -> list:
    """
    Mixed credential records from trade fixtures (4 + 6 = 10).
    """
    return apprenticeships + programs


@fixture
def credential_vectors(credential_records: list) -> np.ndarray:
    """
    Synthetic credential embeddings aligned with `credential_records`.
    """
    return _embeddings(len(credential_records), 77, unit=True)


@fixture
def credentials(
    credential_records : list,
    credential_vectors : np.ndarray
) -> Credentials:
    """
    Bundled credential records and vectors for graph construction.
    """
    return Credentials(
        records = credential_records,
        vectors = credential_vectors
    )


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
    centroids       : np.ndarray,
    cluster_vectors : np.ndarray,
    config          : PipelineConfig,
    credentials     : Credentials,
    job_zone_map    : dict[int, int],
    profiles        : dict[int, ClusterProfile]
) -> CareerPathwayGraph:
    """
    Career pathway graph built from synthetic embeddings.
    """
    return CareerPathwayGraph(
        centroids       = centroids,
        cluster_vectors = cluster_vectors,
        config          = config,
        credentials     = credentials,
        job_zone_map    = job_zone_map,
        profiles        = profiles
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
    return _embeddings(CORPUS_SIZE, 42)


@fixture
def resume_matcher(
    centroids     : np.ndarray,
    cluster_ids   : list[int],
    pathway_graph : CareerPathwayGraph,
    profiles      : dict[int, ClusterProfile],
    svd           : TruncatedSVD
) -> ResumeMatcher:
    """
    Resume matcher built from synthetic embeddings with mock encoder.
    """
    mock: Any = MockEncoder()
    return ResumeMatcher(
        centroids    = centroids,
        cluster_ids  = cluster_ids,
        graph        = pathway_graph,
        model        = mock,
        profiles     = profiles,
        svd          = svd,
        task_labels  = {
            cluster_id: [f"Task {cluster_id}-{i}" for i in range(5)]
            for cluster_id in cluster_ids
        },
        task_vectors = {
            cluster_id: _embeddings(5, cluster_id, unit=True)
            for cluster_id in cluster_ids
        }
    )


@fixture
def soc_vectors() -> np.ndarray:
    """
    Synthetic occupation embeddings (5 occupations, EMBEDDING_DIM).
    """
    return _embeddings(5, 99, unit=True)


@fixture
def svd(unit_vectors: np.ndarray) -> TruncatedSVD:
    """
    Fitted TruncatedSVD for resume projection tests.
    """
    return TruncatedSVD(n_components=COMPONENT_COUNT, random_state=42).fit(unit_vectors)


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
    return _load(list[ApprenticeshipContext], "matching/apprenticeships.json")


@fixture
def programs() -> list[ProgramRecommendation]:
    """
    Load synthetic educational program fixtures.
    """
    return _load(list[ProgramRecommendation], "matching/programs.json")


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
