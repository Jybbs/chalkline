"""
Shared test fixtures for the Chalkline test suite.

Fixtures form an embedding pipeline chain where each step's output is
independently tappable by any test module:

    corpus → raw_vectors → unit_vectors → coordinates → assignments
                                  ↓                          ↓
                   encoded_occupations → job_zone_map → clusters → graph
                                                                     ↓
                           credentials ────────────────────────→ matcher
                                                                     ↓
                                    reference → charts
"""

import numpy as np

from datetime              import date
from pathlib               import Path
from pydantic              import TypeAdapter
from pytest                import fixture
from sklearn.cluster       import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from types                 import SimpleNamespace
from typing                import Any, Callable

from chalkline.collection.schemas import Posting
from chalkline.display.charts     import Charts
from chalkline.display.loaders    import ContentLoader, Layout
from chalkline.display.routes     import Routes
from chalkline.display.theme      import Theme
from chalkline.matching.matcher   import ResumeMatcher
from chalkline.matching.schemas   import MatchResult
from chalkline.pathways.clusters  import Cluster, Clusters, Task
from chalkline.pathways.graph     import CareerPathwayGraph
from chalkline.pathways.loaders   import LaborLoader, StakeholderReference
from chalkline.pathways.schemas   import Credential


CLUSTER_COUNT   = 4
COMPONENT_COUNT = 4
CORPUS_SIZE     = 20
EMBEDDING_DIM   = 16
FIXTURES        = Path(__file__).resolve().parent / "fixtures"


def _embeddings(rows: int, seed: int, unit: bool = False) -> np.ndarray:
    """
    Deterministic random embeddings for test fixtures.
    """
    vectors = np.random.RandomState(seed).randn(rows, EMBEDDING_DIM).astype(np.float32)
    return np.asarray(normalize(vectors)) if unit else vectors


def _load(schema: Any, path: str) -> Any:
    """
    Validate fixture JSON against a Pydantic schema or container type.
    """
    return TypeAdapter(schema).validate_json(
        (FIXTURES / path).read_bytes()
    )


def _postings() -> list[Posting]:
    """
    Load posting fixtures from JSON.
    """
    return _load(list[Posting], "collection/postings.json")


# ── Collection fixtures ─────────────────────────────────────────────


@fixture
def posting() -> Callable[..., Posting]:
    """
    Factory for `Posting` instances with overridable identity fields.
    """
    def _build(
        company     : str         = "Test Co",
        date_posted : date | None = date(2026, 1, 1),
        title       : str         = "Worker"
    ) -> Posting:
        return Posting(
            company     = company,
            date_posted = date_posted,
            description = "x" * 50,
            source_url  = "https://example.com",
            title       = title
        )
    return _build


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
def labor(tmp_path: Path) -> LaborLoader:
    """
    Empty `LaborLoader` backed by an `[]` file so every SOC title resolves
    to a default-valued `LaborRecord` with null wage. Tests do not depend
    on real wage values, so the empty loader keeps `Clusters` construction
    self-contained without fixture data.
    """
    path = tmp_path / "labor.json"
    path.write_text("[]")
    return LaborLoader(path)


@fixture
def lexicon_dir(tmp_path: Path) -> Path:
    """
    Write synthetic lexicon files to a temporary directory.
    """
    (tmp_path / "onet.json").write_text(
        (FIXTURES / "pathways" / "onet_occupations.json").read_text()
    )
    return tmp_path


# ── Embedding pipeline fixtures ──────────────────────────────────────


@fixture
def assignments(coordinates: np.ndarray) -> np.ndarray:
    """
    Ward-linkage HAC label array at k=CLUSTER_COUNT.
    """
    return AgglomerativeClustering(
        linkage    = "ward",
        n_clusters = CLUSTER_COUNT
    ).fit_predict(coordinates)


@fixture
def centroids(assignments: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """
    Mean SVD coordinates per cluster (CLUSTER_COUNT, COMPONENT_COUNT).
    """
    return np.stack([
        coordinates[assignments == cid].mean(axis=0)
        for cid in sorted(np.unique(assignments))
    ])


@fixture
def cluster_ids(assignments: np.ndarray) -> list[int]:
    """
    Sorted unique cluster IDs from assignments.
    """
    return sorted(np.unique(assignments).tolist())


@fixture
def cluster_vectors(
    assignments : np.ndarray,
    raw_vectors : np.ndarray
) -> np.ndarray:
    """
    Mean posting embedding per cluster, L2-normalized (CLUSTER_COUNT,
    EMBEDDING_DIM).
    """
    return np.asarray(normalize(np.stack([
        raw_vectors[assignments == cid].mean(axis=0)
        for cid in sorted(np.unique(assignments))
    ])))


@fixture
def clusters(
    assignments     : np.ndarray,
    centroids       : np.ndarray,
    cluster_vectors : np.ndarray,
    job_zone_map    : dict[int, int],
    labor           : LaborLoader,
    unit_vectors    : np.ndarray
) -> Clusters:
    """
    Unified cluster container with synthetic metadata and task
    embeddings. Postings are empty because the test corpus has
    fewer entries than the synthetic embedding matrix.
    """
    cluster_ids = sorted(np.unique(assignments).tolist())
    titles      = [f"Occupation {cid}" for cid in cluster_ids]
    items = {
        cid: Cluster(
            cluster_id  = cid,
            embeddings  = unit_vectors[assignments == cid],
            job_zone    = job_zone_map[cid],
            modal_title = f"Title {cid}",
            postings    = [],
            sector      = f"Sector {cid % 2}",
            size        = int((assignments == cid).sum()),
            soc_title   = titles[i],
            tasks       = [
                Task(name=f"Task {cid}-{i}", vector=v)
                for i, v in enumerate(_embeddings(5, cid, unit=True))
            ]
        )
        for i, cid in enumerate(cluster_ids)
    }
    return Clusters(
        centroids         = centroids,
        items             = items,
        labor             = labor,
        occupation_titles = titles,
        soc_similarity    = np.eye(len(cluster_ids), dtype=np.float32),
        softmax_tau       = 0.02,
        vectors           = cluster_vectors,
        wage_round        = 10,
        wage_topk         = 3
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
def credentials() -> list[Credential]:
    """
    Synthetic credentials with vectors for graph construction.
    """
    vectors = _embeddings(10, seed=77, unit=True)
    return [
        Credential(
            embedding_text = f"credential {i}",
            kind           = "apprenticeship" if i < 4 else "program",
            label          = f"Credential {i}",
            metadata       = (
                {
                    "min_hours"   : 8000,
                    "rapids_code" : f"0{i}"
                }
                if i < 4
                else {
                    "credential"  : "AAS",
                    "institution" : "SMCC",
                    "url"         : "https://example.com"
                }
            ),
            vector         = vectors[i].tolist()
        )
        for i in range(10)
    ]


@fixture
def job_zone_map(cluster_ids: list[int]) -> dict[int, int]:
    """
    Deterministic Job Zone assignment for synthetic clusters.

    Spreads clusters across Job Zones 2-4 to exercise stepwise k-NN lateral and
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
def pathway_graph(
    clusters    : Clusters,
    credentials : list[Credential]
) -> CareerPathwayGraph:
    """
    Career pathway graph built from synthetic embeddings.
    """
    return CareerPathwayGraph(
        clusters               = clusters,
        credentials            = credentials,
        destination_percentile = 20,
        lateral_neighbors      = 2,
        upward_neighbors       = 2
    )


@fixture
def raw_vectors() -> np.ndarray:
    """
    Synthetic posting embeddings (CORPUS_SIZE, EMBEDDING_DIM).
    """
    return _embeddings(CORPUS_SIZE, 42)


@fixture
def resume_matcher(
    clusters : Clusters,
    svd      : TruncatedSVD
) -> ResumeMatcher:
    """
    Resume matcher built from synthetic embeddings with mock encoder.
    """
    mock_encoder: Any = SimpleNamespace(
        encode=lambda t, unit=True: _embeddings(len(t), seed=len(t), unit=unit)
    )
    return ResumeMatcher(
        clusters = clusters,
        encoder  = mock_encoder,
        svd      = svd
    )


@fixture
def svd(unit_vectors: np.ndarray) -> TruncatedSVD:
    """
    Fitted TruncatedSVD for resume projection tests.
    """
    return TruncatedSVD(
        n_components = COMPONENT_COUNT,
        random_state = 42
    ).fit(unit_vectors)


@fixture
def unit_vectors(raw_vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalized posting embeddings.
    """
    return np.asarray(normalize(raw_vectors))


# ── Display fixtures ─────────────────────────────────────────────────


@fixture
def charts(pathway_graph: CareerPathwayGraph) -> Charts:
    """
    Charts factory wired to the synthetic pathway graph.
    """
    return Charts(
        matched_id = pathway_graph.clusters.cluster_ids[0],
        pathway    = pathway_graph,
        theme      = Theme()
    )


@fixture
def content() -> ContentLoader:
    """
    Centralized content loader for display-layer TOML.
    """
    return ContentLoader()


@fixture
def layout(content: ContentLoader) -> Layout:
    """
    Layout renderer backed by the shared content loader.
    """
    return Layout(content)


@fixture
def match_result(resume_matcher: ResumeMatcher) -> MatchResult:
    """
    Match result from encoding a synthetic resume string.
    """
    return resume_matcher.match("electrician conduit bending NEC code")


@fixture
def posting_factory() -> Callable:
    """
    Factory for minimal `Posting` instances with a given company name.
    """
    def _build(company: str) -> Posting:
        return Posting(
            company     = company,
            date_posted = None,
            description = "x" * 50,
            source_url  = "https://example.com",
            title       = "Test"
        )
    return _build


@fixture
def reference() -> StakeholderReference:
    """
    Stakeholder reference data from display fixture JSONs.
    """
    return StakeholderReference(FIXTURES / "display")


@fixture
def routes(layout: Layout, theme: Theme) -> Routes:
    """
    Route card builder wired to the shared layout and theme.
    """
    return Routes(layout, theme)


@fixture
def theme() -> Theme:
    """
    Dashboard theme with the unified color palette.
    """
    return Theme()
