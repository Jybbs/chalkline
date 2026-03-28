"""
Shared test fixtures for the Chalkline test suite.

Fixtures form an embedding pipeline chain where each step's output is
independently tappable by any test module:

    corpus → raw_vectors → unit_vectors → coordinates → assignments
                                  ↓                          ↓
                           soc_vectors → job_zone_map → clusters → graph
                                                                     ↓
                           credentials ────────────────────────→ matcher
                                                                     ↓
                                    reference → charts
"""

import numpy as np

from datetime              import date
from json                  import loads
from pathlib               import Path
from pydantic              import TypeAdapter
from pytest                import fixture
from sklearn.cluster       import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from types                 import SimpleNamespace
from typing                import Any, Callable

from chalkline.collection.schemas import Corpus, Posting
from chalkline.display.charts     import Charts
from chalkline.display.theme      import Theme
from chalkline.matching.matcher   import ResumeMatcher
from chalkline.matching.schemas   import MatchResult
from chalkline.pathways.graph     import CareerPathwayGraph
from chalkline.pathways.loaders   import StakeholderReference
from chalkline.pathways.loaders   import LexiconLoader
from chalkline.pathways.schemas   import CareerEdge, Cluster, Clusters
from chalkline.pathways.schemas   import Credential, Occupation, Reach, Task
from chalkline.pipeline.schemas   import PipelineConfig


CLUSTER_COUNT   = 4
COMPONENT_COUNT = 4
CORPUS_SIZE     = 20
EMBEDDING_DIM   = 16
FIXTURES        = Path(__file__).resolve().parent / "fixtures"


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
def corpus() -> Corpus:
    """
    Posting corpus built from fixture data.
    """
    return Corpus({p.id: p for p in _postings()})


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
def lexicon_dir(tmp_path: Path) -> Path:
    """
    Write synthetic lexicon files to a temporary directory.
    """
    (tmp_path / "onet.json").write_text(
        (FIXTURES / "pathways" / "onet_occupations.json").read_text()
    )
    return tmp_path


@fixture
def lexicon_loader(lexicon_dir: Path) -> LexiconLoader:
    """
    Load all synthetic lexicon files via `LexiconLoader`.
    """
    return LexiconLoader(lexicon_dir)


@fixture
def occupations(lexicon_loader: LexiconLoader) -> list[Occupation]:
    """
    Synthetic O*NET occupation records from the loader.
    """
    return lexicon_loader.occupations


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
    return normalize(np.stack([
        raw_vectors[assignments == cid].mean(axis=0)
        for cid in sorted(np.unique(assignments))
    ]))


@fixture
def clusters(
    assignments     : np.ndarray,
    centroids       : np.ndarray,
    cluster_vectors : np.ndarray,
    job_zone_map    : dict[int, int]
) -> Clusters:
    """
    Unified cluster container with synthetic metadata and task
    embeddings. Postings are empty because the test corpus has
    fewer entries than the synthetic embedding matrix.
    """
    cluster_ids = sorted(np.unique(assignments).tolist())
    items = {
        cid: Cluster(
            cluster_id   = cid,
            job_zone     = job_zone_map[cid],
            members      = np.where(assignments == cid)[0],
            modal_title  = f"Title {cid}",
            postings     = [],
            sector       = f"Sector {cid % 2}",
            size         = int((assignments == cid).sum()),
            soc_title    = f"Occupation {cid}",
            tasks        = [
                Task(name=f"Task {cid}-{i}", vector=v)
                for i, v in enumerate(_embeddings(5, cid, unit=True))
            ]
        )
        for cid in cluster_ids
    }
    return Clusters(
        centroids = centroids,
        items     = items,
        vectors   = cluster_vectors
    )


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
                {"min_hours": 8000, "rapids_code": f"0{i}"}
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
        destination_percentile = 5,
        lateral_neighbors      = 2,
        source_percentile      = 75,
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
    clusters      : Clusters,
    pathway_graph : CareerPathwayGraph,
    svd           : TruncatedSVD
) -> ResumeMatcher:
    """
    Resume matcher built from synthetic embeddings with mock encoder.
    """
    mock_encoder: Any = SimpleNamespace(
        encode = lambda texts,
        unit   = True: _embeddings(len(texts), seed=len(texts), unit=unit)
    )
    return ResumeMatcher(
        clusters = clusters,
        encoder  = mock_encoder,
        graph    = pathway_graph,
        svd      = svd
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


# ── Display fixtures ─────────────────────────────────────────────────


@fixture
def edge_factory(
    clusters    : Clusters,
    credentials : list[Credential]
) -> Callable:
    """
    Factory for `CareerEdge` instances with a default cluster ID and
    optional credential filtering by kind.
    """
    cluster_id = next(iter(clusters))

    def _build(kind: str | None = None) -> CareerEdge:
        creds = (
            [next(c for c in credentials if c.kind == kind)]
            if kind
            else []
        )
        return CareerEdge(
            cluster_id  = cluster_id,
            credentials = creds,
            weight      = 0.9
        )
    return _build


@fixture
def charts(pathway_graph: CareerPathwayGraph) -> Charts:
    """
    Charts factory wired to the synthetic pathway graph.
    """
    return Charts(
        matched_id = pathway_graph.clusters.cluster_ids[0],
        pathway    = pathway_graph,
        theme      = Theme(dark_fn=lambda: False)
    )


@fixture
def match_result(resume_matcher: ResumeMatcher) -> MatchResult:
    """
    Match result from encoding a synthetic resume string.
    """
    return resume_matcher.match("electrician conduit bending NEC code")


@fixture
def member_names(reference: StakeholderReference) -> tuple[list[dict], list[str]]:
    """
    AGC member records with pre-lowercased names for matching tests.
    """
    members = reference.agc_members
    return members, [m["name"].lower() for m in members]


@fixture
def reach(
    clusters      : Clusters,
    pathway_graph : CareerPathwayGraph
) -> Reach:
    """
    Reach view from the first cluster in the graph.
    """
    return pathway_graph.reach(clusters.cluster_ids[0])


@fixture
def reference() -> StakeholderReference:
    """
    Stakeholder reference data from display fixture JSONs.
    """
    ref = StakeholderReference(FIXTURES / "display")
    ref.agc_members = loads((FIXTURES / "display" / "members.json").read_text())
    ref.career_urls = []
    ref.job_boards  = loads((FIXTURES / "display" / "boards.json").read_text())
    return ref


