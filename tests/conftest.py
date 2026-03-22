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
                                    reference → table_builder, figure_builder
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
from chalkline.display.figures    import FigureBuilder
from chalkline.display.tables     import TableBuilder
from chalkline.matching.matcher   import ResumeMatcher
from chalkline.matching.schemas   import MatchResult
from chalkline.pathways.graph     import CareerPathwayGraph
from chalkline.pathways.loaders   import LexiconLoader
from chalkline.pathways.schemas   import CareerEdge, Cluster, ClusterAssignments
from chalkline.pathways.schemas   import ClusterTasks, Credential
from chalkline.pathways.schemas   import Neighborhood, OnetOccupation
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
def occupations(lexicon_loader: LexiconLoader) -> list[OnetOccupation]:
    """
    Synthetic O*NET occupation records from the loader.
    """
    return lexicon_loader.occupations


# ── Embedding pipeline fixtures ──────────────────────────────────────


@fixture
def assignments(coordinates: np.ndarray) -> ClusterAssignments:
    """
    Ward-linkage HAC assignments at k=CLUSTER_COUNT.
    """
    return ClusterAssignments(AgglomerativeClustering(
        linkage    = "ward",
        n_clusters = CLUSTER_COUNT
    ).fit_predict(coordinates))


@fixture
def centroids(
    assignments : ClusterAssignments,
    coordinates : np.ndarray
) -> np.ndarray:
    """
    Mean SVD coordinates per cluster (CLUSTER_COUNT, COMPONENT_COUNT).
    """
    return assignments.centroids(coordinates)


@fixture
def cluster_ids(assignments: ClusterAssignments) -> list[int]:
    """
    Sorted unique cluster IDs from assignments.
    """
    return assignments.cluster_ids


@fixture
def cluster_vectors(
    assignments : ClusterAssignments,
    raw_vectors : np.ndarray
) -> np.ndarray:
    """
    Mean posting embedding per cluster, L2-normalized (CLUSTER_COUNT,
    EMBEDDING_DIM).
    """
    return assignments.cluster_vectors(raw_vectors)


@fixture
def clusters(
    assignments  : ClusterAssignments,
    job_zone_map : dict[int, int]
) -> dict[int, Cluster]:
    """
    Unified cluster objects with synthetic metadata and task
    embeddings. Postings are empty because the test corpus has
    fewer entries than the synthetic embedding matrix.
    """
    return {
        cid: Cluster(
            cluster_id  = cid,
            job_zone    = job_zone_map[cid],
            members     = assignments.members[cid],
            modal_title = f"Title {cid}",
            postings    = [],
            sector      = f"Sector {cid % 2}",
            size        = len(assignments.members[cid]),
            soc_title   = f"Occupation {cid}",
            tasks       = ClusterTasks(
                labels  = [f"Task {cid}-{i}" for i in range(5)],
                vectors = _embeddings(5, cid, unit=True)
            )
        )
        for cid in assignments.cluster_ids
    }


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
    centroids       : np.ndarray,
    cluster_vectors : np.ndarray,
    clusters        : dict[int, Cluster],
    credentials     : list[Credential]
) -> CareerPathwayGraph:
    """
    Career pathway graph built from synthetic embeddings.
    """
    return CareerPathwayGraph(
        centroids              = centroids,
        cluster_vectors        = cluster_vectors,
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
    centroids     : np.ndarray,
    clusters      : dict[int, Cluster],
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
        centroids = centroids,
        clusters  = clusters,
        graph     = pathway_graph,
        model     = mock_encoder,
        svd       = svd
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
    clusters    : dict[int, Cluster],
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
def figure_builder(pathway_graph: CareerPathwayGraph) -> FigureBuilder:
    """
    Figure builder wired to the synthetic pathway graph.
    """
    return FigureBuilder(
        matched_id = sorted(pathway_graph.clusters)[0],
        pathway    = pathway_graph,
        theme      = lambda: "plotly_white"
    )


@fixture
def match_result(resume_matcher: ResumeMatcher) -> MatchResult:
    """
    Match result from encoding a synthetic resume string.
    """
    return resume_matcher.match("electrician conduit bending NEC code")


@fixture
def member_names(reference: dict) -> tuple[list[dict], list[str]]:
    """
    AGC member records with pre-lowercased names for matching tests.
    """
    members = reference["agc_members"]
    return members, [m["name"].lower() for m in members]


@fixture
def neighborhood(
    clusters      : dict[int, Cluster],
    pathway_graph : CareerPathwayGraph
) -> Neighborhood:
    """
    Neighborhood view from the first cluster in the graph.
    """
    return pathway_graph.neighborhood(sorted(clusters)[0])


@fixture
def pipeline_namespace(
    clusters : dict[int, Cluster],
    config   : PipelineConfig
):
    """
    Lightweight namespace mimicking the `Chalkline` dataclass for
    display tests without requiring the full Hamilton pipeline.
    """
    return SimpleNamespace(
        clusters = clusters,
        config   = config
    )


@fixture
def reference() -> dict:
    """
    Stakeholder reference data loaded from display fixture JSON.
    """
    return {
        "agc_members" : loads(
            (FIXTURES / "display" / "members.json").read_text()
        ),
        "career_urls" : [],
        "job_boards"  : loads(
            (FIXTURES / "display" / "boards.json").read_text()
        )
    }


@fixture
def table_builder(
    match_result : MatchResult,
    pipeline_namespace,
    reference    : dict
) -> TableBuilder:
    """
    Table builder wired to the synthetic pipeline and match result.
    """
    return TableBuilder(
        pipeline  = pipeline_namespace,
        reference = reference,
        result    = match_result
    )
