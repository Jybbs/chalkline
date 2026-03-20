"""
Hamilton DAG node functions for the Chalkline embedding pipeline.

Each function is a node whose parameter names declare dependencies that
Hamilton resolves automatically. The pipeline encodes posting descriptions
with a sentence transformer, reduces with SVD, clusters with Ward HAC,
assigns Job Zones via O*NET cosine matching, builds a stepwise k-NN career
graph with per-edge credential enrichment, and fits a resume matcher for
single-resume inference.
"""

import numpy as np

from collections                 import Counter
from datetime                    import datetime, timezone
from hamilton.function_modifiers import extract_fields
from loguru                      import logger
from pydantic                    import TypeAdapter
from sentence_transformers       import SentenceTransformer
from sklearn.cluster             import AgglomerativeClustering
from sklearn.decomposition       import TruncatedSVD
from sklearn.metrics.pairwise    import cosine_similarity
from sklearn.preprocessing       import normalize

from chalkline.collection.storage import CorpusStorage
from chalkline.extraction.loaders import LexiconLoader
from chalkline.matching.matcher   import ResumeMatcher
from chalkline.pipeline.graph     import CareerPathwayGraph
from chalkline.pipeline.schemas   import ApprenticeshipContext, ClusterProfile
from chalkline.pipeline.schemas   import Credentials, PipelineConfig
from chalkline.pipeline.schemas   import PipelineManifest, ProgramRecommendation
from chalkline.pipeline.trades    import TradeIndex


def assignments(config: PipelineConfig, coordinates: np.ndarray) -> np.ndarray:
    """
    Fit Ward-linkage HAC on SVD coordinates.
    """
    return AgglomerativeClustering(
        linkage    = "ward",
        n_clusters = config.cluster_count
    ).fit_predict(coordinates)


def centroids(assignments: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """
    Mean SVD coordinates per cluster for resume distance computation.
    """
    cluster_ids = sorted(set(assignments))
    return np.stack([
        coordinates[assignments == cluster_id].mean(axis=0)
        for cluster_id in cluster_ids
    ])


def cluster_vectors(assignments: np.ndarray, raw_vectors: np.ndarray) -> np.ndarray:
    """
    Mean posting embedding per cluster in the full embedding space,
    L2-normalized for cosine similarity against occupations and credentials.
    """
    cluster_ids = sorted(set(assignments))
    return np.stack([
        normalize(raw_vectors[assignments == cluster_id].mean(axis=0, keepdims=True))[0]
        for cluster_id in cluster_ids
    ])


def soc_similarity(cluster_vectors: np.ndarray, soc_vectors: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between cluster centroids and O*NET occupations.
    """
    return cosine_similarity(cluster_vectors, soc_vectors)


@extract_fields({
    "corpus"        : dict,
    "corpus_titles" : dict
})
def corpus_data(config: PipelineConfig) -> dict:
    """
    Load posting texts and titles from the corpus directory.

    Raises:
        FileNotFoundError: When no JSON postings exist in the configured directory.
    """
    postings = CorpusStorage(config.postings_dir).load()
    if not (corpus := {
        posting.id: posting.description
        for posting in postings if posting.id is not None
    }):
        raise FileNotFoundError(f"No postings found in {config.postings_dir}")
    return {
        "corpus"        : corpus,
        "corpus_titles" : {
            posting.id: posting.title
            for posting in postings if posting.id is not None
        }
    }


def credentials(
    lexicons : LexiconLoader,
    model    : SentenceTransformer,
    trades   : TradeIndex
) -> Credentials:
    """
    Encode the credential catalog (apprenticeships, programs,
    certifications) with the sentence transformer. Returns a
    `Credentials` bundling the typed records with their embedding
    vectors so the graph can attach full credential metadata to edges.
    """
    records = trades.apprenticeships + trades.programs + lexicons.certifications
    logger.info(f"Encoding {len(records)} credentials...")
    return Credentials(
        records = records,
        vectors = normalize(model.encode(
            [r.embedding_text for r in records],
            show_progress_bar = False
        ))
    )


@extract_fields({
    "coordinates" : np.ndarray,
    "svd"         : TruncatedSVD
})
def reduction(config: PipelineConfig, unit_vectors: np.ndarray) -> dict:
    """
    Fit TruncatedSVD on L2-normalized embeddings and expose both the reduced
    coordinates and the fitted SVD for resume projection.
    """
    svd = TruncatedSVD(n_components=config.component_count, random_state=config.random_seed)
    return {
        "coordinates" : svd.fit_transform(unit_vectors),
        "svd"         : svd
    }


def graph(
    centroids       : np.ndarray,
    cluster_vectors : np.ndarray,
    config          : PipelineConfig,
    credentials     : Credentials,
    job_zone_map    : dict[int, int],
    profiles        : dict[int, ClusterProfile]
) -> CareerPathwayGraph:
    """
    Build the career pathway graph with stepwise k-NN backbone and per-edge
    credential enrichment.
    """
    result = CareerPathwayGraph(
        centroids       = centroids,
        cluster_vectors = cluster_vectors,
        config          = config,
        credentials     = credentials,
        job_zone_map    = job_zone_map,
        profiles        = profiles
    )
    logger.info(
        f"Career graph: {result.graph.number_of_nodes()} nodes, "
        f"{result.graph.number_of_edges()} edges"
    )
    return result


def job_zone_map(
    config         : PipelineConfig,
    lexicons       : LexiconLoader,
    soc_similarity : np.ndarray
) -> dict[int, int]:
    """
    Assign Job Zones to clusters via top-k median cosine against O*NET
    Task+DWA embeddings.
    """
    return {
        cluster_id: int(np.median([
            lexicons.occupations[i].job_zone
            for i in np.argsort(soc_similarity[cluster_id])[-config.soc_neighbors:]
        ]))
        for cluster_id in range(len(soc_similarity))
    }


def lexicons(config: PipelineConfig) -> LexiconLoader:
    """
    Load lexicon files from the configured directory.
    """
    return LexiconLoader(config.lexicon_dir)


def manifest(config: PipelineConfig, corpus: dict[str, str]) -> PipelineManifest:
    """
    Build provenance metadata from fitted artifacts.
    """
    return PipelineManifest(
        component_count = config.component_count,
        corpus_size     = len(corpus),
        embedding_model = config.embedding_model,
        posting_ids     = sorted(corpus),
        timestamp       = datetime.now(timezone.utc).isoformat()
    )


def matcher(
    centroids    : np.ndarray,
    config       : PipelineConfig,
    graph        : CareerPathwayGraph,
    model        : SentenceTransformer,
    profiles     : dict[int, ClusterProfile],
    svd          : TruncatedSVD,
    task_labels  : dict[int, list[str]],
    task_vectors : dict[int, np.ndarray]
) -> ResumeMatcher:
    """
    Build the resume matcher with all artifacts needed for single-resume
    inference.
    """
    return ResumeMatcher(
        centroids    = centroids,
        cluster_ids  = sorted(profiles),
        graph        = graph,
        model        = model,
        profiles     = profiles,
        svd          = svd,
        task_labels  = task_labels,
        task_vectors = task_vectors
    )


def unit_vectors(raw_vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize raw posting embeddings for SVD input.
    """
    return normalize(raw_vectors)


def soc_vectors(lexicons: LexiconLoader, model: SentenceTransformer) -> np.ndarray:
    """
    Encode O*NET occupations using uncapped Task+DWA text, L2-normalized for
    cosine similarity against cluster centroids.
    """
    logger.info(f"Encoding {len(lexicons.occupations)} occupations...")
    return normalize(model.encode(
        [occ.embedding_text for occ in lexicons.occupations],
        show_progress_bar = False
    ))


@extract_fields({
    "task_labels"  : dict,
    "task_vectors" : dict
})
def occupation_tasks(
    job_zone_map   : dict[int, int],
    lexicons       : LexiconLoader,
    model          : SentenceTransformer,
    soc_similarity : np.ndarray
) -> dict:
    """
    Encode per-cluster O*NET Task+DWA embeddings for resume gap analysis.

    For each cluster, identifies the nearest occupation via cosine
    similarity, then encodes that occupation's Task+DWA elements as
    individual embeddings for per-task gap scoring.
    """
    task_data = {
        cluster_id: [t.name for t in nearest.task_elements]
        for cluster_id in range(len(soc_similarity))
        for nearest in [lexicons.nearest_occupation(soc_similarity[cluster_id])]
        if nearest.task_elements
    }
    return {
        "task_labels"  : task_data,
        "task_vectors" : {
            cluster_id: normalize(model.encode(names, show_progress_bar=False))
            for cluster_id, names in task_data.items()
        }
    }


def profiles(
    assignments    : np.ndarray,
    corpus_titles  : dict[str, str],
    job_zone_map   : dict[int, int],
    lexicons       : LexiconLoader,
    soc_similarity : np.ndarray
) -> dict[int, ClusterProfile]:
    """
    Build cluster profiles with Job Zone, sector, occupation title, and
    modal posting title for each cluster.
    """
    doc_ids      = sorted(corpus_titles)
    return {
        cluster_id: ClusterProfile(
            cluster_id  = cluster_id,
            job_zone    = job_zone_map[cluster_id],
            modal_title = Counter(
                corpus_titles[doc_ids[i]] for i in members
            ).most_common(1)[0][0],
            sector      = nearest.sector,
            size        = len(members),
            soc_title   = nearest.title
        )
        for cluster_id in sorted(set(assignments))
        for nearest in [lexicons.nearest_occupation(soc_similarity[cluster_id])]
        for members in [np.where(assignments == cluster_id)[0]]
    }


def raw_vectors(corpus: dict[str, str], model: SentenceTransformer) -> np.ndarray:
    """
    Encode all posting descriptions with the sentence transformer.
    """
    texts = [corpus[doc_id] for doc_id in sorted(corpus)]
    logger.info(f"Encoding {len(texts)} postings...")
    return model.encode(texts, show_progress_bar=False)


def trades(config: PipelineConfig) -> TradeIndex:
    """
    Load apprenticeship and program reference data into a `TradeIndex`.
    """
    load = lambda schema, name: TypeAdapter(schema).validate_json(
        (config.lexicon_dir / name).read_bytes()
    )
    return TradeIndex(
        apprenticeships = load(list[ApprenticeshipContext], "apprenticeships.json"),
        programs        = load(list[ProgramRecommendation], "programs.json")
    )
