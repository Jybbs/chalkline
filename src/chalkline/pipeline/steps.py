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
from hamilton.function_modifiers import extract_fields
from loguru                      import logger
from sklearn.cluster             import AgglomerativeClustering
from sklearn.decomposition       import TruncatedSVD
from sklearn.metrics.pairwise    import cosine_similarity
from sklearn.preprocessing       import normalize

from chalkline.collection.storage import CorpusStorage
from chalkline.extraction.loaders import LexiconLoader
from chalkline.matching.matcher   import ResumeMatcher
from chalkline.pipeline.graph     import CareerPathwayGraph
from chalkline.pipeline.schemas   import ClusterAssignments, ClusterProfile, ClusterTasks
from chalkline.pipeline.schemas   import Corpus, Credentials, Encoder, PipelineConfig
from chalkline.pipeline.trades    import TradeIndex


def assignments(config: PipelineConfig, coordinates: np.ndarray) -> ClusterAssignments:
    """
    Fit Ward-linkage HAC on SVD coordinates.
    """
    return ClusterAssignments(AgglomerativeClustering(
        linkage    = "ward",
        n_clusters = config.cluster_count
    ).fit_predict(coordinates))


def centroids(assignments: ClusterAssignments, coordinates: np.ndarray) -> np.ndarray:
    """
    Mean SVD coordinates per cluster for resume distance computation.
    """
    return assignments.centroids(coordinates)


def cluster_vectors(assignments: ClusterAssignments, raw_vectors: np.ndarray) -> np.ndarray:
    """
    Mean posting embedding per cluster in the full embedding space,
    L2-normalized for cosine similarity against occupations and credentials.
    """
    return assignments.cluster_vectors(raw_vectors)


def corpus(config: PipelineConfig) -> Corpus:
    """
    Load the posting corpus from the configured directory.

    Raises:
        FileNotFoundError: When no valid postings exist.
    """
    postings = CorpusStorage(config.postings_dir).load()
    if not postings:
        raise FileNotFoundError(f"No postings found in {config.postings_dir}")
    return Corpus({p.id: p for p in postings})


def credentials(
    config   : PipelineConfig,
    lexicons : LexiconLoader,
    model    : Encoder
) -> Credentials:
    """
    Encode the credential catalog (apprenticeships, programs,
    certifications) with the sentence transformer. Returns a `Credentials`
    bundling the typed records with their embedding vectors so the graph can
    attach full credential metadata to edges.
    """
    trades  = TradeIndex.from_directory(config.lexicon_dir)
    records = trades.apprenticeships + trades.programs + lexicons.certifications
    logger.info(f"Encoding {len(records)} credentials...")
    return Credentials(
        records = records,
        vectors = model.encode([r.embedding_text for r in records])
    )


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
    assignments    : ClusterAssignments,
    config         : PipelineConfig,
    lexicons       : LexiconLoader,
    soc_similarity : np.ndarray
) -> dict[int, int]:
    """
    Assign Job Zones to clusters via top-k median cosine against O*NET
    Task+DWA embeddings.
    """
    return {
        cid: int(np.median([
            lexicons.occupations[i].job_zone
            for i in np.argsort(soc_similarity[cid])[-config.soc_neighbors:]
        ]))
        for cid in assignments.cluster_ids
    }


def matcher(
    assignments : ClusterAssignments,
    centroids   : np.ndarray,
    graph       : CareerPathwayGraph,
    model       : Encoder,
    profiles    : dict[int, ClusterProfile],
    soc_tasks   : dict[int, ClusterTasks],
    svd         : TruncatedSVD
) -> ResumeMatcher:
    """
    Build the resume matcher with all artifacts needed for single-resume
    inference.
    """
    return ResumeMatcher(
        centroids   = centroids,
        cluster_ids = assignments.cluster_ids,
        graph       = graph,
        model       = model,
        profiles    = profiles,
        soc_tasks   = soc_tasks,
        svd         = svd
    )


def profiles(
    assignments    : ClusterAssignments,
    corpus         : Corpus,
    job_zone_map   : dict[int, int],
    lexicons       : LexiconLoader,
    soc_similarity : np.ndarray
) -> dict[int, ClusterProfile]:
    """
    Build cluster profiles with Job Zone, sector, occupation title, and
    modal posting title for each cluster.
    """
    return {
        cid: ClusterProfile(
            cluster_id  = cid,
            job_zone    = job_zone_map[cid],
            modal_title = Counter(
                corpus.postings[corpus.posting_ids[i]].title
                for i in assignments.members[cid]
            ).most_common(1)[0][0],
            sector      = nearest.sector,
            size        = len(assignments.members[cid]),
            soc_title   = nearest.title
        )
        for cid     in assignments.cluster_ids
        for nearest in [lexicons.nearest_occupation(soc_similarity[cid])]
    }


def raw_vectors(corpus: Corpus, model: Encoder) -> np.ndarray:
    """
    Encode all posting descriptions with the sentence transformer.
    """
    logger.info(f"Encoding {len(corpus.posting_ids)} postings...")
    return model.encode(corpus.descriptions, unit=False)


@extract_fields({
    "coordinates" : np.ndarray,
    "svd"         : TruncatedSVD
})
def reduction(config: PipelineConfig, raw_vectors: np.ndarray) -> dict:
    """
    L2-normalize raw embeddings, fit TruncatedSVD, and expose both the
    reduced coordinates and the fitted SVD for resume projection.
    """
    svd = TruncatedSVD(n_components=config.component_count, random_state=config.random_seed)
    return {
        "coordinates" : svd.fit_transform(normalize(raw_vectors)),
        "svd"         : svd
    }


def soc_similarity(cluster_vectors: np.ndarray, soc_vectors: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between cluster centroids and O*NET occupations.
    """
    return cosine_similarity(cluster_vectors, soc_vectors)


def soc_tasks(
    assignments    : ClusterAssignments,
    lexicons       : LexiconLoader,
    model          : Encoder,
    soc_similarity : np.ndarray
) -> dict[int, ClusterTasks]:
    """
    Encode per-cluster O*NET Task+DWA embeddings for resume gap analysis.

    For each cluster, identifies the nearest occupation via cosine
    similarity, then encodes that occupation's Task+DWA elements as
    individual embeddings for per-task gap scoring.
    """
    return {
        cid: ClusterTasks(
            labels  = (names := [t.name for t in nearest.task_elements]),
            vectors = model.encode(names)
        )
        for cid in assignments.cluster_ids
        for nearest in [lexicons.nearest_occupation(soc_similarity[cid])]
        if nearest.task_elements
    }


def soc_vectors(lexicons: LexiconLoader, model: Encoder) -> np.ndarray:
    """
    Encode O*NET occupations using uncapped Task+DWA text, L2-normalized for
    cosine similarity against cluster centroids.
    """
    logger.info(f"Encoding {len(lexicons.occupations)} occupations...")
    return model.encode(
        [occupation.embedding_text for occupation in lexicons.occupations]
    )
