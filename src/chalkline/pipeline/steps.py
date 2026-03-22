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
from pydantic                    import TypeAdapter
from sklearn.cluster             import AgglomerativeClustering
from sklearn.decomposition       import TruncatedSVD
from sklearn.metrics.pairwise    import cosine_similarity
from sklearn.preprocessing       import normalize

from chalkline.collection.schemas import Corpus
from chalkline.collection.storage import CorpusStorage
from chalkline.matching.matcher   import ResumeMatcher
from chalkline.pathways.graph     import CareerPathwayGraph
from chalkline.pathways.loaders   import LexiconLoader
from chalkline.pathways.schemas   import ClusterAssignments, ClusterProfile
from chalkline.pathways.schemas   import ClusterTasks, Credential
from chalkline.pipeline.schemas   import Encoder, PipelineConfig


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
    if not (postings := CorpusStorage(config.postings_dir).load()):
        raise FileNotFoundError(
            f"No postings found in {config.postings_dir}"
        )
    return Corpus({p.id: p for p in postings})


def credentials(config: PipelineConfig, model: Encoder) -> list[Credential]:
    """
    Load the curated credential catalog, encode with the sentence
    transformer, and attach vectors to each credential instance.
    """
    records = TypeAdapter(list[Credential]).validate_json(
        (config.lexicon_dir / "credentials.json").read_bytes()
    )

    logger.info(f"Encoding {len(records)} credentials...")
    for credential, vector in zip(
        records,
        model.encode([r.embedding_text for r in records])
    ):
        credential.vector = vector.tolist()

    return records


def graph(
    centroids       : np.ndarray,
    cluster_vectors : np.ndarray,
    config          : PipelineConfig,
    credentials     : list[Credential],
    job_zone_map    : dict[int, int],
    profiles        : dict[int, ClusterProfile]
) -> CareerPathwayGraph:
    """
    Build the career pathway graph with stepwise k-NN backbone and per-edge
    credential enrichment.
    """
    result = CareerPathwayGraph(
        centroids              = centroids,
        cluster_vectors        = cluster_vectors,
        credentials            = credentials,
        destination_percentile = config.destination_percentile,
        job_zone_map           = job_zone_map,
        lateral_neighbors      = config.lateral_neighbors,
        profiles               = profiles,
        source_percentile      = config.source_percentile,
        upward_neighbors       = config.upward_neighbors
    )
    logger.info(
        f"Career graph: {result.graph.number_of_nodes()} nodes, "
        f"{result.edge_count} edges"
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
    centroids : np.ndarray,
    graph     : CareerPathwayGraph,
    model     : Encoder,
    profiles  : dict[int, ClusterProfile],
    soc_tasks : dict[int, ClusterTasks],
    svd       : TruncatedSVD
) -> ResumeMatcher:
    """
    Build the resume matcher with all artifacts needed for single-resume
    inference.
    """
    return ResumeMatcher(
        centroids = centroids,
        graph     = graph,
        model     = model,
        profiles  = profiles,
        soc_tasks = soc_tasks,
        svd       = svd
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
    svd = TruncatedSVD(
        n_components = config.component_count,
        random_state = config.random_seed
    )
    return {
        "coordinates" : svd.fit_transform(normalize(raw_vectors)),
        "svd"         : svd
    }


def soc_similarity(
    cluster_vectors : np.ndarray,
    soc_vectors     : np.ndarray
) -> np.ndarray:
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
    return model.encode([o.embedding_text for o in lexicons.occupations])
