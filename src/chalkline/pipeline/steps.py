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
from hamilton.function_modifiers import extract_fields, tag
from json                        import loads
from loguru                      import logger
from sklearn.cluster             import AgglomerativeClustering
from sklearn.decomposition       import TruncatedSVD
from sklearn.metrics.pairwise    import cosine_similarity
from sklearn.preprocessing       import normalize

from chalkline.collection.schemas import Corpus
from chalkline.collection.storage import CorpusStorage
from chalkline.matching.matcher   import ResumeMatcher
from chalkline.pathways.clusters  import Cluster, Clusters, Task
from chalkline.pathways.graph     import CareerPathwayGraph
from chalkline.pathways.loaders   import LexiconLoader
from chalkline.pathways.schemas   import Credential
from chalkline.pipeline.encoder   import SentenceEncoder
from chalkline.pipeline.schemas   import PipelineConfig


def assignments(config: PipelineConfig, coordinates: np.ndarray) -> np.ndarray:
    """
    Fit Ward-linkage HAC on SVD coordinates and return label array.
    """
    return AgglomerativeClustering(
        linkage    = "ward",
        n_clusters = config.cluster_count
    ).fit_predict(coordinates)


def centroids(assignments: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    """
    Mean SVD coordinates per cluster for resume distance computation.
    """
    return np.stack([
        coordinates[assignments == cid].mean(axis=0)
        for cid in sorted(np.unique(assignments))
    ])


def cluster_vectors(assignments: np.ndarray, raw_vectors: np.ndarray) -> np.ndarray:
    """
    Mean posting embedding per cluster in the full embedding space,
    L2-normalized for cosine similarity against occupations and credentials.
    """
    return np.asarray(normalize(np.stack([
        raw_vectors[assignments == cid].mean(axis=0)
        for cid in sorted(np.unique(assignments))
    ])))


def clusters(
    assignments     : np.ndarray,
    centroids       : np.ndarray,
    cluster_vectors : np.ndarray,
    corpus          : Corpus,
    job_zone_map    : dict[int, int],
    lexicons        : LexiconLoader,
    soc_similarity  : np.ndarray,
    soc_tasks       : dict[int, list[Task]]
) -> Clusters:
    """
    Build unified cluster objects from assignments, corpus, and O*NET SOC
    matching. Returns a `Clusters` container with pre-stacked centroid and
    embedding vector matrices.
    """
    cluster_ids = sorted(np.unique(assignments).tolist())
    members     = {
        cid: np.where(assignments == cid)[0]
        for cid in cluster_ids
    }

    items = {
        cid: Cluster(
            cluster_id   = cid,
            job_zone     = job_zone_map[cid],
            members      = members[cid],
            modal_title  = Counter(
                corpus.postings[corpus.posting_ids[i]].title
                for i in members[cid]
            ).most_common(1)[0][0],
            postings     = corpus.at(members[cid]),
            sector       = nearest.sector,
            size         = len(members[cid]),
            soc_title    = nearest.title,
            tasks        = soc_tasks.get(cid, [])
        )
        for cid     in cluster_ids
        for nearest in [lexicons.nearest_occupation(soc_similarity[cid])]
    }

    return Clusters(
        centroids      = centroids,
        items          = items,
        soc_similarity = soc_similarity,
        vectors        = cluster_vectors
    )


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


@tag(batch_label="credentials")
def credentials(config: PipelineConfig, encoder: SentenceEncoder) -> list[Credential]:
    """
    Load the curated credential catalog, encode with the sentence
    transformer, and attach vectors to each credential instance.

    Extra fields beyond the core `Credential` schema are packed into the
    `metadata` dict automatically.
    """
    known_fields = {"embedding_text", "kind", "label"}
    raw          = loads((config.lexicon_dir / "credentials.json").read_bytes())
    records      = [
        Credential(
            embedding_text = entry["embedding_text"],
            kind           = entry["kind"],
            label          = entry["label"],
            metadata       = {k: v for k, v in entry.items() if k not in known_fields}
        )
        for entry in raw
    ]

    logger.info(f"Encoding {len(records)} credentials...")
    for credential, vector in zip(
        records,
        encoder.encode([r.embedding_text for r in records])
    ):
        credential.vector = vector.tolist()

    return records


def graph(
    clusters    : Clusters,
    config      : PipelineConfig,
    credentials : list[Credential]
) -> CareerPathwayGraph:
    """
    Build the career pathway graph with stepwise k-NN backbone and per-edge
    credential enrichment.
    """
    result = CareerPathwayGraph(
        clusters               = clusters,
        credentials            = credentials,
        destination_percentile = config.destination_percentile,
        lateral_neighbors      = config.lateral_neighbors,
        source_percentile      = config.source_percentile,
        upward_neighbors       = config.upward_neighbors
    )
    logger.info(
        f"Career graph: {result.graph.number_of_nodes()} nodes, "
        f"{result.edge_count} edges"
    )
    return result


def job_zone_map(
    assignments    : np.ndarray,
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
        for cid in sorted(np.unique(assignments))
    }


def matcher(
    clusters : Clusters,
    encoder  : SentenceEncoder,
    graph    : CareerPathwayGraph,
    svd      : TruncatedSVD
) -> ResumeMatcher:
    """
    Build the resume matcher with all artifacts needed for single-resume
    inference.
    """
    return ResumeMatcher(
        clusters = clusters,
        encoder  = encoder,
        graph    = graph,
        svd      = svd
    )


@tag(batch_label="postings")
def raw_vectors(corpus: Corpus, encoder: SentenceEncoder) -> np.ndarray:
    """
    Encode all posting descriptions with the sentence transformer.
    """
    logger.info(f"Encoding {len(corpus.posting_ids)} postings...")
    return encoder.encode(corpus.descriptions, unit=False)


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


@tag(batch_label="tasks")
def soc_tasks(
    assignments    : np.ndarray,
    encoder        : SentenceEncoder,
    lexicons       : LexiconLoader,
    soc_similarity : np.ndarray
) -> dict[int, list[Task]]:
    """
    Encode per-cluster O*NET Task+DWA embeddings for resume gap analysis.

    For each cluster, identifies the nearest occupation via cosine
    similarity, then encodes that occupation's Task+DWA elements as
    individual embeddings for per-task gap scoring.
    """
    return {
        cid: [
            Task(
                name       = t.name,
                skill_type = t.type.value,
                vector     = vec
            )
            for t, vec in zip(
                elems,
                encoder.encode([t.name for t in elems])
            )
        ]
        for cid in sorted(np.unique(assignments))
        for nearest in [lexicons.nearest_occupation(soc_similarity[cid])]
        for elems in [nearest.task_elements]
        if elems
    }


@tag(batch_label="occupations")
def soc_vectors(encoder: SentenceEncoder, lexicons: LexiconLoader) -> np.ndarray:
    """
    Encode O*NET occupations using uncapped Task+DWA text, L2-normalized for
    cosine similarity against cluster centroids.
    """
    logger.info(f"Encoding {len(lexicons.occupations)} occupations...")
    return encoder.encode([o.embedding_text for o in lexicons.occupations])
