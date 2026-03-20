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
from json                        import loads
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
from chalkline.pipeline.schemas   import PipelineConfig, PipelineManifest
from chalkline.pipeline.schemas   import ProgramRecommendation
from chalkline.pipeline.trades    import TradeIndex


def assignments(
    config      : PipelineConfig,
    coordinates : np.ndarray
) -> np.ndarray:
    """
    Fit Ward-linkage HAC on SVD coordinates.
    """
    return AgglomerativeClustering(
        linkage    = "ward",
        n_clusters = config.cluster_count
    ).fit_predict(coordinates)


def centroids(
    assignments : np.ndarray,
    coordinates : np.ndarray
) -> np.ndarray:
    """
    Mean SVD coordinates per cluster for resume distance computation.
    """
    cluster_ids = sorted(set(assignments))
    return np.stack([
        coordinates[assignments == cluster_id].mean(axis=0)
        for cluster_id in cluster_ids
    ])


def cluster_vectors(
    assignments : np.ndarray,
    raw_vectors : np.ndarray
) -> np.ndarray:
    """
    Mean posting embedding per cluster in the full embedding space,
    L2-normalized for cosine similarity against occupations and credentials.
    """
    cluster_ids = sorted(set(assignments))
    return np.stack([
        normalize(
            raw_vectors[assignments == cluster_id].mean(axis=0, keepdims=True)
        )[0]
        for cluster_id in cluster_ids
    ])


def corpus(config: PipelineConfig) -> dict[str, str]:
    """
    Load posting texts from the corpus directory.

    Raises:
        FileNotFoundError: When no JSON postings exist in the configured directory.
    """
    postings = CorpusStorage(config.postings_dir).load()
    if not (result := {p.id: p.description for p in postings if p.id is not None}):
        raise FileNotFoundError(f"No postings found in {config.postings_dir}")
    return result


def corpus_titles(config: PipelineConfig) -> dict[str, str]:
    """
    Load posting titles for cluster labeling.
    """
    return {
        p.id: p.title
        for p in CorpusStorage(config.postings_dir).load()
        if p.id is not None
    }


@extract_fields({
    "credential_labels"  : list,
    "credential_types"   : list,
    "credential_vectors" : np.ndarray
})
def credentials(
    config   : PipelineConfig,
    lexicons : LexiconLoader,
    model    : SentenceTransformer
) -> dict:
    """
    Encode the credential catalog (apprenticeships, programs,
    certifications) with the sentence transformer and return parallel lists
    of embeddings, labels, and type strings.
    """
    labels      = []
    texts       = []
    types       = []
    lexicon_dir = config.lexicon_dir

    apprenticeships = loads((lexicon_dir / "apprenticeships.json").read_text())
    for record in apprenticeships:
        labels.append(record["title"])
        texts.append(record["title"])
        types.append("apprenticeship")

    programs = loads((lexicon_dir / "programs.json").read_text())
    for record in programs:
        labels.append(f"{record['program']} ({record['institution']})")
        texts.append(
            f"{record['credential']} {record['program']} "
            f"{record['institution']}"
        )
        types.append("program")

    certifications = loads((lexicon_dir / "certifications.json").read_text())
    for record in certifications:
        acronym = record.get("acronym") or ""
        labels.append(f"{acronym} {record['name']}".strip())
        texts.append(
            f"{acronym} {record['name']} "
            f"{record['organization']}".strip()
        )
        types.append("certification")

    logger.info(f"Encoding {len(texts)} credentials...")
    encoded = normalize(model.encode(texts, show_progress_bar=False))
    return {
        "credential_labels"  : labels,
        "credential_types"   : types,
        "credential_vectors" : encoded
    }


@extract_fields({
    "coordinates" : np.ndarray,
    "svd"         : TruncatedSVD
})
def reduction(
    config       : PipelineConfig,
    unit_vectors : np.ndarray
) -> dict:
    """
    Fit TruncatedSVD on L2-normalized embeddings and expose both the reduced
    coordinates and the fitted SVD for resume projection.
    """
    svd = TruncatedSVD(
        n_components = config.component_count,
        random_state = config.random_seed
    )
    return {
        "coordinates" : svd.fit_transform(unit_vectors),
        "svd"         : svd
    }


def graph(
    centroids          : np.ndarray,
    cluster_vectors    : np.ndarray,
    config             : PipelineConfig,
    credential_labels  : list[str],
    credential_types   : list[str],
    credential_vectors : np.ndarray,
    job_zone_map       : dict[int, int],
    profiles           : dict[int, ClusterProfile]
) -> CareerPathwayGraph:
    """
    Build the career pathway graph with stepwise k-NN backbone and per-edge
    credential enrichment.
    """
    return CareerPathwayGraph(
        cluster_centroids  = centroids,
        cluster_vectors    = cluster_vectors,
        config             = config,
        credential_labels  = credential_labels,
        credential_types   = credential_types,
        credential_vectors = credential_vectors,
        job_zone_map       = job_zone_map,
        profiles           = profiles
    )


def job_zone_map(
    cluster_vectors : np.ndarray,
    config          : PipelineConfig,
    lexicons        : LexiconLoader,
    soc_vectors     : np.ndarray
) -> dict[int, int]:
    """
    Assign Job Zones to clusters via top-k median cosine against O*NET
    Task+DWA embeddings.
    """
    cosine_similarities = cosine_similarity(cluster_vectors, soc_vectors)
    soc_profiles        = lexicons.occupations
    top_k               = config.soc_neighbors
    return {
        c: int(np.median([
            soc_profiles[i].job_zone
            for i in np.argsort(cosine_similarities[c])[-top_k:]
        ]))
        for c in range(len(cluster_vectors))
    }


def lexicons(config: PipelineConfig) -> LexiconLoader:
    """
    Load lexicon files from the configured directory.
    """
    return LexiconLoader(config.lexicon_dir)


def manifest(
    config : PipelineConfig,
    corpus : dict[str, str]
) -> PipelineManifest:
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
        max_gaps     = config.max_gaps,
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


def soc_vectors(
    lexicons : LexiconLoader,
    model    : SentenceTransformer
) -> np.ndarray:
    """
    Encode O*NET occupations using uncapped Task+DWA text, L2-normalized for
    cosine similarity against cluster centroids.
    """
    soc_texts = [
        f"{occ.title}: {', '.join(
            s.name for s in occ.skills
            if s.type.value in ('task', 'dwa')
        )}" for occ in lexicons.occupations
    ]
    logger.info(f"Encoding {len(soc_texts)} occupations...")
    return normalize(model.encode(soc_texts, show_progress_bar=False))


@extract_fields({
    "task_labels"  : dict,
    "task_vectors" : dict
})
def occ_tasks(
    cluster_vectors : np.ndarray,
    job_zone_map    : dict[int, int],
    lexicons        : LexiconLoader,
    model           : SentenceTransformer,
    soc_vectors     : np.ndarray
) -> dict:
    """
    Encode per-cluster O*NET Task+DWA embeddings for resume gap analysis.

    For each cluster, identifies the nearest occupation via cosine
    similarity, then encodes that occupation's Task+DWA elements as
    individual embeddings for per-task gap scoring.
    """
    cosine_similarities = cosine_similarity(cluster_vectors, soc_vectors)
    soc_profiles        = lexicons.occupations
    task_labels         = {}
    task_vectors        = {}

    for cluster_id in range(len(cluster_vectors)):
        nearest_soc = soc_profiles[np.argmax(cosine_similarities[cluster_id])]
        tasks = [
            s for s in nearest_soc.skills
            if s.type.value in ("task", "dwa")
        ]
        if tasks:
            names = [t.name for t in tasks]
            task_labels[cluster_id]  = names
            task_vectors[cluster_id] = normalize(
                model.encode(names, show_progress_bar=False)
            )

    return {
        "task_labels"  : task_labels,
        "task_vectors" : task_vectors
    }


def profiles(
    assignments     : np.ndarray,
    cluster_vectors : np.ndarray,
    corpus_titles   : dict[str, str],
    job_zone_map    : dict[int, int],
    lexicons        : LexiconLoader,
    soc_vectors     : np.ndarray
) -> dict[int, ClusterProfile]:
    """
    Build cluster profiles with Job Zone, sector, occupation title, and
    modal posting title for each cluster.
    """
    doc_ids               = sorted(corpus_titles)
    cosine_similarities   = cosine_similarity(cluster_vectors, soc_vectors)
    soc_profiles          = lexicons.occupations
    cluster_ids           = sorted(set(assignments))

    result = {}
    for cluster_id in cluster_ids:
        nearest_soc    = soc_profiles[np.argmax(cosine_similarities[cluster_id])]
        member_indices = np.where(assignments == cluster_id)[0]
        modal_title = Counter(
            corpus_titles[doc_ids[i]] for i in member_indices
        ).most_common(1)[0][0]

        result[cluster_id] = ClusterProfile(
            cluster_id  = cluster_id,
            job_zone    = job_zone_map[cluster_id],
            modal_title = modal_title,
            sector      = nearest_soc.sector,
            size        = len(member_indices),
            soc_title   = nearest_soc.title
        )

    return result


def raw_vectors(
    corpus : dict[str, str],
    model  : SentenceTransformer
) -> np.ndarray:
    """
    Encode all posting descriptions with the sentence transformer.
    """
    doc_ids = sorted(corpus)
    texts   = [corpus[d] for d in doc_ids]
    logger.info(f"Encoding {len(texts)} postings...")
    return model.encode(texts, show_progress_bar=False)


def trades(config: PipelineConfig) -> TradeIndex:
    """
    Load apprenticeship and program reference data into a `TradeIndex`.
    """
    lexicon_dir = config.lexicon_dir
    return TradeIndex(
        apprenticeships = TypeAdapter(
            list[ApprenticeshipContext]
        ).validate_json((lexicon_dir / "apprenticeships.json").read_bytes()),
        programs = TypeAdapter(
            list[ProgramRecommendation]
        ).validate_json((lexicon_dir / "programs.json").read_bytes())
    )
