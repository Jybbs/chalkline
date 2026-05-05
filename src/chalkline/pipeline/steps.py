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
from pathlib                     import Path
from sklearn.cluster             import AgglomerativeClustering
from sklearn.decomposition       import TruncatedSVD
from sklearn.preprocessing       import normalize

from chalkline.collection.schemas import Corpus
from chalkline.collection.storage import CorpusStorage
from chalkline.matching.matcher   import ResumeMatcher
from chalkline.pathways.clusters  import Cluster, Clusters, Task
from chalkline.pathways.graph     import CareerPathwayGraph
from chalkline.pathways.loaders   import LaborLoader, LexiconLoader
from chalkline.pathways.schemas   import Credential, EncodedOccupation, Occupation
from chalkline.pathways.schemas   import SkillType
from chalkline.pathways.selection import SOCScorer
from chalkline.pipeline.encoder   import SentenceEncoder


def assignments(
    cluster_count   : int,
    component_count : int,
    consensus_seeds : int,
    raw_vectors     : np.ndarray
) -> np.ndarray:
    """
    Consensus cluster labels via evidence accumulation across
    `consensus_seeds` independent SVD-then-Ward chains, finalized by
    average-linkage HAC on the resulting pairwise co-association matrix.
    Stabilizes membership for postings near fuzzy cluster boundaries that
    single-seed runs assign inconsistently.
    """
    normed = normalize(raw_vectors)
    co     = np.zeros((raw_vectors.shape[0],) * 2, dtype=np.float32)
    logger.info(
        f"Consensus clustering: {consensus_seeds} seeds at k={cluster_count}"
    )
    for seed in range(consensus_seeds):
        coordinates = TruncatedSVD(
            n_components = component_count,
            random_state = seed
        ).fit_transform(normed)
        labels = AgglomerativeClustering(
            linkage    = "ward",
            n_clusters = cluster_count
        ).fit_predict(coordinates)
        co += np.equal.outer(labels, labels)
    return AgglomerativeClustering(
        linkage    = "average",
        metric     = "precomputed",
        n_clusters = cluster_count
    ).fit_predict(1.0 - co / consensus_seeds)


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
    assignments         : np.ndarray,
    centroids           : np.ndarray,
    cluster_vectors     : np.ndarray,
    corpus              : Corpus,
    credentials         : list[Credential],
    labor               : LaborLoader,
    lexicons            : LexiconLoader,
    nearest_occupations : dict[int, Occupation],
    raw_vectors         : np.ndarray,
    soc_similarity      : np.ndarray,
    soc_softmax_tau     : float,
    soc_tasks           : dict[int, list[Task]],
    soc_wage_round      : int,
    soc_wage_topk       : int,
    wage_tier_count     : int
) -> Clusters:
    """
    Build unified cluster objects from assignments, corpus, and O*NET SOC
    matching. Returns a `Clusters` container with pre-stacked centroid and
    embedding vector matrices, and per-cluster posting embeddings sliced out
    of `raw_vectors` so display-layer projections never have to re-encode
    the corpus. `Clusters.__post_init__` fans the softmax-derived
    `soc_weights`, `wage`, K-means-derived `wage_tier`, and zipf-windowed
    TF-IDF `display_title` onto each child cluster using the occupation
    titles, wage vector, and the softmax/wage/tier thresholds threaded
    through from this step.
    """
    items: dict[int, Cluster] = {}
    for cid in sorted(np.unique(assignments).tolist()):
        member_idx = np.where(assignments == cid)[0]
        postings   = corpus.at(member_idx)
        nearest    = nearest_occupations[cid]
        items[cid] = Cluster(
            cluster_id  = cid,
            embeddings  = raw_vectors[member_idx],
            modal_title = Counter(p.title for p in postings).most_common(1)[0][0],
            postings    = postings,
            sector      = nearest.sector,
            size        = len(postings),
            soc_title   = nearest.title,
            tasks       = soc_tasks.get(cid, [])
        )

    return Clusters(
        centroids         = centroids,
        credentials       = credentials,
        items             = items,
        labor             = labor,
        occupation_titles = [o.title for o in lexicons.occupations],
        soc_similarity    = soc_similarity,
        softmax_tau       = soc_softmax_tau,
        vectors           = cluster_vectors,
        wage_round        = soc_wage_round,
        wage_tier_count   = wage_tier_count,
        wage_topk         = soc_wage_topk
    )


def corpus(corpus_mtime: float, postings_dir: str) -> Corpus:
    """
    Load the posting corpus from the configured directory.

    `corpus_mtime` is threaded through so Hamilton's content-addressed
    cache invalidates this node when `corpus.json` changes on disk, since
    the path alone doesn't reflect file writes. `postings_dir` arrives as
    a string so Hamilton's input hasher can fingerprint it, then converts
    to `Path` here for filesystem operations.

    Raises:
        FileNotFoundError: When no valid postings exist.
    """
    path = Path(postings_dir)
    if not (postings := CorpusStorage(path).load()):
        raise FileNotFoundError(f"No postings found in {path}")
    return Corpus({p.id: p for p in postings})


@tag(batch_label="credentials")
def credentials(encoder: SentenceEncoder, lexicon_dir: str) -> list[Credential]:
    """
    Load the curated credential catalog, encode with the sentence
    transformer, and attach vectors to each credential instance.

    The curated `credentials.json` stores kind-specific extras inside a
    `metadata` object per record; the loader threads that through directly
    so apprenticeship RAPIDS codes, program URLs, and other type-specific
    fields reach the display layer intact. `lexicon_dir` arrives as a
    string for Hamilton input hashing and is wrapped in `Path` here.
    """
    raw     = loads((Path(lexicon_dir) / "credentials.json").read_bytes())
    records = [
        Credential(
            embedding_text = entry["embedding_text"],
            kind           = entry["kind"],
            label          = entry["label"],
            metadata       = entry.get("metadata", {})
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
    clusters               : Clusters,
    credentials            : list[Credential],
    destination_percentile : int,
    lateral_neighbors      : int,
    upward_neighbors       : int
) -> CareerPathwayGraph:
    """
    Build the career pathway graph with stepwise k-NN backbone and on-demand
    destination-affinity credential filtering.
    """
    result = CareerPathwayGraph(
        clusters               = clusters,
        credentials            = credentials,
        destination_percentile = destination_percentile,
        lateral_neighbors      = lateral_neighbors,
        upward_neighbors       = upward_neighbors
    )
    logger.info(
        f"Career graph: {result.graph.number_of_nodes()} nodes, "
        f"{result.edge_count} edges"
    )
    return result


def matcher(
    clusters : Clusters,
    encoder  : SentenceEncoder,
    svd      : TruncatedSVD
) -> ResumeMatcher:
    """
    Build the resume matcher with all artifacts needed for single-resume
    inference. Reach composition lives on the orchestrator because the
    matcher itself has no reason to know about the career graph.
    """
    return ResumeMatcher(
        clusters = clusters,
        encoder  = encoder,
        svd      = svd
    )


def nearest_occupations(
    assignments    : np.ndarray,
    lexicons       : LexiconLoader,
    soc_similarity : np.ndarray
) -> dict[int, Occupation]:
    """
    Top-similarity O*NET occupation per cluster, computed once and reused by
    `clusters` and `soc_tasks` so the argmax over the similarity matrix runs
    only once per cluster.
    """
    return {
        cid: lexicons.nearest_occupation(soc_similarity[cid])
        for cid in sorted(np.unique(assignments).tolist())
    }


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
def reduction(
    component_count : int,
    random_seed     : int,
    raw_vectors     : np.ndarray
) -> dict:
    """
    L2-normalize raw embeddings, fit TruncatedSVD, and expose both the
    reduced coordinates and the fitted SVD for resume projection.
    """
    svd = TruncatedSVD(
        n_components = component_count,
        random_state = random_seed
    )
    return {
        "coordinates" : svd.fit_transform(normalize(raw_vectors)),
        "svd"         : svd
    }


@tag(batch_label="occupations")
def encoded_occupations(
    encoder  : SentenceEncoder,
    lexicons : LexiconLoader
) -> list[EncodedOccupation]:
    """
    Encode every occupation's Task elements in a single batched pass,
    pairing each `Occupation` schema with its L2-normalized task matrix.

    Late-interaction scoring in `soc_similarity` requires each task to live
    as its own vector rather than being pooled into a single occupation
    embedding. DWA elements are excluded because O*NET's DWA taxonomy is
    shared across occupations and the generic coordination activities
    attached to construction-trade SOCs pool into a centroid that outranks
    specialty SOCs whose tasks carry the discriminative signal.

    Batching every occupation's tasks into one `encoder.encode` call
    amortizes tokenizer warm-up and ONNX session overhead across the whole
    lexicon rather than paying it 21 times.
    """
    task_lists = [
        [s.name for s in o.skills if s.type == SkillType.TASK]
        for o in lexicons.occupations
    ]
    flat_names = [n for names in task_lists for n in names]
    logger.info(
        f"Encoding {len(flat_names)} tasks across "
        f"{len(lexicons.occupations)} occupations..."
    )
    flat_vectors = encoder.encode(flat_names)
    offsets      = np.cumsum([0] + [len(names) for names in task_lists])
    return [
        EncodedOccupation(
            occupation = occupation,
            tasks      = flat_vectors[offsets[i]:offsets[i + 1]]
        )
        for i, occupation in enumerate(lexicons.occupations)
    ]


def soc_similarity(
    assignments          : np.ndarray,
    encoded_occupations  : list[EncodedOccupation],
    raw_vectors          : np.ndarray
) -> np.ndarray:
    """
    ColBERTv2-style late-interaction similarity between cluster postings and
    O*NET occupations, delegated to `SOCScorer`.

    The concat-then-pool formulation that preceded this was dominated by
    occupations whose DWA descriptions happened to pool to a generic
    construction-trade centroid and could not recover SOCs whose specific
    task language was diluted by shared administrative DWAs. Late
    interaction lets each posting find its own best-matching task rather
    than averaging over a pooled vector, so task specificity survives.
    """
    return SOCScorer(occupations=encoded_occupations).score(
        assignments = assignments,
        raw_vectors = raw_vectors
    )


@tag(batch_label="tasks")
def soc_tasks(
    encoder             : SentenceEncoder,
    nearest_occupations : dict[int, Occupation]
) -> dict[int, list[Task]]:
    """
    Encode per-cluster O*NET Task+DWA embeddings for resume gap analysis in
    a single batched pass.

    For each cluster, takes the pre-computed nearest occupation and stages
    that occupation's Task+DWA elements into a flat batch, encodes the whole
    corpus at once, then splits the result back along cluster boundaries.
    Batching avoids re-paying tokenizer and ONNX session overhead per
    cluster; the per-task vectors land on `Cluster.tasks` for BM25-weighted
    gap scoring by the resume matcher.
    """
    ordered_cids = [
        cid for cid, nearest in nearest_occupations.items()
        if nearest.task_elements
    ]
    per_cluster_elements = [
        nearest_occupations[cid].task_elements for cid in ordered_cids
    ]
    flat_names = [
        element.name for elements in per_cluster_elements
        for element in elements
    ]
    flat_vectors = encoder.encode(flat_names)
    offsets      = np.cumsum([0] + [len(elems) for elems in per_cluster_elements])
    return {
        cid: [
            Task(name=element.name, vector=vector)
            for element, vector in zip(
                per_cluster_elements[i],
                flat_vectors[offsets[i]:offsets[i + 1]]
            )
        ]
        for i, cid in enumerate(ordered_cids)
    }
