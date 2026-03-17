"""
Pipeline evaluation diagnostics.

PCA component loadings, Louvain-HAC alignment analysis, feature
density metrics, and corpus statistics. Not part of the production
pipeline.
"""

import networkx as nx
import numpy    as np

from collections     import Counter
from logging         import getLogger
from sklearn.metrics import adjusted_rand_score

from chalkline.extraction.vectorize import SkillVectorizer
from chalkline.pathways.graph       import CareerPathwayGraph
from chalkline.reduction.pca        import PcaReducer


logger = getLogger(__name__)


def alignment(graph: CareerPathwayGraph) -> dict:
    """
    ARI between Louvain communities and HAC cluster partitions
    projected onto the shared skill space. Returns empty diagnostics
    when the network is unavailable (after deserialization).
    """
    if graph.network is None:
        return {"ari": 0.0, "modularity": None}

    louvain = graph.network.partition_map
    hac     = {
        s: p.cluster_id
        for p in graph.profiles.values()
        for s in p.skills
    }
    shared = louvain.keys() & hac.keys()

    if not shared:
        logger.warning("No shared skills for ARI computation")
        return {"ari": 0.0, "modularity": None}

    if len(graph.profiles) > len(shared) / 2:
        logger.warning(
            f"ARI computed with {len(graph.profiles)} clusters "
            f"against {len(shared)} shared skills; "
            f"near-singleton clustering makes ARI "
            f"structurally near-zero"
        )

    labels = [(louvain[s], hac[s]) for s in shared]
    result = {"ari": adjusted_rand_score(*zip(*labels)), "modularity": None}

    if (skill_graph := graph.network.graph()).size():
        result["modularity"] = nx.community.modularity(
            skill_graph,
            communities = graph.network.partition,
            weight      = "weight"
        )
    return result


def corpus_statistics(vectorizer: SkillVectorizer) -> dict:
    """
    Vocabulary size, matrix sparsity, mean skills per posting,
    and per-skill frequency counts.
    """
    binary     = vectorizer.binary_matrix
    rows, cols = binary.shape
    frequency  = Counter(skill for d in vectorizer.dicts for skill in d)

    return {
        "matrix_sparsity"         : 1 - binary.nnz / (rows * cols),
        "mean_skills_per_posting" : sum(map(len, vectorizer.dicts)) / len(vectorizer.dicts),
        "skill_frequency"         : dict(sorted(frequency.items())),
        "vocabulary_size"         : cols
    }


def loadings(reducer: PcaReducer, top_n: int = 10) -> list[dict]:
    """
    Top loading terms for each selected PCA component, revealing
    which skills drive each dimension.
    """
    names = np.array(reducer.feature_names)
    return [
        {
            "index"          : i,
            "terms"          : names[indices].tolist(),
            "variance_ratio" : reducer.explained_variance_ratio[i],
            "weights"        : row[indices].tolist()
        }
        for i, row in enumerate(
            reducer.pipeline.named_steps["svd"].components_
        )
        for indices in [np.argsort(np.abs(row))[::-1][:top_n]]
    ]


def log_density(vectorizer: SkillVectorizer):
    """
    Log vocabulary size, mean skills per posting, and the fraction
    of posting pairs with zero skill overlap. Warns above 50%.
    """
    stats = corpus_statistics(vectorizer)
    logger.info(f"Vocabulary: {stats['vocabulary_size']} features")
    logger.info(
        f"Mean skills/posting: "
        f"{stats['mean_skills_per_posting']:.1f}"
    )

    B            = vectorizer.binary_matrix
    intersection = B @ B.T
    n            = intersection.shape[0]
    total_pairs  = n * (n - 1) // 2
    nonzero_pairs = (intersection.nnz - n) // 2
    zero_frac     = (
        1 - nonzero_pairs / total_pairs if total_pairs else 0.0
    )

    logger.info(f"Zero-overlap fraction: {zero_frac:.1%}")

    if zero_frac > 0.5:
        logger.warning(
            "Zero-overlap fraction exceeds 50%; consider "
            "sentence-embedding alternative for the geometry "
            "track"
        )
