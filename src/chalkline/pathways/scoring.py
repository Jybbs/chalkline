"""
ColBERTv2-style late-interaction scoring for cluster-to-occupation matching.

Computes per-cluster matching strength against each O*NET occupation via
mean-of-max over per-posting and per-task embeddings. Late interaction
captures task-specific relevance within noisy posting text and prevents
shared administrative DWAs from pooling into an undiscriminative occupation
centroid that outranks specialty SOCs whose tasks carry the real signal.

`SOCScorer` consumes a list of `EncodedOccupation` pairs (declared in
`schemas.py`) and produces the cluster-to-occupation similarity matrix.
"""

import numpy as np

from dataclasses           import dataclass, field
from sklearn.preprocessing import normalize

from chalkline.pathways.schemas import EncodedOccupation


@dataclass(kw_only=True)
class SOCScorer:
    """
    ColBERTv2-style MaxSim mean-of-max scorer for clusters against O*NET
    occupations, referencing Santhanam et al. 2022's late-interaction
    formulation translated to the occupation-matching setting.

    Stacks the per-occupation task matrices into one flat matrix at
    construction so `score()` resolves every cluster-occupation pair with a
    single BLAS-backed matmul and a contiguous `np.maximum.reduceat`, rather
    than a Python double loop of per-pair matmuls. Per-occupation boundaries
    are preserved via `owner_starts`, a row-aligned index into the flat
    matrix suitable for reduceat segmentation.

    Every O*NET occupation in the curated reference set lists at least one
    task element, so the scorer requires non-empty task matrices and does
    not handle degenerate zero-task occupations.
    """

    occupations: list[EncodedOccupation]

    flat_tasks   : np.ndarray = field(init=False)
    owner_starts : np.ndarray = field(init=False)

    def __post_init__(self):
        """
        Flatten per-occupation task matrices into one contiguous array plus
        a starting-index array for `np.maximum.reduceat` segmentation.
        """
        if not all(e.tasks.size for e in self.occupations):
            raise ValueError(
                "SOCScorer requires every occupation to list at least one "
                "task; empty task matrix found."
            )
        sizes             = np.array([e.tasks.shape[0] for e in self.occupations])
        self.flat_tasks   = np.vstack([e.tasks for e in self.occupations])
        self.owner_starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])

    def score(
        self,
        assignments : np.ndarray,
        raw_vectors : np.ndarray
    ) -> np.ndarray:
        """
        Compute mean-of-max similarity for every (cluster, occupation) pair.

        Normalizes posting vectors once, runs a single matmul against the
        flat task matrix, reduces columnwise per occupation via
        `np.maximum.reduceat` to obtain each posting's best-matching task
        score per occupation, then averages within each cluster.

        Args:
            assignments : Cluster label per posting in `raw_vectors` order.
            raw_vectors : Unnormalized posting embeddings from the encoder, shape
                          `(n_postings, embedding_dim)`.

        Returns:
            Similarity matrix of shape `(n_clusters, n_occupations)` with cluster rows
            in sorted assignment-id order and occupation columns in `self.occupations`
            order.
        """
        postings       = normalize(raw_vectors)
        per_task       = postings @ self.flat_tasks.T
        per_occupation = np.maximum.reduceat(per_task, self.owner_starts, axis=1)
        cluster_ids    = sorted(np.unique(assignments).tolist())

        return np.stack([
            per_occupation[assignments == cid].mean(axis=0)
            for cid in cluster_ids
        ]).astype(np.float32)
