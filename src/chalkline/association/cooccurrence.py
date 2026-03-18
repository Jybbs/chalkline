"""
PMI co-occurrence network with Louvain community detection.

Computes pairwise co-occurrence counts via sparse matrix multiplication,
derives NPMI as the primary association measure and constructs a
NetworkX graph with Louvain community detection for career track
grouping. The NPMI graph feeds edge weights in the pathway graph,
and the PPMI DataFrame feeds gap ranking in resume matching.
"""

import networkx as nx
import numpy    as np
import pandas   as pd

from functools                import cached_property
from kneed                    import KneeLocator
from loguru                   import logger
from math                     import ceil
from scipy.sparse             import csc_array, csr_array, spmatrix



class CooccurrenceNetwork:
    """
    PMI-weighted skill co-occurrence network with Louvain detection.

    Receives the binary skill matrix from `SkillVectorizer`, computes
    the thresholded co-occurrence matrix eagerly, and exposes NPMI,
    graph construction, and Louvain community detection as cached
    properties and on-demand methods.

    When `min_cooccurrence` is `"auto"`, the threshold is selected
    by finding the knee of the modularity-vs-threshold curve via
    `kneed`, identifying the point where community quality begins to
    degrade. The floor of 3 protects PMI stability at small corpus
    sizes where a single observation can shift a pair's score by 25%.
    """

    def __init__(
        self,
        binary_matrix    : spmatrix,
        feature_names    : list[str],
        min_cooccurrence : float | str = "auto",
        random_seed      : int         = 42
    ):
        """
        Compute the thresholded co-occurrence matrix from the
        binary presence/absence matrix produced by
        `SkillVectorizer.binary_matrix`.

        When `min_cooccurrence` is `"auto"`, the threshold is
        selected by finding the knee of the
        modularity-vs-threshold curve. Otherwise the float
        fraction is applied to the corpus size with a floor of
        3 to protect PMI stability.

        Args:
            binary_matrix    : CSR binary presence/absence matrix.
            feature_names    : Vocabulary in column order.
            min_cooccurrence : `"auto"` or float corpus fraction.
            random_seed      : Seed for Louvain reproducibility.
        """
        self.feature_names = feature_names
        self.n_docs        = binary_matrix.shape[0]
        self.random_seed   = random_seed

        self.doc_freq, self.cooccurrence, self.threshold = (
            self._build_cooccurrence(binary_matrix, min_cooccurrence)
        )

    @cached_property
    def npmi_matrix(self) -> csr_array:
        """
        Normalized pointwise mutual information with positive
        clipping.

            NPMI(x, y) = log(n * C_xy / (df_x * df_y))
                         / log(n / C_xy)

        Bounded to [0, 1] by clipping. Pairs where both skills
        appear in every posting use a floored denominator rather
        than dividing by zero.

        Returns:
            Sparse `csr_array` of NPMI+ values in [0, 1].
        """
        npmi = self.cooccurrence.astype(float)
        npmi.data = np.clip(
            self._pmi_values() / np.maximum(
                np.log(self.n_docs / npmi.data), 1e-10
            ), 0, 1.0
        )
        npmi.eliminate_zeros()
        return npmi

    def _build_cooccurrence(
        self,
        binary_matrix    : spmatrix,
        min_cooccurrence : float | str
    ) -> tuple[np.ndarray, csr_array, int]:
        """
        Compute document frequencies and the thresholded
        co-occurrence matrix from the binary presence/absence
        matrix.

        Converts to CSC for efficient column access, derives
        document frequencies from column pointers, multiplies
        B^T @ B to get pairwise co-occurrence counts, zeros the
        diagonal, and applies a threshold selected via modularity
        knee detection (auto) or a corpus fraction with a floor
        of 3.

        Args:
            binary_matrix    : Binary presence/absence matrix.
            min_cooccurrence : `"auto"` or float corpus fraction.

        Returns:
            Tuple of (document frequency array, thresholded CSR
            co-occurrence matrix, threshold value).
        """
        B = csc_array(binary_matrix)
        C = (B.T @ B).tocsr()
        C.setdiag(0)
        C.eliminate_zeros()

        threshold = (
            self._find_threshold(C) if min_cooccurrence == "auto"
            else max(3, ceil(min_cooccurrence * self.n_docs))
        )

        C.data[C.data < threshold] = 0
        C.eliminate_zeros()
        return np.diff(B.indptr), C, threshold

    def _find_threshold(self, C: csr_array) -> int:
        """
        Select co-occurrence threshold via modularity knee detection.

        Sweeps integer thresholds from the floor of 3 up to the point
        where the graph has no edges, computes Louvain modularity at
        each level, and returns the knee of the modularity-vs-threshold
        curve via `kneed.KneeLocator`.

        The knee is the point where community quality transitions
        from stable to rapidly degrading, balancing graph density
        against community structure. Falls back to the floor of 3
        when the sweep produces fewer than 3 candidate thresholds or
        `kneed` cannot locate a knee.

        Args:
            C: Pre-computed co-occurrence matrix with diagonal zeroed
               but not yet thresholded.

        Returns:
            Integer threshold, at minimum 3.
        """
        max_cooccurrence = int(C.data.max()) if C.nnz > 0 else 0
        thresholds       = list(range(3, max_cooccurrence + 1))

        if len(thresholds) < 3:
            return 3

        modularities = []
        for t in thresholds:
            filtered = C.copy()
            filtered.data[filtered.data < t] = 0
            filtered.eliminate_zeros()

            if filtered.nnz == 0:
                modularities.append(0.0)
                continue

            G = nx.from_scipy_sparse_array(filtered, edge_attribute="weight")
            modularities.append(
                nx.community.modularity(G, self._louvain(G), weight="weight")
            )

        knee = KneeLocator(
            thresholds,
            modularities,
            curve     = "convex",
            direction = "decreasing",
            S         = 1.0
        ).knee

        if knee is None:
            logger.info("Knee detection found no inflection, using floor of 3")
            return 3

        logger.info(f"Auto-selected co-occurrence threshold: {knee}")
        return int(knee)

    def _louvain(self, G: nx.Graph) -> list[set]:
        """
        Louvain community detection with fixed seed and weight.

        Args:
            G: NetworkX graph with `weight` edge attributes.

        Returns:
            List of sets, each containing node identifiers for one
            community.
        """
        return nx.community.louvain_communities(
            G, seed = self.random_seed, weight = "weight"
        )

    def _pmi_values(self) -> np.ndarray:
        """
        Raw PMI values for all thresholded co-occurrence pairs.

            PMI(x, y) = log(n * C_xy / (df_x * df_y))

        Returns the flat data array aligned with
        `self.cooccurrence.data` for direct assignment into a
        structural copy of the cooccurrence matrix.
        """
        C    = self.cooccurrence
        df   = self.doc_freq.astype(float)
        r, c = C.nonzero()
        return np.log(self.n_docs * C.data.astype(float) / (df[r] * df[c]))

    def graph(self) -> nx.Graph:
        """
        Build a NetworkX graph from the NPMI matrix.

        Relabels integer node indices to canonical skill names.

        Returns:
            Undirected `nx.Graph` with skill-name nodes and
            `weight` edge attributes.
        """
        return nx.relabel_nodes(
            nx.from_scipy_sparse_array(
                self.npmi_matrix,
                edge_attribute = "weight"
            ),
            dict(enumerate(self.feature_names))
        )

    def ppmi_dataframe(self) -> pd.DataFrame:
        """
        Positive PMI materialized as a labeled dense DataFrame for
        `ResumeMatcher` gap ranking. Computed and materialized in a
        single step because no other consumer needs the sparse PPMI
        matrix.

        Returns:
            Square `pd.DataFrame` with skill names as both index
            and columns, values clipped to non-negative.
        """
        ppmi = self.cooccurrence.astype(float)
        ppmi.data = np.maximum(self._pmi_values(), 0)
        return pd.DataFrame(
            ppmi.toarray(),
            columns = self.feature_names,
            index   = self.feature_names
        )
