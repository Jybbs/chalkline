"""
PMI co-occurrence network with Louvain community detection.

Computes pairwise co-occurrence counts via sparse matrix multiplication,
derives PMI, PPMI, NPMI, and Dunning's G-test as association measures,
and constructs a NetworkX graph with Louvain community detection for
career track grouping. The NPMI graph feeds directly into the pathway
graph in CL-11, and the NPMI DataFrame feeds gap ranking in CL-10.
"""

import networkx as nx
import numpy    as np
import pandas   as pd

from functools                import cached_property
from kneed                    import KneeLocator
from logging                  import getLogger
from math                     import ceil
from scipy.sparse             import csc_array, csr_array, triu
from scipy.special            import xlogy
from typing                   import TYPE_CHECKING

from chalkline.association.schemas import CommunityLabel
from chalkline.association.schemas import GraphDiagnostics, MeasureComparison

if TYPE_CHECKING:
    from scipy.sparse import spmatrix


logger = getLogger(__name__)


class CooccurrenceNetwork:
    """
    PMI-weighted skill co-occurrence network with Louvain detection.

    Receives the binary skill matrix from `SkillVectorizer`, computes
    the thresholded co-occurrence matrix eagerly, and exposes PMI
    variants, graph construction, and Louvain community detection as
    cached properties and on-demand methods. When `min_cooccurrence`
    is `"auto"`, the threshold is selected by finding the knee of the
    modularity-vs-threshold curve via `kneed`, which identifies the
    point where community quality begins to degrade. The floor of 3
    protects PMI stability at small corpus sizes where a single
    observation can shift a pair's score by 25%.
    """

    def __init__(
        self,
        binary_matrix   : "spmatrix",
        feature_names   : list[str],
        min_cooccurrence: float | str = "auto",
        random_seed     : int         = 42
    ):
        """
        Compute the thresholded co-occurrence matrix.

        Args:
            binary_matrix   : CSR binary presence/absence matrix from
                              `SkillVectorizer.binary_matrix`.
            feature_names   : Vocabulary in column order, matching
                              matrix column indices.
            min_cooccurrence: Either `"auto"` to select the threshold
                              via modularity knee detection, or a float
                              fraction of corpus size floored at 3.
            random_seed     : Seed for Louvain reproducibility.
        """
        self.feature_names = feature_names
        self.n_docs        = binary_matrix.shape[0]
        self.random_seed   = random_seed

        B = csc_array(binary_matrix)
        self.doc_freq = np.diff(B.indptr)

        C = (B.T @ B).tocsr()
        C.setdiag(0)
        C.eliminate_zeros()

        if min_cooccurrence == "auto":
            self.threshold = self._find_threshold(C)
        else:
            self.threshold = max(3, ceil(min_cooccurrence * self.n_docs))

        C.data[C.data < self.threshold] = 0
        C.eliminate_zeros()

        self.cooccurrence = C

    # -----------------------------------------------------------------
    # Cached properties
    # -----------------------------------------------------------------

    @cached_property
    def gtest_matrix(self) -> csr_array:
        """
        Dunning's G-test statistics for all thresholded pairs.

        Computes 2x2 contingency tables from document frequency
        marginals across all upper-triangle pairs via vectorized
        array operations. The result is symmetric.
        """
        C  = triu(self.cooccurrence, format = "coo")
        n  = float(self.n_docs)
        df = self.doc_freq.astype(float)

        a = C.data.astype(float)
        b = df[C.row] - a
        c = df[C.col] - a
        d = n - df[C.row] - df[C.col] + a

        valid      = (b >= 0) & (c >= 0) & (d >= 0)
        rows, cols = C.row[valid], C.col[valid]
        a, b, c, d = a[valid], b[valid], c[valid], d[valid]

        df_r = df[rows]
        df_c = df[cols]
        obs  = np.vstack([a, b, c, d])
        exp  = np.vstack([
            df_r * df_c,
            df_r * (n - df_c),
            (n - df_r) * df_c,
            (n - df_r) * (n - df_c)
        ]) / n

        g = 2 * (xlogy(obs, obs) - xlogy(obs, exp)).sum(axis = 0)

        size  = self.cooccurrence.shape[0]
        upper = csr_array(
            (g, (rows, cols)),
            shape = (size, size)
        )
        return upper + upper.T

    @cached_property
    def npmi_matrix(self) -> csr_array:
        """
        Normalized pointwise mutual information with positive
        clipping.

        Divides PMI by the negative log joint probability for each
        pair, bounding values to [-1, 1]. Pairs where both skills
        appear in every posting are capped at 1.0 rather than
        dividing by zero. Negative values are clipped to zero
        (NPMI+) for graph construction.
        """
        pmi   = self.pmi_matrix.copy()
        denom = np.log(self.n_docs / self.cooccurrence.data.astype(float))

        pmi.data = np.where(denom > 0, pmi.data / denom, 1.0)
        np.clip(pmi.data, 0, 1.0, out = pmi.data)
        pmi.eliminate_zeros()
        return pmi

    @cached_property
    def pmi_matrix(self) -> csr_array:
        """
        Pointwise mutual information as sparse matrix algebra.

        Constructs diagonal normalization matrices from document
        frequencies and applies log directly to the sparse data
        attribute to avoid densifying the matrix.
        """
        n  = self.n_docs
        df = self.doc_freq.astype(float)
        C  = self.cooccurrence.copy().astype(float)

        rows, cols = C.nonzero()
        C.data = np.log(
            (n * C.data) / (df[rows] * df[cols])
        )
        return C

    @cached_property
    def ppmi_matrix(self) -> csr_array:
        """
        Positive pointwise mutual information.

        Clips negative PMI values to zero and eliminates the
        resulting structural zeros.
        """
        ppmi = self.pmi_matrix.copy()
        ppmi.data = np.maximum(ppmi.data, 0)
        ppmi.eliminate_zeros()
        return ppmi

    # -----------------------------------------------------------------
    # Private methods
    # -----------------------------------------------------------------

    def _find_threshold(self, C: csr_array) -> int:
        """
        Select the co-occurrence threshold via modularity knee detection.

        Sweeps integer thresholds from the floor of 3 up to the point
        where the graph has no edges, computes Louvain modularity at
        each level, and returns the knee of the modularity-vs-threshold
        curve. The knee is the point where community quality transitions
        from stable to rapidly degrading, balancing graph density against
        community structure.

        Falls back to the floor of 3 when the sweep produces fewer than
        3 candidate thresholds or `kneed` cannot locate a knee.

        Args:
            C: Pre-computed co-occurrence matrix with diagonal zeroed
               but not yet thresholded.
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
        """
        return nx.community.louvain_communities(
            G, seed = self.random_seed, weight = "weight"
        )

    # -----------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------

    def communities(
        self,
        matrix: "spmatrix | None" = None
    ) -> list[CommunityLabel]:
        """
        Louvain community detection with weighted-degree labeling.

        Each community is labeled by its top-3 skills ranked by
        weighted degree within the community. Defaults to the NPMI
        graph when no matrix is provided.

        Args:
            matrix: Optional sparse weight matrix to override the
                    default NPMI matrix.
        """
        G = self.graph(matrix = matrix)

        labels = []
        for idx, members in enumerate(
            sorted(
                self._louvain(G),
                key     = len,
                reverse = True
            )
        ):
            degrees = dict(G.subgraph(members).degree(weight = "weight"))
            labels.append(CommunityLabel(
                community_id        = idx,
                size                = len(members),
                top_skills          = sorted(
                    degrees, key = degrees.get, reverse = True
                )[:3],
                weighted_degree_sum = sum(degrees.values())
            ))

        return labels

    def compare_measures(self) -> list[MeasureComparison]:
        """
        Compare graph structures across PMI variants.

        Builds graphs from PPMI, NPMI, and G-test matrices and
        reports edge count, density, and community count for each.
        """
        results = []
        for name, matrix in [
            ("ppmi",   self.ppmi_matrix),
            ("npmi",   self.npmi_matrix),
            ("g-test", self.gtest_matrix)
        ]:
            G = self.graph(matrix = matrix)
            results.append(MeasureComparison(
                density       = nx.density(G),
                edge_count    = G.number_of_edges(),
                measure       = name,
                n_communities = len(
                    self._louvain(G)
                ) if G.number_of_edges() > 0 else 0
            ))

        return results

    def diagnostics(
        self,
        graph: nx.Graph | None = None
    ) -> GraphDiagnostics:
        """
        Network-level diagnostics for the co-occurrence graph.

        Reports edge count, connected components, isolate count,
        largest component size, modularity, coverage, and
        performance. Logs a warning if more than 30% of skills
        are isolate nodes.

        Args:
            graph: Optional pre-built graph. Defaults to the NPMI
                   graph.
        """
        G = graph or self.graph()

        components = list(nx.connected_components(G))
        isolates   = list(nx.isolates(G))

        modularity  = None
        coverage    = 0.0
        performance = 0.0

        if G.number_of_edges() > 0:
            louvain    = self._louvain(G)
            modularity = nx.community.modularity(
                G, louvain, weight = "weight"
            )
            coverage, performance = nx.community.partition_quality(
                G, louvain
            )

        if G.number_of_nodes() > 0 and (
            isolate_rate := len(isolates) / G.number_of_nodes()
        ) > 0.30:
            logger.warning(
                f"{isolate_rate:.0%} of skills are isolate nodes "
                f"({len(isolates)} of {G.number_of_nodes()})"
            )

        return GraphDiagnostics(
            connected_components = len(components),
            coverage             = coverage,
            edge_count           = G.number_of_edges(),
            isolate_count        = len(isolates),
            largest_component    = max(
                (len(c) for c in components), default = 0
            ),
            modularity           = modularity,
            node_count           = G.number_of_nodes(),
            performance          = performance
        )

    def graph(
        self,
        matrix: "spmatrix | None" = None
    ) -> nx.Graph:
        """
        Build a NetworkX graph from a sparse weight matrix.

        Relabels integer node indices to canonical skill names.
        Defaults to the NPMI matrix when no matrix is provided.

        Args:
            matrix: Optional sparse weight matrix to override the
                    default NPMI matrix.
        """
        G = nx.from_scipy_sparse_array(
            matrix if matrix is not None else self.npmi_matrix,
            edge_attribute = "weight"
        )
        return nx.relabel_nodes(G, dict(enumerate(self.feature_names)))

    def pmi_dataframe(self) -> pd.DataFrame:
        """
        Symmetric NPMI DataFrame indexed by skill names.

        The dense conversion is scoped to this call. Used by
        CL-10 for gap ranking where skill-pair weights inform
        which gaps are most actionable.
        """
        return pd.DataFrame(
            self.npmi_matrix.toarray(),
            columns = self.feature_names,
            index   = self.feature_names
        )

    def trade_alignment(self, apprenticeships: list[dict]) -> dict:
        """
        Compare Louvain communities against apprenticeship trades.

        For each apprenticeship trade, checks whether the trade
        title appears as a graph node and reports which Louvain
        community it belongs to. This is a diagnostic measure
        rather than a blocking constraint because the vocabulary
        may not contain every trade title as a matchable skill.

        Args:
            apprenticeships: List of dicts with `title` and
                             `rapids_code` keys from the stakeholder
                             reference data.
        """
        G = self.graph()
        node_to_community = {
            member: comm_id
            for comm_id, members in enumerate(
                sorted(self._louvain(G), key = len, reverse = True)
            )
            for member in members
        }

        alignments = [
            {
                "community"   : node_to_community.get(
                    title := trade["title"].lower()
                ),
                "matched"     : title in G,
                "rapids_code" : trade["rapids_code"],
                "title"       : trade["title"]
            }
            for trade in apprenticeships
        ]

        matched_count = sum(1 for a in alignments if a["matched"])
        logger.info(
            f"Trade alignment: {matched_count}/{len(apprenticeships)} "
            f"trades found in vocabulary"
        )

        return {
            "alignments"    : alignments,
            "matched_count" : matched_count,
            "total_trades"  : len(apprenticeships)
        }
