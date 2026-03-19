"""
Career pathway graph from HAC clusters and NPMI co-occurrence edges.

Constructs a directed weighted DiGraph where nodes are job clusters from
hierarchical agglomerative clustering and edges connect clusters whose skill
profiles share significant NPMI co-occurrence. Edge weights are bounded to
[0, 1] for interpretability in downstream widest-path routing and career
report display. Edge direction follows a strict total order on (Job Zone,
cluster ID), guaranteeing acyclicity.
"""

import networkx as nx
import numpy    as np

from bisect      import bisect_left
from collections import defaultdict
from functools   import cached_property
from itertools   import combinations
from kneed       import KneeLocator
from loguru      import logger

from chalkline.association.cooccurrence import CooccurrenceNetwork
from chalkline.pathways.schemas         import LongestPath
from chalkline.pipeline.schemas         import ClusterProfile


class CareerPathwayGraph:
    """
    Directed weighted career graph with longest-path analysis and export.

    Accepts pre-enriched `ClusterProfile` records and a co-occurrence
    network, then constructs a DiGraph with one node per cluster and
    thresholded PMI edges. Edge direction follows a strict total order on
    (Job Zone, cluster ID), so the graph is always acyclic.
    """

    def __init__(
        self,
        profiles    : dict[int, ClusterProfile],
        graph       : nx.DiGraph | None = None,
        max_density : float = 0.05,
        network     : CooccurrenceNetwork | None = None
    ):
        """
        Build or accept a pre-built career pathway graph.

        When `network` is provided, edges are computed via NPMI
        co-occurrence and thresholded by knee detection with a `max_density`
        ceiling. When `graph` is provided instead, the DiGraph is used
        directly without rebuilding, as in the `Pipeline.load()` path.

        Args:
            profiles    : Enriched cluster characteristics from the
                          orchestrator.
            graph       : Pre-built DiGraph from deserialized artifacts.
                          Skips `_build_graph` when provided.
            max_density : Density ceiling for edge pruning.
            network     : NPMI matrix, vocabulary, and Louvain partition.
                          Required when `graph` is not provided.
        """
        self.max_density = max_density
        self.network     = network
        self.profiles    = profiles
        self.graph       = graph or self._build_graph()

        logger.info(
            f"Career graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

    @cached_property
    def longest_path(self) -> LongestPath:
        """
        Longest weighted path through the career graph.
        """
        if not self.graph:
            return LongestPath()

        path = nx.dag_longest_path(
            self.graph, weight="weight", default_weight=0
        )
        return LongestPath(
            edges       = self.graph.number_of_edges(),
            path        = path,
            path_weight = nx.path_weight(self.graph, path, "weight")
        )

    @cached_property
    def size(self) -> int:
        """
        Number of clusters in the career graph.
        """
        return len(self.profiles)

    @cached_property
    def ranked_ids(self) -> list[int]:
        """
        Cluster IDs sorted ascending by (Job Zone, cluster ID).
        """
        return sorted(self.profiles, key=lambda c: self.profiles[c].rank)

    @cached_property
    def skill_indices(self) -> dict[int, np.ndarray]:
        """
        Vocab-mapped column indices per cluster as contiguous arrays,
        precomputed for batch NPMI lookups.
        """
        vocab = self.vocab
        return {
            cid: np.fromiter(
                (vocab[s] for s in profile.skills if s in vocab),
                dtype=np.intp,
            )
            for cid, profile in self.profiles.items()
        }

    @cached_property
    def vocab(self) -> dict[str, int]:
        """
        Reverse index from skill name to column position in the
        co-occurrence matrix.
        """
        return {
            name: i
            for i, name in enumerate(self.network.feature_names)
        }

    def _build_graph(self) -> nx.DiGraph:
        """
        Construct the directed weighted career graph.

        Nodes carry all profile and enrichment attributes via
        `model_dump`. Edge weights are the top-k mean NPMI across
        each cluster pair's skill cross-product, thresholded by knee
        detection with a density cap as a secondary guard. Edges are
        directed from lower to higher rank.
        """
        (G := nx.DiGraph()).add_nodes_from(
            (cid, profile.model_dump(mode="json"))
            for cid, profile in self.profiles.items()
        )

        pairs   = list(combinations(self.ranked_ids, 2))
        weights = self._compute_all_weights(pairs)

        if (positive := weights > 0).any():
            threshold = self._find_threshold(
                max_edges = int(self.max_density * self.size * (self.size - 1) / 2),
                weights   = sorted(weights[positive])
            )
            keep = np.flatnonzero(weights >= threshold)
            G.add_weighted_edges_from(
                [(*pairs[i], weights[i]) for i in keep],
                direction_source="job_zone"
            )

        if not G.number_of_edges():
            logger.warning(
                f"Career graph has zero edges from {self.size} clusters and "
                f"{sum(p.size for p in self.profiles.values())} postings; "
                f"co-occurrence floor of {self.network.threshold} may exceed "
                f"corpus density"
            )

        return G

    def _compute_all_weights(self, pairs: list[tuple[int, int]]) -> np.ndarray:
        """
        Batch-compute top-k mean NPMI for all cluster pairs.

        Groups pairs by (|Cᵢ|, |Cⱼ|) so that each group shares a
        common submatrix shape, enabling 3D broadcast indexing across
        the group in a single numpy operation. The per-pair weight is

            w(i, j) = Σ top-k NPMI⁺(Cᵢ, Cⱼ) / min(k, pᵢⱼ)

        where k = min(10, |Cᵢ|, |Cⱼ|) and pᵢⱼ is the count of
        positive entries. This unifies the top-k and mean-of-positives
        paths because when pᵢⱼ < k the sorted tail pads with zeros
        that vanish from the sum, and dividing by pᵢⱼ recovers the
        mean of positives exactly.

        Args:
            pairs: Ordered (ci, cj) pairs from `combinations`.

        Returns:
            Weight array aligned with `pairs`, zero where fewer
            than 3 positive NPMI values exist.
        """
        npmi    = self.network.npmi_matrix.toarray().clip(0)
        sizes   = {c: v.size for c, v in self.skill_indices.items()}
        weights = np.zeros(len(pairs))

        shape_groups = defaultdict(list)
        for i, (ci, cj) in enumerate(pairs):
            shape_groups[(sizes[ci], sizes[cj])].append((i, ci, cj))

        for (row_size, col_size), group in shape_groups.items():
            pair_indices, flat = self._group_submatrices(npmi, group)
            positive_count     = np.count_nonzero(flat, axis=1)
            if not (valid := positive_count >= 3).any():
                continue

            k_cap       = min(10, row_size, col_size)
            effective_k = np.minimum(positive_count[valid], k_cap)
            weights[pair_indices[valid]] = (
                np.sort(flat[valid], axis=1)[:, -k_cap:].sum(axis=1)
                / effective_k
            )

        return weights

    def _find_threshold(self, max_edges: int, weights: list[float]) -> float:
        """
        Select edge weight threshold via knee detection.

        Finds the knee of the sorted weight curve where values transition
        from the noise floor to the signal regime, matching the pattern used
        by `CooccurrenceNetwork` for co-occurrence thresholds and by
        `ClusterComparison` for DBSCAN epsilon. Falls back to the 75th
        percentile when fewer than 3 candidates exist or no knee is found.

        Args:
            max_edges : Density-derived edge count ceiling, zero when
                        no cap applies.
            weights   : Candidate edge weights in ascending order.

        Returns:
            Weight threshold at or above the knee.
        """
        if len(weights) < 3:
            return weights[0]

        if (knee := KneeLocator(
            curve     = "convex",
            direction = "increasing",
            x         = range(len(weights)),
            y         = weights
        ).knee) is None:
            logger.info(
                "Edge weight knee detection found no inflection, "
                "using 75th percentile"
            )
            threshold = np.percentile(weights, 75)
        else:
            threshold = weights[knee]

        if max_edges and len(weights) - bisect_left(weights, threshold) > max_edges:
            threshold = weights[-max_edges]
            logger.info(
                f"Density cap raised threshold to {threshold:.3f} "
                f"({max_edges} edges at density {self.max_density})"
            )

        return threshold

    def _group_submatrices(
        self,
        D   : np.ndarray,
        grp : list[tuple[int, int, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract cross-cluster NPMI submatrices for one shape group.

        All pairs in `grp` share the same `(|skills_i|, |skills_j|)`
        dimensions, so their submatrices can be stacked into a single
        3D tensor via broadcast indexing and reshaped into a 2D batch.

        Args:
            D   : Dense NPMI matrix clipped to non-negative values.
            grp : Tuples of (pair index, source cluster, target cluster)
                  sharing a common skill-set shape.

        Returns:
            Pair of (row indices into the weights array, flattened
            submatrix batch of shape `(len(grp), si_sz * sj_sz)`).
        """
        pair_indices, row_skills, col_skills = (np.array(x) for x in zip(
            *((i, self.skill_indices[ci], self.skill_indices[cj])
              for i, ci, cj in grp)
        ))
        submatrices = D[
            row_skills[:, :, None], 
            col_skills[:, None, :]
        ].reshape(len(pair_indices), -1)
        
        return pair_indices, submatrices


