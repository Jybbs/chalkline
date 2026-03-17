"""
Career pathway graph from HAC clusters and NPMI co-occurrence edges.

Constructs a directed weighted DiGraph where nodes are job clusters from
hierarchical agglomerative clustering and edges connect clusters whose
skill profiles share significant NPMI co-occurrence. Edge weights are
bounded to [0, 1] for interpretability in downstream widest-path routing
and career report display. Edge direction follows a strict total order on
(Job Zone, cluster ID), guaranteeing acyclicity.
"""

import networkx as nx
import numpy    as np

from functools import cached_property
from itertools import combinations
from kneed     import KneeLocator
from logging   import getLogger

from chalkline.association.cooccurrence import CooccurrenceNetwork
from chalkline.pathways.schemas         import LongestPath
from chalkline.pipeline.schemas         import ClusterProfile


logger = getLogger(__name__)


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
        co-occurrence and thresholded by knee detection with a
        `max_density` ceiling. When `graph` is provided instead,
        the DiGraph is used directly without rebuilding, as in
        the `Pipeline.load()` path.

        Args:
            profiles    : Enriched cluster characteristics from
                          the orchestrator.
            graph       : Pre-built DiGraph from deserialized
                          artifacts. Skips `_build_graph` when
                          provided.
            max_density : Density ceiling for edge pruning.
            network     : NPMI matrix, vocabulary, and Louvain
                          partition. Required when `graph` is
                          not provided.
        """
        self.max_density = max_density
        self.network     = network
        self.profiles    = profiles
        self.graph       = graph or self._build_graph()

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
    def skill_indices(self) -> dict[int, list[int]]:
        """
        Vocab-mapped column indices per cluster, precomputed for
        pairwise NPMI lookups in `_edge_weight`.
        """
        return {
            cid: [self.vocab[s] for s in profile.skills if s in self.vocab]
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

        Nodes carry all profile and enrichment attributes via `model_dump`.
        Edge threshold is selected by knee detection on the sorted candidate
        weight curve, with a density cap as a secondary guard. Edges are
        directed from lower to higher rank.
        """
        (G := nx.DiGraph()).add_nodes_from(
            (cid, profile.model_dump(mode="json"))
            for cid, profile in self.profiles.items()
        )

        cluster_ids = sorted(self.profiles, key=lambda c: self.profiles[c].rank)
        candidates  = [
            (ci, cj, wc[0])
            for ci, cj in combinations(cluster_ids, 2)
            if (wc := self._edge_weight(ci, cj))[0] > 0 and wc[1] >= 3
        ]

        if candidates:
            threshold = self._find_threshold(
                max_edges = int(self.max_density * self.size * (self.size - 1) / 2),
                weights   = sorted(w for _, _, w in candidates)
            )

            G.add_weighted_edges_from(
                [(s, t, w) for s, t, w in candidates if w >= threshold],
                direction_source="job_zone"
            )

        if not G.number_of_edges():
            logger.warning(
                f"Career graph has zero edges from "
                f"{self.size} clusters and "
                f"{sum(p.size for p in self.profiles.values())} "
                f"postings; co-occurrence floor of "
                f"{self.network.threshold} may exceed corpus density"
            )

        return G

    def _edge_weight(self, ci: int, cj: int) -> tuple[float, int]:
        """
        Top-k mean NPMI for a cluster pair.

        Uses NPMI rather than PPMI so that edge weights are bounded to
        [0, 1] and interpretable as association strength independent of
        skill frequency. The top-k aggregation retains the strongest
        inter-cluster bridges where k = min(10, |Ci|, |Cj|). Operates
        directly on the sparse NPMI matrix to avoid dense materialization.

        Args:
            ci : First cluster ID.
            cj : Second cluster ID.

        Returns:
            Tuple of (top-k mean NPMI, count of positive pairs).
        """
        si, sj    = self.skill_indices[ci], self.skill_indices[cj]
        values    = self.network.npmi_matrix[np.ix_(si, sj)].data
        positives = values[values > 0]

        if not len(positives):
            return 0.0, 0

        k = min(10, len(si), len(sj), len(positives))
        return np.partition(positives, -k)[-k:].mean(), len(positives)
    
    def _find_threshold(
        self,
        max_edges : int,
        weights   : list[float]
    ) -> float:
        """
        Select edge weight threshold via knee detection.

        Finds the knee of the sorted weight curve where values transition
        from the noise floor to the signal regime, matching the pattern used
        by `CooccurrenceNetwork` for co-occurrence thresholds and by
        `ClusterComparison` for DBSCAN epsilon. Falls back to the 75th
        percentile when fewer than 3 candidates exist or no knee is found.

        Args:
            weights : Candidate edge weights in ascending order.

        Returns:
            Weight threshold at or above the knee.
        """
        if len(weights) < 3:
            return weights[0]

        knee = KneeLocator(
            curve     = "convex",
            direction = "increasing",
            x         = range(len(weights)),
            y         = weights
        ).knee

        if knee is None:
            logger.info(
                "Edge weight knee detection found no inflection, "
                "using 75th percentile"
            )
            threshold = np.percentile(weights, 75)
        else:
            threshold = weights[knee]

        if max_edges and sum(1 for w in weights if w >= threshold) > max_edges:
            threshold = weights[-max_edges]
            logger.info(
                f"Density cap raised threshold to {threshold:.3f} "
                f"({max_edges} edges at density "
                f"{self.max_density})"
            )

        return threshold


