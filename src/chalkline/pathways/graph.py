"""
Career pathway graph from sentence-embedding cluster centroids.

Constructs a directed weighted graph where nodes are career clusters from
Ward-linkage HAC on sentence embeddings and edges connect clusters via
stepwise k-NN in cosine similarity space. Credential metadata
(apprenticeships, programs, certifications) is attached per-edge using a
dual-threshold filter on destination selectivity and source relevance.
"""

import networkx as nx
import numpy    as np

from dataclasses              import dataclass
from functools                import cached_property
from operator                 import attrgetter, eq, gt
from sklearn.metrics.pairwise import cosine_similarity

from chalkline.pathways.clusters import Clusters
from chalkline.pathways.schemas  import CareerEdge, Credential, Reach


@dataclass(kw_only=True)
class CareerPathwayGraph:
    """
    Directed weighted career graph with per-edge credential enrichment.

    Accepts a `Clusters` with pre-stacked centroid and embedding matrices,
    credential embeddings, and graph construction hyperparameters. Builds a
    stepwise k-NN backbone with bidirectional lateral edges and
    unidirectional upward edges, then enriches each edge with credentials
    filtered by destination selectivity and source relevance thresholds.

    Args:
        clusters               : Cluster map with centroids and vectors.
        credentials            : Typed records with aligned embedding vectors.
        destination_percentile : Top-p threshold for destination affinity.
        lateral_neighbors      : k for same-JZ bidirectional edges.
        source_percentile      : Floor percentile for source relevance.
        upward_neighbors       : k for next-JZ unidirectional edges.
    """

    clusters               : Clusters
    credentials            : list[Credential]
    destination_percentile : int
    lateral_neighbors      : int
    source_percentile      : int
    upward_neighbors       : int

    @property
    def brokerage(self) -> list[tuple[int, float]]:
        """
        Cluster IDs with brokerage scores, sorted descending by centrality.
        """
        return sorted(self.centrality.items(), key=lambda x: x[1], reverse=True)

    @cached_property
    def centrality(self) -> dict[int, float]:
        """
        Betweenness centrality per node, computed once and cached.

        Both `brokerage` (sorted view for the methods tab) and
        `Charts.landscape` (raw lookup for marker sizing) read from this
        single computation, eliminating a duplicate Brandes' run per
        session.
        """
        from networkx import betweenness_centrality
        return betweenness_centrality(self.graph, weight="weight")

    @property
    def edge_count(self) -> int:
        """
        Number of edges in the career pathway graph.
        """
        return self.graph.number_of_edges()

    @property
    def edge_weights(self) -> list[float]:
        """
        Cosine similarity weights for all edges.
        """
        return [w for _, _, w in self.graph.edges(data="weight")]

    @cached_property
    def graph(self) -> nx.DiGraph:
        """
        Stepwise k-NN DiGraph with credential-enriched edges.

        Built lazily on first access. `_add_edges` lays down the backbone
        from the centroid cosine similarity matrix, then `_enrich_edges`
        attaches per-edge credential metadata via the dual-threshold
        filter.
        """
        g = nx.DiGraph()
        g.add_nodes_from(self.node_ids)
        self._add_edges(g, cosine_similarity(self.clusters.centroids))
        self._enrich_edges(g, cosine_similarity(
            np.array([c.vector for c in self.credentials if c.vector]),
            self.clusters.vectors
        ))
        return g

    @cached_property
    def node_ids(self) -> np.ndarray:
        """
        Cluster IDs as a numpy array for vectorized index operations.
        """
        return np.array(self.clusters.cluster_ids)

    @cached_property
    def widest_path_tree(self) -> nx.Graph:
        """
        Maximum spanning tree of the undirected graph view.

        Cached so route clicks reuse one MST computation across all
        widest-path queries within a session. The tree is a pure function
        of the frozen graph.
        """
        return nx.maximum_spanning_tree(
            self.graph.to_undirected(), weight="weight"
        )

    def _add_edges(self, g: nx.DiGraph, pairwise: np.ndarray):
        """
        Add stepwise k-NN backbone edges to `g`.

        Each cluster gets `lateral_neighbors` bidirectional edges to its
        most similar same-JZ clusters and `upward_neighbors` unidirectional
        edges to its most similar clusters at the next JZ level. The
        stepwise constraint prevents tier-skipping shortcuts.
        """
        similarity  = pairwise - np.eye(len(pairwise))
        zones       = np.array([self.clusters.job_zone_map[c] for c in self.node_ids])
        zone_levels = sorted(set(zones))
        next_zone   = dict(zip(zone_levels, zone_levels[1:]))

        for source in self.node_ids:
            zone      = self.clusters.job_zone_map[source]
            lateral   = self.node_ids[(zones == zone) & (self.node_ids > source)]
            proximity = similarity[source, lateral]

            for i in np.argsort(-proximity)[:self.lateral_neighbors]:
                g.add_edge(source, lateral[i], weight=proximity[i])
                g.add_edge(lateral[i], source, weight=proximity[i])

            if zone in next_zone:
                upward    = self.node_ids[zones == next_zone[zone]]
                proximity = similarity[source, upward]

                for i in np.argsort(-proximity)[:self.upward_neighbors]:
                    g.add_edge(source, upward[i], weight=proximity[i])

    def _edge(self, source: int, target: int) -> CareerEdge:
        """
        Build one typed `CareerEdge` from a source/target pair, unpacking
        the graph edge data (weight + credentials).
        """
        return CareerEdge(cluster_id=target, **self.graph[source][target])

    def _enrich_edges(self, g: nx.DiGraph, credential_similarity: np.ndarray):
        """
        Attach per-edge credential metadata to `g` using the dual-threshold
        filter.

        For each edge (source → target), a credential 𝐜 passes when both
        conditions hold:

            cos(𝐜, 𝐭) ≥ P₁₀₀₋ₚ(cos(·, 𝐭))
            cos(𝐜, 𝐬) ≥ Pₛ(cos(·, ·))

        The first threshold selects credentials with high destination
        affinity. The second ensures source relevance against the global
        credential-to-cluster distribution.
        """
        mask = credential_similarity >= np.percentile(
            a = credential_similarity,
            q = self.source_percentile
        )
        affinity_floors = np.percentile(
            a    = credential_similarity,
            axis = 0,
            q    = 100 - self.destination_percentile
        )

        records = np.array(self.credentials, dtype=object)

        for s, t, edge_data in g.edges(data=True):
            affinity = credential_similarity[:, t]
            passing  = np.flatnonzero(
                (affinity >= affinity_floors[t]) & mask[:, s]
            )
            ranked   = passing[np.argsort(-affinity[passing])]
            edge_data["credentials"] = [
                r.model_dump(mode="json")
                for r in records[ranked]
            ]

    def hops_from(self, source: int) -> dict[int, int]:
        """
        BFS hop distance from `source` to every reachable cluster.

        Operates on the undirected projection of the directed pathway
        graph so distance reflects connectivity rather than edge
        direction. The map widget uses these distances to fade nodes by
        their proximity to the matched cluster.
        """
        return nx.single_source_shortest_path_length(
            self.graph.to_undirected(), source
        )

    def path_edges(self, path: list[int]) -> list[CareerEdge]:
        """
        Typed edges for each hop along a cluster ID path.

        Reconstructs `CareerEdge` objects from the raw graph edge data,
        keeping credential metadata and cosine similarity weights. Used
        by the display layer to build multi-hop route details without
        reaching into the internal `nx.DiGraph`.

        Args:
            path: Ordered cluster IDs from source to destination.

        Returns:
            One `CareerEdge` per consecutive pair in the path.
        """
        return [self._edge(s, t) for s, t in zip(path, path[1:])]

    def reach(self, cluster_id: int) -> Reach:
        """
        Local reach exploration from a given cluster.

        Returns advancement paths (edges to higher JZ clusters) and lateral
        pivots (edges to same JZ clusters) with their per-edge credential
        metadata sorted by edge weight.
        """
        zone  = self.clusters.job_zone_map[cluster_id]
        edges = sorted(
            (self._edge(cluster_id, t) for t in self.graph.successors(cluster_id)),
            key     = attrgetter("weight"),
            reverse = True
        )
        pick = lambda op: [
            e for e in edges
            if op(self.clusters.job_zone_map[e.cluster_id], zone)
        ]
        return Reach(advancement=pick(gt), lateral=pick(eq))

    def try_widest_path(self, source: int, target: int) -> list[int]:
        """
        Most achievable multi-hop career progression via max-min bottleneck,
        returning an empty list when no path exists.

        The max-min bottleneck path between two nodes equals the unique
        path on the maximum spanning tree, because the MST maximizes the
        minimum edge weight across any cut. Building the MST on the
        undirected view lets the route traverse lateral edges in either
        direction, which may produce routes that reverse upward edges
        because the path describes skill affinity rather than strict
        progression order.

        Args:
            source : Origin cluster ID.
            target : Destination cluster ID.

        Returns:
            Ordered cluster IDs from source to target inclusive, or empty
            when nodes are absent or disconnected.
        """
        if source == target:
            return [source]
        try:
            return nx.shortest_path(self.widest_path_tree, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
