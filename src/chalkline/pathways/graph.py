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
from itertools                import pairwise
from operator                 import attrgetter, itemgetter
from sklearn.metrics.pairwise import cosine_similarity
from typing                   import SupportsFloat, SupportsInt

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
        return sorted(self.centrality.items(), key=itemgetter(1), reverse=True)

    @cached_property
    def centrality(self) -> dict[int, float]:
        """
        Betweenness centrality per node, computed once and cached.

        Both `brokerage` (sorted view for the methods tab) and
        `Charts.landscape` (raw lookup for marker sizing) read from this
        single computation, eliminating a duplicate Brandes' run per
        session.
        """
        return nx.betweenness_centrality(self.graph, weight="weight")

    @cached_property
    def credential_matrix(self) -> tuple[list[Credential], np.ndarray]:
        """
        Filtered credentials and their pre-stacked vector matrix.

        Both edge enrichment (during graph construction) and the
        display-layer `RelevantCredentials` factory rank credentials
        against cluster vectors via cosine similarity. The filtered
        list and the stacked matrix are derived purely from
        `self.credentials`, which is fixed at fit time, so the np.array
        copy of ~325 vectors only happens once per session instead of
        twice (once during graph build, once per data tab render).
        """
        with_vectors = [c for c in self.credentials if c.vector]
        return with_vectors, np.array([c.vector for c in with_vectors])

    @cached_property
    def credential_similarity(self) -> np.ndarray:
        """
        Cosine similarity between every credential vector and every
        cluster vector.

        Shape `(n_credentials, n_clusters)` aligned with
        `credential_matrix[0]` on the row axis and
        `clusters.cluster_ids` on the column axis. Computed once during
        graph construction. The dual-threshold edge enrichment reads
        every column to filter credentials per edge, and the display
        layer's `RelevantCredentials.from_cluster` slices a single
        column per render to rank credentials for the matched career
        family. Sharing the matrix avoids the per-render cosine call
        and keeps both consumers reading the same authoritative source.
        """
        _, matrix = self.credential_matrix
        return cosine_similarity(matrix, self.clusters.vectors)

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
        g.add_nodes_from(self.clusters.cluster_ids)
        self._add_edges(g, self.clusters.centroid_cosine)
        self._enrich_edges(g, self.credential_similarity)
        return g

    @cached_property
    def node_ids(self) -> np.ndarray:
        """
        Cluster IDs as a numpy array for vectorized index operations.
        """
        return np.array(self.clusters.cluster_ids)

    @cached_property
    def undirected_graph(self) -> nx.Graph:
        """
        Undirected projection of the directed pathway graph.

        Cached so `widest_path_tree` and `hops_from` share one
        `to_undirected()` materialization across the session instead of
        rebuilding it on every map render.
        """
        return self.graph.to_undirected()

    @cached_property
    def widest_path_tree(self) -> nx.Graph:
        """
        Maximum spanning tree of the undirected graph view.

        Cached so route clicks reuse one MST computation across all
        widest-path queries within a session. The tree is a pure function
        of the frozen graph.
        """
        return nx.maximum_spanning_tree(self.undirected_graph, weight="weight")

    def _add_edges(self, g: nx.DiGraph, similarity: np.ndarray):
        """
        Add stepwise k-NN backbone edges to `g`.

        Each cluster gets `lateral_neighbors` bidirectional edges to its
        most similar same-JZ clusters and `upward_neighbors` unidirectional
        edges to its most similar clusters at the next JZ level. The
        stepwise constraint prevents tier-skipping shortcuts.
        """
        job_zones   = self.clusters.job_zone_map
        zones       = np.array([job_zones[c] for c in self.node_ids])
        zone_levels = sorted(set(zones))
        next_zone   = dict(pairwise(zone_levels))

        for source in self.node_ids:
            zone      = job_zones[source]
            lateral   = self.node_ids[(zones == zone) & (self.node_ids > source)]
            proximity = similarity[source, lateral]

            for i in np.argsort(-proximity)[:self.lateral_neighbors]:
                self._link(g, source,     lateral[i], proximity[i])
                self._link(g, lateral[i], source,     proximity[i])

            if zone in next_zone:
                upward    = self.node_ids[zones == next_zone[zone]]
                proximity = similarity[source, upward]

                for i in np.argsort(-proximity)[:self.upward_neighbors]:
                    self._link(g, source, upward[i], proximity[i])

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
        dest_mask = credential_similarity >= np.percentile(
            a    = credential_similarity,
            axis = 0,
            q    = 100 - self.destination_percentile
        )
        source_mask = credential_similarity >= np.percentile(
            a = credential_similarity,
            q = self.source_percentile
        )

        records = np.array(self.credential_matrix[0], dtype=object)

        for s, t, edge_data in g.edges(data=True):
            passing = np.flatnonzero(dest_mask[:, t] & source_mask[:, s])
            ranked  = passing[np.argsort(-credential_similarity[passing, t])]
            edge_data["credentials"] = [
                r.model_dump(mode="json")
                for r in records[ranked]
            ]

    def _link(
        self,
        g      : nx.DiGraph,
        source : SupportsInt,
        target : SupportsInt,
        weight : SupportsFloat
    ):
        """
        Add a weighted edge to `g`, coercing numpy scalars to Python types.

        The backbone builder iterates `node_ids` (an `np.ndarray`) and
        slices it with boolean masks, so source and target arrive as
        `np.int64` and weight as `np.float64`. NetworkX accepts those, but
        downstream consumers (`json.dumps` in the map widget, Pydantic
        `CareerEdge` validation) expect native Python types. Casting at
        the insertion boundary keeps the rest of the graph type-clean.
        """
        g.add_edge(int(source), int(target), weight=float(weight))

    def hops_from(self, source: int) -> dict[int, int]:
        """
        BFS hop distance from `source` to every reachable cluster.

        Operates on the undirected projection of the directed pathway
        graph so distance reflects connectivity rather than edge
        direction. The map widget uses these distances to fade nodes by
        their proximity to the matched cluster.
        """
        return nx.single_source_shortest_path_length(self.undirected_graph, source)

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
        return [self._edge(s, t) for s, t in pairwise(path)]

    def reach(self, cluster_id: int) -> Reach:
        """
        Local reach exploration from a given cluster.

        Returns advancement paths (edges to higher JZ clusters) and lateral
        pivots (edges to same JZ clusters) with their per-edge credential
        metadata sorted by edge weight.
        """
        job_zones = self.clusters.job_zone_map
        zone      = job_zones[cluster_id]
        edges     = sorted(
            (self._edge(cluster_id, t) for t in self.graph.successors(cluster_id)),
            key     = attrgetter("weight"),
            reverse = True
        )
        return Reach(
            advancement = [e for e in edges if job_zones[e.cluster_id] >  zone],
            lateral     = [e for e in edges if job_zones[e.cluster_id] == zone]
        )

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
