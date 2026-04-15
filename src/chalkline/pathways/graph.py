"""
Career pathway graph from sentence-embedding cluster centroids.

Constructs a directed weighted graph where nodes are career clusters from
Ward-linkage HAC on sentence embeddings and edges connect clusters via
stepwise k-NN in cosine similarity space. Credentials are computed per
(source, target) pair on demand via `credentials_for`, applying a
dual-threshold filter on destination selectivity and source relevance to
the cluster-credential cosine matrix.
"""

import networkx as nx
import numpy    as np

from dataclasses              import dataclass
from functools                import cached_property
from itertools                import pairwise
from operator                 import attrgetter, itemgetter
from sklearn.metrics.pairwise import cosine_similarity

from chalkline.pathways.clusters import Clusters
from chalkline.pathways.schemas  import CareerEdge, Credential, Reach


@dataclass(kw_only=True)
class CareerPathwayGraph:
    """
    Directed weighted career graph backbone with on-demand credential
    filtering.

    Accepts a `Clusters` with pre-stacked centroid and embedding matrices,
    credential embeddings, and graph construction hyperparameters. Builds
    a stepwise k-NN backbone with bidirectional lateral edges and
    unidirectional upward edges. Per-route credential lists come from
    `credentials_for(source_id, target_id)` at click time, applying the
    dual-threshold filter to the cluster-credential cosine matrix.

    Args:
        clusters               : Cluster map with centroids and vectors.
        credentials            : Typed records with aligned embedding vectors.
        destination_percentile : Top-p threshold for destination affinity.
        lateral_neighbors      : k for same Job Zone bidirectional edges.
        source_percentile      : Floor percentile for source relevance.
        upward_neighbors       : k for next Job Zone unidirectional edges.
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
        display-layer `RelevantCredentials` factory rank credentials against
        cluster vectors via cosine similarity. The filtered list and the
        stacked matrix are derived purely from `self.credentials`, which is
        fixed at fit time, so the np.array copy of ~325 vectors only happens
        once per session instead of twice (once during graph build, once per
        data tab render).
        """
        with_vectors = [c for c in self.credentials if c.vector]
        return with_vectors, np.array([c.vector for c in with_vectors])

    @cached_property
    def credential_similarity(self) -> np.ndarray:
        """
        Cosine similarity between every credential vector and every cluster
        vector.

        Shape `(n_credentials, n_clusters)` aligned with
        `credential_matrix[0]` on the row axis and `clusters.cluster_ids` on
        the column axis. Computed once during graph construction. The
        dual-threshold edge enrichment reads every column to filter
        credentials per edge, and the display layer's
        `RelevantCredentials.from_cluster` slices a single column per render
        to rank credentials for the matched career family. Sharing the
        matrix avoids the per-render cosine call and keeps both consumers
        reading the same authoritative source.
        """
        creds, matrix = self.credential_matrix
        if not creds:
            return np.empty((0, len(self.clusters.cluster_ids)))
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
        Stepwise k-NN DiGraph backbone built lazily on first access.
        Edges carry only cosine-similarity weights; credentials are
        computed on demand per (source, target) pair via
        `credentials_for`, not attached at fit time.
        """
        g = nx.DiGraph()
        g.add_nodes_from(self.clusters.cluster_ids)
        self._add_edges(g, self.clusters.centroid_cosine)
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
        Undirected projection of the directed pathway graph, cached so
        the map widget's `hops_from` BFS materializes the projection
        once per session.
        """
        return self.graph.to_undirected()

    def _add_edges(self, g: nx.DiGraph, similarity: np.ndarray):
        """
        Add stepwise k-NN backbone edges to `g`.

        Each cluster gets `lateral_neighbors` bidirectional edges to its
        most similar clusters at the same Job Zone and `upward_neighbors`
        unidirectional edges to its most similar clusters at the next Job
        Zone level. The stepwise constraint prevents tier-skipping
        shortcuts.
        """
        job_zones = self.clusters.job_zone_map
        zones     = np.array([job_zones[c] for c in self.node_ids])
        next_zone = dict(pairwise(sorted(set(zones))))

        for source in self.node_ids:
            zone      = job_zones[source]
            lateral   = self.node_ids[(zones == zone) & (self.node_ids > source)]
            proximity = similarity[source, lateral]

            for i in np.argsort(-proximity)[:self.lateral_neighbors]:
                g.add_edge(int(source), int(lateral[i]), weight=float(proximity[i]))
                g.add_edge(int(lateral[i]), int(source), weight=float(proximity[i]))

            if zone in next_zone:
                upward    = self.node_ids[zones == next_zone[zone]]
                proximity = similarity[source, upward]

                for i in np.argsort(-proximity)[:self.upward_neighbors]:
                    g.add_edge(int(source), int(upward[i]), weight=float(proximity[i]))

    def _edge(self, source: int, target: int) -> CareerEdge:
        """
        Build one typed `CareerEdge` from a source/target pair, unpacking
        the graph edge weight; the `credentials` field defaults to empty
        because per-edge enrichment is no longer attached at fit time.
        """
        return CareerEdge(cluster_id=target, **self.graph[source][target])

    def credentials_for(self, source_id: int, target_id: int) -> list[Credential]:
        """
        Credentials that bridge a specific source-to-target transition,
        filtered by the dual-threshold rule on destination affinity and
        source relevance and ranked by descending cosine to the target.

        Routes the matched cluster to any clicked destination through the
        same calibration that previously enriched per-edge metadata at fit
        time, with no path traversal. For self-routes (`source_id ==
        target_id`) the filter collapses to credentials with high affinity
        to that cluster on both axes.

        Args:
            source_id : Cluster ID the user is matched against.
            target_id : Cluster ID the user clicked, may equal source.
        """
        creds, _ = self.credential_matrix
        if not creds:
            return []

        similarity       = self.credential_similarity
        s_idx            = self.clusters.cluster_index[source_id]
        t_idx            = self.clusters.cluster_index[target_id]
        dest_threshold   = np.percentile(
            similarity[:, t_idx], 100 - self.destination_percentile
        )
        source_threshold = np.percentile(similarity, self.source_percentile)
        passing          = np.flatnonzero(
            (similarity[:, t_idx] >= dest_threshold) &
            (similarity[:, s_idx] >= source_threshold)
        )
        ranked = passing[np.argsort(-similarity[passing, t_idx])]
        return [creds[i] for i in ranked]

    def hops_from(self, source: int) -> dict[int, int]:
        """
        BFS hop distance from `source` to every reachable cluster.

        Operates on the undirected projection of the directed pathway graph
        so distance reflects connectivity rather than edge direction. The
        map widget uses these distances to fade nodes by their proximity to
        the matched cluster.
        """
        return nx.single_source_shortest_path_length(self.undirected_graph, source)

    def reach(self, cluster_id: int) -> Reach:
        """
        Local reach exploration from a given cluster.

        Returns advancement paths (edges to higher Job Zone clusters) and
        lateral pivots (edges to same Job Zone clusters) with their per-edge
        credential metadata sorted by edge weight.
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

