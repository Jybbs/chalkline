"""
Career pathway graph from sentence-embedding cluster centroids.

Constructs a directed weighted graph where nodes are career clusters from
Ward-linkage HAC on sentence embeddings and edges connect clusters via
stepwise k-NN in cosine similarity space. Credentials are computed per
target on demand via `credentials_for`, applying a destination-affinity
percentile filter to the cluster-credential cosine matrix so each route sees
the candidate set most aligned with where the user is going.
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
    credential embeddings, and graph construction hyperparameters. Builds a
    stepwise k-NN backbone with bidirectional lateral edges and
    unidirectional upward edges. Per-route credential lists come from
    `credentials_for(target_id)` at click time, keeping credentials whose
    similarity to the target cluster sits in the top
    `destination_percentile`.

    Args:
        clusters               : Cluster map with centroids and vectors.
        credentials            : Typed records with aligned embedding vectors.
        destination_percentile : Top-p threshold for destination affinity.
        lateral_neighbors      : k for same wage-tier bidirectional edges.
        rrf_k                  : RRF damping constant (Cormack 2009 default).
        upward_neighbors       : k for next wage-tier unidirectional edges.
    """

    clusters               : Clusters
    credentials            : list[Credential]
    destination_percentile : int
    lateral_neighbors      : int
    rrf_k                  : int
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
    def credential_pool(self) -> list[Credential]:
        """
        Credentials with non-null vectors, the subset usable for similarity
        matching. Fixed at fit time and shared across every consumer that
        needs credential identity (calibrate_coverage, credentials_for, the
        display-layer `RelevantCredentials` factory).
        """
        return [c for c in self.credentials if c.vector]

    @cached_property
    def credential_similarity(self) -> np.ndarray:
        """
        Cosine similarity between every credential vector and every cluster
        vector, shape `(n_credentials, n_clusters)`. Row axis aligned with
        `credential_pool`; column axis with `clusters.cluster_ids`. Computed
        once per fit and shared between `credentials_for` and the display
        layer so neither has to rerun the cosine sweep per render.
        """
        if not self.credential_pool:
            return np.empty((0, len(self.clusters.cluster_ids)))
        return cosine_similarity(self.credential_vectors, self.clusters.vectors)

    @cached_property
    def credential_task_maxsim(self) -> dict[int, np.ndarray]:
        """
        Max cosine similarity between each credential and each cluster's
        O*NET task vectors, keyed by cluster id. Used as the second
        ranking signal in the RRF candidate filter so destination-aligned
        credentials surface alongside centroid-aligned ones.

        Only clusters whose nearest occupation contributed Task or DWA
        elements appear in the dict; clusters without tasks fall back to
        the centroid signal alone in `credentials_for`.
        """
        if not self.credential_pool:
            return {}
        return {
            cid: cosine_similarity(
                self.credential_vectors, self.clusters[cid].task_matrix
            ).max(axis=1)
            for cid in self.clusters.cluster_ids
            if self.clusters[cid].tasks
        }

    @cached_property
    def credential_vectors(self) -> np.ndarray:
        """
        Row-stacked vectors aligned with `credential_pool`, cached so the
        np.array copy happens once per session.
        """
        return np.array([c.vector for c in self.credential_pool])

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
        Stepwise k-NN DiGraph backbone built lazily on first access. Edges
        carry only cosine-similarity weights. Credentials are computed on
        demand per target via `credentials_for`, not attached at fit time.
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
        Undirected projection of the directed pathway graph, cached so the
        map widget's `hops_from` BFS materializes the projection once per
        session.
        """
        return self.graph.to_undirected()

    def _add_edges(self, g: nx.DiGraph, similarity: np.ndarray):
        """
        Add stepwise k-NN backbone edges to `g`.

        Each cluster gets `lateral_neighbors` bidirectional edges to its
        most similar clusters at the same wage tier and `upward_neighbors`
        unidirectional edges to its most similar clusters at the next wage
        tier. The stepwise constraint prevents tier-skipping shortcuts.
        """
        tiers     = self.clusters.wage_tier_map
        per_node  = np.array([tiers[c] for c in self.node_ids])
        next_tier = dict(pairwise(sorted(set(per_node))))

        for source in self.node_ids:
            tier      = tiers[source]
            lateral   = self.node_ids[(per_node == tier) & (self.node_ids > source)]
            proximity = similarity[source, lateral]

            for i in np.argsort(-proximity)[:self.lateral_neighbors]:
                g.add_edge(int(source), int(lateral[i]), weight=float(proximity[i]))
                g.add_edge(int(lateral[i]), int(source), weight=float(proximity[i]))

            if tier in next_tier:
                upward    = self.node_ids[per_node == next_tier[tier]]
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

    def credentials_for(self, target_id: int) -> list[Credential]:
        """
        Credentials aligned with a specific destination, filtered via
        Reciprocal Rank Fusion over centroid cosine and destination-task
        MaxSim rankings, then gated to the top `destination_percentile` of
        the fused score and returned in descending order.

        RRF (Cormack, Clarke, Buettcher 2009) absorbs the score-scale
        mismatch between centroid cosine and task MaxSim without
        calibration, letting both signals contribute on equal footing.
        Falls back to centroid-only ranking when the destination has no
        task elements.

        Args:
            target_id: Cluster ID the user clicked, may equal the matched cluster.
        """
        if not (creds := self.credential_pool):
            return []

        t_idx          = self.clusters.cluster_index[target_id]
        centroid_score = self.credential_similarity[:, t_idx]
        task_score     = self.credential_task_maxsim.get(target_id)

        if task_score is None:
            scores = centroid_score
        else:
            rank_centroid = (-centroid_score).argsort().argsort()
            rank_task     = (-task_score).argsort().argsort()
            scores        = (1.0 / (self.rrf_k + rank_centroid)
                           + 1.0 / (self.rrf_k + rank_task))

        threshold = np.percentile(scores, 100 - self.destination_percentile)
        passing   = np.flatnonzero(scores >= threshold)
        ranked    = passing[np.argsort(-scores[passing])]
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

        Returns advancement paths (edges to higher wage-tier clusters) and
        lateral pivots (edges to same wage-tier clusters) with their per-edge
        credential metadata sorted by edge weight.
        """
        tiers = self.clusters.wage_tier_map
        tier  = tiers[cluster_id]
        edges = sorted(
            (self._edge(cluster_id, t) for t in self.graph.successors(cluster_id)),
            key     = attrgetter("weight"),
            reverse = True
        )
        return Reach(
            advancement = [e for e in edges if tiers[e.cluster_id] >  tier],
            lateral     = [e for e in edges if tiers[e.cluster_id] == tier]
        )

