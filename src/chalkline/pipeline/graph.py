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

from dataclasses              import dataclass, field
from operator                 import eq, gt
from sklearn.metrics.pairwise import cosine_similarity

from chalkline.matching.schemas import CareerEdge, Neighborhood
from chalkline.pipeline.schemas import ClusterProfile, Credentials, PipelineConfig


@dataclass(kw_only=True)
class CareerPathwayGraph:
    """
    Directed weighted career graph with per-edge credential enrichment.

    Accepts cluster centroids in both reduced and full embedding spaces,
    credential embeddings, and Job Zone assignments. Builds a stepwise k-NN
    backbone with bidirectional lateral edges and unidirectional upward
    edges, then enriches each edge with credentials filtered by destination
    selectivity and source relevance thresholds.

    Args:
        centroids       : (n_clusters, n_components) in SVD-reduced space.
        cluster_vectors : (n_clusters, embedding_dim) L2-normalized.
        config          : Pipeline hyperparameters for graph construction.
        credentials     : Typed records with aligned embedding vectors.
        job_zone_map    : Cluster ID → Job Zone.
        profiles        : For node construction and neighborhood display.
    """

    centroids       : np.ndarray
    cluster_vectors : np.ndarray
    config          : PipelineConfig
    credentials     : Credentials
    job_zone_map    : dict[int, int]
    profiles        : dict[int, ClusterProfile]

    graph    : nx.DiGraph = field(init=False)
    node_ids : np.ndarray = field(init=False)

    def __post_init__(self):
        """
        Compute similarity matrices, build the stepwise backbone, and attach
        credential metadata to every edge.
        """
        self.node_ids = np.array(sorted(self.profiles))
        self.graph    = nx.DiGraph()

        self.graph.add_nodes_from(
            (c, self.profiles[c].model_dump(mode="json"))
            for c in self.node_ids
        )

        self._add_edges(cosine_similarity(self.centroids))
        self._enrich_edges(
            cosine_similarity(self.credentials.vectors, self.cluster_vectors)
        )

    @property
    def edge_count(self) -> int:
        """
        Number of edges in the career pathway graph.
        """
        return self.graph.number_of_edges()

    def _add_edges(self, pairwise: np.ndarray):
        """
        Add stepwise k-NN backbone edges to the graph.

        Each cluster gets `lateral_neighbors` bidirectional edges to its
        most similar same-JZ clusters and `upward_neighbors` unidirectional
        edges to its most similar clusters at the next JZ level. The
        stepwise constraint prevents tier-skipping shortcuts.
        """
        similarity  = pairwise - np.eye(len(pairwise))
        zones       = np.array([self.job_zone_map[c] for c in self.node_ids])
        zone_levels = sorted(set(zones))
        next_zone   = dict(zip(zone_levels, zone_levels[1:]))

        for source in self.node_ids:
            source_zone = self.job_zone_map[source]
            lateral     = self.node_ids[(zones == source_zone) & (self.node_ids > source)]
            proximity   = similarity[source, lateral]

            for i in np.argsort(-proximity)[:self.config.lateral_neighbors]:
                self.graph.add_edge(source, lateral[i], weight=proximity[i])
                self.graph.add_edge(lateral[i], source, weight=proximity[i])

            if source_zone in next_zone:
                upward    = self.node_ids[zones == next_zone[source_zone]]
                proximity = similarity[source, upward]

                for i in np.argsort(-proximity)[:self.config.upward_neighbors]:
                    self.graph.add_edge(source, upward[i], weight=proximity[i])

    def _enrich_edges(self, credential_similarity: np.ndarray):
        """
        Attach per-edge credential metadata using the dual-threshold filter.

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
            q = self.config.source_percentile
        )
        affinity_floors = np.percentile(
            a    = credential_similarity,
            q    = 100 - self.config.destination_percentile,
            axis = 0
        )

        records = np.array(self.credentials.records, dtype=object)
        kinds   = np.array([r.credential_kind for r in self.credentials.records])

        for s, t, edge_data in self.graph.edges(data=True):
            affinity = credential_similarity[:, t]
            passing  = np.flatnonzero((affinity >= affinity_floors[t]) & mask[:, s])
            ranked   = passing[np.argsort(-affinity[passing])]
            matched  = kinds[ranked]

            edge_data.update({
                f"{kind}s": [
                    r.model_dump(mode="json")
                    for r in records[ranked[matched == kind]]
                ]
                for kind in ("apprenticeship", "certification", "program")
            })

    def neighborhood(self, cluster_id: int) -> Neighborhood:
        """
        Local neighborhood exploration from a given cluster.

        Returns advancement paths (edges to higher JZ clusters) and lateral
        pivots (edges to same JZ clusters) with their per-edge credential
        metadata sorted by edge weight.
        """
        edges = [
            CareerEdge(
                profile = self.profiles[target],
                **self.graph[cluster_id][target]
            )
            for target in self.graph.successors(cluster_id)
        ]

        ranked = lambda compare: sorted(
            [e for e in edges if compare(
                self.job_zone_map[e.profile.cluster_id],
                self.job_zone_map[cluster_id]
            )],
            key     = lambda edge: edge.weight,
            reverse = True
        )

        return Neighborhood(
            advancement = ranked(gt),
            lateral     = ranked(eq)
        )
