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

from chalkline.pathways.schemas import CareerEdge, Clusters
from chalkline.pathways.schemas import Credential, Neighborhood


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

    graph        : nx.DiGraph     = field(init=False)
    job_zone_map : dict[int, int] = field(init=False)
    node_ids     : np.ndarray     = field(init=False)

    def __post_init__(self):
        """
        Compute similarity matrices, build the stepwise backbone, and attach
        credential metadata to every edge.
        """
        self.node_ids     = np.array(self.clusters.cluster_ids)
        self.job_zone_map = {
            cid: self.clusters[cid].job_zone
            for cid in self.clusters
        }
        self.graph = nx.DiGraph()

        self.graph.add_nodes_from(
            (cid, {
                "cluster_id"  : c.cluster_id,
                "job_zone"    : c.job_zone,
                "modal_title" : c.modal_title,
                "sector"      : c.sector,
                "size"        : c.size,
                "soc_title"   : c.soc_title
            })
            for cid, c in sorted(self.clusters.items.items())
        )

        self._add_edges(cosine_similarity(self.clusters.centroids))
        self._enrich_edges(cosine_similarity(
            np.array([c.vector for c in self.credentials if c.vector]),
            self.clusters.vectors
        ))

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
            zone      = self.job_zone_map[source]
            lateral   = self.node_ids[(zones == zone) & (self.node_ids > source)]
            proximity = similarity[source, lateral]

            for i in np.argsort(-proximity)[:self.lateral_neighbors]:
                self.graph.add_edge(source, lateral[i], weight=proximity[i])
                self.graph.add_edge(lateral[i], source, weight=proximity[i])

            if zone in next_zone:
                upward    = self.node_ids[zones == next_zone[zone]]
                proximity = similarity[source, upward]

                for i in np.argsort(-proximity)[:self.upward_neighbors]:
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
            q = self.source_percentile
        )
        affinity_floors = np.percentile(
            a    = credential_similarity,
            q    = 100 - self.destination_percentile,
            axis = 0
        )

        records = np.array(self.credentials, dtype=object)

        for s, t, edge_data in self.graph.edges(data=True):
            affinity = credential_similarity[:, t]
            passing  = np.flatnonzero(
                (affinity >= affinity_floors[t]) & mask[:, s]
            )
            ranked = passing[np.argsort(-affinity[passing])]
            edge_data["credentials"] = [
                r.model_dump(mode="json")
                for r in records[ranked]
            ]

    def neighborhood(self, cluster_id: int) -> Neighborhood:
        """
        Local neighborhood exploration from a given cluster.

        Returns advancement paths (edges to higher JZ clusters) and lateral
        pivots (edges to same JZ clusters) with their per-edge credential
        metadata sorted by edge weight.
        """
        edges = [
            CareerEdge(
                cluster_id = target,
                **self.graph[cluster_id][target]
            )
            for target in self.graph.successors(cluster_id)
        ]

        ranked = lambda compare: sorted(
            [e for e in edges if compare(
                self.job_zone_map[e.cluster_id],
                self.job_zone_map[cluster_id]
            )],
            key     = lambda edge: edge.weight,
            reverse = True
        )

        return Neighborhood(
            advancement = ranked(gt),
            lateral     = ranked(eq)
        )
