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
from chalkline.pipeline.schemas import ClusterProfile, PipelineConfig


@dataclass(kw_only=True)
class CareerPathwayGraph:
    """
    Directed weighted career graph with per-edge credential enrichment.

    Accepts cluster centroids in both reduced and full embedding spaces,
    credential embeddings, and Job Zone assignments. Builds a stepwise
    k-NN backbone with bidirectional lateral edges and unidirectional
    upward edges, then enriches each edge with credentials filtered by
    destination selectivity and source relevance thresholds.

    Args:
        centroids          : (n_clusters, n_components) in SVD-reduced space.
        cluster_vectors    : (n_clusters, embedding_dim) L2-normalized.
        config             : Pipeline hyperparameters for graph construction.
        credential_labels  : Parallel with `credential_vectors`.
        credential_types   : "apprenticeship", "program", or "certification".
        credential_vectors : (n_credentials, embedding_dim) L2-normalized.
        job_zone_map       : Cluster ID → Job Zone.
        profiles           : For node construction and neighborhood display.
    """

    centroids          : np.ndarray
    cluster_vectors    : np.ndarray
    config             : PipelineConfig
    credential_labels  : list[str]
    credential_types   : list[str]
    credential_vectors : np.ndarray
    job_zone_map       : dict[int, int]
    profiles           : dict[int, ClusterProfile]

    graph: nx.DiGraph = field(init=False)

    def __post_init__(self):
        """
        Compute similarity matrices, build the stepwise backbone, and
        attach credential metadata to every edge.
        """
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(
            (c, self.profiles[c].model_dump(mode="json"))
            for c in sorted(self.profiles)
        )

        self._add_edges(cosine_similarity(self.centroids))
        self._enrich_edges(
            cosine_similarity(self.credential_vectors, self.cluster_vectors)
        )

    def _add_edges(self, pairwise: np.ndarray):
        """
        Add stepwise k-NN backbone edges to the graph.

        Each cluster gets `lateral_neighbors` bidirectional edges to
        its most similar same-JZ clusters and `upward_neighbors`
        unidirectional edges to its most similar clusters at the next
        JZ level. The stepwise constraint prevents tier-skipping
        shortcuts.
        """
        similarity  = pairwise - np.eye(len(pairwise))
        ids         = np.array(sorted(self.profiles))
        zones       = np.array([self.job_zone_map[c] for c in ids])
        zone_levels = sorted(set(zones))
        next_zone   = dict(zip(zone_levels, zone_levels[1:]))

        for source in ids:
            source_zone = self.job_zone_map[source]
            lateral     = ids[(zones == source_zone) & (ids > source)]
            proximity   = similarity[source, lateral]

            for i in np.argsort(-proximity)[:self.config.lateral_neighbors]:
                self.graph.add_edge(source, lateral[i], weight=proximity[i])
                self.graph.add_edge(lateral[i], source, weight=proximity[i])

            if source_zone in next_zone:
                upward    = ids[zones == next_zone[source_zone]]
                proximity = similarity[source, upward]
                
                for i in np.argsort(-proximity)[:self.config.upward_neighbors]:
                    self.graph.add_edge(source, upward[i], weight=proximity[i])

    def _enrich_edges(self, credential_similarity: np.ndarray):
        """
        Attach per-edge credential metadata using the dual-threshold
        filter.

        For each edge (source → target), a credential 𝐜 passes when
        both conditions hold:

            cos(𝐜, 𝐭) ≥ P₁₀₀₋ₚ(cos(·, 𝐭))
            cos(𝐜, 𝐬) ≥ Pₛ(cos(·, ·))

        The first threshold selects credentials with high destination
        affinity. The second ensures source relevance against the
        global credential-to-cluster distribution.
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

        types  = np.array(self.credential_types)
        labels = np.array(self.credential_labels, dtype=object)

        for s, t, edge_data in self.graph.edges(data=True):
            affinity = credential_similarity[:, t]
            passing  = np.flatnonzero((affinity >= affinity_floors[t]) & mask[:, s])
            ranked   = passing[np.argsort(-affinity[passing])]
            kinds    = types[ranked]

            edge_data.update({
                f"{kind}s": labels[ranked[kinds == kind]].tolist()
                for kind in ("apprenticeship", "certification", "program")
            })

    def neighborhood(self, cluster_id: int) -> Neighborhood:
        """
        Local neighborhood exploration from a given cluster.

        Returns advancement paths (edges to higher JZ clusters) and
        lateral pivots (edges to same JZ clusters) with their per-edge
        credential metadata sorted by edge weight.
        """
        edges = [
            CareerEdge(
                cluster_id  = target,
                modal_title = (profile := self.profiles[target]).modal_title,
                size        = profile.size,
                soc_title   = profile.soc_title,
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
