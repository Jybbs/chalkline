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

from loguru                   import logger
from sklearn.metrics.pairwise import cosine_similarity

from chalkline.matching.schemas import CareerEdge, Neighborhood
from chalkline.pipeline.schemas import ClusterProfile, PipelineConfig


class CareerPathwayGraph:
    """
    Directed weighted career graph with per-edge credential enrichment.

    Accepts cluster centroids in both reduced and full embedding spaces,
    credential embeddings, and Job Zone assignments. Builds a stepwise k-NN
    backbone with bidirectional lateral edges and unidirectional upward
    edges, then enriches each edge with credentials filtered by destination
    selectivity and source relevance thresholds.
    """

    def __init__(
        self,
        cluster_centroids  : np.ndarray,
        cluster_vectors    : np.ndarray,
        config             : PipelineConfig,
        credential_labels  : list[str],
        credential_types   : list[str],
        credential_vectors : np.ndarray,
        job_zone_map       : dict[int, int],
        profiles           : dict[int, ClusterProfile]
    ):
        """
        Build the career pathway graph with credential-enriched edges.

        Args:
            cluster_centroids  : (n_clusters, n_components) in SVD-reduced space.
            cluster_vectors    : (n_clusters, embedding_dim) L2-normalized.
            config             : Pipeline hyperparameters for graph construction.
            credential_labels  : Parallel with `credential_vectors`.
            credential_types   : "apprenticeship", "program", or "certification".
            credential_vectors : (n_credentials, embedding_dim) L2-normalized.
            job_zone_map       : Cluster ID → Job Zone.
            profiles           : For node construction and neighborhood display.
        """
        self.cluster_centroids  = cluster_centroids
        self.cluster_vectors    = cluster_vectors
        self.config             = config
        self.credential_labels  = credential_labels
        self.credential_types   = credential_types
        self.credential_vectors = credential_vectors
        self.job_zone_map       = job_zone_map
        self.profiles           = profiles

        self.cluster_sim = cosine_similarity(cluster_centroids)
        np.fill_diagonal(self.cluster_sim, 0)

        self.credential_similarity = cosine_similarity(
            credential_vectors, cluster_vectors
        )

        self.graph = self._build_backbone()
        self._enrich_edges()

        logger.info(
            f"Career graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

    def _build_backbone(self) -> nx.DiGraph:
        """
        Construct the stepwise k-NN backbone.

        Each cluster gets `lateral_neighbors` bidirectional edges to
        its most similar same-JZ clusters and `upward_neighbors`
        unidirectional edges to its most similar clusters at the next
        JZ level. The stepwise constraint prevents tier-skipping
        shortcuts.
        """
        cluster_ids = sorted(self.profiles)
        graph       = nx.DiGraph()
        graph.add_nodes_from(
            (cluster_id, self.profiles[cluster_id].model_dump(mode="json"))
            for cluster_id in cluster_ids
        )

        job_zone_levels = sorted(set(self.job_zone_map.values()))
        next_zone       = dict(zip(job_zone_levels, job_zone_levels[1:]))

        for source in cluster_ids:
            source_zone = self.job_zone_map[source]

            same_zone = [
                target for target in cluster_ids
                if self.job_zone_map[target] == source_zone
                and target > source
            ]
            same_zone.sort(
                key     = lambda target: self.cluster_sim[source, target],
                reverse = True
            )
            for target in same_zone[:self.config.lateral_neighbors]:
                weight = self.cluster_sim[source, target]
                graph.add_edge(source, target, weight=weight)
                graph.add_edge(target, source, weight=weight)

            if source_zone in next_zone:
                upward = [
                    target for target in cluster_ids
                    if self.job_zone_map[target] == next_zone[source_zone]
                ]
                upward.sort(
                    key     = lambda target: self.cluster_sim[source, target],
                    reverse = True
                )
                for target in upward[:self.config.upward_neighbors]:
                    weight = self.cluster_sim[source, target]
                    graph.add_edge(source, target, weight=weight)

        return graph

    def _enrich_edges(self):
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
        all_similarities = self.credential_similarity.flatten()
        source_floor = np.percentile(
            all_similarities, self.config.source_percentile
        )

        for source, target, edge_data in self.graph.edges(data=True):
            destination_threshold = np.percentile(
                self.credential_similarity[:, target], 100 - self.config.destination_percentile
            )

            dest_column = self.credential_similarity[:, target]
            src_column  = self.credential_similarity[:, source]
            passing_mask = (
                (dest_column >= destination_threshold)
                & (src_column >= source_floor)
            )
            passing_indices = np.flatnonzero(passing_mask)
            ranked_indices = passing_indices[
                np.argsort(dest_column[passing_indices])[::-1]
            ]

            apprenticeships = []
            certifications  = []
            programs        = []

            for idx in ranked_indices:
                label = self.credential_labels[idx]
                match self.credential_types[idx]:
                    case "apprenticeship" : apprenticeships.append(label)
                    case "certification"  : certifications.append(label)
                    case "program"        : programs.append(label)

            edge_data["apprenticeships"] = apprenticeships
            edge_data["certifications"]  = certifications
            edge_data["programs"]        = programs

    def neighborhood(self, cluster_id: int) -> Neighborhood:
        """
        Local neighborhood exploration from a given cluster.

        Returns advancement paths (edges to higher JZ clusters) and lateral
        pivots (edges to same JZ clusters) with their per-edge credential
        metadata sorted by edge weight.
        """
        source_zone = self.job_zone_map[cluster_id]
        advancement = []
        lateral     = []

        for target in self.graph.successors(cluster_id):
            edge_data   = self.graph[cluster_id][target]
            target_zone = self.job_zone_map[target]
            profile     = self.profiles[target]

            career_edge = CareerEdge(
                apprenticeships = edge_data.get("apprenticeships", []),
                certifications  = edge_data.get("certifications", []),
                cluster_id      = target,
                modal_title     = profile.modal_title,
                programs        = edge_data.get("programs", []),
                size            = profile.size,
                soc_title       = profile.soc_title,
                weight          = edge_data["weight"]
            )

            if target_zone > source_zone:
                advancement.append(career_edge)
            elif target_zone == source_zone:
                lateral.append(career_edge)

        advancement.sort(key=lambda e: -e.weight)
        lateral.sort(key=lambda e: -e.weight)

        return Neighborhood(
            advancement = advancement,
            lateral     = lateral
        )
