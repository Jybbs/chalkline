"""
Validate career pathway graph construction and per-pair credential
filtering from synthetic embedding fixtures.

Tests focus on invariants that would silently corrupt downstream
reach exploration if broken, including edge directionality, Job Zone
ordering, backbone connectivity, and the dual-threshold credential
filter.
"""

import numpy as np

from networkx import number_weakly_connected_components

from chalkline.pathways.clusters import Clusters
from chalkline.pathways.graph    import CareerPathwayGraph


class TestCareerPathwayGraph:
    """
    Structural invariants of the stepwise k-NN career graph.
    """

    def test_backbone_connected(self, pathway_graph: CareerPathwayGraph):
        """
        The stepwise backbone must produce exactly one weakly connected
        component so every cluster is reachable.
        """
        assert number_weakly_connected_components(pathway_graph.graph) == 1

    def test_brokerage_sorted(self, pathway_graph: CareerPathwayGraph):
        """
        Brokerage scores are sorted descending by centrality value.
        """
        scores = [score for _, score in pathway_graph.brokerage]
        assert scores == sorted(scores, reverse=True)

    def test_credentials_for_filters_by_threshold(
        self,
        cluster_ids   : list[int],
        pathway_graph : CareerPathwayGraph
    ):
        """
        Per-pair credential filter returns Credential instances ranked
        by descending similarity to the target cluster.
        """
        source, target = cluster_ids[0], cluster_ids[-1]
        credentials    = pathway_graph.credentials_for(source, target)
        assert all(c.vector for c in credentials)
        if len(credentials) >= 2:
            similarity = pathway_graph.credential_similarity
            t_idx      = pathway_graph.clusters.cluster_index[target]
            creds_pool = pathway_graph.credential_matrix[0]
            scores     = [
                similarity[creds_pool.index(c), t_idx] for c in credentials
            ]
            assert scores == sorted(scores, reverse=True)

    def test_edge_weights_bounded(self, pathway_graph: CareerPathwayGraph):
        """
        All edge weights are valid cosine similarities in [-1, 1].
        """
        for w in pathway_graph.edge_weights:
            assert -1 <= w <= 1

    def test_hops_from_self_distance(
        self,
        cluster_ids   : list[int],
        pathway_graph : CareerPathwayGraph
    ):
        """
        Hops from a node to itself is zero, and all reachable nodes
        carry non-negative integer distances.
        """
        distances = pathway_graph.hops_from(cluster_ids[0])
        assert distances[cluster_ids[0]] == 0
        assert all(d >= 0 for d in distances.values())

    def test_no_credentials_builds_graph(self, clusters: Clusters):
        """
        A graph with no credentials still builds a valid backbone, and
        the per-pair filter returns empty for any pair.
        """
        graph = CareerPathwayGraph(
            clusters               = clusters,
            credentials            = [],
            destination_percentile = 5,
            lateral_neighbors      = 2,
            source_percentile      = 75,
            upward_neighbors       = 2
        )
        assert graph.graph.number_of_edges() > 0
        ids = clusters.cluster_ids
        assert graph.credentials_for(ids[0], ids[-1]) == []

    def test_reach_types(
        self,
        job_zone_map  : dict[int, int],
        pathway_graph : CareerPathwayGraph
    ):
        """
        Reach advancement edges point to higher Job Zone clusters and
        lateral edges point to same Job Zone clusters.
        """
        for cluster_id in pathway_graph.clusters:
            reach       = pathway_graph.reach(cluster_id)
            source_zone = job_zone_map[cluster_id]
            for edge in reach.advancement:
                assert job_zone_map[edge.cluster_id] > source_zone
            for edge in reach.lateral:
                assert job_zone_map[edge.cluster_id] == source_zone

    def test_reach_weight_order(self, pathway_graph: CareerPathwayGraph):
        """
        Advancement and lateral edges are sorted descending by cosine
        similarity weight.
        """
        for cluster_id in pathway_graph.clusters:
            reach = pathway_graph.reach(cluster_id)
            for edges in (reach.advancement, reach.lateral):
                weights = [e.weight for e in edges]
                assert weights == sorted(weights, reverse=True)

    def test_upward_stepwise(
        self,
        job_zone_map  : dict[int, int],
        pathway_graph : CareerPathwayGraph
    ):
        """
        Upward edges connect only to the next Job Zone level, never
        skipping tiers.
        """
        levels    = sorted(set(job_zone_map.values()))
        next_zone = dict(zip(levels, levels[1:]))

        for source, target in pathway_graph.graph.edges():
            source_zone = job_zone_map[source]
            target_zone = job_zone_map[target]
            if target_zone > source_zone:
                assert target_zone == next_zone[source_zone]


