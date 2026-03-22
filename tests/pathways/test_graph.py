"""
Validate career pathway graph construction and credential enrichment from
synthetic embedding fixtures.

Tests focus on invariants that would silently corrupt downstream
neighborhood exploration if broken, including edge directionality, JZ
ordering, backbone connectivity, and credential metadata attachment.
"""

from networkx import number_weakly_connected_components

from chalkline.pathways.graph import CareerPathwayGraph


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

    def test_credential_metadata(self, pathway_graph: CareerPathwayGraph):
        """
        Every edge must carry a credentials list, even if empty.
        """
        for _, _, data in pathway_graph.graph.edges(data=True):
            assert "credentials" in data
            assert isinstance(data["credentials"], list)

    def test_edge_count_positive(self, pathway_graph: CareerPathwayGraph):
        """
        The graph must have at least one edge from the stepwise k-NN
        backbone.
        """
        assert pathway_graph.edge_count > 0

    def test_neighborhood_types(self, pathway_graph: CareerPathwayGraph):
        """
        Neighborhood advancement edges point to higher JZ clusters and
        lateral edges point to same JZ clusters.
        """
        for cluster_id in pathway_graph.clusters:
            neighborhood = pathway_graph.neighborhood(cluster_id)
            source_zone  = pathway_graph.job_zone_map[cluster_id]
            for edge in neighborhood.advancement:
                assert pathway_graph.job_zone_map[edge.cluster_id] > source_zone
            for edge in neighborhood.lateral:
                assert pathway_graph.job_zone_map[edge.cluster_id] == source_zone

    def test_node_count(self, pathway_graph: CareerPathwayGraph):
        """
        One graph node per cluster profile.
        """
        assert pathway_graph.graph.number_of_nodes() == len(pathway_graph.clusters)

    def test_upward_stepwise(self, pathway_graph: CareerPathwayGraph):
        """
        Upward edges connect only to the next JZ level, never skipping
        tiers.
        """
        job_zone_levels = sorted(set(pathway_graph.job_zone_map.values()))
        next_zone       = dict(zip(job_zone_levels, job_zone_levels[1:]))

        for source, target in pathway_graph.graph.edges():
            source_zone = pathway_graph.job_zone_map[source]
            target_zone = pathway_graph.job_zone_map[target]
            if target_zone > source_zone:
                assert target_zone == next_zone[source_zone]
