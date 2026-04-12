"""
Validate career pathway graph construction and credential enrichment from
synthetic embedding fixtures.

Tests focus on invariants that would silently corrupt downstream
reach exploration if broken, including edge directionality, JZ
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

    def test_path_edges_round_trip(
        self,
        cluster_ids   : list[int],
        pathway_graph : CareerPathwayGraph
    ):
        """
        Reconstructed CareerEdges along a widest path return one edge
        per hop with target cluster IDs matching the path tail.
        """
        path  = pathway_graph.try_widest_path(cluster_ids[0], cluster_ids[-1])
        edges = pathway_graph.path_edges(path)
        assert len(edges)                    == len(path) - 1
        assert [e.cluster_id for e in edges] == path[1:]

    def test_reach_types(
        self,
        job_zone_map  : dict[int, int],
        pathway_graph : CareerPathwayGraph
    ):
        """
        Reach advancement edges point to higher JZ clusters and lateral
        edges point to same JZ clusters.
        """
        for cluster_id in pathway_graph.clusters:
            reach       = pathway_graph.reach(cluster_id)
            source_zone = job_zone_map[cluster_id]
            for edge in reach.advancement:
                assert job_zone_map[edge.cluster_id] > source_zone
            for edge in reach.lateral:
                assert job_zone_map[edge.cluster_id] == source_zone

    def test_upward_stepwise(
        self,
        job_zone_map  : dict[int, int],
        pathway_graph : CareerPathwayGraph
    ):
        """
        Upward edges connect only to the next JZ level, never skipping
        tiers.
        """
        levels    = sorted(set(job_zone_map.values()))
        next_zone = dict(zip(levels, levels[1:]))

        for source, target in pathway_graph.graph.edges():
            source_zone = job_zone_map[source]
            target_zone = job_zone_map[target]
            if target_zone > source_zone:
                assert target_zone == next_zone[source_zone]

    def test_widest_path_endpoints(
        self,
        cluster_ids   : list[int],
        pathway_graph : CareerPathwayGraph
    ):
        """
        Widest path between connected nodes starts at source and ends
        at target with at least two entries.
        """
        path = pathway_graph.try_widest_path(cluster_ids[0], cluster_ids[-1])
        assert path[0] == cluster_ids[0]
        assert path[-1] == cluster_ids[-1]
        assert len(path) >= 2

    def test_widest_path_nonexistent_target(
        self,
        cluster_ids   : list[int],
        pathway_graph : CareerPathwayGraph
    ):
        """
        Querying a node absent from the graph returns an empty list.
        """
        assert pathway_graph.try_widest_path(cluster_ids[0], 9999) == []

    def test_widest_path_self_loop(
        self,
        cluster_ids   : list[int],
        pathway_graph : CareerPathwayGraph
    ):
        """
        Widest path from a node to itself returns a single-element
        list.
        """
        assert pathway_graph.try_widest_path(cluster_ids[0], cluster_ids[0]) == [
            cluster_ids[0]
        ]
