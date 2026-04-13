"""
Validate career pathway graph construction and credential enrichment from
synthetic embedding fixtures.

Tests focus on invariants that would silently corrupt downstream
reach exploration if broken, including edge directionality, Job Zone
ordering, backbone connectivity, and credential metadata attachment.
"""

import numpy as np

from networkx import number_weakly_connected_components

from chalkline.pathways.graph   import CareerPathwayGraph
from chalkline.pathways.schemas import Reach


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

    def test_coverage_empty_labels(self, pathway_graph: CareerPathwayGraph):
        """
        Empty route labels produce an empty coverage mapping.
        """
        assert pathway_graph.credential_coverage([], np.empty((0, 4))) == {}

    def test_coverage_matching_labels(self, pathway_graph: CareerPathwayGraph):
        """
        Matching route labels produce a dict keyed by label with set
        values.
        """
        cluster = pathway_graph.clusters[pathway_graph.clusters.cluster_ids[0]]
        labels  = [c.label for c in pathway_graph.credential_matrix[0]][:3]
        vectors = np.stack([t.vector for t in cluster.tasks])
        result  = pathway_graph.credential_coverage(labels, vectors)
        assert len(result) == 3
        assert all(isinstance(s, set) for s in result.values())
        assert any(result.values())

    def test_credential_metadata(self, pathway_graph: CareerPathwayGraph):
        """
        Every edge carries a credentials list of serialized dicts
        with required schema fields and the vector excluded.
        """
        for _, _, data in pathway_graph.graph.edges(data=True):
            assert isinstance(data["credentials"], list)
            for cred in data["credentials"]:
                assert {"embedding_text", "kind", "label"} <= cred.keys()
                assert "vector" not in cred

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

    def test_stepping_stone_none(self, pathway_graph: CareerPathwayGraph):
        """
        Returns None when an empty reach provides no candidates.
        """
        cluster = pathway_graph.clusters[pathway_graph.clusters.cluster_ids[0]]
        task_vectors = np.stack([t.vector for t in cluster.tasks])
        assert pathway_graph.stepping_stone(cluster, Reach(), task_vectors) is None

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
        cid = cluster_ids[0]
        assert pathway_graph.try_widest_path(cid, cid) == [cid]
