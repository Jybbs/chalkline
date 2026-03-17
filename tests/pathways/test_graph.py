"""
Validate career pathway graph construction and enrichment from the
synthetic 20-posting fixture chain.

Tests focus on invariants that would silently corrupt downstream routing
and career reports if broken, rather than confirming library guarantees
or fixture shape.
"""

from networkx import is_directed_acyclic_graph

from chalkline.association.cooccurrence import CooccurrenceNetwork
from chalkline.clustering.hierarchical  import HierarchicalClusterer
from chalkline.pathways.graph           import CareerPathwayGraph


class TestCareerPathwayGraph:
    """
    Validate career pathway graph construction and enrichment from
    the synthetic 20-posting fixture chain.
    """

    def test_is_acyclic(self, pathway_graph: CareerPathwayGraph):
        """
        Total ordering on (Job Zone, cluster ID) guarantees acyclicity.
        """
        assert is_directed_acyclic_graph(pathway_graph.graph)

    def test_node_count(self, pathway_graph: CareerPathwayGraph):
        """
        One node per HAC cluster.
        """
        assert pathway_graph.graph.number_of_nodes() == len(
            pathway_graph.profiles
        )

    def test_cluster_skills_union(
        self,
        clusterer        : HierarchicalClusterer,
        extracted_skills : dict[str, list[str]],
        pathway_graph    : CareerPathwayGraph
    ):
        """
        Each node's skill set is the union of its constituent postings'
        canonical skills. A bug here silently corrupts Job Zone
        assignment, edge weights, and enrichments.
        """
        for node, data in pathway_graph.graph.nodes(data=True):
            expected = {
                skill
                for doc, cid in zip(
                    clusterer.document_ids, clusterer.assignments
                )
                if cid == node
                for skill in extracted_skills.get(doc, [])
            }
            assert set(data["skills"]) == expected

    def test_edge_attrs_complete(
        self,
        pathway_graph: CareerPathwayGraph
    ):
        """
        Every edge carries positive `weight` and `direction_source`.
        """
        for _, _, data in pathway_graph.graph.edges(data=True):
            assert data["weight"] > 0
            assert data["direction_source"] == "job_zone"

    def test_edge_job_zone_order(
        self,
        pathway_graph: CareerPathwayGraph
    ):
        """
        Edge direction follows Job Zone ordering: source Job Zone is
        less than or equal to target Job Zone.
        """
        for source, target in pathway_graph.graph.edges():
            assert (pathway_graph.graph.nodes[source]["job_zone"]
                    <= pathway_graph.graph.nodes[target]["job_zone"])

    def test_longest_path_empty(self, network: CooccurrenceNetwork):
        """
        An empty graph returns a zero-weight empty path rather than
        raising from `dag_longest_path`.
        """
        empty = CareerPathwayGraph(
            network  = network,
            profiles = {}
        )
        assert empty.longest_path.path == []
        assert empty.longest_path.path_weight == 0.0

    def test_longest_path_weight(
        self,
        pathway_graph: CareerPathwayGraph
    ):
        """
        The path weight equals the sum of edge weights along the
        longest path.
        """
        lp   = pathway_graph.longest_path
        path = lp.path
        assert abs(lp.path_weight - sum(
            pathway_graph.graph.edges[u, v].get("weight", 0)
            for u, v in zip(path, path[1:])
        )) < 1e-10

