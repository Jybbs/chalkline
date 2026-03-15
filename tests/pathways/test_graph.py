"""
Validate career pathway graph construction, enrichment, and export from
the synthetic 20-posting fixture chain.

Tests focus on invariants that would silently corrupt downstream routing
and career reports if broken, rather than confirming library guarantees
or fixture shape.
"""

from json                          import loads
from networkx                      import is_directed_acyclic_graph
from networkx.readwrite.json_graph import node_link_graph
from pathlib                       import Path

from chalkline.association.cooccurrence import CooccurrenceNetwork
from chalkline.clustering.hierarchical  import HierarchicalClusterer
from chalkline.pathways.graph           import CareerPathwayGraph


class TestCareerPathwayGraph:
    """
    Validate career pathway graph construction, enrichment, and export
    from the synthetic 20-posting fixture chain.
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

    def test_alignment_ari_bounded(
        self,
        pathway_graph: CareerPathwayGraph
    ):
        """
        ARI is bounded in [-1, 1].
        """
        assert -1.0 <= pathway_graph.alignment.ari <= 1.0

    def test_modularity_computed(
        self,
        pathway_graph: CareerPathwayGraph
    ):
        """
        Louvain modularity is computed when the skill graph has edges.
        """
        if pathway_graph.network.graph().number_of_edges() > 0:
            assert pathway_graph.alignment.modularity is not None

    def test_longest_path_empty(self, network: CooccurrenceNetwork):
        """
        An empty graph returns a zero-weight empty path rather than
        raising from `dag_longest_path`.
        """
        empty = CareerPathwayGraph(network = network, profiles = {})
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

    def test_skill_to_cluster(
        self,
        pathway_graph: CareerPathwayGraph
    ):
        """
        Every skill in every profile maps to its owning cluster ID,
        with last-write-wins for skills shared across clusters.
        """
        for cid, profile in pathway_graph.profiles.items():
            for skill in profile.skills:
                assert skill in pathway_graph.skill_to_cluster

    def test_export_creates_files(
        self,
        pathway_graph : CareerPathwayGraph,
        tmp_path      : Path
    ):
        """
        Export creates both GraphML and JSON files.
        """
        result = pathway_graph.export(tmp_path / "export")
        assert result.graphml_path.exists()
        assert result.json_path.exists()

    def test_json_roundtrip(
        self,
        pathway_graph : CareerPathwayGraph,
        tmp_path      : Path
    ):
        """
        JSON roundtrip via `node_link_data` / `node_link_graph` preserves
        node count, edge count, and nested program lists.
        """
        G = node_link_graph(loads(
            pathway_graph.export(tmp_path / "export").json_path.read_text()
        ))

        assert G.number_of_nodes() == pathway_graph.graph.number_of_nodes()
        assert G.number_of_edges() == pathway_graph.graph.number_of_edges()

        for node in pathway_graph.graph:
            assert len(G.nodes[node].get("programs", [])) == len(
                pathway_graph.graph.nodes[node].get("programs", [])
            )
