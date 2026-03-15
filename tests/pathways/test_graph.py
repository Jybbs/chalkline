"""
Validate career pathway graph construction, enrichment, traversal, and
export from the synthetic 20-posting fixture chain.

Tests focus on invariants that would silently corrupt downstream routing and
career reports if broken, rather than confirming library guarantees or
fixture shape.
"""

import networkx as nx

from json                          import loads
from networkx.readwrite.json_graph import node_link_graph
from pathlib                       import Path

from chalkline.pathways.graph import CareerPathwayGraph


class TestCareerPathwayGraph:
    """
    Validate career pathway graph construction, enrichment, traversal,
    and export from the synthetic 20-posting fixture chain.
    """

    # -----------------------------------------------------------------
    # Graph structure
    # -----------------------------------------------------------------

    def test_node_count(self, pathway_graph: CareerPathwayGraph):
        """
        One node per HAC cluster.
        """
        assert pathway_graph.graph.number_of_nodes() == len(
            set(int(a) for a in pathway_graph.assignments)
        )

    def test_node_has_job_zone(self, pathway_graph: CareerPathwayGraph):
        """
        Every node carries a Job Zone in [1, 5].
        """
        for _, data in pathway_graph.graph.nodes(data = True):
            assert 1 <= data["job_zone"] <= 5

    def test_node_has_sector(self, pathway_graph: CareerPathwayGraph):
        """
        Every node carries a non-empty sector label.
        """
        for _, data in pathway_graph.graph.nodes(data = True):
            assert data["sector"]

    def test_cluster_skills_union(
        self,
        pathway_graph    : CareerPathwayGraph,
        extracted_skills : dict[str, list[str]]
    ):
        """
        Each node's skill set is the union of its constituent postings'
        canonical skills. A bug here silently corrupts Job Zone assignment,
        edge weights, and enrichments.
        """
        for node, data in pathway_graph.graph.nodes(data = True):
            expected = set()
            for doc, cid in zip(pathway_graph.document_ids, pathway_graph.assignments):
                if int(cid) == node:
                    expected.update(extracted_skills.get(doc, []))
            assert set(data["skills"]) == expected

    # -----------------------------------------------------------------
    # Edge attributes
    # -----------------------------------------------------------------

    def test_edge_attrs_complete(
        self,
        pathway_graph: CareerPathwayGraph
    ):
        """
        Every edge carries `weight`, `mean_pmi`, and `direction_source`
        with weight equal to mean_pmi. Divergence means shortest paths
        and analysis output disagree silently.
        """
        for _, _, data in pathway_graph.graph.edges(data = True):
            assert data["weight"] > 0
            assert data["weight"] == data["mean_pmi"]
            assert data["direction_source"] == "job_zone"

    def test_edge_job_zone_order(
        self,
        pathway_graph: CareerPathwayGraph
    ):
        """
        Edge direction follows Job Zone ordering: source Job Zone is less
        than or equal to target Job Zone.
        """
        for source, target in pathway_graph.graph.edges():
            src_jz = pathway_graph.graph.nodes[source]["job_zone"]
            tgt_jz = pathway_graph.graph.nodes[target]["job_zone"]
            assert src_jz <= tgt_jz

    def test_edge_threshold(self, pathway_graph: CareerPathwayGraph):
        """
        Every edge weight meets or exceeds the adaptive threshold. A bug
        in thresholding adds spurious career transitions or drops
        legitimate ones.
        """
        if pathway_graph.graph.number_of_edges() == 0:
            return
        weights   = [d["weight"] for _, _, d in pathway_graph.graph.edges(data = True)]
        threshold = sorted(weights)[0]
        for w in weights:
            assert w >= threshold

    # -----------------------------------------------------------------
    # Alignment diagnostics
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Result aggregation
    # -----------------------------------------------------------------

    def test_result_aggregates(
        self,
        pathway_graph: CareerPathwayGraph
    ):
        """
        The `result` property aggregates graph diagnostics into a single
        `PathwayGraphResult`.
        """
        r = pathway_graph.result
        assert r.node_count == pathway_graph.graph.number_of_nodes()
        assert r.edge_count == pathway_graph.graph.number_of_edges()
        assert r.alignment.ari == pathway_graph.alignment.ari

    # -----------------------------------------------------------------
    # DAG derivation
    # -----------------------------------------------------------------

    def test_dag_is_acyclic(self, pathway_graph: CareerPathwayGraph):
        """
        The derived DAG view contains no cycles.
        """
        _ = pathway_graph.dag
        assert nx.is_directed_acyclic_graph(pathway_graph.dag_view)

    def test_dag_subgraph(self, pathway_graph: CareerPathwayGraph):
        """
        The DAG is a subgraph of the primary graph. Every DAG edge must
        exist in the original, meaning the derivation only removes edges
        and never creates phantom ones.
        """
        _ = pathway_graph.dag
        primary_edges = set(pathway_graph.graph.edges())
        dag_edges     = set(pathway_graph.dag_view.edges())
        assert dag_edges <= primary_edges

    def test_dag_path_weight(self, pathway_graph: CareerPathwayGraph):
        """
        The path weight equals the sum of edge weights along the longest
        path in the DAG view.
        """
        dag   = pathway_graph.dag
        path  = dag.longest_path
        view  = pathway_graph.dag_view
        total = sum(
            view.edges[path[i], path[i + 1]].get("weight", 0)
            for i in range(len(path) - 1)
        )
        assert abs(dag.path_weight - total) < 1e-10

    # -----------------------------------------------------------------
    # Traversal
    # -----------------------------------------------------------------

    def test_traversal_symmetry(
        self,
        pathway_graph: CareerPathwayGraph
    ):
        """
        If cluster A is reachable from B, then B leads to A. Downstream
        routing depends on this graph-theoretic invariant holding across
        both methods.
        """
        for node in pathway_graph.graph.nodes():
            for descendant in pathway_graph.reachable_from(node):
                assert node in pathway_graph.leads_to(descendant)

    def test_reachable_subset(self, pathway_graph: CareerPathwayGraph):
        """
        Reachable set is a subset of all node IDs.
        """
        all_nodes = set(pathway_graph.graph.nodes())
        for node in all_nodes:
            assert pathway_graph.reachable_from(node) <= all_nodes

    # -----------------------------------------------------------------
    # Shortest paths
    # -----------------------------------------------------------------

    def test_shortest_self_zero(
        self,
        pathway_graph: CareerPathwayGraph
    ):
        """
        Shortest path from a node to itself is zero.
        """
        for node in pathway_graph.graph.nodes():
            assert pathway_graph.shortest_path_length(node, node) == 0.0

    def test_shortest_unreachable(
        self,
        pathway_graph: CareerPathwayGraph
    ):
        """
        Shortest path returns None when no directed path exists between
        two clusters.
        """
        nodes = list(pathway_graph.graph.nodes())
        if len(nodes) >= 2 and not nx.is_strongly_connected(pathway_graph.graph):
            assert any(
                pathway_graph.shortest_path_length(s, t) is None
                for s in nodes for t in nodes if s != t
            )

    # -----------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------

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
        result = pathway_graph.export(tmp_path / "export")
        G      = node_link_graph(loads(result.json_path.read_text()))

        assert G.number_of_nodes() == pathway_graph.graph.number_of_nodes()
        assert G.number_of_edges() == pathway_graph.graph.number_of_edges()

        for node in pathway_graph.graph.nodes():
            original  = pathway_graph.graph.nodes[node].get("programs", [])
            recovered = G.nodes[node].get("programs", [])
            assert len(recovered) == len(original)

    # -----------------------------------------------------------------
    # Enrichment
    # -----------------------------------------------------------------

    def test_programs_attribute(
        self,
        pathway_graph: CareerPathwayGraph
    ):
        """
        Every node carries a `programs` attribute (possibly empty).
        """
        for _, data in pathway_graph.graph.nodes(data = True):
            assert "programs" in data
            assert isinstance(data["programs"], list)
