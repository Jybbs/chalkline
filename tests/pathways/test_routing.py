"""
Validate centrality analysis, widest-path routing, and learning plan
generation on deterministic DAG fixtures and the synthetic 20-posting
fixture chain.
"""

from networkx import DiGraph

from chalkline.pathways.routing    import CareerRouter
from chalkline.pathways.schemas    import LearningPlan
from chalkline.pipeline.enrichment import EnrichmentContext
from chalkline.pipeline.schemas    import ApprenticeshipContext
from chalkline.pipeline.schemas    import ClusterProfile, ProgramRecommendation


def _make_linear_router() -> CareerRouter:
    """
    Build a 4-node linear DAG router for deterministic tests.

    C0(JZ=1) --0.8--> C1(JZ=2) --0.5--> C2(JZ=3) --0.9--> C3(JZ=4)

    Skills: C0={concrete,welding}, C1={scaffolding,welding},
            C2={electrical safety,scaffolding},
            C3={electrical safety,electrical wiring}
    Bridging: 0->1={scaffolding}, 1->2={electrical safety},
              2->3={electrical wiring}
    Widest path bottleneck: 0.5 (the C1->C2 edge)
    """
    profiles = {
        0: ClusterProfile(
            apprenticeship = ApprenticeshipContext(
                rapids_code = "001",
                term_hours  = "2000",
                title       = "Laborer"
            ),
            cluster_id = 0,
            job_zone   = 1,
            sector     = "Heavy Highway Construction",
            size       = 5,
            skills     = ["concrete", "welding"]
        ),
        1: ClusterProfile(
            cluster_id = 1,
            job_zone   = 2,
            sector     = "Heavy Highway Construction",
            size       = 5,
            skills     = ["scaffolding", "welding"]
        ),
        2: ClusterProfile(
            cluster_id = 2,
            job_zone   = 3,
            sector     = "Building Construction",
            size       = 5,
            skills     = ["electrical safety", "scaffolding"]
        ),
        3: ClusterProfile(
            apprenticeship = ApprenticeshipContext(
                rapids_code = "002",
                term_hours  = "8000",
                title       = "Electrician"
            ),
            cluster_id = 3,
            job_zone   = 4,
            programs   = [ProgramRecommendation(
                credential  = "AAS",
                institution = "CMCC",
                program     = "Electrical Technology",
                url         = "https://example.com"
            )],
            sector     = "Building Construction",
            size       = 5,
            skills     = ["electrical safety", "electrical wiring"]
        )
    }

    (G := DiGraph()).add_nodes_from(
        (cid, profile.model_dump(mode="json"))
        for cid, profile in profiles.items()
    )
    G.add_weighted_edges_from([
        (0, 1, 0.8),
        (1, 2, 0.5),
        (2, 3, 0.9)
    ], direction_source="job_zone")
    G.edges[0, 1]["term_hours_delta"] = 1000
    G.edges[2, 3]["term_hours_delta"] = 3000

    apps  = [p.apprenticeship for p in profiles.values() if p.apprenticeship]
    progs = [p for prof in profiles.values() for p in prof.programs]

    return CareerRouter(
        enrichment = EnrichmentContext(apps, progs),
        graph      = G,
        profiles   = profiles
    )


def _make_diamond_router() -> CareerRouter:
    """
    Build a 4-node diamond DAG with two paths for route comparison.

    C0(JZ=1) --0.8--> C1(JZ=2) --0.3--> C3(JZ=4)
    C0(JZ=1) --0.4--> C2(JZ=3) --0.9--> C3(JZ=4)

    Path via C1: bottleneck = min(0.8, 0.3) = 0.3
    Path via C2: bottleneck = min(0.4, 0.9) = 0.4
    Widest path: C0 -> C2 -> C3 (bottleneck 0.4)
    """
    profiles = {
        0: ClusterProfile(
            cluster_id = 0,
            job_zone   = 1,
            sector     = "Heavy Highway Construction",
            size       = 5,
            skills     = ["a", "b"]
        ),
        1: ClusterProfile(
            cluster_id = 1,
            job_zone   = 2,
            sector     = "Heavy Highway Construction",
            size       = 5,
            skills     = ["b", "c"]
        ),
        2: ClusterProfile(
            cluster_id = 2,
            job_zone   = 3,
            sector     = "Building Construction",
            size       = 5,
            skills     = ["a", "d"]
        ),
        3: ClusterProfile(
            cluster_id = 3,
            job_zone   = 4,
            sector     = "Building Construction",
            size       = 5,
            skills     = ["c", "d", "e"]
        )
    }

    (G := DiGraph()).add_nodes_from(
        (cid, profile.model_dump(mode="json"))
        for cid, profile in profiles.items()
    )
    G.add_weighted_edges_from([
        (0, 1, 0.8),
        (0, 2, 0.4),
        (1, 3, 0.3),
        (2, 3, 0.9)
    ], direction_source="job_zone")

    return CareerRouter(
        enrichment = EnrichmentContext([], []),
        graph      = G,
        profiles   = profiles
    )


class TestCareerRouter:
    """
    Validate centrality, widest-path routing, bridging skills,
    and learning plans.
    """

    # ---------------------------------------------------------
    # Centrality
    # ---------------------------------------------------------

    def test_centrality_bounded(self):
        """
        Betweenness and PageRank values fall within [0, 1].
        """
        c = _make_linear_router().centrality
        assert all(0 <= v <= 1 for v in c.betweenness.values())
        assert all(0 <= v <= 1 for v in c.pagerank.values())

    def test_centrality_edgeless(self):
        """
        An edgeless graph produces uniform PageRank and zero
        betweenness without raising.
        """
        profiles = {
            0: ClusterProfile(
                cluster_id = 0,
                job_zone   = 1,
                sector     = "Test",
                size       = 1,
                skills     = ["a"]
            ),
            1: ClusterProfile(
                cluster_id = 1,
                job_zone   = 2,
                sector     = "Test",
                size       = 1,
                skills     = ["b"]
            )
        }

        (G := DiGraph()).add_nodes_from(
            (cid, profile.model_dump(mode="json"))
            for cid, profile in profiles.items()
        )
        c = CareerRouter(
            enrichment = EnrichmentContext([], []),
            graph      = G,
            profiles   = profiles
        ).centrality
        assert all(v == 0.0 for v in c.betweenness.values())
        assert abs(sum(c.pagerank.values()) - 1.0) < 1e-10

    def test_centrality_keys(self):
        """
        Every node ID appears in each centrality dict.
        """
        router = _make_linear_router()
        c      = router.centrality
        nodes  = set(router.graph.nodes())
        assert set(c.betweenness) == nodes
        assert set(c.in_degree) == nodes
        assert set(c.out_degree) == nodes
        assert set(c.pagerank) == nodes

    def test_centrality_node_attrs(self):
        """
        Centrality values are stored as node attributes on the
        graph for downstream consumption.
        """
        G = _make_linear_router().graph
        for node in G.nodes():
            assert "betweenness" in G.nodes[node]
            assert "pagerank" in G.nodes[node]

    # ---------------------------------------------------------
    # Widest path
    # ---------------------------------------------------------

    def test_widest_bottleneck(self):
        """
        Bottleneck weight equals the minimum edge weight along
        the widest path. On the linear DAG, the bottleneck from
        C0 to C3 is 0.5 (the C1->C2 edge).
        """
        route = _make_linear_router().widest_path(0, 3)
        assert route is not None
        assert route.bottleneck_weight == 0.5

    def test_widest_diamond(self):
        """
        On the diamond DAG, the widest path prefers the route
        via C2 (bottleneck 0.4) over C1 (bottleneck 0.3).
        """
        route = _make_diamond_router().widest_path(0, 3)
        assert route is not None
        assert route.path == [0, 2, 3]
        assert route.bottleneck_weight == 0.4

    def test_widest_disconnected(self):
        """
        Disconnected pairs return `None`.
        """
        assert _make_linear_router().widest_path(3, 0) is None

    def test_widest_path_linear(self):
        """
        On the 4-node linear DAG, the widest path from C0 to C3
        traverses all four nodes in order.
        """
        route = _make_linear_router().widest_path(0, 3)
        assert route is not None
        assert route.path == [0, 1, 2, 3]
        assert route.hops == 3

    def test_widest_self(self):
        """
        Source equal to target returns a single-node route.
        """
        route = _make_linear_router().widest_path(0, 0)
        assert route is not None
        assert route.path == [0]
        assert route.hops == 0

    def test_bridging_set_diff(self):
        """
        Bridging skills at each step equal the set difference
        between target and source skill profiles.
        """
        route = _make_linear_router().widest_path(0, 3)
        assert route is not None
        assert route.steps[0].bridging_skills == ["scaffolding"]
        assert route.steps[1].bridging_skills == ["electrical safety"]
        assert route.steps[2].bridging_skills == ["electrical wiring"]

    def test_bridging_step_hours(self):
        """
        Steps with `term_hours_delta` produce int hours; steps
        without produce `None`.
        """
        route = _make_linear_router().widest_path(0, 3)
        assert route is not None
        assert route.steps[0].estimated_hours == 1000
        assert route.steps[1].estimated_hours is None
        assert route.steps[2].estimated_hours == 3000

    # ---------------------------------------------------------
    # Enrichment
    # ---------------------------------------------------------

    def test_enrichment_apprenticeship(self):
        """
        The C2->C3 step's bridging skill "electrical wiring"
        matches the "Electrician" apprenticeship via the "elec"
        prefix.
        """
        route = _make_linear_router().widest_path(0, 3)
        assert route is not None
        step_2_3 = route.steps[2]
        trades   = [a.title for a in step_2_3.apprenticeships]
        assert "Electrician" in trades

    def test_enrichment_no_match(self):
        """
        Steps whose bridging skills do not prefix-match any
        apprenticeship or program produce empty lists.
        """
        route = _make_linear_router().widest_path(0, 3)
        assert route is not None
        assert route.steps[0].apprenticeships == []
        assert route.steps[0].programs == []

    def test_enrichment_program(self):
        """
        The C2->C3 step's bridging skill "electrical wiring"
        matches the "Electrical Technology" program via the
        "elec" prefix.
        """
        route = _make_linear_router().widest_path(0, 3)
        assert route is not None
        step_2_3 = route.steps[2]
        progs    = [p.program for p in step_2_3.programs]
        assert "Electrical Technology" in progs

    # ---------------------------------------------------------
    # Learning plan
    # ---------------------------------------------------------

    def test_plan_derived(self):
        """
        `LearningPlan` derives `bridging_skills` and
        `estimated_hours` from the route's transition steps.
        """
        plan = _make_linear_router().learning_plan(0, 3)
        assert plan is not None
        assert plan.bridging_skills == [
            "electrical safety", "electrical wiring", "scaffolding"
        ]
        assert plan.estimated_hours == 4000

    def test_plan_disconnected(self):
        """
        Disconnected pairs return `None`.
        """
        assert _make_linear_router().learning_plan(3, 0) is None

    def test_plan_hours_none(self):
        """
        `estimated_hours` is `None` when no transition step
        carries training hour data.
        """
        plan = _make_diamond_router().learning_plan(0, 3)
        assert plan is not None
        assert plan.estimated_hours is None

    def test_plan_step_count(self):
        """
        Number of steps equals the number of hops (path length
        minus one).
        """
        plan = _make_linear_router().learning_plan(0, 3)
        assert plan is not None
        assert len(plan.route.steps) == plan.route.hops

    def test_plan_fixture(self, router: CareerRouter):
        """
        `learning_plan` on the fixture graph returns a valid plan
        or `None` without crashing.
        """
        nodes = list(router.graph.nodes())
        assert len(nodes) >= 2
        result = router.learning_plan(nodes[0], nodes[-1])
        assert result is None or isinstance(result, LearningPlan)
