"""
Career routing with centrality analysis and widest-path computation.

Consumes the career pathway DAG from `CareerPathwayGraph`, computes
four centrality measures as node attributes, and provides on-demand
widest-path routing with skill gap bridging and enrichment annotation
per transition step. The widest-path algorithm maximizes the minimum
edge weight along the route, identifying the most achievable career
progression rather than the fewest transitions.
"""

from functools          import cached_property
from logging            import getLogger
from networkx           import all_simple_paths, ancestors
from networkx           import betweenness_centrality, descendants
from networkx           import get_node_attributes, has_path
from networkx           import in_degree_centrality, out_degree_centrality
from networkx           import pagerank, set_edge_attributes
from networkx           import set_node_attributes, shortest_path, subgraph_view
from networkx.utils     import pairwise

from chalkline.pathways.graph   import CareerPathwayGraph
from chalkline.pathways.schemas import CareerRoute, CentralityMetrics
from chalkline.pathways.schemas import LearningPlan, TransitionStep


logger = getLogger(__name__)


class CareerRouter:
    """
    Centrality analysis and widest-path routing on the career DAG.

    Receives a `CareerPathwayGraph`, computes betweenness, in-degree,
    out-degree, and PageRank centrality at construction time, and
    exposes on-demand widest-path routing with progressive learning
    plan generation. Bridging skills and enrichment annotations are
    precomputed per edge during construction for downstream assembly
    into transition steps.
    """

    def __init__(self, pathway_graph: CareerPathwayGraph):
        """
        Compute centrality and edge enrichment attributes eagerly.

        Both passes are negligible at the expected graph scale
        (5-21 nodes under healthy clustering). Centrality scores
        are stored as node attributes, while bridging skills and
        enrichment matches are stored as edge attributes.

        Args:
            pathway_graph: Career DAG with enriched profiles.
        """
        self.pathway_graph = pathway_graph
        self.widest_cache: dict[tuple[int, int], list[int] | None] = {}

        self._compute_centrality()
        self._compute_edge_enrichment()

    @cached_property
    def centrality(self) -> CentralityMetrics:
        """
        Four centrality measures over the career DAG.

        Wraps the node-attribute values computed at construction
        into a typed schema for downstream consumption.
        """
        G = self.pathway_graph.graph
        return CentralityMetrics(
            betweenness = get_node_attributes(G, "betweenness"),
            in_degree   = get_node_attributes(G, "in_degree"),
            out_degree  = get_node_attributes(G, "out_degree"),
            pagerank    = get_node_attributes(G, "pagerank")
        )

    def _build_step(self, source: int, target: int) -> TransitionStep:
        """
        Assemble a `TransitionStep` from precomputed edge
        attributes.

        Args:
            source : Source cluster ID.
            target : Target cluster ID.

        Returns:
            Populated transition step with enrichment.
        """
        edge = self.pathway_graph.graph[source][target]
        return TransitionStep(
            apprenticeships = edge["apprenticeships"],
            bridging_skills = edge["bridging_skills"],
            estimated_hours = edge["estimated_hours"],
            programs        = edge["programs"],
            source_cluster  = source,
            target_cluster  = target,
            weight          = edge["weight"]
        )

    def _compute_centrality(self):
        """
        Compute four centrality measures and store as node
        attributes on the career DAG.

        Guards PageRank against edgeless graphs where the power
        iteration would degenerate to uniform damping.
        """
        G = self.pathway_graph.graph

        if G.number_of_nodes() == 0:
            return

        set_node_attributes(G, betweenness_centrality(G, weight="weight"), "betweenness")
        set_node_attributes(G, in_degree_centrality(G), "in_degree")
        set_node_attributes(G, out_degree_centrality(G), "out_degree")

        if G.number_of_edges() > 0:
            set_node_attributes(G, pagerank(G, weight="weight"), "pagerank")
        else:
            uniform = 1 / G.number_of_nodes()
            set_node_attributes(
                G, {n: uniform for n in G.nodes()}, "pagerank"
            )

    def _compute_edge_enrichment(self):
        """
        Precompute bridging skills and enrichment per edge.

        For each edge, stores the skill set difference between
        target and source cluster profiles, matched apprenticeships,
        matched programs, and estimated training hours. Enrichment
        matching uses 4-char prefix overlap against the bridging
        skill set, following the same approach as `ResumeMatcher`.
        """
        G        = self.pathway_graph.graph
        profiles = self.pathway_graph.profiles

        prefix   = lambda t: {w[:4] for w in t.lower().split() if len(w) >= 4}
        apps     = G.graph["apprenticeships"]
        progs    = G.graph["programs"]
        trade_pf = {a.rapids_code: prefix(a.title) for a in apps}
        prog_pf  = {
            (p.institution, p.program): prefix(p.program)
            for p in progs
        }

        enrichment = {}
        for s, t, edge in G.edges(data=True):
            bridging = sorted(
                set(profiles[t].skills) - set(profiles[s].skills)
            )
            skill_pf = {p for skill in bridging for p in prefix(skill)}
            delta    = edge.get("term_hours_delta")

            enrichment[s, t] = {
                "apprenticeships" : [
                    a for a in apps
                    if skill_pf & trade_pf[a.rapids_code]
                ],
                "bridging_skills" : bridging,
                "estimated_hours" : int(delta) if delta is not None else None,
                "programs"        : [
                    p for p in progs
                    if skill_pf & prog_pf[p.institution, p.program]
                ],
            }

        set_edge_attributes(G, enrichment)

    def _route_from_path(self, path: list[int]) -> CareerRoute:
        """
        Build a `CareerRoute` from an ordered list of cluster IDs.

        Constructs `TransitionStep` records for each adjacent pair
        and derives the bottleneck weight as the minimum step weight.

        Args:
            path: Ordered cluster IDs from source to target.

        Returns:
            Career route with steps and bottleneck weight.
        """
        steps = [self._build_step(u, v) for u, v in pairwise(path)]
        return CareerRoute(
            bottleneck_weight = min(
                (s.weight for s in steps), default=0.0
            ),
            hops  = len(path) - 1,
            path  = path,
            steps = steps
        )

    def all_routes(self, source: int, target: int) -> list[CareerRoute]:
        """
        Enumerate all simple paths between two clusters with
        bottleneck weights and transition steps.

        The DAG property guarantees termination. Routes are sorted
        by descending bottleneck weight so the most achievable
        career progression appears first.

        Args:
            source : Source cluster ID.
            target : Target cluster ID.

        Returns:
            All simple paths as `CareerRoute` records, or an empty
            list if the target is unreachable.
        """
        routes = [
            self._route_from_path(path)
            for path in all_simple_paths(
                self.pathway_graph.graph, source, target
            )
        ]
        routes.sort(key=lambda r: r.bottleneck_weight, reverse=True)
        return routes

    def leads_to(self, cluster_id: int) -> set[int]:
        """
        All clusters that lead to this node (upstream ancestors).

        Args:
            cluster_id: Target cluster to find predecessors of.

        Returns:
            Set of cluster IDs that can reach this node.
        """
        return ancestors(self.pathway_graph.graph, cluster_id)

    def learning_plan(self, source: int, target: int) -> LearningPlan | None:
        """
        Progressive learning plan along the widest career route.

        Delegates route construction to `widest_path`, then
        aggregates bridging skills and estimated hours across
        transition steps. Logs when the widest and shortest paths
        diverge across pairs with multiple simple paths, since the
        divergence reveals where the two optimization objectives
        disagree.

        Args:
            source : Source cluster ID (from resume match).
            target : Target cluster ID (career goal).

        Returns:
            Learning plan with per-step detail, or `None` if the
            target is unreachable from the source.
        """
        route = self.widest_path(source, target)
        if route is None:
            return None

        sp = shortest_path(
            self.pathway_graph.graph, source, target
        )
        if sp != route.path:
            logger.info(
                f"Widest path {route.path} diverges from "
                f"shortest path {sp} between clusters "
                f"{source} and {target}"
            )

        hours = [
            s.estimated_hours for s in route.steps
            if s.estimated_hours is not None
        ]

        return LearningPlan(
            all_bridging_skills = sorted({
                s for step in route.steps
                for s in step.bridging_skills
            }),
            route                 = route,
            total_estimated_hours = sum(hours) if hours else None
        )

    def reachable_from(self, cluster_id: int) -> set[int]:
        """
        All clusters reachable from this node (downstream
        descendants).

        Args:
            cluster_id: Source cluster to find successors of.

        Returns:
            Set of cluster IDs reachable from this node.
        """
        return descendants(self.pathway_graph.graph, cluster_id)

    def widest_path(self, source: int, target: int) -> CareerRoute | None:
        """
        Widest (maximum bottleneck) career route between two
        clusters.

        Searches edge weight thresholds in descending order,
        filtering the DAG via `subgraph_view` at each level and
        checking reachability with `has_path`. The first threshold
        where a path exists yields the maximum bottleneck value.

        Args:
            source : Source cluster ID.
            target : Target cluster ID.

        Returns:
            Career route with bottleneck weight and transition
            steps, or `None` if unreachable.
        """
        if source == target:
            return self._route_from_path([source])

        key = (source, target)
        if key in self.widest_cache:
            cached = self.widest_cache[key]
            return self._route_from_path(cached) if cached else None

        G = self.pathway_graph.graph

        for w in sorted(
            {w for _, _, w in G.edges(data="weight")}, reverse=True
        ):
            view = subgraph_view(
                G,
                filter_edge=lambda u, v, w=w: G[u][v]["weight"] >= w
            )
            if has_path(view, source, target):
                path = shortest_path(view, source, target)
                self.widest_cache[key] = path
                return self._route_from_path(path)

        self.widest_cache[key] = None
        return None
