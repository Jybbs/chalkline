"""
Career routing with centrality analysis and widest-path computation.

Consumes a career pathway DAG and cluster profiles, computes
four centrality measures as node attributes, and provides on-demand
widest-path routing with skill gap bridging and enrichment annotation
per transition step. The widest-path algorithm maximizes the minimum
edge weight along the route, identifying the most achievable career
progression rather than the fewest transitions.
"""

import networkx as nx

from itertools import accumulate, takewhile
from logging   import getLogger

from chalkline.pathways.schemas    import CareerRoute, CentralityMetrics
from chalkline.pathways.schemas    import LearningPlan, TransitionStep
from chalkline.pipeline.enrichment import EnrichmentContext, prefix_set
from chalkline.pipeline.schemas    import ClusterProfile


logger = getLogger(__name__)


class CareerRouter:
    """
    Centrality analysis and widest-path routing on the career DAG.

    Accepts a NetworkX DiGraph and cluster profiles directly,
    computes betweenness, in-degree, out-degree, and PageRank
    centrality at construction time, and exposes on-demand
    widest-path routing with progressive learning plan generation.
    Bridging skills and enrichment annotations are precomputed per
    edge during construction for downstream assembly into
    transition steps.
    """

    def __init__(
        self,
        enrichment : EnrichmentContext,
        graph      : nx.DiGraph,
        profiles   : dict[int, ClusterProfile]
    ):
        """
        Compute centrality and edge enrichment attributes eagerly.

        Both passes are negligible at the expected graph scale
        (5-21 nodes under healthy clustering). Centrality scores
        are stored as node attributes, while bridging skills and
        enrichment matches are stored as edge attributes. The
        shared `EnrichmentContext` provides precomputed prefix
        lookup dicts for apprenticeship and program matching,
        eliminating redundant prefix dict construction.

        Args:
            enrichment : Shared prefix lookup context.
            graph      : Career DAG with weighted edges.
            profiles   : Enriched cluster characteristics keyed
                         by cluster ID.
        """
        self.enrichment  = enrichment
        self.graph       = graph
        self.profiles    = profiles
        self.centrality  = self._compute_centrality()
        self._compute_edge_enrichment()

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
        edge = self.graph[source][target]
        return TransitionStep(
            apprenticeships = edge["apprenticeships"],
            bridging_skills = edge["bridging_skills"],
            estimated_hours = edge["estimated_hours"],
            programs        = edge["programs"],
            source_cluster  = source,
            target_cluster  = target,
            weight          = edge["weight"]
        )

    def _compute_centrality(self) -> CentralityMetrics:
        """
        Compute four centrality measures, store as node attributes
        on the career DAG, and return the typed schema.
        """
        if not self.graph.number_of_edges():
            n = self.graph.number_of_nodes()
            logger.warning(
                f"Edgeless graph; all centrality measures are "
                f"degenerate (PageRank uniform at 1/{n})"
            )

        metrics = CentralityMetrics(
            betweenness = nx.betweenness_centrality(self.graph, weight="weight"),
            in_degree   = nx.in_degree_centrality(self.graph),
            out_degree  = nx.out_degree_centrality(self.graph),
            pagerank    = nx.pagerank(self.graph, weight="weight")
        )

        for name, values in metrics:
            nx.set_node_attributes(self.graph, values, name)

        return metrics

    def _compute_edge_enrichment(self):
        """
        Precompute bridging skills and enrichment per edge.

        For each edge, stores the skill set difference between
        target and source cluster profiles, matched apprenticeships,
        matched programs, and estimated training hours. Enrichment
        matching delegates to the shared `EnrichmentContext` prefix
        lookups rather than rebuilding them locally.
        """
        dump = lambda items: [x.model_dump(mode="json") for x in items]
        for s, t, data in self.graph.edges(data=True):
            bridging = sorted(
                self.profiles[t].skills - self.profiles[s].skills
            )
            skill_pf = {
                p for skill in bridging
                for p in prefix_set(skill)
            }

            data |= {
                "apprenticeships" : dump(self.enrichment.find_apprenticeships(skill_pf)),
                "bridging_skills" : bridging,
                "estimated_hours" : data.get("term_hours_delta"),
                "programs"        : dump(self.enrichment.find_programs(skill_pf))
            }

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
        steps = [
            self._build_step(u, v)
            for u, v in nx.utils.pairwise(path)
        ]
        return CareerRoute(
            bottleneck_weight = min((s.weight for s in steps), default=0.0),
            hops              = len(path) - 1,
            path              = path,
            steps             = steps
        )

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
        if (route := self.widest_path(source, target)) is None:
            return None

        if (sp := nx.shortest_path(self.graph, source, target)) != route.path:
            logger.info(
                f"Widest path {route.path} diverges from "
                f"shortest path {sp} between clusters "
                f"{source} and {target}"
            )

        return LearningPlan(route=route)

    def widest_path(self, source: int, target: int) -> CareerRoute | None:
        """
        Widest (maximum bottleneck) career route between two
        clusters via topological DP.

        Walks nodes in topological order, relaxing each edge with
        max-min bottleneck semantics in O(V + E). DAG acyclicity
        guarantees a single topological pass suffices.

        Args:
            source : Source cluster ID.
            target : Target cluster ID.

        Returns:
            Career route with bottleneck weight and transition
            steps, or `None` if unreachable.
        """
        best = {source: (float("inf"), None)}

        for node in nx.topological_sort(self.graph):
            if (state := best.get(node)) is None:
                continue
            best |= {
                n: (c, node)
                for n, d in self.graph[node].items()
                if (c := min(state[0], d["weight"])) > best.get(n, (0,))[0]
            }

        return self._route_from_path([
            *takewhile(
                lambda n: n is not None,
                accumulate(best, lambda n, _: best[n][1], initial=target)
            )
        ][::-1]) if best.get(target) else None
