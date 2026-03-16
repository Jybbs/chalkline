"""
Career pathway graph from HAC clusters and NPMI co-occurrence edges.

Constructs a directed weighted DiGraph where nodes are job clusters from
hierarchical agglomerative clustering and edges connect clusters whose
skill profiles share significant NPMI co-occurrence. Edge weights are
bounded to [0, 1] for interpretability in downstream widest-path routing
and career report display. Edge direction follows a strict total order
on (Job Zone, cluster ID), guaranteeing acyclicity.
"""

import networkx as nx
import numpy    as np

from functools       import cached_property
from itertools       import combinations
from json            import dumps
from kneed           import KneeLocator
from logging         import getLogger
from pathlib         import Path
from sklearn.metrics import adjusted_rand_score

from chalkline.association.cooccurrence import CooccurrenceNetwork
from chalkline.pathways.schemas         import AlignmentDiagnostics, GraphExport
from chalkline.pathways.schemas         import LongestPath
from chalkline.pipeline.schemas         import ApprenticeshipContext
from chalkline.pipeline.schemas         import ClusterProfile, ProgramRecommendation


logger = getLogger(__name__)


class CareerPathwayGraph:
    """
    Directed weighted career graph with longest-path analysis and export.

    Accepts pre-enriched `ClusterProfile` records and a co-occurrence
    network, then constructs a DiGraph with one node per cluster and
    thresholded PMI edges. Edge direction follows a strict total order
    on (Job Zone, cluster ID), so the graph is always acyclic.
    """

    def __init__(
        self,
        network     : CooccurrenceNetwork,
        profiles    : dict[int, ClusterProfile],
        max_density : float = 0.05
    ):
        """
        Build the career pathway graph from pre-enriched upstream
        artifacts.

        Edge weights are thresholded via knee detection on the
        sorted weight curve, matching the pattern used by
        `CooccurrenceNetwork._find_threshold`. The
        `max_density` ceiling serves as a secondary guard
        when the knee still produces a graph too dense for
        career pathway navigation.

        Args:
            network     : NPMI matrix, vocabulary, and Louvain partition.
            profiles    : Enriched cluster characteristics from the
                          orchestrator.
            max_density : Density ceiling for edge pruning.
        """
        self.network     = network
        self.profiles    = profiles
        self.max_density = max_density
        self.cluster_ids = sorted(self.profiles)
        self.graph       = self._build_graph()

    @cached_property
    def apprenticeships(self) -> list[ApprenticeshipContext]:
        """
        Deduplicated apprenticeship contexts across all cluster
        profiles, keyed by RAPIDS code.
        """
        return list({
            p.apprenticeship.rapids_code: p.apprenticeship
            for p in self.profiles.values()
            if p.apprenticeship
        }.values())

    @cached_property
    def alignment(self) -> AlignmentDiagnostics:
        """
        ARI between Louvain communities and HAC cluster partitions
        projected onto the shared skill space.

        For each skill present in both the NPMI graph and at least
        one cluster, assigns a Louvain community ID and an HAC
        cluster ID. The adjusted Rand index quantifies agreement
        between these two partitions. Modularity is computed on the
        Louvain partition as a standalone quality measure.
        """
        louvain = self.network.partition_map
        hac     = self.skill_to_cluster
        shared  = louvain.keys() & hac.keys()

        if not shared:
            logger.warning("No shared skills for ARI computation")
            return AlignmentDiagnostics()

        if len(self.profiles) > len(shared) / 2:
            logger.warning(
                f"ARI computed with {len(self.profiles)} clusters "
                f"against {len(shared)} shared skills; "
                f"near-singleton clustering makes ARI "
                f"structurally near-zero"
            )

        labels = [(louvain[s], hac[s]) for s in shared]
        result = AlignmentDiagnostics(
            ari = float(adjusted_rand_score(*zip(*labels)))
        )

        if (skill_graph := self.network.graph()).size():
            result.modularity = float(nx.community.modularity(
                skill_graph,
                communities = self.network.partition,
                weight      = "weight"
            ))
        return result

    @cached_property
    def longest_path(self) -> LongestPath:
        """
        Longest weighted path through the career graph.
        """
        if not self.graph:
            return LongestPath()

        path = nx.dag_longest_path(
            self.graph, weight="weight", default_weight=0
        )
        return LongestPath(
            edge_count  = self.graph.number_of_edges(),
            path        = path,
            path_weight = nx.path_weight(self.graph, path, "weight")
        )

    @cached_property
    def programs(self) -> list[ProgramRecommendation]:
        """
        Deduplicated program recommendations across all cluster
        profiles, keyed by (institution, program name).
        """
        return list({
            (p.institution, p.program): p
            for profile in self.profiles.values()
            for p in profile.programs
        }.values())

    @cached_property
    def skill_to_cluster(self) -> dict[str, int]:
        """
        Inverse mapping from skill name to owning cluster ID.

        For skills present in multiple clusters, the highest cluster
        ID wins. Used by `alignment` to project the HAC partition
        onto the shared skill space for ARI comparison against
        Louvain communities.
        """
        return {
            s: p.cluster_id
            for p in self.profiles.values()
            for s in p.skills
        }

    def _build_graph(self) -> nx.DiGraph:
        """
        Construct the directed weighted career graph.

        Nodes carry all profile and enrichment attributes via
        `model_dump`. Edge threshold is selected by knee detection
        on the sorted candidate weight curve, with a density cap
        as a secondary guard. Edges are directed from lower to
        higher rank.
        """
        n = len(self.profiles)
        n_postings = sum(p.size for p in self.profiles.values())
        if n > n_postings * 0.75:
            logger.warning(
                f"Cluster count ({n}) exceeds 75% of corpus "
                f"size ({n_postings}); near-singleton clustering "
                f"degrades edge weight discrimination"
            )

        (G := nx.DiGraph()).add_nodes_from(
            (cid, profile.model_dump(mode="json"))
            for cid, profile in self.profiles.items()
        )

        rank       = lambda c: self.profiles[c].rank
        candidates = [
            (min(ci, cj, key=rank), max(ci, cj, key=rank), wc[0])
            for ci, cj in combinations(self.cluster_ids, 2)
            if (wc := self._edge_weight(ci, cj))[0] > 0 and wc[1] >= 3
        ]

        if candidates:
            weights   = sorted(w for _, _, w in candidates)
            threshold = self._find_threshold(weights)

            max_edges = int(
                self.max_density * n * (n - 1) / 2
            ) if n > 1 else 0
            if max_edges and sum(1 for w in weights if w >= threshold) > max_edges:
                threshold = weights[-max_edges]
                logger.info(
                    f"Density cap raised threshold to {threshold:.3f} "
                    f"({max_edges} edges at density "
                    f"{self.max_density})"
                )

            G.add_weighted_edges_from(
                [(s, t, w) for s, t, w in candidates if w >= threshold],
                direction_source="job_zone"
            )

            hours = {
                cid: int(p.apprenticeship.term_hours.split("-")[0])
                for cid, p in self.profiles.items()
                if p.apprenticeship
            }
            nx.set_edge_attributes(G, {
                (s, t): hours[t] - hours[s]
                for s, t in G.edges()
                if s in hours and t in hours
            }, "term_hours_delta")

        if not G.number_of_edges():
            logger.warning(
                f"Career graph has zero edges from {n} clusters "
                f"and {n_postings} postings; co-occurrence floor "
                f"of {self.network.threshold} may exceed corpus "
                f"density"
            )

        return G

    def _find_threshold(self, weights: list[float]) -> float:
        """
        Select edge weight threshold via knee detection.

        Finds the knee of the sorted weight curve where values
        transition from the noise floor to the signal regime,
        matching the pattern used by `CooccurrenceNetwork` for
        co-occurrence thresholds and by `ClusterComparison` for
        DBSCAN epsilon. Falls back to the 75th percentile when
        fewer than 3 candidates exist or no knee is found.

        Args:
            weights: Candidate edge weights in ascending order.

        Returns:
            Weight threshold at or above the knee.
        """
        if len(weights) < 3:
            return weights[0]

        knee = KneeLocator(
            curve     = "convex",
            direction = "increasing",
            x         = range(len(weights)),
            y         = weights
        ).knee

        if knee is None:
            logger.info(
                "Edge weight knee detection found no inflection, "
                "using 75th percentile"
            )
            return float(np.percentile(weights, 75))

        return weights[knee]

    def _edge_weight(self, ci: int, cj: int) -> tuple[float, int]:
        """
        Top-k mean NPMI for a cluster pair.

        Uses NPMI rather than PPMI so that edge weights are
        bounded to [0, 1] and interpretable as association
        strength independent of skill frequency. The top-k
        aggregation retains the strongest inter-cluster bridges
        where k = min(10, |Ci|, |Cj|).

        Args:
            ci : First cluster ID.
            cj : Second cluster ID.

        Returns:
            Tuple of (top-k mean NPMI, count of positive pairs).
        """
        idx_i = self.network.indices_for(self.profiles[ci].skills)
        idx_j = self.network.indices_for(self.profiles[cj].skills)

        if not idx_i or not idx_j:
            return 0.0, 0

        values = (v := self.network.pairwise_npmi(idx_i, idx_j))[v > 0]

        if len(values) == 0:
            return 0.0, 0

        k = min(
            10, len(self.profiles[ci].skills),
            len(self.profiles[cj].skills), len(values)
        )
        topk = np.partition(values, -k)[-k:]
        return float(topk.mean()), len(values)

    def export(self, output_dir: Path) -> GraphExport:
        """
        Export the career graph as GraphML and JSON to the
        specified output directory, creating it if needed.

        GraphML includes only scalar node and edge attributes for
        interoperability with Gephi and Cytoscape. JSON preserves
        full attribute fidelity including nested program lists via
        `node_link_data`.

        Args:
            output_dir: Target directory for export artifacts.

        Returns:
            Export paths for GraphML and JSON artifacts.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        graphml_path = output_dir / "career_graph.graphml"
        scalar_graph = self.graph.copy()
        nx.remove_node_attributes(
            scalar_graph, "apprenticeship", "programs", "skills", "terms"
        )

        for _, _, data in scalar_graph.edges(data=True):
            for attr in ("apprenticeships", "bridging_skills", "programs"):
                data.pop(attr, None)
        nx.write_graphml(scalar_graph, graphml_path)

        json_path = output_dir / "career_graph.json"
        nld       = nx.node_link_data(self.graph)
        dump      = lambda items: [x.model_dump(mode="json") for x in items]
        nld["apprenticeships"] = dump(self.apprenticeships)
        nld["programs"]        = dump(self.programs)
        json_path.write_text(dumps(nld, indent=2))

        return GraphExport(
            graphml_path = graphml_path,
            json_path    = json_path
        )
