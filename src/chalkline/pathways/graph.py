"""
Career pathway graph from HAC clusters and PMI co-occurrence edges.

Constructs a directed weighted DiGraph where nodes are job clusters from
hierarchical agglomerative clustering and edges connect clusters whose
skill profiles share significant PMI co-occurrence. Edge direction
follows a strict total order on (Job Zone, cluster ID), guaranteeing
acyclicity.
"""

import numpy as np

from functools                     import cached_property
from itertools                     import combinations
from json                          import dumps
from logging                       import getLogger
from networkx                      import dag_longest_path, DiGraph
from networkx                      import node_link_data, path_weight
from networkx                      import remove_node_attributes
from networkx                      import set_edge_attributes, write_graphml
from networkx.algorithms.community import modularity
from pathlib                       import Path
from sklearn.metrics               import adjusted_rand_score

from chalkline.association.cooccurrence import CooccurrenceNetwork
from chalkline.pathways.schemas         import AlignmentDiagnostics, GraphExport
from chalkline.pathways.schemas         import LongestPath
from chalkline.pipeline.schemas         import ClusterProfile


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
        network  : CooccurrenceNetwork,
        profiles : dict[int, ClusterProfile]
    ):
        """
        Build the career pathway graph from pre-enriched upstream
        artifacts.

        Args:
            network  : Co-occurrence network providing PPMI matrix,
                       feature vocabulary, and Louvain partition.
            profiles : Pre-enriched domain characteristics per
                       cluster, including skills, Job Zone, sector,
                       apprenticeship match, and program matches.
        """
        self.network  = network
        self.profiles = profiles

        self.cluster_ids = sorted(self.profiles)
        self.graph       = self._build_graph()

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
            return AlignmentDiagnostics(ari=0.0)

        labels = [(louvain[s], hac[s]) for s in shared]
        result = AlignmentDiagnostics(
            ari = float(adjusted_rand_score(*zip(*labels)))
        )

        if (skill_graph := self.network.graph()).size():
            result.modularity = float(modularity(
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
            return LongestPath(path=[], path_weight=0.0)

        path = dag_longest_path(
            self.graph, weight="weight", default_weight=0
        )
        return LongestPath(
            path        = path,
            path_weight = path_weight(self.graph, path, "weight")
        )

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

    def _build_graph(self) -> DiGraph:
        """
        Construct the directed weighted career graph.

        Nodes carry all profile and enrichment attributes via
        `model_dump`. Edges are thresholded by the 75th percentile
        of pairwise top-k PPMI weights, directed from lower to
        higher rank.
        """
        (G := DiGraph()).add_nodes_from(
            (cid, profile.model_dump())
            for cid, profile in self.profiles.items()
        )

        rank       = lambda c: self.profiles[c].rank
        candidates = [
            (min(ci, cj, key=rank), max(ci, cj, key=rank), wc[0])
            for ci, cj in combinations(self.cluster_ids, 2)
            if (wc := self._edge_weight(ci, cj))[0] > 0 and wc[1] >= 3
        ]

        if candidates:
            threshold = float(np.percentile(
                [w for _, _, w in candidates], 75
            ))
            G.add_weighted_edges_from(
                [(s, t, w) for s, t, w in candidates if w >= threshold],
                direction_source="job_zone"
            )

            hours = {
                cid: int(p.apprenticeship.term_hours)
                for cid, p in self.profiles.items()
                if p.apprenticeship
            }
            set_edge_attributes(G, {
                (s, t): str(hours[t] - hours[s])
                for s, t in G.edges()
                if s in hours and t in hours
            }, "term_hours_delta")

        return G

    def _edge_weight(self, ci: int, cj: int) -> tuple[float, int]:
        """
        Top-k mean PPMI for a cluster pair.

        Extracts the PPMI submatrix for inter-cluster skill pairs,
        filters to positive values, and returns the mean of the
        top-k values where k = min(10, |Ci|, |Cj|).

        Args:
            ci : First cluster ID.
            cj : Second cluster ID.

        Returns:
            Tuple of (top-k mean PPMI, count of positive pairs).
        """
        idx_i = self.network.indices_for(self.profiles[ci].skills)
        idx_j = self.network.indices_for(self.profiles[cj].skills)

        if not idx_i or not idx_j:
            return 0.0, 0

        values = (v := self.network.pairwise_ppmi(idx_i, idx_j))[v > 0]

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
        Export the career graph as GraphML and JSON.

        GraphML includes only scalar node and edge attributes for
        interoperability with Gephi and Cytoscape. JSON preserves full
        attribute fidelity including nested program lists via
        `node_link_data`.

        Args:
            output_dir: Directory for export artifacts. Created if it
                        does not exist.

        Returns:
            Export paths for GraphML and JSON serialization artifacts.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        graphml_path = output_dir / "career_graph.graphml"
        scalar_graph = self.graph.copy()
        remove_node_attributes(
            scalar_graph, "apprenticeship", "programs", "skills", "terms"
        )
        write_graphml(scalar_graph, graphml_path)

        json_path = output_dir / "career_graph.json"
        json_path.write_text(
            dumps(node_link_data(self.graph), indent=2)
        )

        return GraphExport(
            graphml_path = graphml_path,
            json_path    = json_path
        )
