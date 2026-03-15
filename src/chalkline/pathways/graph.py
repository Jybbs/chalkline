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
from networkx                      import set_edge_attributes, set_node_attributes
from networkx                      import write_graphml
from networkx.algorithms.community import louvain_communities, modularity
from pathlib                       import Path
from sklearn.metrics               import adjusted_rand_score

from chalkline.association.cooccurrence import CooccurrenceNetwork
from chalkline.clustering.schemas       import ClusterLabel
from chalkline.pathways.schemas         import AlignmentDiagnostics, GraphExport
from chalkline.pathways.schemas         import LongestPath
from chalkline.pipeline.schemas         import ApprenticeshipContext, ClusterProfile
from chalkline.pipeline.schemas         import ProgramRecommendation


logger = getLogger(__name__)


def _prefixes(text: str) -> set[str]:
    """
    Extract 4-character word prefixes for fuzzy trade matching.

    Catches inflectional variants across the construction domain
    (welding/welder, electrical/electrician, scaffolding/scaffold).

    Args:
        text: Input string to extract prefixes from.

    Returns:
        Set of 4-character lowercase prefixes.
    """
    return {w[:4] for w in text.lower().split() if len(w) >= 4}


class CareerPathwayGraph:
    """
    Directed weighted career graph with longest-path analysis and export.

    Accepts pre-computed `ClusterProfile` records, apprenticeship
    reference data, and cluster labels, then constructs a DiGraph with
    one node per cluster and thresholded PMI edges. Edge direction
    follows a strict total order on (Job Zone, cluster ID), so the
    graph is always acyclic.
    """

    def __init__(
        self,
        apprenticeships : list[ApprenticeshipContext],
        cluster_labels  : list[ClusterLabel],
        network         : CooccurrenceNetwork,
        profiles        : dict[int, ClusterProfile],
        programs        : list[ProgramRecommendation]
    ):
        """
        Build the career pathway graph from pre-computed upstream
        artifacts.
        """
        self.apprenticeships = apprenticeships
        self.cluster_labels  = cluster_labels
        self.network         = network
        self.profiles        = profiles
        self.programs        = programs

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
        skill_graph = self.network.graph()
        partition = sorted(
            louvain_communities(
                skill_graph, seed=self.network.random_seed, weight="weight"
            ),
            key=len, reverse=True
        )

        louvain = {s: i for i, members in enumerate(partition) for s in members}
        hac = {s: cid for cid in self.cluster_ids for s in self.profiles[cid].skills}
        shared = sorted(louvain.keys() & hac.keys())

        if not shared:
            logger.warning(
                "No shared skills between Louvain communities and "
                "HAC clusters for ARI computation"
            )
            return AlignmentDiagnostics(ari=0.0)

        ari = float(adjusted_rand_score(
            [louvain[s] for s in shared], [hac[s] for s in shared]
        ))

        if ari <= 0.3:
            logger.warning(
                f"Low ARI ({ari:.3f}) between Louvain communities and "
                f"HAC clusters, limited agreement between partitions"
            )

        return AlignmentDiagnostics(
            ari        = ari,
            modularity = float(modularity(
                skill_graph, partition, weight="weight"
            )) if skill_graph.size() else None
        )

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


    def _build_graph(self) -> DiGraph:
        """
        Construct the directed weighted career graph.

        Adds nodes via `add_nodes_from`, computes thresholded PMI
        edges, and enriches with apprenticeship and program metadata.
        """
        label_map = {cl.cluster_id: cl for cl in self.cluster_labels}
        job_zones = {cid: p.job_zone for cid, p in self.profiles.items()}

        G = DiGraph()
        G.add_nodes_from(
            (
                cid,
                {
                    "cluster_id" : cid,
                    "job_zone"   : profile.job_zone,
                    "sector"     : profile.sector,
                    "size"       : profile.size,
                    "skills"     : profile.skills,
                    "terms"      : l.terms if (l := label_map.get(cid)) else []
                }
            )
            for cid, profile in self.profiles.items()
        )

        candidates = [
            (ci, cj, wc[0])
            for ci, cj in combinations(self.cluster_ids, 2)
            if (wc := self._edge_weight(ci, cj))[0] > 0 and wc[1] >= 3
        ]

        if candidates:
            threshold = float(np.percentile(
                [w for _, _, w in candidates], 75
            ))
            G.add_weighted_edges_from(
                [
                    (*sorted((ci, cj), key=lambda c: (job_zones[c], c)), weight)
                    for ci, cj, weight in candidates
                    if weight >= threshold
                ],
                direction_source="job_zone"
            )

        self._enrich_nodes(G)
        return G

    def _edge_weight(self, ci: int, cj: int) -> tuple[float, int]:
        """
        Top-k mean PPMI for a cluster pair.

        Extracts the PPMI submatrix for inter-cluster skill pairs, filters
        to positive values, and returns the mean of the top-k values where
        k = min(10, |Ci|, |Cj|).

        Args:
            ci : First cluster ID.
            cj : Second cluster ID.

        Returns:
            Tuple of (top-k mean PPMI, count of positive pairs).
        """
        idx_i = self._skill_indices(ci)
        idx_j = self._skill_indices(cj)

        if not idx_i or not idx_j:
            return 0.0, 0

        values = (v := self.network.ppmi_matrix[np.ix_(idx_i, idx_j)].data)[v > 0]

        if len(values) == 0:
            return 0.0, 0

        k = min(
            10, len(self.profiles[ci].skills),
            len(self.profiles[cj].skills), len(values)
        )
        topk = np.sort(values)[-k:]
        return float(topk.mean()), len(values)

    def _enrich_nodes(self, G: DiGraph):
        """
        Attach apprenticeship and educational program metadata to
        graph nodes, and term hours deltas to edges connecting
        apprenticeship-mapped nodes.

        Uses bulk NetworkX attribute APIs (`set_node_attributes`,
        `set_edge_attributes`) to separate computation from graph
        mutation.

        Args:
            G: Directed graph to enrich in place.
        """
        trade_pf = {
            a.rapids_code: _prefixes(a.title)
            for a in self.apprenticeships
        }
        prog_pf = {
            (p.institution, p.program): _prefixes(p.program)
            for p in self.programs
        }

        node_pf_map = {
            node: {
                p
                for text in (*data.get("terms", []), *data.get("skills", []))
                for p in _prefixes(text)
            }
            for node, data in G.nodes(data=True)
        }

        matches = {
            node: match
            for node, node_pf in node_pf_map.items()
            if (match := next(
                (a for a in self.apprenticeships
                 if node_pf & trade_pf[a.rapids_code]),
                None
            ))
        }

        set_node_attributes(G, {
            node: {
                "trade"      : match.title,
                "term_hours" : match.term_hours
            }
            for node, match in matches.items()
        })

        set_node_attributes(G, {
            node: [
                p.model_dump()
                for p in self.programs
                if node_pf & prog_pf[p.institution, p.program]
            ]
            for node, node_pf in node_pf_map.items()
        }, "programs")

        set_edge_attributes(G, {
            (s, t): str(int(matches[t].term_hours) - int(matches[s].term_hours))
            for s, t in G.edges()
            if s in matches and t in matches
        }, "term_hours_delta")

    def _skill_indices(self, cid: int) -> list[int]:
        """
        Map a cluster's skills to PPMI matrix column positions.

        Args:
            cid: Cluster ID to resolve.

        Returns:
            Column indices for skills present in the network
            vocabulary.
        """
        return [
            self.network.feature_index[s]
            for s in self.profiles[cid].skills
            if s in self.network.feature_index
        ]


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
        remove_node_attributes(scalar_graph, "programs", "skills", "terms")
        write_graphml(scalar_graph, graphml_path)

        json_path = output_dir / "career_graph.json"
        json_path.write_text(
            dumps(node_link_data(self.graph), indent=2)
        )

        return GraphExport(
            graphml_path = graphml_path,
            json_path    = json_path
        )
