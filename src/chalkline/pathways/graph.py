"""
Career pathway graph from HAC clusters and PMI co-occurrence edges.

Constructs a directed weighted DiGraph where nodes are job clusters from
hierarchical agglomerative clustering and edges connect clusters whose skill
profiles share significant PMI co-occurrence. Edge direction defaults to
O*NET Job Zone levels, with enrichments from apprenticeship term hours and
educational program metadata attached to nodes and edges. The primary graph
retains cycles for legitimate lateral transitions, with a derived DAG view
for longest-path visualization.
"""

import networkx as nx
import numpy    as np

from collections             import Counter
from functools               import cached_property
from itertools               import combinations
from json                    import dumps
from logging                 import getLogger
from pathlib                 import Path
from statistics              import median

from networkx.readwrite.json_graph import node_link_data
from sklearn.metrics               import adjusted_rand_score

from chalkline.association.cooccurrence import CooccurrenceNetwork
from chalkline.clustering.schemas       import ClusterLabel
from chalkline.extraction.occupations   import OccupationIndex
from chalkline.pathways.schemas         import AlignmentDiagnostics
from chalkline.pathways.schemas         import DagResult, GraphExport
from chalkline.pathways.schemas         import PathwayGraphResult, SocMatch
from chalkline.pipeline.schemas         import ApprenticeshipContext
from chalkline.pipeline.schemas         import ProgramRecommendation


logger = getLogger(__name__)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _node_prefixes(G: nx.DiGraph, node: int) -> set[str]:
    """
    Union of 4-char word prefixes from a node's terms and skills.

    Args:
        G    : Career graph containing node attribute data.
        node : Node ID to extract prefixes from.

    Returns:
        Set of 4-character prefixes from the node's terms and skills.
    """
    data = G.nodes[node]
    return {
        prefix
        for text in (*data.get("terms", []), *data.get("skills", []))
        for prefix in _prefix_set(text)
    }


def _prefix_set(text: str) -> set[str]:
    """
    Extract 4-character word prefixes from text.

    Filters to words of 4+ characters and truncates to 4-char prefixes,
    catching inflectional variants across the construction domain
    (welding/welder, electrical/electrician, scaffolding/scaffold).

    Args:
        text: Input string to extract prefixes from.

    Returns:
        Set of 4-character lowercase prefixes.
    """
    return {w[:4] for w in text.lower().split() if len(w) >= 4}


# -------------------------------------------------------------------------
# Career Pathway Graph
# -------------------------------------------------------------------------

class CareerPathwayGraph:
    """
    Directed weighted career graph with enrichment and traversal.

    Receives HAC cluster assignments and PMI co-occurrence weights from the
    upstream pipeline, constructs a DiGraph with one node per cluster,
    computes inter-cluster edge weights via top-k mean PPMI, assigns Job
    Zone direction from overlap coefficient matching against O*NET concrete
    skill profiles, and attaches sector labels, apprenticeship term hours,
    and educational program metadata. The primary graph retains cycles, with
    a derived DAG for longest-path analysis.
    """

    def __init__(
        self,
        apprenticeships  : list[dict],
        assignments      : np.ndarray,
        cluster_labels   : list[ClusterLabel],
        document_ids     : list[str],
        extracted_skills : dict[str, list[str]],
        network          : CooccurrenceNetwork,
        occupation_index : OccupationIndex,
        programs         : list[ProgramRecommendation],
        sector_labels    : list[str] | None = None
    ):
        """
        Build the career pathway graph from upstream artifacts.

        Eagerly computes cluster skill sets, Job Zone assignments, sector
        labels, edge weights, enrichments, and all-pairs shortest path
        lengths. ARI alignment and DAG derivation are deferred to cached
        properties because they serve diagnostic and visualization roles
        respectively.

        Args:
            apprenticeships  : Raw dicts from `apprenticeships.json` with
                               `title`, `rapids_code`, and `term_hours`.
            assignments      : Cluster ID per posting from
                               `HierarchicalClusterer.assignments`.
            cluster_labels   : TF-IDF centroid labels from
                               `HierarchicalClusterer.labels()`.
            document_ids     : Posting identifiers in matrix row order.
            extracted_skills : Mapping from document identifier to sorted
                               canonical skill names.
            network          : Co-occurrence network providing PPMI matrix,
                               feature names, Louvain communities, and
                               skill graph.
            occupation_index : O*NET occupation lookup for SOC matching
                               and sector labels.
            programs         : Normalized educational programs from
                               `load_programs()`.
            sector_labels    : Pre-computed SOC codes per posting from
                               `compute_sector_labels()`, aligned with
                               `document_ids`. Skips per-posting nearest-SOC
                               computation when provided.
        """
        self.apprenticeships  = apprenticeships
        self.assignments      = assignments
        self.cluster_labels   = cluster_labels
        self.document_ids     = document_ids
        self.extracted_skills = extracted_skills
        self.network          = network
        self.occupation_index = occupation_index
        self.programs         = programs

        self.skill_index       = {n: i for i, n in enumerate(network.feature_names)}
        self.cluster_ids       = sorted(set(int(a) for a in assignments))
        self.label_map         = {l.cluster_id: l for l in cluster_labels}
        self.cluster_skills    = self._aggregate_cluster_skills()
        self.concrete_profiles = {
            soc: {s.name.lower() for s in occ.skills if s.type.is_concrete}
            for soc, occ in occupation_index.occupation_map.items()
        }

        self.job_zones       = {c: self._assign_job_zone(c) for c in self.cluster_ids}
        self.cluster_sectors = self._compute_cluster_sectors(sector_labels)

        self.apprenticeship_models = [
            ApprenticeshipContext(
                rapids_code = a["rapids_code"],
                term_hours  = a["term_hours"],
                trade       = a["title"]
            )
            for a in apprenticeships
        ]
        self.trade_prefixes = {
            a.rapids_code: _prefix_set(a.trade)
            for a in self.apprenticeship_models
        }
        self.program_prefixes = {
            (p.institution, p.program): _prefix_set(p.program)
            for p in programs
        }

        self.graph = self._build_graph()

        cycles = list(nx.simple_cycles(self.graph))
        self.cycle_count = len(cycles)
        if cycles:
            logger.info(f"Found {len(cycles)} cycle(s) in the career graph")
            for cycle in cycles:
                logger.debug(f"Cycle: {cycle}")

        self.shortest_paths = dict(
            nx.all_pairs_dijkstra_path_length(self.graph, weight="weight")
        )

    # -----------------------------------------------------------------
    # Cached properties
    # -----------------------------------------------------------------

    @cached_property
    def alignment(self) -> AlignmentDiagnostics:
        """
        ARI between Louvain communities and HAC cluster partitions projected
        onto the shared skill space.

        For each skill present in both the NPMI graph and at least one
        cluster, assigns a Louvain community ID and an HAC cluster ID (the
        cluster where the skill appears most frequently). The adjusted
        Rand index quantifies agreement between these two partitions.
        Modularity is computed on the Louvain partition of the NPMI skill
        graph as a standalone measure of community structure quality.

        Returns:
            Diagnostic alignment with ARI and optional Louvain modularity.
        """
        skill_graph = self.network.graph()
        louvain_partition = sorted(
            nx.community.louvain_communities(
                skill_graph, seed=self.network.random_seed, weight="weight"
            ),
            key=len, reverse=True
        )

        skill_to_community: dict[str, int] = {}
        for comm_id, members in enumerate(louvain_partition):
            for skill in members:
                skill_to_community[skill] = comm_id

        skill_counts: dict[str, Counter[int]] = {}
        for cid in self.cluster_ids:
            for skill in self.cluster_skills[cid]:
                skill_counts.setdefault(skill, Counter())[cid] += 1
        skill_to_cluster = {
            skill: counts.most_common(1)[0][0]
            for skill, counts in skill_counts.items()
        }

        shared = sorted(set(skill_to_community) & set(skill_to_cluster))
        if not shared:
            logger.warning(
                "No shared skills between Louvain communities and "
                "HAC clusters for ARI computation"
            )
            return AlignmentDiagnostics(ari=0.0)

        louvain_labels = [skill_to_community[s] for s in shared]
        hac_labels     = [skill_to_cluster[s] for s in shared]
        ari = float(adjusted_rand_score(louvain_labels, hac_labels))

        modularity = None
        if skill_graph.number_of_edges() > 0:
            modularity = float(nx.community.modularity(
                skill_graph, louvain_partition, weight="weight"
            ))

        if ari <= 0.3:
            logger.warning(
                f"Low ARI ({ari:.3f}) between Louvain communities and "
                f"HAC clusters, limited agreement between partitions"
            )

        return AlignmentDiagnostics(ari=ari, modularity=modularity)

    @cached_property
    def dag(self) -> DagResult:
        """
        Directed acyclic graph derived from the primary career graph.

        Copies the primary graph and iteratively removes the lowest-weight
        edge per cycle until the graph is acyclic. Computes the longest
        path through the resulting DAG via `dag_longest_path` as the
        deepest career progression chain.

        Returns:
            DAG derivation output with edges removed, longest path,
            and weight.
        """
        dag_view      = self.graph.copy()
        edges_removed = 0

        while not nx.is_directed_acyclic_graph(dag_view):
            try:
                cycle = next(nx.simple_cycles(dag_view))
            except StopIteration:
                break

            min_edge = min(
                ((cycle[i], cycle[(i + 1) % len(cycle)])
                 for i in range(len(cycle))),
                key=lambda e: dag_view.edges[e].get("weight", 0)
            )
            dag_view.remove_edge(*min_edge)
            edges_removed += 1

        self.dag_view = dag_view

        if dag_view.number_of_nodes() == 0:
            return DagResult(
                edges_removed = edges_removed,
                longest_path  = [],
                path_weight   = 0.0
            )

        longest = nx.dag_longest_path(
            dag_view, weight="weight", default_weight=0
        )
        path_weight = sum(
            dag_view.edges[longest[i], longest[i + 1]].get("weight", 0)
            for i in range(len(longest) - 1)
        )

        return DagResult(
            edges_removed = edges_removed,
            longest_path  = longest,
            path_weight   = path_weight
        )

    @cached_property
    def result(self) -> PathwayGraphResult:
        """
        Aggregated diagnostic summary of the career pathway graph.

        Returns:
            Diagnostic summary with alignment, DAG, and graph statistics.
        """
        return PathwayGraphResult(
            alignment    = self.alignment,
            cycles_found = self.cycle_count,
            dag          = self.dag,
            density      = nx.density(self.graph),
            edge_count   = self.graph.number_of_edges(),
            node_count   = self.graph.number_of_nodes()
        )

    # -----------------------------------------------------------------
    # Private methods
    # -----------------------------------------------------------------

    def _aggregate_cluster_skills(self) -> dict[int, set[str]]:
        """
        Build the union skill set for each cluster from constituent
        postings.

        Returns:
            Mapping from cluster ID to union skill set.
        """
        cluster_skills: dict[int, set[str]] = {cid: set() for cid in self.cluster_ids}
        for doc, cid in zip(self.document_ids, self.assignments):
            cluster_skills[int(cid)].update(self.extracted_skills.get(doc, []))
        return cluster_skills

    def _assign_job_zone(self, cluster_id: int) -> int:
        """
        Assign a Job Zone via overlap coefficient against concrete SOC
        profiles.

        Computes |A & B| / min(|A|, |B|) between the cluster's union skill
        set and each SOC code's concrete skill profile (Tasks, Technology
        Skills, Tools, DWAs only). Takes the median Job Zone of the top-3
        SOC matches.

        Args:
            cluster_id: Cluster to assign.

        Returns:
            Job Zone integer from 1 to 5.
        """
        cluster_skills = {s.lower() for s in self.cluster_skills[cluster_id]}
        if not cluster_skills:
            return 2

        matches = []
        for soc, profile in self.concrete_profiles.items():
            if not profile:
                continue
            denominator = min(len(cluster_skills), len(profile))
            occ         = self.occupation_index.get(soc)
            matches.append(SocMatch(
                job_zone = occ.job_zone,
                overlap  = len(cluster_skills & profile) / denominator,
                soc_code = occ.soc_code
            ))

        if not matches:
            return 2

        top_3 = sorted(matches, key=lambda m: m.overlap, reverse=True)[:3]
        return int(median(m.job_zone for m in top_3))

    def _build_graph(self) -> nx.DiGraph:
        """
        Construct the directed weighted career graph.

        Adds one node per cluster with Job Zone, sector, skill, and term
        attributes. Computes top-k mean PPMI for each cluster pair, applies
        the adaptive 75th-percentile threshold, and creates directed edges
        from lower to higher Job Zone. Enriches nodes with apprenticeship
        and program metadata.

        Returns:
            Directed weighted `nx.DiGraph` with enriched nodes and edges.
        """
        G = nx.DiGraph()

        for cid in self.cluster_ids:
            label = self.label_map.get(cid)
            G.add_node(
                cid,
                cluster_id = cid,
                job_zone   = self.job_zones[cid],
                sector     = self.cluster_sectors[cid],
                size       = int((self.assignments == cid).sum()),
                skills     = sorted(self.cluster_skills[cid]),
                terms      = label.terms if label else []
            )

        candidates = []
        for ci, cj in combinations(self.cluster_ids, 2):
            weight, positive_count = self._compute_edge_weight(ci, cj)
            if weight > 0 and positive_count >= 3:
                candidates.append((ci, cj, weight))

        if candidates:
            threshold = float(np.percentile([w for _, _, w in candidates], 75))
            for ci, cj, weight in candidates:
                if weight < threshold:
                    continue

                jz_i = self.job_zones[ci]
                jz_j = self.job_zones[cj]

                if jz_i < jz_j:
                    source, target = ci, cj
                elif jz_i > jz_j:
                    source, target = cj, ci
                else:
                    source, target = min(ci, cj), max(ci, cj)

                G.add_edge(
                    source, target,
                    direction_source = "job_zone",
                    mean_pmi         = weight,
                    weight           = weight
                )

        self._enrich_apprenticeships(G)
        self._enrich_programs(G)
        return G

    def _compute_cluster_sectors(
        self,
        sector_labels: list[str] | None
    ) -> dict[int, str]:
        """
        Assign majority sector to each cluster.

        When pre-computed `sector_labels` are provided (SOC codes per
        posting from `compute_sector_labels`), maps each SOC to its sector
        and aggregates by cluster assignment. This avoids re-running the
        per-posting nearest-SOC computation which dominates construction
        time via `cdist`.

        Falls back to on-demand `occupation_index.nearest()` per posting
        when no labels are provided.

        Args:
            sector_labels: Pre-computed SOC codes aligned with
                           `document_ids`, or None to compute on demand.

        Returns:
            Mapping from cluster ID to majority sector label.
        """
        if sector_labels is not None:
            posting_sectors = [
                self.occupation_index.get(soc).sector
                for soc in sector_labels
            ]
        else:
            posting_sectors = []
            for doc in self.document_ids:
                skills = set(self.extracted_skills.get(doc, []))
                if skills:
                    soc = self.occupation_index.nearest(skills)
                    posting_sectors.append(self.occupation_index.get(soc).sector)
                else:
                    posting_sectors.append("Unknown")

        cluster_sectors: dict[int, str] = {}
        for cid in self.cluster_ids:
            counts = Counter(
                sector for sector, assignment
                in zip(posting_sectors, self.assignments)
                if int(assignment) == cid
            )
            cluster_sectors[cid] = counts.most_common(1)[0][0] if counts else "Unknown"
        return cluster_sectors

    def _compute_edge_weight(self, ci: int, cj: int) -> tuple[float, int]:
        """
        Compute top-k mean PPMI for a cluster pair.

        Extracts the PPMI submatrix for inter-cluster skill pairs, filters
        to positive values, and returns the mean of the top-k values
        where k = min(10, |Ci|, |Cj|).

        Args:
            ci : First cluster ID.
            cj : Second cluster ID.

        Returns:
            Tuple of (top-k mean PPMI, count of positive pairs).
        """
        skills_i  = self.cluster_skills[ci]
        skills_j  = self.cluster_skills[cj]
        indices_i = [self.skill_index[s] for s in skills_i if s in self.skill_index]
        indices_j = [self.skill_index[s] for s in skills_j if s in self.skill_index]

        if not indices_i or not indices_j:
            return 0.0, 0

        values = self.network.ppmi_matrix[np.ix_(indices_i, indices_j)].toarray().ravel()

        positive = values[values > 0]
        if len(positive) == 0:
            return 0.0, 0

        k    = min(10, len(skills_i), len(skills_j), len(positive))
        topk = np.partition(positive, -k)[-k:]
        return float(topk.mean()), int(len(positive))

    def _enrich_apprenticeships(self, G: nx.DiGraph):
        """
        Attach apprenticeship trade and term hours to matching nodes and
        edges.

        For each node, checks whether any apprenticeship trade title has
        word-prefix overlap with the node's terms or skills. For edges
        connecting two apprenticeship-mapped nodes, computes the term
        hours delta.

        Args:
            G: Directed graph to enrich in place.
        """
        node_trades: dict[int, ApprenticeshipContext] = {}

        for node in G.nodes():
            prefixes = _node_prefixes(G, node)
            for a in self.apprenticeship_models:
                if prefixes & self.trade_prefixes[a.rapids_code]:
                    G.nodes[node]["term_hours"] = a.term_hours
                    G.nodes[node]["trade"]      = a.trade
                    node_trades[node] = a
                    break

        for source, target in G.edges():
            if source in node_trades and target in node_trades:
                delta = (int(node_trades[target].term_hours)
                         - int(node_trades[source].term_hours))
                G.edges[source, target]["term_hours_delta"] = str(delta)

    def _enrich_programs(self, G: nx.DiGraph):
        """
        Attach educational program metadata to matching nodes.

        For each node, finds programs whose names have word-prefix overlap
        with the node's terms or skills, and attaches the matching programs
        as a list of dicts.

        Args:
            G: Directed graph to enrich in place.
        """
        for node in G.nodes():
            prefixes = _node_prefixes(G, node)
            G.nodes[node]["programs"] = [
                {
                    "credential"  : p.credential,
                    "institution" : p.institution,
                    "program"     : p.program,
                    "url"         : p.url
                }
                for p in self.programs
                if prefixes & self.program_prefixes.get(
                    (p.institution, p.program), set()
                )
            ]

    # -----------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------

    def export(self, output_dir: Path) -> GraphExport:
        """
        Export the career graph as GraphML and JSON.

        GraphML includes only scalar node and edge attributes for
        interoperability with Gephi and Cytoscape. JSON preserves full
        attribute fidelity including nested program lists via
        `node_link_data`.

        Args:
            output_dir: Directory for export artifacts. Created if it does
                        not exist.

        Returns:
            Export paths for GraphML and JSON serialization artifacts.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        graphml_path = output_dir / "career_graph.graphml"
        scalar_graph = self.graph.copy()
        for node in scalar_graph.nodes():
            for attr in ("programs", "skills", "terms"):
                scalar_graph.nodes[node].pop(attr, None)
        nx.write_graphml(scalar_graph, str(graphml_path))

        json_path = output_dir / "career_graph.json"
        json_path.write_text(dumps(node_link_data(self.graph), indent=2))

        return GraphExport(
            graphml_path = graphml_path,
            json_path    = json_path
        )

    def leads_to(self, cluster_id: int) -> set[int]:
        """
        Cluster IDs that have a directed path to the given cluster.

        Uses `nx.ancestors` on the primary cyclic graph, returning the set
        of clusters from which one can reach the given cluster via directed
        edges.

        Args:
            cluster_id: Target cluster.

        Returns:
            Set of ancestor cluster IDs.
        """
        return nx.ancestors(self.graph, cluster_id)

    def reachable_from(self, cluster_id: int) -> set[int]:
        """
        Cluster IDs reachable via directed paths from the given cluster.

        Uses `nx.descendants` on the primary cyclic graph, returning the
        set of clusters one can reach from the given cluster via directed
        edges.

        Args:
            cluster_id: Source cluster.

        Returns:
            Set of descendant cluster IDs.
        """
        return nx.descendants(self.graph, cluster_id)

    def shortest_path_length(self, source: int, target: int) -> float | None:
        """
        Precomputed shortest path length between two clusters.

        Returns None when no directed path exists from `source` to `target`
        in the primary cyclic graph.

        Args:
            source : Origin cluster ID.
            target : Destination cluster ID.

        Returns:
            Weighted path length, or None if unreachable.
        """
        return self.shortest_paths.get(source, {}).get(target)
