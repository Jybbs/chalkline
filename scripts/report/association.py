"""
DS5230 association mining comparison and network diagnostics.

Apriori frequent itemset mining, Dunning's G-test computation, PMI
measure comparison, Louvain community labeling, network diagnostics,
and apprenticeship trade alignment. Not part of the production
pipeline.
"""

import networkx as nx
import numpy    as np
import pandas   as pd

from itertools                 import takewhile
from logging                   import getLogger
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.sparse              import csr_array, spmatrix, triu
from scipy.special             import xlogy

from chalkline.association.cooccurrence import CooccurrenceNetwork


logger = getLogger(__name__)


def _graph_from_matrix(network: CooccurrenceNetwork, matrix) -> nx.Graph:
    """
    Build a skill-labeled graph from an arbitrary sparse weight
    matrix using the network's feature vocabulary.
    """
    return nx.relabel_nodes(
        nx.from_scipy_sparse_array(matrix, edge_attribute = "weight"),
        dict(enumerate(network.feature_names))
    )


def _louvain(network: CooccurrenceNetwork, G: nx.Graph) -> list[set]:
    """
    Louvain community detection using the network's random seed.
    """
    return nx.community.louvain_communities(
        G, seed = network.random_seed, weight = "weight"
    )


def _partition_map(network: CooccurrenceNetwork) -> dict[str, int]:
    """
    Flat mapping from skill name to Louvain community index on
    the NPMI graph.
    """
    return {
        s: i
        for i, members in enumerate(_louvain(network, network.graph()))
        for s in members
    }


def _ppmi_matrix(network: CooccurrenceNetwork) -> csr_array:
    """
    Compute sparse PPMI from the network's co-occurrence matrix
    and document frequencies.
    """
    C          = network.cooccurrence
    df         = network.doc_freq.astype(float)
    rows, cols = C.nonzero()
    pmi = np.log(
        (network.n_docs * C.data.astype(float))
        / (df[rows] * df[cols])
    )
    ppmi = C.astype(float)
    ppmi.data = np.maximum(pmi, 0)
    ppmi.eliminate_zeros()
    return ppmi


class AprioriComparison:
    """
    Apriori frequent itemset mining for comparing association rules
    against PPMI positive pairs.
    """

    def __init__(self, binary_matrix: spmatrix, feature_names: list[str]):
        self.boolean_df = pd.DataFrame(
            binary_matrix.toarray().astype(bool),
            columns = feature_names
        )

    def mine(self, min_support: float = 0.05) -> dict:
        """
        Mine frequent itemsets and generate association rules.

        Iteratively halves support down to 0.01 until rules with
        lift > 1.0 and confidence >= 0.3 are found.
        """
        itemsets = pd.DataFrame()
        rules    = pd.DataFrame()
        for support in takewhile(
            lambda s: s >= 0.01,
            (min_support / 2**i for i in range(20))
        ):
            itemsets = apriori(
                self.boolean_df,
                min_support  = support,
                use_colnames = True
            )
            if itemsets.empty:
                continue
            rules = association_rules(
                itemsets,
                metric        = "confidence",
                min_threshold = 0.3
            )
            if not (rules := rules[rules["lift"] > 1.0]).empty:
                break

        return {
            "min_support"   : support,
            "n_itemsets"    : len(itemsets),
            "n_rules"       : len(rules),
            "rules_summary" : (top := rules.head(20)).assign(
                antecedents = top["antecedents"].map(sorted),
                consequents = top["consequents"].map(sorted)
            )[
                ["antecedents", "confidence", "consequents", "lift", "support"]
            ].to_dict(orient="records")
        }


def gtest_matrix(network: CooccurrenceNetwork) -> csr_array:
    """
    Dunning's G-test statistics for all thresholded co-occurrence
    pairs. Returns a symmetric sparse matrix.
    """
    C  = triu(network.cooccurrence, format = "coo")
    n  = network.n_docs
    df = network.doc_freq.astype(float)

    a = C.data.astype(float)
    b = df[C.row] - a
    c = df[C.col] - a
    d = n - df[C.row] - df[C.col] + a

    valid      = (b >= 0) & (c >= 0) & (d >= 0)
    rows, cols = C.row[valid], C.col[valid]
    a, b, c, d = a[valid], b[valid], c[valid], d[valid]

    df_r = df[rows]
    df_c = df[cols]
    obs  = np.vstack([a, b, c, d])
    exp  = np.vstack([
        df_r * df_c,
        df_r * (n - df_c),
        (n - df_r) * df_c,
        (n - df_r) * (n - df_c)
    ]) / n

    g = 2 * (xlogy(obs, obs) - xlogy(obs, exp)).sum(axis = 0)

    size  = network.cooccurrence.shape[0]
    upper = csr_array(
        (g, (rows, cols)),
        shape = (size, size)
    )
    return upper + upper.T


def compare_measures(network: CooccurrenceNetwork) -> list[dict]:
    """
    Compare PPMI, NPMI, and G-test graph structures by edge count,
    density, and community count.
    """
    return [
        {
            "density"       : nx.density(G),
            "edge_count"    : G.number_of_edges(),
            "measure"       : name,
            "n_communities" : len(
                _louvain(network, G)
            ) if G.number_of_edges() > 0 else 0
        }
        for name, matrix in [
            ("ppmi",   _ppmi_matrix(network)),
            ("npmi",   network.npmi_matrix),
            ("g-test", gtest_matrix(network))
        ]
        for G in [_graph_from_matrix(network, matrix)]
    ]


def communities(
    network : CooccurrenceNetwork,
    matrix  : spmatrix | None = None
) -> list[dict]:
    """
    Louvain communities labeled by their top-3 weighted-degree
    skills. Defaults to the NPMI graph.
    """
    G = _graph_from_matrix(network, matrix) if matrix is not None else network.graph()
    louvain = _louvain(network, G)

    return [
        {
            "community_id"        : idx,
            "size"                : len(members),
            "top_skills"          : sorted(
                degrees, key = degrees.get, reverse = True
            )[:3],
            "weighted_degree_sum" : sum(degrees.values())
        }
        for idx, members in enumerate(
            sorted(louvain, key = len, reverse = True)
        )
        for degrees in [
            dict(G.subgraph(members).degree(weight = "weight"))
        ]
    ]


def diagnostics(
    network : CooccurrenceNetwork,
    graph   : nx.Graph | None = None
) -> dict:
    """
    Network-level diagnostics: edge count, components, isolates,
    modularity, coverage, and performance. Warns if > 30% isolates.
    """
    G = graph or network.graph()

    components    = list(nx.connected_components(G))
    isolate_count = sum(1 for c in components if len(c) == 1)

    modularity  = None
    coverage    = 0.0
    performance = 0.0

    if G.number_of_edges() > 0:
        louvain    = _louvain(network, G)
        modularity = nx.community.modularity(
            G, louvain, weight = "weight"
        )
        coverage, performance = nx.community.partition_quality(
            G, louvain
        )

    if G.number_of_nodes() > 0 and (
        isolate_rate := isolate_count / G.number_of_nodes()
    ) > 0.30:
        logger.warning(
            f"{isolate_rate:.0%} of skills are isolate nodes "
            f"({isolate_count} of {G.number_of_nodes()})"
        )

    return {
        "connected_components" : len(components),
        "coverage"             : coverage,
        "edge_count"           : G.number_of_edges(),
        "isolate_count"        : isolate_count,
        "largest_component"    : max(
            (len(c) for c in components), default = 0
        ),
        "modularity"           : modularity,
        "node_count"           : G.number_of_nodes(),
        "performance"          : performance
    }


def trade_alignment(
    network         : CooccurrenceNetwork,
    apprenticeships : list[dict]
) -> dict:
    """
    Check which apprenticeship trades appear as graph nodes and
    report their Louvain community assignments.
    """
    nodes = set(network.graph().nodes())
    pmap  = _partition_map(network)

    alignments = [
        {
            "community"   : pmap.get(t := trade["title"].lower()),
            "matched"     : t in nodes,
            "rapids_code" : trade["rapids_code"],
            "title"       : trade["title"]
        }
        for trade in apprenticeships
    ]

    matched_count = sum(1 for a in alignments if a["matched"])
    logger.info(
        f"Trade alignment: {matched_count}/{len(apprenticeships)} "
        f"trades found in vocabulary"
    )

    return {
        "alignments"    : alignments,
        "matched_count" : matched_count,
        "total_trades"  : len(apprenticeships)
    }
