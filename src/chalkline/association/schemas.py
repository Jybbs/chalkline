"""
Schemas for co-occurrence network results and Apriori comparison.

Pydantic models capturing outputs from PMI-based graph construction,
Louvain community detection, and Apriori frequent itemset mining.
"""

from pydantic import BaseModel, Field

from chalkline import NonEmptyStr


class AprioriResult(BaseModel, extra="forbid"):
    """
    Output from Apriori frequent itemset mining.

    Captures the support threshold used, itemset and rule counts,
    Jaccard overlap against PPMI positive pairs, and a summary of the
    top association rules with their metrics.
    """

    min_support : float
    n_itemsets  : int = Field(ge = 0)
    n_rules     : int = Field(ge = 0)

    overlap_jaccard : float | None = None
    rules_summary   : list[dict]   = Field(default_factory = list)


class CommunityLabel(BaseModel, extra="forbid"):
    """
    Human-readable label for a single Louvain community.

    The `top_skills` are the three highest weighted-degree nodes
    within the community, serving as the community's descriptor for
    downstream labeling in the pathway graph.
    """

    community_id        : int = Field(ge = 0)
    size                : int = Field(ge = 1)
    top_skills          : list[str]
    weighted_degree_sum : float


class GraphDiagnostics(BaseModel, extra="forbid"):
    """
    Network-level diagnostics for the co-occurrence graph.

    Modularity is `None` when the graph is degenerate (no edges or a
    single connected component with all nodes in one community).
    """

    connected_components : int = Field(ge = 0)
    coverage             : float
    edge_count           : int = Field(ge = 0)
    isolate_count        : int = Field(ge = 0)
    largest_component    : int = Field(ge = 0)
    node_count           : int = Field(ge = 0)
    performance          : float

    modularity: float | None = None


class MeasureComparison(BaseModel, extra="forbid"):
    """
    Summary statistics for a single PMI variant's graph.

    Used to compare PPMI, NPMI, and G-test graph structures side by
    side.
    """

    density       : float
    edge_count    : int = Field(ge = 0)
    measure       : NonEmptyStr
    n_communities : int = Field(ge = 0)
