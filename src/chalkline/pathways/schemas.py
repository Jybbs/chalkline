"""
Schemas for career pathway graph construction and diagnostics.

Defines node and edge attribute models, alignment diagnostics between
Louvain communities and HAC clusters, DAG derivation results, and export
artifact paths consumed by the career report and pathway routing modules.
"""

from pathlib  import Path
from pydantic import BaseModel, Field

from chalkline                 import NonEmptyStr
from chalkline.pipeline.schemas import ProgramRecommendation


class AlignmentDiagnostics(BaseModel, extra="forbid"):
    """
    Diagnostic alignment between Louvain communities and HAC clusters.

    Captures the adjusted Rand index between the two partitions projected
    onto the shared skill space, plus the standalone Louvain modularity on
    the NPMI skill graph. An ARI above 0.3 indicates sufficient alignment
    between the geometric and co-occurrence views of the career landscape.
    """

    ari        : float
    modularity : float | None = None


class CareerEdge(BaseModel, extra="forbid"):
    """
    Attributes stored on each directed edge in the career graph.

    The `weight` and `mean_pmi` fields carry the same value, with `weight`
    serving the NetworkX convention for shortest-path and centrality
    computations and `mean_pmi` providing semantic clarity for analysis
    output.
    """

    direction_source : NonEmptyStr
    mean_pmi         : float
    weight           : float

    term_hours_delta : str | None = None


class CareerNode(BaseModel, extra="forbid"):
    """
    Attributes stored on each node in the career graph.

    Each node represents a single HAC cluster (career family) with its
    aggregated skill profile, Job Zone assignment, sector label, and
    optional enrichments from apprenticeship and educational program
    reference data.
    """

    cluster_id : int = Field(ge = 0)
    job_zone   : int = Field(ge = 1, le = 5)
    sector     : NonEmptyStr
    size       : int = Field(ge = 1)
    skills     : list[str]
    terms      : list[str]

    programs   : list[ProgramRecommendation] = Field(default_factory = list)
    term_hours : str | None                  = None
    trade      : str | None                  = None


class DagResult(BaseModel, extra="forbid"):
    """
    Output from the directed acyclic graph derivation.

    The DAG is a lossy projection of the primary cyclic graph, produced by
    iteratively removing the lowest-weight edge per cycle. The longest path
    through this DAG identifies the deepest career progression chain.
    """

    edges_removed : int = Field(ge = 0)
    longest_path  : list[int]
    path_weight   : float


class GraphExport(BaseModel, extra="forbid"):
    """
    Paths to exported graph serialization artifacts.

    GraphML contains only scalar node and edge attributes for
    interoperability with Gephi and Cytoscape. JSON preserves full attribute
    fidelity including nested program lists and serves as the primary
    pipeline serialization format.
    """

    graphml_path : Path
    json_path    : Path


class PathwayGraphResult(BaseModel, extra="forbid"):
    """
    Top-level diagnostic summary for the career pathway graph.

    Aggregates alignment diagnostics, DAG derivation output, and graph-level
    statistics into a single inspectable result.
    """

    alignment  : AlignmentDiagnostics
    dag        : DagResult
    density    : float
    edge_count : int = Field(ge = 0)
    node_count : int = Field(ge = 1)

    cycles_found : int = Field(ge = 0, default = 0)


class SocMatch(BaseModel, extra="forbid"):
    """
    Overlap coefficient result for a single cluster-to-SOC match.

    Used during cluster-level Job Zone assignment, where each cluster's
    union skill set is compared against SOC concrete skill profiles via
    the overlap coefficient.
    """

    job_zone : int = Field(ge = 1, le = 5)
    overlap  : float = Field(ge = 0, le = 1)
    soc_code : NonEmptyStr
