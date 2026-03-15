"""
Schemas for career pathway graph diagnostics and export.

Defines alignment diagnostics between Louvain communities and HAC
clusters, longest-path results, export artifact paths, and the top-level
diagnostic summary consumed by the career report.
"""

from pathlib  import Path
from pydantic import BaseModel


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


class LongestPath(BaseModel, extra="forbid"):
    """
    Longest weighted path through the career graph.

    Edge direction follows a strict total order on (Job Zone, cluster ID),
    guaranteeing acyclicity. The longest path identifies the deepest
    career progression chain from entry-level to advanced roles.
    """

    path        : list[int]
    path_weight : float
