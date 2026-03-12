"""
Schemas for clustering results and validity metrics.
"""

from pydantic import BaseModel, Field

from chalkline import NonEmptyStr


class ClusterLabel(BaseModel, extra="forbid"):
    """
    Human-readable label for a single cluster derived from TF-IDF
    centroid terms.

    The `leader_node_id` maps to the linkage node serving as the
    subtree root for this cluster, enabling direct annotation of
    dendrogram branches with cluster labels.
    """

    cluster_id     : int = Field(ge = 0)
    leader_node_id : int = Field(ge = 0)
    size           : int = Field(ge = 1)
    terms          : list[str]
    weights        : list[float]


class ComparisonResult(BaseModel, extra="forbid"):
    """
    Output from a single comparison clustering method.

    Metric fields are optional because degenerate clusterings
    (fewer than 2 non-noise clusters) cannot produce internal
    validity scores.
    """

    assignments : list[int]
    method      : NonEmptyStr
    n_clusters  : int = Field(ge = 0)

    ari_vs_sectors    : float | None = None
    calinski_harabasz : float | None = None
    davies_bouldin    : float | None = None
    noise_count       : int          = Field(ge = 0, default = 0)
    silhouette        : float | None = None


class CopheneticResult(BaseModel, extra="forbid"):
    """
    Cophenetic correlation for a single linkage method.

    Measures how faithfully the dendrogram preserves pairwise
    distances from the original coordinate space.
    """

    correlation : float
    method      : NonEmptyStr


class ValidationMetrics(BaseModel, extra="forbid"):
    """
    Internal validity metrics for a hierarchical cluster partition.

    All metric fields are optional because degenerate clusterings
    (fewer than 2 clusters) cannot produce internal validity scores.
    `silhouette_samples` is an empty list in the degenerate case.
    """

    calinski_harabasz  : float | None = None
    davies_bouldin     : float | None = None
    silhouette         : float | None = None
    silhouette_samples : list[float]  = Field(default_factory = list)
