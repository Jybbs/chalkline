"""
Schemas for clustering results.

Defines the Pydantic model for cluster labels derived from TF-IDF centroid
terms.
"""

from pydantic import BaseModel, Field


class ClusterLabel(BaseModel, extra="forbid"):
    """
    Human-readable label for a single cluster derived from TF-IDF centroid
    terms.

    The `leader_node_id` maps to the linkage node serving as the subtree
    root for this cluster, enabling direct annotation of dendrogram branches
    with cluster labels.
    """

    cluster_id     : int = Field(ge = 0)
    leader_node_id : int = Field(ge = 0)
    size           : int = Field(ge = 1)
    terms          : list[str]
    weights        : list[float]
