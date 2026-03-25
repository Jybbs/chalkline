"""
Data models for resume matching results.

Captures cluster distance rankings, per-task cosine gap analysis against
O*NET Task+DWA embeddings, and the full match result with reach
exploration.
"""

from pydantic import BaseModel, Field

from chalkline.pathways.schemas import Reach


class ClusterDistance(BaseModel, extra="forbid"):
    """
    Distance from the resume to a single cluster centroid.

    Provides both the cluster identifier and the Euclidean distance in the
    reduced SVD space, enabling the career report to show proximity to all
    career families rather than only the assigned one.
    """

    cluster_id : int = Field(ge=0)
    distance   : float


class MatchResult(BaseModel, extra="forbid"):
    """
    Full result of projecting a resume into the embedding space and matching
    to career families.

    Captures the nearest career family, distances to all clusters, the local
    reach with credential-enriched edges, and per-task gap analysis against
    the matched cluster's O*NET occupation profile.
    """

    cluster_distances : list[ClusterDistance]
    cluster_id        : int = Field(ge=0)
    reach             : Reach
    sector            : str

    coordinates  : list[float]   = Field(default_factory=list)
    demonstrated : list[TaskGap] = Field(default_factory=list)
    gaps         : list[TaskGap] = Field(default_factory=list)

    @property
    def match_distance(self) -> float:
        """
        Euclidean distance to the nearest cluster centroid.
        """
        return self.cluster_distances[0].distance


class TaskGap(BaseModel, extra="forbid"):
    """
    A single O*NET Task or DWA with its cosine similarity to the resume
    embedding.

    Tasks where cos(𝐫, 𝐭ᵢ) ≥ S̃ are demonstrated competencies. Tasks below
    are gaps representing advancement directions.
    """

    name       : str
    similarity : float
