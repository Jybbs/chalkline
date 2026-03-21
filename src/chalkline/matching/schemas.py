"""
Data models for resume matching results.

Captures cluster assignments, neighborhood exploration with per-edge
credential metadata, and per-task cosine gap analysis against O*NET Task+DWA
embeddings.
"""

from pydantic import BaseModel, Field

from chalkline.extraction.schemas import Certification
from chalkline.pipeline.schemas   import ApprenticeshipContext, ClusterProfile
from chalkline.pipeline.schemas   import ProgramRecommendation


class CareerEdge(BaseModel, extra="forbid"):
    """
    A single edge in the neighborhood view with credential metadata.

    Each edge connects the matched cluster to a neighboring cluster,
    carrying the target cluster's profile, the cosine similarity weight,
    and the credentials that bridge the specific transition, filtered by
    the destination_percentile and source_percentile dual-threshold rule.
    """

    profile : ClusterProfile
    weight  : float

    apprenticeships : list[ApprenticeshipContext] = Field(default_factory=list)
    certifications  : list[Certification]         = Field(default_factory=list)
    programs        : list[ProgramRecommendation] = Field(default_factory=list)


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
    neighborhood with credential-enriched edges, and per-task gap analysis
    against the matched cluster's O*NET occupation profile.
    """

    cluster_distances : list[ClusterDistance]
    cluster_id        : int = Field(ge=0)
    neighborhood      : Neighborhood
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


class Neighborhood(BaseModel, extra="forbid"):
    """
    Local neighborhood exploration view from the matched cluster.

    Shows advancement paths (edges to higher Job Zone clusters) and lateral
    pivots (edges to same Job Zone clusters), each with per-edge credential
    metadata identifying the training that bridges each specific transition.
    """

    advancement : list[CareerEdge] = Field(default_factory=list)
    lateral     : list[CareerEdge] = Field(default_factory=list)

    @property
    def all_edges(self) -> list[CareerEdge]:
        """
        Combined advancement and lateral edges.
        """
        return self.advancement + self.lateral


class TaskGap(BaseModel, extra="forbid"):
    """
    A single O*NET Task or DWA with its cosine similarity to the resume
    embedding.

    Tasks where cos(𝐫, 𝐭ᵢ) ≥ S̃ are demonstrated competencies. Tasks below
    are gaps representing advancement directions.
    """

    name       : str
    similarity : float
