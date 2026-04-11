"""
Data models for resume matching results.

Captures cluster distance rankings, per-task cosine gap analysis against
O*NET Task+DWA embeddings, and the full match result with reach exploration.
"""

from pydantic import BaseModel, Field

from chalkline.pathways.schemas import Reach


class MatchResult(BaseModel, extra="forbid"):
    """
    Full result of projecting a resume into the embedding space and matching
    to career families.

    Captures the nearest career family, distances to all clusters, and the
    local reach with credential-enriched edges.
    """

    cluster_distances : list[float]
    cluster_id        : int = Field(ge=0)
    reach             : Reach

    coordinates: list[float] = Field(default_factory=list)

    @property
    def confidence(self) -> int:
        """
        Match confidence as a 0-100 percentage, inversely proportional to
        distance relative to the farthest cluster.
        """
        distances = self.cluster_distances
        return round(100 * (1 - min(distances) / max(distances)))


class ScoredTask(BaseModel, extra="forbid"):
    """
    A single O*NET Task or DWA with its cosine similarity to the resume
    embedding and a flag indicating whether the resume demonstrated
    competency (similarity >= median) or revealed a gap (similarity <
    median).
    """

    demonstrated : bool
    name         : str
    similarity   : float = Field(ge=-1, le=1)

    @property
    def pct(self) -> float:
        """
        Similarity as a 0-100 percentage.
        """
        return round(self.similarity * 100, 1)
