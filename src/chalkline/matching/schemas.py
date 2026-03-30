"""
Data models for resume matching results.

Captures cluster distance rankings, per-task cosine gap analysis against
O*NET Task+DWA embeddings, and the full match result with reach exploration.
"""

from collections import Counter
from operator    import attrgetter
from pydantic    import BaseModel, Field, field_validator
from statistics  import fmean

from chalkline.pathways.schemas import Reach


class ClusterDistance(BaseModel, extra="forbid"):
    """
    Distance from the resume to a single cluster centroid.

    Provides both the cluster identifier and the Euclidean distance in the
    reduced SVD space, enabling the career report to show proximity to all
    career families rather than only the assigned one.
    """

    cluster_id : int   = Field(ge=0)
    distance   : float = Field(ge=0)
    job_zone   : int
    sector     : str
    soc_title  : str


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
    scored_tasks      : list[ScoredTask]
    sector            : str

    coordinates : list[float] = Field(default_factory=list)

    @field_validator("cluster_distances", mode="after")
    @classmethod
    def _sort_by_distance(cls, v: list[ClusterDistance]) -> list[ClusterDistance]:
        """
        Ensure ascending distance order so `match_distance`, `max_distance`,
        and `top(n)` operate on sorted data.
        """
        return sorted(v, key=attrgetter("distance"))

    @property
    def all_similarities(self) -> list[float]:
        """
        Similarity percentages (0-100) across all tasks.
        """
        return [round(t.similarity * 100, 1) for t in self.scored_tasks]

    @property
    def cluster_ids(self) -> list[int]:
        """
        Cluster IDs in distance-sorted order.
        """
        return [cd.cluster_id for cd in self.cluster_distances]

    @property
    def confidence(self) -> int:
        """
        Match confidence as a 0-100 percentage, inversely proportional to
        distance relative to the farthest cluster.
        """
        return round(100 * (1 - self.match_distance / self.max_distance))

    @property
    def demonstrated_count(self) -> int:
        """
        Number of tasks above the median similarity threshold.
        """
        return sum(t.demonstrated for t in self.scored_tasks)

    @property
    def distances_by_sector(self) -> dict[str, list[float]]:
        """
        Cluster distances grouped by sector name.
        """
        by_sector: dict[str, list[float]] = {}
        for cd in self.cluster_distances:
            by_sector.setdefault(cd.sector, []).append(cd.distance)
        return by_sector

    @property
    def gap_count(self) -> int:
        """
        Number of tasks below the median similarity threshold.
        """
        return len(self.scored_tasks) - self.demonstrated_count

    @property
    def gap_type_counts(self) -> dict[str, int]:
        """
        Number of gap tasks per skill type.
        """
        return Counter(
            t.skill_type for t in self.scored_tasks
            if not t.demonstrated
        )

    @property
    def match_distance(self) -> float:
        """
        Euclidean distance to the nearest cluster centroid.
        """
        return self.cluster_distances[0].distance

    @property
    def max_distance(self) -> float:
        """
        Euclidean distance to the farthest cluster centroid.
        """
        return max(cd.distance for cd in self.cluster_distances)

    @property
    def mean_similarity(self) -> float:
        """
        Mean similarity percentage (0-100) across all tasks.
        """
        sims = self.all_similarities
        return round(fmean(sims), 1) if sims else 0.0

    @property
    def tasks_by_type(self) -> dict[str, list[ScoredTask]]:
        """
        All scored tasks grouped by skill type, preserving sort order within
        each group.
        """
        groups: dict[str, list[ScoredTask]] = {}
        for t in self.scored_tasks:
            groups.setdefault(t.skill_type, []).append(t)
        return dict(sorted(groups.items()))


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

    skill_type : str = "other"

    @property
    def pct(self) -> float:
        """
        Similarity as a 0-100 percentage.
        """
        return round(self.similarity * 100, 1)
