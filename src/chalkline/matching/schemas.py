"""
Data models for resume matching results.

Captures cluster distance rankings, per-task cosine gap analysis against
O*NET Task+DWA embeddings, BM25 weighting configuration, and the full match
result with reach exploration.
"""

from pydantic import BaseModel, Field

from chalkline.pathways.schemas import Reach


class BM25Config(BaseModel, extra="forbid"):
    """
    BM25 scoring parameters for weighting cosine similarity by lexical
    relevance.

    The `length_weight` parameter controls how much task description length
    affects scoring (0 = no normalization, 1 = full normalization). The
    `saturation` parameter controls how quickly term frequency gains
    diminish (higher = slower saturation). With stem sets where each term
    appears at most once, `saturation` primarily affects the ratio of IDF
    contribution to length penalty.
    """

    length_weight : float = Field(default=0.75, ge=0, le=1)
    saturation    : float = Field(default=1.5, gt=0)

    @property
    def base_penalty(self) -> float:
        """
        Length-independent portion of the BM25 denominator, `1 -
        length_weight`. When `length_weight` is 0, the denominator collapses
        to `1 + saturation` regardless of document length.
        """
        return 1 - self.length_weight

    @property
    def numerator(self) -> float:
        """
        BM25 numerator `saturation + 1`, constant across all terms and
        documents for a given config.
        """
        return self.saturation + 1


class MatchResult(BaseModel, extra="forbid"):
    """
    Full result of projecting a resume into the embedding space and matching
    to career families.

    Captures the nearest career family, distances to all clusters, and the
    local reach with credential-enriched edges.
    """

    cluster_distances : list[float]
    cluster_id        : int = Field(ge=0)

    coordinates : list[float] = Field(default_factory=list)
    reach       : Reach       = Field(default_factory=Reach)

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
