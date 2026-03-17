"""
Data models for resume matching results.

Captures cluster assignments, nearest-neighbor postings with Jaccard
similarities, and ranked skill gaps with apprenticeship and educational
program enrichment annotations.
"""

from pydantic import BaseModel, Field, computed_field

from chalkline.pipeline.schemas import ApprenticeshipContext
from chalkline.pipeline.schemas import ProgramRecommendation


class ClusterDistance(BaseModel, extra="forbid"):
    """
    Distance from the resume to a single cluster centroid.

    Provides both the cluster identifier and the geometric distance
    in PCA space, enabling the career report to show proximity to
    all career families rather than only the assigned one.
    """

    cluster_id : int = Field(ge = 0)
    distance   : float


class NeighborMatch(BaseModel, extra="forbid"):
    """
    A single nearest-neighbor posting within the assigned career
    family.

    Carries both the geometric distance in PCA space and the Jaccard
    similarity on discrete skill sets, giving two complementary views
    of how closely the resume resembles the posting.
    """

    distance    : float
    document_id : str
    jaccard     : float = Field(ge = 0, le = 1)
    skills      : list[str]


class RankedGap(BaseModel, extra="forbid"):
    """
    A single skill gap ranked by PPMI relevance to the resume.

    The relevance score is the mean PPMI between the gap skill and
    the resume's existing skills, scoped to the matched cluster's
    TF-IDF centroid terms. Apprenticeship and program annotations
    provide actionable next steps when the gap aligns with a known
    training pathway.
    """

    relevance : float
    skill     : str

    apprenticeships : list[ApprenticeshipContext] = Field(default_factory = list)
    programs        : list[ProgramRecommendation] = Field(default_factory = list)


class MatchResult(BaseModel, extra="forbid"):
    """
    Full result of projecting a resume into PCA space and matching
    to career families.

    Captures the nearest career family, distances to all clusters,
    top-5 nearest postings with Jaccard scores, the raw skill gap,
    PPMI-ranked gaps with enrichment annotations, and aggregate
    apprenticeship and program recommendations across all gaps.
    """

    cluster_distances : list[ClusterDistance]
    cluster_id        : int = Field(ge = 0)
    nearest_neighbors : list[NeighborMatch]
    resume_skills     : list[str]

    pca_coordinates : list[float]     = Field(default_factory = list)
    ranked_gaps     : list[RankedGap] = Field(default_factory = list)
    sector          : str | None      = None
    unrankable_gaps : list[str]       = Field(default_factory = list)

    @computed_field
    @property
    def programs(self) -> list[ProgramRecommendation]:
        """
        Deduplicated program recommendations across all ranked
        gaps, keyed by `(institution, program)` pair.
        """
        return list({
            (p.institution, p.program): p
            for gap in self.ranked_gaps for p in gap.programs
        }.values())

    @computed_field
    @property
    def skill_gaps(self) -> list[str]:
        """
        Sorted union of ranked and unrankable gap skill names
        for display in the career report.
        """
        return sorted(
            {g.skill for g in self.ranked_gaps}
            | set(self.unrankable_gaps)
        )

    @computed_field
    @property
    def trade_paths(self) -> list[ApprenticeshipContext]:
        """
        Deduplicated apprenticeship paths across all ranked
        gaps, keyed by RAPIDS code.
        """
        return list({
            a.rapids_code: a
            for gap in self.ranked_gaps for a in gap.apprenticeships
        }.values())
