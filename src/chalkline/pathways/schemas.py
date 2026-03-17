"""
Schemas for career pathway graph routing.

Defines centrality metrics, career route and transition step models
for widest-path routing, longest-path results, and progressive
learning plans.
"""

from functools import cached_property
from pydantic  import BaseModel, Field

from chalkline.pipeline.schemas import ApprenticeshipContext, ProgramRecommendation


class TransitionStep(BaseModel, extra="forbid"):
    """
    A single career transition between adjacent clusters on a
    widest-path route.

    Bridging skills are the set difference between the target and
    source cluster skill profiles. Apprenticeship and program
    annotations are matched against the bridging skills via 4-char
    prefix overlap, identifying training pathways that address the
    specific skills needed at this transition step.
    """

    bridging_skills : list[str]
    source_cluster  : int = Field(ge=0)
    target_cluster  : int = Field(ge=0)
    weight          : float

    apprenticeships : list[ApprenticeshipContext] = Field(default_factory=list)
    estimated_hours : int | None                  = None
    programs        : list[ProgramRecommendation] = Field(default_factory=list)


class CareerRoute(BaseModel, extra="forbid"):
    """
    A complete path through the career DAG with aggregate metrics.

    The bottleneck weight is the minimum edge weight along the path,
    representing the weakest skill co-occurrence link in the career
    progression. Used by widest-path routing to maximize the minimum
    association strength across all transitions.
    """

    bottleneck_weight : float
    hops              : int = Field(ge=0)
    path              : list[int]

    steps : list[TransitionStep] = Field(default_factory=list)


class CentralityMetrics(BaseModel, extra="forbid"):
    """
    Four centrality measures computed over the career pathway DAG.

    Betweenness identifies gateway roles connecting multiple career
    tracks. In-degree measures convergence (many paths lead here).
    Out-degree measures launch potential (many paths forward).
    PageRank incorporates full graph topology as a prestige measure.
    """

    betweenness : dict[int, float] = Field(default_factory=dict)
    in_degree   : dict[int, float] = Field(default_factory=dict)
    out_degree  : dict[int, float] = Field(default_factory=dict)
    pagerank    : dict[int, float] = Field(default_factory=dict)


class LearningPlan(BaseModel, extra="forbid"):
    """
    Progressive learning plan along a career route.

    Aggregates all bridging skills across transition steps into a
    unified development sequence with per-step enrichment detail
    and route-level summary metrics.
    """

    route : CareerRoute

    @cached_property
    def bridging_skills(self) -> list[str]:
        """
        Sorted union of bridging skills across all transition
        steps in the route.
        """
        return sorted({
            s for step in self.route.steps
            for s in step.bridging_skills
        })

    @cached_property
    def estimated_hours(self) -> int | None:
        """
        Sum of estimated training hours across steps where
        available, or `None` when no step carries hour data.
        """
        hours = [
            s.estimated_hours for s in self.route.steps
            if s.estimated_hours is not None
        ]
        return sum(hours) if hours else None


class LongestPath(BaseModel, extra="forbid"):
    """
    Longest weighted path through the career graph.

    Edge direction follows a strict total order on (Job Zone, cluster ID),
    guaranteeing acyclicity. The longest path identifies the deepest
    career progression chain from entry-level to advanced roles.
    """

    edges       : int        = Field(default=0, ge=0)
    path        : list[int]  = Field(default_factory=list)
    path_weight : float      = 0.0
