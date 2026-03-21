"""
Plain-text career report generation for download.

Produces a formatted summary of the resume match result
suitable for the sidebar download button.
"""

from chalkline.matching.schemas import MatchResult
from chalkline.pipeline.schemas import ClusterProfile


def build_report_text(
    profile : ClusterProfile,
    result  : MatchResult
) -> str:
    """
    Build a downloadable plain-text career report.

    Summarizes the matched career family, demonstrated skills,
    and skill gaps with per-task similarity scores.

    Args:
        profile : Matched cluster's profile.
        result  : Full match result with gaps and demonstrated tasks.

    Returns:
        Newline-joined report string.
    """
    return "\n".join([
        "Chalkline Career Report",
        "=" * 50,
        "",
        f"Career Family: {profile.soc_title}",
        f"Sector: {profile.sector}",
        f"Job Zone: {profile.job_zone}",
        f"Match Distance: {result.match_distance:.4f}",
        "",
        f"Demonstrated Skills ({len(result.demonstrated)}):",
        *[f"  + {d.name} ({d.similarity:.3f})" for d in result.demonstrated],
        "",
        f"Skill Gaps ({len(result.gaps)}):",
        *[f"  - {g.name} ({g.similarity:.3f})" for g in result.gaps]
    ])
