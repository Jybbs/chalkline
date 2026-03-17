"""
Shared enrichment context for apprenticeship and program matching.

Precomputes 4-character prefix lookup dicts once so that cluster
profile enrichment, resume gap annotation, and career route edge
enrichment all share the same matching state rather than
independently rebuilding it.
"""

from chalkline.pipeline.schemas import ApprenticeshipContext
from chalkline.pipeline.schemas import ProgramRecommendation


def prefix_set(text: str) -> set[str]:
    """
    Extract 4-character word prefixes from text.

    Filters to words of 4+ characters and truncates to 4-char
    prefixes, catching inflectional variants across the
    construction domain (welding/welder, electrical/electrician,
    scaffolding/scaffold).

    Args:
        text: Raw text to extract prefixes from.

    Returns:
        Set of lowercased 4-character prefixes.
    """
    return {w[:4] for w in text.lower().split() if len(w) >= 4}


class EnrichmentContext:
    """
    Precomputed prefix lookup dicts for matching skills to
    apprenticeships and educational programs.

    Built once by the orchestrator after deduplicating reference
    data, then passed to `build_profiles`, `ResumeMatcher`, and
    `CareerRouter` so that each consumer uses the same cached
    lookups rather than independently rebuilding them.
    """

    def __init__(
        self,
        apprenticeships : list[ApprenticeshipContext],
        programs        : list[ProgramRecommendation]
    ):
        """
        Precompute prefix dicts from apprenticeship titles and
        program names.

        Args:
            apprenticeships : Deduplicated apprenticeship records.
            programs        : Deduplicated program records.
        """
        self.apprenticeships = apprenticeships
        self.programs        = programs

        self.trade_prefixes = {
            a.rapids_code: prefix_set(a.title)
            for a in apprenticeships
        }
        self.program_prefixes = {
            (p.institution, p.program): prefix_set(p.program)
            for p in programs
        }

    def find_apprenticeship(
        self, prefixes: set[str]
    ) -> ApprenticeshipContext | None:
        """
        First apprenticeship whose trade title shares a prefix
        with the given set.

        Used for cluster-level matching where each cluster gets
        at most one apprenticeship annotation.

        Args:
            prefixes: Precomputed prefix set from cluster
                      terms and skills.

        Returns:
            First matching apprenticeship, or `None`.
        """
        return next(
            (a for a in self.apprenticeships
             if prefixes & self.trade_prefixes[a.rapids_code]),
            None
        )

    def find_apprenticeships(
        self, prefixes: set[str]
    ) -> list[ApprenticeshipContext]:
        """
        All apprenticeships whose trade titles share a prefix
        with the given set.

        Used for gap-level and edge-level matching where multiple
        apprenticeships may apply.

        Args:
            prefixes: Precomputed prefix set from a skill name
                      or bridging skill set.

        Returns:
            All matching apprenticeship records.
        """
        return [
            a for a in self.apprenticeships
            if prefixes & self.trade_prefixes[a.rapids_code]
        ]

    def find_programs(
        self, prefixes: set[str]
    ) -> list[ProgramRecommendation]:
        """
        All programs whose names share a prefix with the given
        set.

        Args:
            prefixes: Precomputed prefix set from a skill name,
                      cluster terms, or bridging skill set.

        Returns:
            All matching program records.
        """
        return [
            p for p in self.programs
            if prefixes & self.program_prefixes[p.institution, p.program]
        ]
