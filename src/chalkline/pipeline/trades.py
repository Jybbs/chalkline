"""
Trade index for apprenticeship and program matching.

Matches skill strings against apprenticeship trades and educational
programs via 4-character prefix overlap. Each record carries its own
`prefixes` cached property; this class handles the query side.
"""

from collections.abc import Iterable

from chalkline.pipeline.schemas import ApprenticeshipContext
from chalkline.pipeline.schemas import ProgramRecommendation


def _prefix_set(text: str) -> set[str]:
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


class TradeIndex:
    """
    Matches skill strings against apprenticeship trades and
    educational programs via 4-character prefix overlap.

    Built once by the orchestrator after loading reference data,
    then passed to `build_profiles`, `ResumeMatcher`, and
    `CareerRouter` so that each consumer shares the same
    reference lists.
    """

    def __init__(
        self,
        apprenticeships : list[ApprenticeshipContext],
        programs        : list[ProgramRecommendation]
    ):
        """
        Args:
            apprenticeships : Deduplicated apprenticeship records.
            programs        : Deduplicated program records.
        """
        self.apprenticeships = apprenticeships
        self.programs        = programs

    def find_apprenticeships(
        self, terms: Iterable[str]
    ) -> list[ApprenticeshipContext]:
        """
        All apprenticeships whose trade titles share prefixes
        with the given terms.

        Args:
            terms: Skill names, cluster labels, or other text
                   to match against trade titles.

        Returns:
            All matching apprenticeship records.
        """
        query = {
            p for term in terms
            for p in _prefix_set(term)
        }
        return [
            a for a in self.apprenticeships
            if query & a.prefixes
        ]

    def find_programs(
        self, terms: Iterable[str]
    ) -> list[ProgramRecommendation]:
        """
        All programs whose names share prefixes with the given
        terms.

        Args:
            terms: Skill names, cluster labels, or other text
                   to match against program names.

        Returns:
            All matching program records.
        """
        query = {
            p for term in terms
            for p in _prefix_set(term)
        }
        return [
            p for p in self.programs
            if query & p.prefixes
        ]
