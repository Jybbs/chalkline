"""
Trade index for apprenticeship and program matching.

Holds pre-normalized apprenticeship and educational program reference data
and matches skill strings against both via 4-character prefix overlap. Each
record carries a pre-computed `prefixes` set from curation, and the
`TradeIndex` class handles the query side.
"""

from collections.abc import Iterable
from pathlib         import Path
from pydantic        import TypeAdapter

from chalkline.pipeline.schemas import ApprenticeshipContext, ProgramRecommendation


class TradeIndex:
    """
    Matches skill strings against apprenticeship trades and educational
    programs via 4-character prefix overlap.

    Built once by the orchestrator, then passed to `build_profiles`,
    `ResumeMatcher`, and `CareerRouter` so that each consumer shares the
    same reference lists.
    """

    def __init__(
        self,
        apprenticeships : list[ApprenticeshipContext],
        programs        : list[ProgramRecommendation]
    ):
        """
        Args:
            apprenticeships : With pre-computed 4-char prefix sets from curation.
            programs        : With pre-computed 4-char prefix sets from curation.
        """
        self.apprenticeships = apprenticeships
        self.programs        = programs

    @classmethod
    def from_directory(cls, lexicon_dir: Path) -> "TradeIndex":
        """
        Load apprenticeship and program reference data from a directory.

        Args:
            lexicon_dir: Must contain `apprenticeships.json` and `programs.json`.

        Returns:
            Populated `TradeIndex` for prefix-based matching.
        """
        load = lambda schema, name: TypeAdapter(schema).validate_json(
            (lexicon_dir / name).read_bytes()
        )
        return cls(
            apprenticeships = load(list[ApprenticeshipContext], "apprenticeships.json"),
            programs        = load(list[ProgramRecommendation], "programs.json")
        )

    def lookup(
        self,
        terms: Iterable[str]
    ) -> tuple[list[ApprenticeshipContext], list[ProgramRecommendation]]:
        """
        All apprenticeships and programs whose names share prefixes with the
        given terms.

        Computes the query prefix set once and matches against both
        reference collections in a single pass. Each record's pre-computed
        `prefixes` field provides the reference-side prefix set.

        Args:
            terms: Skill names, cluster labels, or other text to match against reference
                   records.

        Returns:
            Tuple of (matching apprenticeships, matching programs).
        """
        query = {
            w[:4] for term in terms
            for w in term.lower().split()
            if len(w) >= 4
        }
        return (
            [a for a in self.apprenticeships if query & a.prefixes],
            [p for p in self.programs        if query & p.prefixes]
        )
