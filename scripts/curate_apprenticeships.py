"""
Curate apprenticeship reference data from stakeholder extractions.

Reads the raw `apprenticeships.json` produced by
`parse_agc_workbook.py`, pre-computes `min_hours` and 4-character
prefix sets for runtime matching, and writes the pipeline-ready
`apprenticeships.json` to `data/lexicons/`.

    uv run python scripts/curate_apprenticeships.py
"""

from json    import dumps, loads
from pathlib import Path


def prefixes(text: str) -> list[str]:
    """
    Sorted 4-character word prefixes for prefix matching.
    """
    return sorted({
        w[:4] for w in text.lower().split()
        if len(w) >= 4
    })


class ApprenticeshipCurator:
    """
    Pre-compute runtime fields for apprenticeship reference data.

    Resolves the `term_hours` string range into an integer
    `min_hours` lower bound and generates 4-character prefix sets
    from the trade title for `TradeIndex` matching.
    """

    def __init__(self, root: Path):
        """
        Args:
            root: Worktree root containing `data/` directories.
        """
        self.output = root / "data/lexicons/apprenticeships.json"
        self.source = root / "data/stakeholder/reference/apprenticeships.json"

    def run_all(self):
        """
        Transform raw apprenticeships and write
        `data/lexicons/apprenticeships.json`.
        """
        records = sorted(
            [
                {
                    "min_hours"   : int(raw["term_hours"].split("-")[0]),
                    "prefixes"    : prefixes(raw["title"]),
                    "rapids_code" : raw["rapids_code"],
                    "title"       : raw["title"]
                }
                for raw in loads(self.source.read_text())
            ],
            key=lambda r: r["title"]
        )

        self.output.write_text(dumps(records, indent=2) + "\n")
        print(f"Wrote {len(records)} apprenticeships to {self.output}")


if __name__ == "__main__":

    ApprenticeshipCurator(Path(__file__).resolve().parents[1]).run_all()
