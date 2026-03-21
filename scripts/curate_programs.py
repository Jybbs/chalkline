"""
Curate a unified program lexicon from stakeholder extractions.

Reads the raw `cc_programs.json` and `umaine_programs.json` produced
by `parse_agc_workbook.py`, normalizes each source schema into the
`ProgramRecommendation` field layout, pre-computes 4-character prefix
sets for runtime matching, and writes a merged, sorted `programs.json`
to `data/lexicons/`.

    uv run python scripts/curate_programs.py
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


class ProgramCurator:
    """
    Normalize and merge community college and university program
    reference data into the pipeline-ready schema.

    Reads two extraction outputs with different field layouts and
    maps each into the `credential`, `institution`, `prefixes`,
    `program`, `url` structure consumed by `TradeIndex`.
    """

    def __init__(self, root: Path):
        """
        Args:
            root: Worktree root containing `data/` directories.
        """
        self.output    = root / "data/lexicons/programs.json"
        self.reference = root / "data/stakeholder/reference"

    def run_all(self):
        """
        Normalize both source files and write
        `data/lexicons/programs.json`.
        """
        cc_path = self.reference / "cc_programs.json"
        um_path = self.reference / "umaine_programs.json"

        cc = loads(cc_path.read_text()) if cc_path.exists() else {}
        um = loads(um_path.read_text()) if um_path.exists() else []

        programs = sorted(
            [
                *[
                    {
                        "credential"  : e["credential"],
                        "institution" : e["college"],
                        "prefixes"    : prefixes(e["program"]),
                        "program"     : e["program"],
                        "url"         : e["url"]
                    }
                    for e in cc.get("degrees", [])
                ],
                *[
                    {
                        "credential"  : e["best_for"],
                        "institution" : "Statewide",
                        "prefixes"    : prefixes(e["initiative"]),
                        "program"     : e["initiative"],
                        "url"         : e.get("url", "")
                    }
                    for e in cc.get("initiatives", [])
                ],
                *[
                    {
                        "credential"  : e["degree"],
                        "institution" : e["campus"],
                        "prefixes"    : prefixes(e["program"]),
                        "program"     : e["program"],
                        "url"         : e["url"]
                    }
                    for e in um
                ]
            ],
            key=lambda p: p["program"]
        )

        self.output.write_text(dumps(programs, indent=2) + "\n")
        print(f"Wrote {len(programs)} programs to {self.output}")


if __name__ == "__main__":

    ProgramCurator(Path(__file__).resolve().parents[1]).run_all()
