"""
Join parsed labor market data into a unified lexicon.

Reads per-source JSON from `data/labor/` (wages, projections, outlook),
joins by SOC code, and writes `data/lexicons/labor.json` keyed by SOC
with nested wage, projection, and outlook sections per occupation.

    uv run python scripts/curate_labor.py
"""

from json    import dumps, loads
from pathlib import Path


class LaborCurator:
    """
    Join wages, projections, and outlook data into a single per-SOC
    record for display-layer enrichment.
    """

    def __init__(self, root: Path):
        """
        Args:
            root: Worktree root containing `data/` directories.
        """
        self.sources = {
            name: {
                r["soc_code"]: r for r in loads(
                    (root / "data/labor" / f"{name}.json").read_text()
                )
            }
            for name in ("outlook", "projections", "wages")
        }
        self.codes  = loads(
            (root / "data/stakeholder/reference/onet_codes.json").read_text()
        )
        self.output = root / "data/lexicons/labor.json"

    def run_all(self):
        """
        Join all sources by SOC code and write `labor.json`.
        """
        records = [
            {
                "soc_code"  : (soc := entry["soc_code"]),
                "soc_title" : entry["title"],
                **{
                    name: {
                        k: v for k, v in index.get(soc, {}).items()
                        if k not in ("soc_code", "soc_title")
                    } or None
                    for name, index in self.sources.items()
                }
            }
            for entry in sorted(self.codes, key=lambda c: c["soc_code"])
        ]

        self.output.parent.mkdir(exist_ok=True, parents=True)
        self.output.write_text(dumps(records, indent=2) + "\n")

        print(f"Wrote {len(records)} records to {self.output}")
        for name in sorted(self.sources):
            count = sum(1 for r in records if r[name])
            print(f"  {name}: {count}/{len(records)}")


if __name__ == "__main__":

    LaborCurator(Path(__file__).resolve().parents[1]).run_all()
