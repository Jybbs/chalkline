"""
Parse raw BLS and O*NET labor market files into structured JSON.

Reads manually downloaded files from `data/labor/raw/` and writes
per-source JSON to `data/labor/`. The three outputs are:

- `wages.json` from BLS OEWS state-level Excel (Maine rows only)
- `projections.json` from BLS employment projections Table 1.2
- `outlook.json` from O*NET Bright Outlook CSVs and Related Occupations

All three filter to the 21 stakeholder SOC codes. The downstream
`curate_labor.py` script joins them into a unified `labor.json`.

Run from the worktree root:

    uv run python scripts/parse_labor.py
"""

import pandas as pd

from csv     import DictReader
from json    import dumps, loads
from pathlib import Path


class LaborParser:
    """
    Parse raw BLS and O*NET files into per-source JSON artifacts.

    Each parse method reads one or more raw files, filters to the 21
    stakeholder SOC codes, and returns a list of dicts ready for JSON
    serialization. BLS codes omit the `.00` suffix that O*NET codes
    carry, so a mapping handles the translation.
    """

    def __init__(self, root: Path):
        """
        Args:
            root: Worktree root containing `data/` directories.
        """
        self.raw = root / "data/labor/raw"
        self.out = root / "data/labor"
        self.codes = {
            c["soc_code"]: c for c in loads(
                (root / "data/stakeholder/reference/onet_codes.json").read_text()
            )
        }
        self.bls_to_onet = {
            soc.replace(".00", ""): soc for soc in self.codes
        }

    @staticmethod
    def _clean(value) -> str | None:
        """
        Return a stripped string or `None` for BLS suppression markers,
        NaN, and empty values.
        """
        if pd.isna(value):
            return None
        text = str(value).strip()
        return None if text in ("", "\u2014", "nan", "*") else text

    def _numeric(self, value) -> float | int | None:
        """
        Coerce a BLS cell to a number, returning `None` for suppressed
        or missing entries.
        """
        if (cleaned := self._clean(value)) is None:
            return None
        try:
            number = float(cleaned.replace(",", ""))
            return int(number) if number.is_integer() else number
        except ValueError:
            return None

    def _parse_bls(
        self,
        code    : str,
        df      : pd.DataFrame,
        numeric : dict[str, str],
        static  : dict | None = None,
        text    : dict[str, str] | None = None
    ) -> list[dict]:
        """
        Filter a BLS DataFrame to stakeholder SOC codes and reshape
        into typed output records.

        Renames columns per the `numeric` and `text` mappings, applies
        coercion per group, maps BLS codes to O*NET SOC codes, and
        exports sorted records. Callers pre-filter the DataFrame
        (e.g., Maine rows only or "Line item" rows) before passing.

        Args:
            code    : Column containing BLS SOC codes.
            df      : Pre-filtered BLS DataFrame read as `dtype=str`.
            numeric : BLS column to output name for numeric coercion.
            static  : Constant key-value pairs added to every record.
            text    : BLS column to output name for text cleaning.
        """
        result = (
            df[df[code].isin(self.bls_to_onet)]
            .rename(columns={**numeric, **(text or {})})
        )

        for col in numeric.values():
            result[col] = result[col].apply(self._numeric)
        for col in (text or {}).values():
            result[col] = result[col].apply(self._clean)

        result["soc_code"]  = result[code].map(self.bls_to_onet)
        result["soc_title"] = result["soc_code"].map(
            {soc: c["title"] for soc, c in self.codes.items()}
        )

        if static:
            result = result.assign(**static)

        output = sorted([
            *numeric.values(), *(text or {}).values(),
            *(static or {}).keys(), "soc_code", "soc_title"
        ])
        return [
            {k: None if pd.isna(v) else v for k, v in r.items()}
            for r in result[output].sort_values("soc_code").to_dict("records")
        ]

    def _write(self, name: str, records: list[dict]):
        (path := self.out / f"{name}.json").write_text(
            dumps(records, indent=2) + "\n"
        )
        print(f"  {name}.json: {len(records)} records -> {path}")

    def parse_outlook(self) -> list[dict]:
        """
        Build Bright Outlook status and related occupations from O*NET
        raw files.

        Reads the master Bright Outlook CSV to determine which
        stakeholder codes carry the designation, per-category CSVs for
        reason mapping, and the Related Occupations tab-delimited file
        for career transition targets.

        Returns:
            Outlook records for all 21 SOC codes, sorted by code.
        """
        with open(self.raw / "onet/bright_outlook.csv") as f:
            bright = {
                code for row in DictReader(f)
                if (code := row["O*NET-SOC 2019 Code"]) in self.codes
            }

        reasons: dict[str, list[str]] = {}
        category_files = {
            "Rapid Growth"   : "bright_outlook_growth.csv",
            "Many Openings"  : "bright_outlook_openings.csv",
            "New & Emerging" : "bright_outlook_new_emerging.csv"
        }
        for reason, filename in category_files.items():
            with open(self.raw / "onet" / filename) as f:
                for row in DictReader(f):
                    if (code := row["O*NET-SOC 2019 Code"]) in self.codes:
                        reasons.setdefault(code, []).append(reason)

        related_df = pd.read_csv(
            self.raw / "onet/related_occupations.txt",
            dtype = str,
            sep   = "\t"
        )
        related = {
            soc: sorted(
                group[["Related O*NET-SOC Code", "Relatedness Tier"]]
                .rename(columns={
                    "Related O*NET-SOC Code" : "soc_code",
                    "Relatedness Tier"       : "tier"
                })
                .to_dict("records"),
                key=lambda e: e["soc_code"]
            )
            for soc, group in related_df[
                related_df["O*NET-SOC Code"].isin(self.codes)
            ].groupby("O*NET-SOC Code")
        }

        return [
            {
                "bright_outlook"      : soc in bright,
                "outlook_reasons"     : sorted(set(reasons.get(soc, []))),
                "related_occupations" : related.get(soc, []),
                "soc_code"            : soc,
                "soc_title"           : self.codes[soc]["title"]
            }
            for soc in sorted(self.codes)
        ]

    def parse_projections(self) -> list[dict]:
        """
        Extract 10-year employment projections from BLS Table 1.2.

        Reads the multi-sheet projections workbook, filters "Line item"
        rows to stakeholder SOC codes, and extracts employment figures,
        growth rates, median wages, and worker characteristics.

        Returns:
            Projection records sorted by SOC code.
        """
        df = pd.read_excel(
            self.raw / "bls/projections_2024_2034.xlsx",
            dtype      = str,
            header     = 1,
            sheet_name = "Table 1.2"
        )

        return self._parse_bls(
            code    = "2024 National Employment Matrix code",
            df      = df[df["Occupation type"] == "Line item"],
            numeric = {
                "Employment, 2024"                         : "base_employment",
                "Employment, 2034"                         : "projected_employment",
                "Employment change, numeric, 2024\u201334" : "change_number",
                "Employment change, percent, 2024\u201334" : "change_percent",
                "Median annual wage, dollars, 2024[1]"     : "median_annual_wage",
                "Occupational openings, 2024\u201334 annual average" : "openings"
            },
            static  = {
                "base_year"      : 2024,
                "projected_year" : 2034
            },
            text    = {
                "Typical education needed for entry": "education",
                "Typical on-the-job training needed to attain "
                "competency in the occupation": "training"
            }
        )

    def parse_wages(self) -> list[dict]:
        """
        Extract Maine wage percentiles from the BLS OEWS state file.

        Filters to Maine rows matching stakeholder SOC codes and
        reshapes into records with median, percentile, and employment
        fields.

        Returns:
            Wage records sorted by SOC code.
        """
        df = pd.read_excel(
            self.raw / "bls/oews_state_2024.xlsx", dtype=str
        )

        return self._parse_bls(
            code    = "OCC_CODE",
            df      = df[df["AREA_TITLE"] == "Maine"],
            numeric = {
                "A_MEDIAN" : "annual_median",
                "A_PCT10"  : "annual_10",
                "A_PCT25"  : "annual_25",
                "A_PCT75"  : "annual_75",
                "A_PCT90"  : "annual_90",
                "H_MEDIAN" : "hourly_median",
                "H_PCT10"  : "hourly_10",
                "H_PCT25"  : "hourly_25",
                "H_PCT75"  : "hourly_75",
                "H_PCT90"  : "hourly_90",
                "TOT_EMP"  : "employment"
            }
        )

    def run_all(self):
        """
        Parse all raw files and write per-source JSON to `data/labor/`.
        """
        self.out.mkdir(exist_ok=True, parents=True)

        total = 0
        for name in ("outlook", "projections", "wages"):
            records = getattr(self, f"parse_{name}")()
            self._write(name, records)
            total += len(records)

        print(f"\nParsed {total} total records from 7 raw files")


if __name__ == "__main__":

    LaborParser(Path(__file__).resolve().parents[1]).run_all()
