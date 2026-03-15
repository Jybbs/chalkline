"""
Parse the CareerOneStop certifications flat file into structured JSON.

Reads the manually downloaded Excel file from
`data/certifications/raw/careeronestop.xlsx`, extracts certification
records with names, acronyms, organizations, types, and descriptions,
and writes structured JSON to `data/certifications/careeronestop.json`.

Run from the worktree root:

    uv run python scripts/parse_certifications.py
"""

from json    import dumps
from pandas  import ExcelFile, notna
from pathlib import Path


class CertificationParser:
    """
    Extract structured certification records from the CareerOneStop Excel
    flat file.

    Reads the single-sheet workbook and transforms each row into a clean
    dictionary with consistent field names and null handling. The output
    preserves all 6,444 certifications without filtering, leaving SOC-code
    scoping and description mining to the downstream curation script.
    """

    def __init__(self, root: Path):
        """
        Locate the source file and output path.

        Args:
            root: Worktree root containing `data/` directories.
        """
        self.output = root / "data/certifications/careeronestop.json"
        self.source = root / "data/certifications/raw/careeronestop.xlsx"

    def extract(self) -> list[dict]:
        """
        Read the Excel file and extract certification records.

        Each row becomes a dict with `acronym`, `description`, `id`, `name`,
        `organization`, `type`, and `url` keys. Fields with missing or
        whitespace-only values are set to `None`.

        Returns:
            Sorted list of certification dicts.
        """
        df = ExcelFile(self.source).parse("Certification Finder Data")

        def clean(value) -> str | None:
            return s if notna(value) and (s := str(value).strip()) else None

        return sorted(
            (
                {
                    "acronym"      : clean(row["ACRONYM"]),
                    "description"  : clean(row["CERT_DESCRIPTION"]),
                    "id"           : str(int(row["CERT_ID"])),
                    "name"         : clean(row["CERT_NAME"]),
                    "organization" : clean(row["ORG_NAME"]),
                    "type"         : clean(row["TYPE"]),
                    "url"          : clean(row["URL"])
                }
                for _, row in df.iterrows()
            ),
            key=lambda r: (r["name"] or "")
        )

    def run_all(self):
        """
        Parse the flat file and write the
        `data/certifications/careeronestop.json` output.
        """
        records = self.extract()

        self.output.parent.mkdir(exist_ok=True, parents=True)
        self.output.write_text(dumps(records, indent=2) + "\n")

        with_acronyms     = sum(1 for r in records if r["acronym"])
        with_descriptions = sum(1 for r in records if r["description"])

        print(f"Parsed {len(records)} certifications")
        print(f"  With acronyms: {with_acronyms}")
        print(f"  With descriptions: {with_descriptions}")
        print(f"  Wrote {self.output}")


if __name__ == "__main__":

    CertificationParser(
        Path(__file__).resolve().parent.parent
    ).run_all()
