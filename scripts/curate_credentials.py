"""
Curate the unified credential lexicon for the Chalkline pipeline.

Reads raw apprenticeship and program reference data from stakeholder
extractions, fetches and filters CareerOneStop certifications via O*NET
OnLine, pre-computes embedding text and display labels, and writes a
merged `credentials.json` to `data/lexicons/`.

    uv run python scripts/curate_credentials.py
"""

from bs4                import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from json               import dumps, loads
from pathlib            import Path
from re                 import search
from typing             import Any
from urllib.request     import Request, urlopen
from wordfreq           import zipf_frequency


class CredentialCurator:
    """
    Curate apprenticeships, certifications, and educational programs
    into a single credential lexicon.

    Apprenticeships and programs are simple field transforms from
    stakeholder extractions. Certifications require fetching O*NET
    OnLine pages to establish the SOC-to-certification mapping, then
    joining against the parsed CareerOneStop flat file for names,
    acronyms, and organizations.
    """

    def __init__(self, root: Path):
        """
        Args:
            root: Worktree root containing `data/` directories.
        """
        stakeholder    = root / "data/stakeholder/reference"
        self.codes     = self._load(stakeholder / "onet_codes.json", key="soc_code")
        self.output    = root / "data/lexicons/credentials.json"
        self.parsed    = self._load(root / "data/certifications/careeronestop.json", key="id")
        self.reference = stakeholder

    @staticmethod
    def _cert_record(raw: dict, ambiguous: set[str]) -> dict:
        """
        Build a single certification credential record.

        Acronyms that collide with common English are stripped from
        the label and embedding text so they do not pollute
        downstream similarity matching.

        Args:
            raw       : Parsed CareerOneStop certification entry.
            ambiguous : Set of acronyms to suppress.
        """
        acronym = raw["acronym"]
        if acronym in ambiguous:
            acronym = None

        return {
            "embedding_text" : " ".join(
                x for x in (acronym, raw["name"], raw["organization"])
                if x
            ),
            "kind"           : "certification",
            "label"          : (
                f"{acronym} {raw['name']}" if acronym
                else raw["name"]
            )
        }

    def _curate_apprenticeships(self) -> list[dict]:
        """
        Transform raw apprenticeship data into credential records.
        """
        return [
            {
                "embedding_text" : raw["title"],
                "kind"           : "apprenticeship",
                "label"          : raw["title"],
                "metadata"       : {
                    "min_hours"   : int(raw["term_hours"].split("-")[0]),
                    "rapids_code" : raw["rapids_code"]
                }
            }
            for raw in self._load(self.reference / "apprenticeships.json")
        ]

    def _curate_certifications(self) -> list[dict]:
        """
        Fetch, filter, and transform CareerOneStop certifications.
        """
        print(f"Fetching certifications for {len(self.codes)} SOC codes...")
        codes = sorted(self.codes)
        with ThreadPoolExecutor(max_workers=5) as pool:
            all_cert_ids = {
                cert_id
                for batch in pool.map(self._fetch_cert_ids, codes)
                for cert_id in batch
            }

        print(f"Unique certifications across all SOC codes: {len(all_cert_ids)}")

        certs = sorted(
            (self.parsed[cid] for cid in all_cert_ids if cid in self.parsed),
            key=lambda r: r["name"]
        )
        ambiguous = {
            acronym for r in certs
            if (acronym := r["acronym"]) and self._is_ambiguous(acronym)
        }

        if ambiguous:
            print(
                f"  Excluded {len(ambiguous)} ambiguous "
                f"acronyms: {sorted(ambiguous)}"
            )

        return [self._cert_record(r, ambiguous) for r in certs]

    def _curate_programs(self) -> list[dict]:
        """
        Normalize and merge community college and university programs.
        """
        cc = self._load(self.reference / "cc_programs.json", {})
        um = self._load(self.reference / "umaine_programs.json", [])

        def program(credential: str, institution: str, name: str, url: str):
            return {
                "embedding_text" : f"{credential} {name} {institution}",
                "kind"           : "program",
                "label"          : name,
                "metadata"       : {
                    "credential"  : credential,
                    "institution" : institution,
                    "url"         : url
                }
            }

        records = [
            program(e["credential"], e["college"], e["program"], e["url"])
            for e in cc.get("degrees", [])
        ]
        records += [
            program(e["best_for"], "Statewide", e["initiative"], e.get("url", ""))
            for e in cc.get("initiatives", [])
        ]
        records += [
            program(e["degree"], e["campus"], e["program"], e["url"])
            for e in um
        ]
        return records

    @staticmethod
    def _fetch_cert_ids(soc_code: str) -> set[str]:
        """
        Fetch certification IDs for one SOC code from O*NET OnLine.

        Args:
            soc_code: O*NET SOC code (e.g., `47-2111.00`).

        Returns:
            Set of cert ID strings found on the page.
        """
        url = f"https://www.onetonline.org/link/localcert/{soc_code}"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(req, timeout=30) as resp:
                html = resp.read()
        except Exception as exc:
            print(f"  {soc_code}: fetch failed ({exc})")
            return set()

        ids = {
            m.group(1)
            for a in BeautifulSoup(html, "html.parser").find_all("a", href=True)
            if (m := search(r"/link/certinfo/(\d+)-", str(a["href"])))
        }
        print(f"  {soc_code}: {len(ids)} certifications")
        return ids

    @staticmethod
    def _is_ambiguous(term: str) -> bool:
        """
        Test whether a single-word term collides with common English.

        Args:
            term: Acronym or short certification name.

        Returns:
            `True` if the term should be excluded.
        """
        return " " not in term and zipf_frequency(term, "en") >= 4.0

    @staticmethod
    def _load(
        path    : Path,
        default : Any = ...,
        key     : str | None = None
    ):
        """
        Read and parse a JSON file, optionally indexing by a field.

        Args:
            path    : JSON file to read.
            default : Value to return when the file does not exist.
                      Omit to let `FileNotFoundError` propagate.
            key     : When provided, return a `{record[key]: record}`
                      dict instead of the raw list.
        """
        if default is not ... and not path.exists():
            return default
        records = loads(path.read_text())
        if key:
            return {r[key]: r for r in records}
        return records

    def run_all(self):
        """
        Curate all credential types and write
        `data/lexicons/credentials.json`.
        """
        apprenticeships = self._curate_apprenticeships()
        certifications  = self._curate_certifications()
        programs        = self._curate_programs()

        credentials = sorted(
            apprenticeships + certifications + programs,
            key=lambda r: (r["kind"], r["label"])
        )

        self.output.parent.mkdir(exist_ok=True, parents=True)
        self.output.write_text(dumps(credentials, indent=2) + "\n")

        print(f"\nWrote {len(credentials)} credentials to {self.output}")
        print(f"  Apprenticeships: {len(apprenticeships)}")
        print(f"  Certifications:  {len(certifications)}")
        print(f"  Programs:        {len(programs)}")


if __name__ == "__main__":

    CredentialCurator(Path(__file__).resolve().parents[1]).run_all()
