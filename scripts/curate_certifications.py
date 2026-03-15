"""
Curate the CareerOneStop certification lexicon for Chalkline's SOC codes.

Reads parsed certification records from
`data/certifications/careeronestop.json`, fetches O*NET OnLine
certification listing pages for each stakeholder SOC code to establish
the certification-to-occupation mapping, filters to
construction-relevant certifications, decomposes descriptions into
matchable sub-phrases via POS-based chunking, and writes
`data/lexicons/certifications.json`.

Run from the worktree root:

    uv run python scripts/curate_certifications.py
"""

from bs4                import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from json               import dumps, loads
from nltk               import download, pos_tag, RegexpParser, word_tokenize
from pathlib            import Path
from re                 import search
from urllib.request     import Request, urlopen
from wordfreq           import zipf_frequency


class CertificationCurator:
    """
    Filter CareerOneStop certifications to stakeholder SOC codes and
    decompose descriptions into matchable sub-phrases.

    Fetches O*NET OnLine certification listing pages for each SOC code
    to establish which certifications map to which occupations, then
    joins against the parsed CareerOneStop flat file for acronyms and
    description text. Descriptions are decomposed via the same NP/VP
    chunking grammar used by `curate_onet.py`, producing multi-word
    phrases that feed the Aho-Corasick automaton alongside the certification
    names and acronyms.
    """

    def __init__(self, root: Path):
        """
        Load dependencies, SOC codes, and parsed certifications.

        Args:
            root: Worktree root containing `data/` directories.
        """
        for corpus in ("averaged_perceptron_tagger_eng", "punkt_tab"):
            download(corpus, quiet=True)

        self.codes = {
            c["soc_code"]: c for c in loads(
                (root / "data/stakeholder/reference/onet_codes.json")
                .read_text()
            )
        }
        self.output = root / "data/lexicons/certifications.json"
        self.parsed = {
            r["id"]: r for r in loads(
                (root / "data/certifications/careeronestop.json")
                .read_text()
            )
        }
        self.parser = RegexpParser(r"""
            NP: {<DT>?<JJ>*<NN.*>+}
            VP: {<VB.*><NP>}
        """)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _decompose(self, text: str) -> list[str]:
        """
        Extract matchable sub-phrases from description text.

        Uses the same NP/VP grammar as `curate_onet.py` to chunk POS-tagged
        tokens. Determiners are stripped and only phrases with at least two
        tokens are retained, filtering generic single-word nouns. Phrases
        composed entirely of common English words (all tokens with
        Zipf >= 4.0) are excluded to avoid boilerplate like "knowledge
        base" and "work experience."

        Args:
            text: Certification description text.

        Returns:
            Deduplicated list of extracted sub-phrase strings.
        """
        tree = self.parser.parse(pos_tag(word_tokenize(text)))
        return list(dict.fromkeys(
            phrase
            for subtree in tree.subtrees(
                lambda t: t.label() in ("NP", "VP")
            )
            if len(
                words := [w for w, tag in subtree.leaves() if tag != "DT"]
            ) >= 2
            if any(
                zipf_frequency(w, "en") < 4.0
                for w in (phrase := " ".join(words).lower()).split()
                if len(w) >= 3
            )
        ))

    def _fetch_cert_ids(self, soc_code: str) -> set[str]:
        """
        Fetch certification IDs for one SOC code from O*NET OnLine.

        Parses the HTML table on the certification listing page and
        extracts numeric cert IDs from the detail link hrefs.

        Args:
            soc_code: O*NET SOC code (e.g., `47-2111.00`).

        Returns:
            Set of cert ID strings found on the page.
        """
        req = Request(
            f"https://www.onetonline.org/link/localcert/{soc_code}",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        try:
            with urlopen(req, timeout=30) as resp:
                soup = BeautifulSoup(resp.read(), "html.parser")
            ids = {
                m.group(1)
                for a in soup.find_all("a", href=True)
                if (m := search(r"/link/certinfo/(\d+)-", a["href"]))
            }
            print(f"  {soc_code}: {len(ids)} certifications")
            return ids
        except Exception as exc:
            print(f"  {soc_code}: fetch failed ({exc})")
            return set()

    def _is_ambiguous(self, term: str) -> bool:
        """
        Test whether a single-word term collides with common English.

        Uses wordfreq Zipf frequency with the same 4.0 threshold as
        `curate_onet.py` for consistency.

        Args:
            term: Acronym or short certification name.

        Returns:
            `True` if the term should be excluded.
        """
        return " " not in term and zipf_frequency(term, "en") >= 4.0

    # -------------------------------------------------------------------------
    # Entry point
    # -------------------------------------------------------------------------

    def run_all(self):
        """
        Filter certifications, decompose descriptions, and write the
        `data/lexicons/certifications.json` lexicon file.
        """
        print(f"Fetching certifications for {len(self.codes)} SOC codes...")
        codes = sorted(self.codes)
        with ThreadPoolExecutor(max_workers=5) as pool:
            cert_to_socs = {}
            for soc, cert_ids in zip(codes, pool.map(self._fetch_cert_ids, codes)):
                for cert_id in cert_ids:
                    cert_to_socs.setdefault(cert_id, set()).add(soc)

        print(f"Unique certifications across all SOC codes: {len(cert_to_socs)}")

        certifications    = []
        excluded_acronyms = []

        for cert_id in sorted(
            cert_to_socs,
            key=lambda cid: self.parsed.get(cid, {}).get("name", "")
        ):
            if cert_id not in self.parsed:
                continue

            record  = self.parsed[cert_id]
            acronym = record["acronym"]

            if acronym and self._is_ambiguous(acronym):
                excluded_acronyms.append(acronym)
                acronym = None

            phrases = (
                self._decompose(record["description"]) or None
                if record["description"]
                else None
            )

            certifications.append({
                "acronym"      : acronym,
                "name"         : record["name"],
                "organization" : record["organization"],
                "phrases"      : phrases,
                "soc_codes"    : sorted(cert_to_socs[cert_id]),
                "type"         : record["type"]
            })

        self.output.parent.mkdir(exist_ok=True, parents=True)
        self.output.write_text(dumps(certifications, indent=2) + "\n")

        with_acronyms = sum(1 for c in certifications if c["acronym"])
        with_phrases  = sum(1 for c in certifications if c["phrases"])
        total_phrases = sum(len(c["phrases"]) for c in certifications if c["phrases"])

        print(f"\nWrote {len(certifications)} certifications to {self.output}")
        print(f"  With acronyms: {with_acronyms}")
        print(f"  With description phrases: {with_phrases} ({total_phrases} total)")

        if excluded_acronyms:
            print(f"  Excluded {len(excluded_acronyms)} ambiguous"
                  f" acronyms: {sorted(set(excluded_acronyms))}")


if __name__ == "__main__":

    CertificationCurator(Path(__file__).resolve().parents[1]).run_all()
