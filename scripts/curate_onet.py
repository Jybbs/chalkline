"""
Curate the O*NET occupation-skill mapping for Chalkline's 21 SOC codes.

Downloads element-type files from the O*NET 30.0 database, filters to the
stakeholder-curated SOC codes, merges Skills, Knowledge, Abilities, Tasks,
Emerging Tasks, Technology Skills, Detailed Work Activities, and Tools Used
into a structured `skills` array, decomposes Task and DWA entries into
matchable sub-phrases via POS-based chunking, filters ambiguous single-word
tool and technology entries using wordfreq Zipf frequency, and writes
`data/lexicons/onet.json`.

Run from the worktree root:

    uv run python scripts/curate_onet.py
"""

from collections        import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from json               import dumps, loads
from nltk               import download, pos_tag, RegexpParser, word_tokenize
from pandas             import read_csv
from pathlib            import Path
from urllib.parse       import quote
from urllib.request     import urlopen
from wordfreq           import zipf_frequency


class OnetCurator:
    """
    Fetch, filter, and merge O*NET element files into a structured
    occupation-skill lexicon for Aho-Corasick extraction.

    Downloads tab-delimited files from the O*NET 30.0 database, filters
    to stakeholder SOC codes, merges element types with POS-based phrase
    decomposition for Tasks and DWAs, and excludes ambiguous single-word
    tools and technologies via wordfreq Zipf frequency.
    """

    def __init__(self, root: Path):
        """
        Download NLTK data, compile the chunk parser, and load stakeholder
        SOC codes.

        Args:
            root: Worktree root containing `data/` directories.
        """
        for corpus in ("averaged_perceptron_tagger_eng", "punkt_tab"):
            download(corpus, quiet=True)

        self.codes = {
            c["soc_code"]: c for c in loads(
                (root / "data/stakeholder/reference/onet_codes.json").read_text()
            )
        }
        self.output = root / "data/lexicons/onet.json"
        self.parser = RegexpParser(r"""
            NP: {<DT>?<JJ>*<NN.*>+}
            VP: {<VB.*><NP>}
        """)

    def _decompose(self, text: str) -> list[str]:
        """
        Extract matchable sub-phrases from a sentence-length entry.

        Uses the compiled `RegexpParser` with NP and VP grammar rules to
        chunk POS-tagged tokens. Determiners are stripped from chunks and
        only phrases with at least two tokens are retained, filtering out
        generic single-word nouns like "equipment" and "materials."

        Args:
            text: A sentence-length O*NET Task or DWA.

        Returns:
            List of extracted sub-phrase strings.
        """
        tree = self.parser.parse(pos_tag(word_tokenize(text)))
        return [
            " ".join(words).lower()
            for subtree in tree.subtrees(
                lambda t: t.label() in ("NP", "VP")
            )
            if len(
                words := [w for w, tag in subtree.leaves() if tag != "DT"]
            ) >= 2
        ]

    def _download(self) -> dict:
        """
        Download and filter the ten O*NET database files.

        Fetches each file in parallel via `ThreadPoolExecutor` and filters
        to stakeholder SOC codes where applicable.

        Returns:
            Mapping from internal key to filtered `DataFrame`.
        """
        files = {
            "abilities"      : "Abilities.txt",
            "dwa_reference"  : "DWA Reference.txt",
            "emerging_tasks" : "Emerging Tasks.txt",
            "job_zones"      : "Job Zones.txt",
            "knowledge"      : "Knowledge.txt",
            "skills"         : "Skills.txt",
            "tasks"          : "Task Statements.txt",
            "tasks_to_dwas"  : "Tasks to DWAs.txt",
            "tech_skills"    : "Technology Skills.txt",
            "tools_used"     : "Tools Used.txt"
        }

        print("Downloading O*NET 30.0 database files...")
        with ThreadPoolExecutor() as pool:
            return dict(pool.map(self._fetch, files.items()))

    def _entry(
        self,
        name       : str,
        type_label : str,
        importance : float | None = None,
        level      : float | None = None
    ) -> dict:
        """
        Build a skill entry dict for the output lexicon.

        Tasks and DWAs are decomposed into matchable sub-phrases via
        POS-based chunking. Other entry types receive `None` for the
        `phrases` field.

        Args:
            name       : The O*NET element name.
            type_label : Entry type for downstream dispatch.
            importance : IM scale value for KSA entries.
            level      : LV scale value for KSA entries.

        Returns:
            Skill entry dict with `importance`, `level`, `name`, `phrases`,
            and `type` keys.
        """
        return {
            "importance" : importance,
            "level"      : level,
            "name"       : name,
            "phrases"    : (
                self._decompose(name)
                if type_label in ("dwa", "task") else None
            ),
            "type"       : type_label
        }

    def _fetch(self, pair: tuple[str, str]) -> tuple:
        """
        Fetch a single O*NET file and filter to stakeholder codes.

        Designed as a `ThreadPoolExecutor.map` target where each item is
        a `(key, filename)` pair from the files dict.

        Args:
            pair: Internal key and O*NET filename.

        Returns:
            The key and filtered `DataFrame` as a tuple.
        """
        name, filename = pair
        with urlopen(
            f"https://www.onetcenter.org/dl_files/database"
            f"/db_30_0_text/{quote(filename)}"
        ) as resp:
            df = read_csv(resp, delimiter="\t", dtype=str)

        return name, (
            df[df["O*NET-SOC Code"].isin(self.codes)]
            if "O*NET-SOC Code" in df.columns else df
        )

    def _is_ambiguous(self, name: str) -> bool:
        """
        Test whether a single-word tool or technology name collides with
        common English.

        Uses wordfreq Zipf frequency as the ambiguity signal. The
        threshold of 4.0 corresponds to roughly one occurrence per 10,000
        words in modern English, sitting above domain terms like "rebar"
        (2.57) while catching general words like "level" (4.84) and
        "iron" (4.27) that would produce false Aho-Corasick matches.

        Args:
            name: The O*NET entry name to test.

        Returns:
            `True` if the term should be excluded from the lexicon.
        """
        return " " not in name and zipf_frequency(name, "en") >= 4.0

    def _merge(self, raw: dict) -> tuple[dict, set]:
        """
        Merge all element types into per-occupation skill lists.

        Processes KSA entries with importance/level scales first, then
        simple entry types with ambiguity filtering for tools and
        technologies.

        Args:
            raw: Downloaded O*NET data keyed by element type.

        Returns:
            Per-SOC merged skill list and excluded ambiguous terms.
        """
        merged = defaultdict(list)

        for source_key, type_label in (
            ("abilities", "ability"),
            ("knowledge", "knowledge"),
            ("skills",    "skill")
        ):
            df = raw[source_key].query("`Recommend Suppress` != 'Y'")
            for soc, group in df.groupby("O*NET-SOC Code"):
                im, lv = (
                    group[group["Scale ID"] == scale]
                    .set_index("Element Name")["Data Value"]
                    .astype(float).to_dict()
                    for scale in ("IM", "LV")
                )
                merged[soc].extend(
                    self._entry(
                        importance = im.get(name),
                        level      = lv.get(name),
                        name       = name,
                        type_label = type_label
                    )
                    for name in {*im, *lv}
                )

        raw["dwas"] = (
            raw["tasks_to_dwas"]
            .drop_duplicates(subset=["O*NET-SOC Code", "DWA ID"])
            .merge(raw["dwa_reference"], on="DWA ID")
        )

        excluded = set()
        for source_key, name_column, type_label in (
            ("dwas",           "DWA Title", "dwa"),
            ("emerging_tasks", "Task",      "task"),
            ("tasks",          "Task",      "task"),
            ("tech_skills",    "Example",   "technology"),
            ("tools_used",     "Example",   "tool")
        ):
            for soc, group in (
                raw[source_key]
                .drop_duplicates(subset=["O*NET-SOC Code", name_column])
                .groupby("O*NET-SOC Code")
            ):
                for name in group[name_column]:
                    if (
                        type_label in ("technology", "tool")
                        and self._is_ambiguous(name)
                    ):
                        excluded.add(name)
                        continue
                    merged[soc].append(self._entry(name, type_label))

        return merged, excluded

    def run_all(self):
        """
        Fetch O*NET element files and write `data/lexicons/onet.json`.
        """
        raw              = self._download()
        merged, excluded = self._merge(raw)

        if excluded:
            print(f"  Excluded {len(excluded)} ambiguous tool/tech entries:")
            for name in sorted(excluded):
                print(f"    {name:20s}  zipf={zipf_frequency(name, "en"):.2f}")

        job_zones = (
            raw["job_zones"]
            .set_index("O*NET-SOC Code")["Job Zone"]
            .astype(int)
            .to_dict()
        )

        occupations = [
            {
                "job_zone" : job_zones.get(soc),
                "sector"   : self.codes[soc]["sector"],
                "skills"   : sorted(merged[soc], key=lambda s: (s["type"], s["name"])),
                "soc_code" : soc,
                "title"    : self.codes[soc]["title"]
            }
            for soc in sorted(self.codes)
        ]

        self.output.parent.mkdir(exist_ok=True, parents=True)
        self.output.write_text(dumps(occupations, indent=2) + "\n")

        print(f"  Wrote {len(occupations)} occupations to {self.output}")
        for occ in occupations:
            counts  = Counter(s["type"] for s in occ["skills"])
            summary = ", ".join(
                f"{v} {k}" for k, v in sorted(counts.items())
            )
            print(
                f"    {occ["soc_code"]}  {occ["title"]:45s}"
                f"  JZ={occ["job_zone"]}  [{summary}]"
            )


if __name__ == "__main__":

    OnetCurator(Path(__file__).resolve().parent.parent).run_all()
