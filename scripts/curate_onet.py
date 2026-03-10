"""
Curate the O*NET occupation-skill mapping for Chalkline's 21 SOC codes.

Downloads element-type files from the O*NET 30.0 database, filters to the
stakeholder-curated SOC codes in `data/stakeholder/reference/onet_codes.json`,
merges Skills, Knowledge, Abilities, Tasks, Technology Skills, Detailed Work
Activities, Tools Used, and Alternate Titles into a structured `skills` array,
decomposes Task and DWA entries into matchable sub-phrases via POS-based
chunking, and writes `data/lexicons/onet.json`.

Run from the worktree root:

    uv run python scripts/curate_onet.py
"""

from collections import Counter, defaultdict
from json        import dumps, loads
from nltk        import download, pos_tag, RegexpParser, word_tokenize
from pandas      import read_csv
from pathlib     import Path
from urllib      import parse, request


ONET_FILES = {
    "abilities"     : "Abilities.txt",
    "alt_titles"    : "Alternate Titles.txt",
    "dwa_reference" : "DWA Reference.txt",
    "job_zones"     : "Job Zones.txt",
    "knowledge"     : "Knowledge.txt",
    "skills"        : "Skills.txt",
    "tasks"         : "Task Statements.txt",
    "tasks_to_dwas" : "Tasks to DWAs.txt",
    "tech_skills"   : "Technology Skills.txt",
    "tools_used"    : "Tools Used.txt"
}

BASE = "https://www.onetcenter.org/dl_files/database/db_30_0_text"

entry = lambda name, type_label, importance=None, level=None: {
    "importance" : importance,
    "level"      : level,
    "name"       : name,
    "phrases"    : decompose(name) if type_label in ("dwa", "task") else None,
    "type"       : type_label
}


def decompose(text: str) -> list[str]:
    """
    Extract matchable sub-phrases from a sentence-length entry.

    Uses `nltk.RegexpParser` with NP and VP grammar rules to chunk
    POS-tagged tokens. Determiners are stripped from chunks and only
    phrases with at least two tokens are retained, filtering out
    generic single-word nouns like "equipment" and "materials."

    Args:
        text: A sentence-length O*NET Task or DWA.

    Returns:
        List of extracted sub-phrase strings.
    """
    tree = RegexpParser(r"""
        NP: {<DT>?<JJ>*<NN.*>+}
        VP: {<VB.*><NP>}
    """).parse(pos_tag(word_tokenize(text)))
    return [
        " ".join(words).lower()
        for subtree in tree.subtrees(
            lambda t: t.label() in ("NP", "VP")
        )
        if len(words := [
            w for w, tag in subtree.leaves() if tag != "DT"
        ]) >= 2
    ]


def main():
    """
    Fetch O*NET element files and write `data/lexicons/onet.json`.

    Downloads ten tab-delimited files from the O*NET 30.0 database,
    filters each to the 21 stakeholder SOC codes, merges eight element
    types into a structured `skills` array per occupation, decomposes
    Task and DWA entries into matchable sub-phrases, and writes the
    result as a sorted JSON array.
    """
    root = Path(__file__).resolve().parent.parent

    download("averaged_perceptron_tagger_eng", quiet=True)
    download("punkt_tab", quiet=True)

    codes = {c["soc_code"]: c for c in loads(
        (root / "data/stakeholder/reference/onet_codes.json").read_text()
    )}

    print("Downloading O*NET 30.0 database files...")
    raw = {}
    for name, filename in ONET_FILES.items():
        print(f"  Downloading {filename}...")
        with request.urlopen(
            f"{BASE}/{parse.quote(filename)}"
        ) as resp:
            df = read_csv(resp, delimiter="\t", dtype=str)
        raw[name] = (
            df[df["O*NET-SOC Code"].isin(codes)]
            if "O*NET-SOC Code" in df.columns else df
        )
    merged = defaultdict(list)

    for source_key, type_label in (
        ("abilities", "ability"),
        ("knowledge", "knowledge"),
        ("skills",    "skill")
    ):
        df = raw[source_key].query("`Recommend Suppress` != 'Y'")
        for soc, group in df.groupby("O*NET-SOC Code"):
            im, lv = (
                dict(zip(
                    (sub := group[group["Scale ID"] == scale])["Element Name"],
                    sub["Data Value"].astype(float)
                ))
                for scale in ("IM", "LV")
            )
            merged[soc].extend(
                entry(name, type_label, im.get(name), lv.get(name))
                for name in {*im, *lv}
            )

    raw["dwas"] = (
        raw["tasks_to_dwas"]
        .drop_duplicates(subset=["O*NET-SOC Code", "DWA ID"])
        .merge(raw["dwa_reference"], on="DWA ID")
    )

    for source_key, name_column, type_label in (
        ("alt_titles",  "Alternate Title", "alternate_title"),
        ("dwas",        "DWA Title",       "dwa"),
        ("tasks",       "Task",            "task"),
        ("tech_skills", "Example",         "technology"),
        ("tools_used",  "Example",         "tool")
    ):
        for soc, group in (
            raw[source_key]
            .drop_duplicates(subset=["O*NET-SOC Code", name_column])
            .groupby("O*NET-SOC Code")
        ):
            merged[soc].extend(
                entry(name, type_label) for name in group[name_column]
            )

    job_zones = (
        raw["job_zones"]
        .set_index("O*NET-SOC Code")["Job Zone"]
        .astype(int)
        .to_dict()
    )

    occupations = [
        {
            "job_zone" : job_zones.get(soc),
            "sector"   : codes[soc]["sector"],
            "skills"   : sorted(merged[soc], key=lambda s: (s["type"], s["name"])),
            "soc_code" : soc,
            "title"    : codes[soc]["title"]
        }
        for soc in sorted(codes)
    ]

    output = root / "data/lexicons/onet.json"
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(dumps(occupations, indent=2) + "\n")

    print(f"  Wrote {len(occupations)} occupations to {output}")
    for occ in occupations:
        print(
            f"    {occ['soc_code']}  {occ['title']:45s}"
            f"  JZ={occ['job_zone']}  [{', '.join(
                f'{v} {k}' for k, v in sorted(
                    Counter(s['type'] for s in occ['skills']).items()
                )
            )}]"
        )


if __name__ == "__main__":

    main()
