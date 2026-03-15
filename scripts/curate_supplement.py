"""
Curate a domain supplement lexicon from multiple mechanized sources.

Mines six sources for construction vocabulary absent from the OSHA/O*NET
index, namely cross-SOC nouns from O*NET Task and DWA text, apprenticeship
trade titles and their decomposed words, community college and university
program name words, technical abbreviations from O*NET tool and technology
entries, and high-frequency words from job posting titles. All sources
apply wordfreq Zipf frequency filtering to exclude common English. Writes
a sorted JSON array to `data/lexicons/supplement.json`.

Run from the worktree root:

    uv run python scripts/curate_supplement.py
"""

from collections import Counter, defaultdict
from json        import dumps, loads
from nltk        import download, pos_tag, word_tokenize
from nltk.stem   import WordNetLemmatizer
from pathlib     import Path
from re          import sub
from wordfreq    import zipf_frequency


class SupplementCurator:
    """
    Mine construction vocabulary from multiple mechanized sources for
    supplement lexicon generation.

    Six sources are mined in priority order, namely cross-SOC nouns from
    O*NET task text, apprenticeship trade titles and their decomposed
    individual words, community college and university program name words,
    technical abbreviations from O*NET tool and technology entries, and
    high-frequency words from job posting titles. All sources filter
    against wordfreq Zipf frequency and the existing OSHA/O*NET index to
    avoid duplicates and common English false positives.
    """

    def __init__(self, root: Path):
        """
        Load dependencies and initialize the lemmatizer.

        Args:
            root: Worktree root containing `data/` directories.
        """
        for corpus in (
            "averaged_perceptron_tagger_eng",
            "omw-1.4",
            "punkt_tab",
            "wordnet"
        ):
            download(corpus, quiet=True)

        self.lemmatizer = WordNetLemmatizer()
        self.onet_data  = loads((root / "data/lexicons/onet.json").read_text())
        self.output     = root / "data/lexicons/supplement.json"
        self.root       = root
        self.trades     = [
            t["title"] for t in loads(
                (root / "data/stakeholder/reference/apprenticeships.json").read_text()
            )
        ]

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    def _build_existing_index(self) -> set[str]:
        """
        Load the current OSHA and O*NET lexicons and collect all indexed
        terms for deduplication.

        Returns:
            Set of lowercased terms already in the extraction index.
        """
        from chalkline.extraction.lexicons import LexiconRegistry
        from chalkline.extraction.loaders  import load_onet, load_osha

        registry = LexiconRegistry(
            load_onet(self.root / "data/lexicons/onet.json"),
            load_osha(self.root / "data/lexicons/osha.json")
        )

        return (
            set(registry.lemma_index)
            | {v.lower() for v in registry.lemma_index.values()}
        )

    def _collect(
        self,
        existing : set[str],
        heading  : str,
        items,
        label    : str,
        terms    : set[str],
        fmt      = None,
        key      = None
    ):
        """
        Filter novel terms from a mined source into the term set.

        Args:
            existing : Lowercased terms from the OSHA/O*NET index.
            heading  : Header line printed before processing.
            items    : Iterable of mined items (strings or tuples).
            label    : Source description for the summary line.
            terms    : Accumulating set of accepted terms.
            fmt      : Optional formatter for printing each item.
            key      : Optional extractor from item to term string.
        """
        print(heading)
        added = 0
        for item in items:
            term = key(item) if key else item
            if self._novel(existing, term, terms):
                terms.add(term)
                added += 1
                print(f"  {fmt(item) if fmt else term}")
        print(f"  {added} terms from {label}")

    def _extract_nouns(self, text: str) -> list[str]:
        """
        Clean, lemmatize, and Zipf-filter words from a title string.

        Splits on whitespace after normalizing slashes and hyphens, strips
        non-alpha characters, and retains words with Zipf frequency below
        the ambiguity threshold.

        Args:
            text: A program name, trade title, or similar label.

        Returns:
            Filtered lemmatized nouns.
        """
        return [
            lemma
            for word in sub(r"[/\-]", " ", text.lower()).split()
            if (clean := sub(r"[^a-z]", "", word))
            and zipf_frequency(
                lemma := self.lemmatizer.lemmatize(clean, pos="n"),
                "en"
            ) < 4.0
        ]

    # -----------------------------------------------------------------
    # Mining sources
    # -----------------------------------------------------------------

    def _mine_onet_abbreviations(self) -> list[str]:
        """
        Extract technical abbreviations from O*NET tool and technology
        entry names.

        Scans each tool and technology entry for all-uppercase tokens
        between 4 and 6 characters, capturing domain abbreviations like
        SCADA, HCSS, and LIDAR that appear as individual words within
        multi-word entry names. The 4-character minimum avoids 3-letter
        abbreviations that produce substring false positives in
        Aho-Corasick matching because they appear inside common English
        words ("gis" in "logistics", "mis" in "commission").

        Returns:
            Sorted list of unique abbreviation strings, lowercased.
        """
        return sorted({
            clean.lower()
            for occupation in self.onet_data
            for skill in occupation["skills"]
            if skill["type"] in ("technology", "tool")
            for word in skill["name"].split()
            if (clean := sub(r"[^A-Za-z0-9]", "", word)).isupper()
            and 4 <= len(clean) <= 6
            and zipf_frequency(clean, "en") < 4.0
        })

    def _mine_onet_nouns(self) -> list[tuple[str, int, int]]:
        """
        Extract nouns from O*NET Task and DWA text that appear across
        multiple SOC codes.

        Tokenizes and POS-tags each task/DWA entry, collects nouns with 3+
        characters, lemmatizes to noun form, and filters to terms appearing
        in 3+ SOC codes with Zipf frequency below the ambiguity threshold.

        Returns:
            List of (term, soc_count, task_count) tuples sorted
            by descending SOC coverage.
        """
        noun_socs  = defaultdict(set)
        noun_tasks = defaultdict(int)

        for occupation in self.onet_data:
            soc = occupation["soc_code"]
            for skill in occupation["skills"]:
                if skill["type"] not in ("task", "dwa"):
                    continue
                for word, tag in pos_tag(word_tokenize(skill["name"])):
                    if tag.startswith("NN") and len(word) >= 3:
                        lemma = self.lemmatizer.lemmatize(
                            word.lower(), pos="n"
                        )
                        noun_socs[lemma].add(soc)
                        noun_tasks[lemma] += 1

        return [
            (noun, len(noun_socs[noun]), noun_tasks[noun])
            for noun in sorted(
                noun_socs,
                key=lambda n: (-len(noun_socs[n]), -noun_tasks[n])
            )
            if len(noun_socs[noun]) >= 3
            and zipf_frequency(noun, "en") < 4.0
        ]

    def _mine_posting_titles(self) -> list[tuple[str, int]]:
        """
        Extract high-frequency words from job posting titles.

        Tokenizes each posting title, lemmatizes words to noun form, and
        retains terms appearing in 3+ postings with Zipf frequency below
        the ambiguity threshold.

        Returns:
            List of (term, posting_count) tuples sorted by descending
            frequency.
        """
        counts = Counter(
            word
            for posting in loads(
                (self.root / "data/postings/corpus.json").read_text()
            )
            for word in self._extract_nouns(posting.get("title", ""))
        )
        return [
            (word, count)
            for word, count in counts.most_common()
            if count >= 3
        ]

    def _mine_program_words(self) -> list[str]:
        """
        Extract domain words from community college and university program
        names.

        Returns:
            Sorted list of unique program name words.
        """
        ref = self.root / "data/stakeholder/reference"
        return sorted({
            word
            for title in (
                [p["program"] for p in loads(
                    (ref / "cc_programs.json").read_text()
                )["degrees"]]
                + [p["program"] for p in loads(
                    (ref / "umaine_programs.json").read_text()
                )]
            )
            for word in self._extract_nouns(title)
        })

    def _mine_trade_words(self) -> list[str]:
        """
        Decompose stakeholder trade titles into individual words.

        Returns:
            Sorted list of unique trade title words.
        """
        return sorted({
            word
            for title in self.trades
            for word in self._extract_nouns(title)
        })

    def _novel(self, existing: set[str], term: str, terms: set[str]) -> bool:
        """
        Test whether a term is absent from both the existing index and
        the accumulating term set.

        Args:
            existing : Lowercased terms from the OSHA/O*NET index.
            term     : Candidate term to test.
            terms    : Accepted terms from prior sources.

        Returns:
            `True` if the term should be added.
        """
        lem = " ".join(
            self.lemmatizer.lemmatize(w, pos="n")
            for w in (low := term.lower()).split()
        )
        forms = {low, lem}
        return forms.isdisjoint(existing) and forms.isdisjoint(terms)

    # -----------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------

    def run_all(self):
        """
        Mine terms from all sources and write the
        `data/lexicons/supplement.json` lexicon file.
        """
        existing = self._build_existing_index()
        terms    = set()

        self._collect(
            existing,
            "Mining O*NET task/DWA text for cross-SOC nouns...",
            self._mine_onet_nouns(), "O*NET task text", terms,
            fmt=lambda x: (
                f"{x[0]:<25s}  SOCs={x[1]:>2d}  tasks={x[2]:>3d}"
            ),
            key=lambda x: x[0]
        )

        self._collect(
            existing,
            "Extracting stakeholder apprenticeship trades...",
            self.trades, "apprenticeship trades", terms,
            key=str.lower
        )

        self._collect(
            existing,
            "Decomposing stakeholder trade title words...",
            self._mine_trade_words(), "trade title words", terms
        )

        self._collect(
            existing,
            "Extracting CC/UMaine program name words...",
            self._mine_program_words(), "program name words", terms
        )

        self._collect(
            existing,
            "Extracting O*NET tool/tech abbreviations...",
            self._mine_onet_abbreviations(), "O*NET abbreviations",
            terms
        )

        self._collect(
            existing,
            "Extracting posting title words...",
            self._mine_posting_titles(), "posting titles", terms,
            fmt=lambda x: f"{x[0]:<25s}  postings={x[1]:>3d}",
            key=lambda x: x[0]
        )

        self.output.parent.mkdir(exist_ok=True, parents=True)
        self.output.write_text(dumps(sorted(terms), indent=2) + "\n")
        print(f"Wrote {len(terms)} terms to {self.output}")


if __name__ == "__main__":

    SupplementCurator(Path(__file__).resolve().parents[1]).run_all()
