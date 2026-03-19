"""
Skill extraction from job posting text via Aho-Corasick matching.

Builds surface form variants for each lexicon term, loads them into a
`pyahocorasick` automaton with longest-match semantics, and runs a
single-pass match per posting with word boundary enforcement. The output
is a mapping from document identifiers to deduplicated, sorted canonical
skill names.
"""

from ahocorasick import Automaton
from html        import unescape
from loguru      import logger
from nltk.stem   import PorterStemmer
from re          import IGNORECASE, MULTILINE, search, sub

from chalkline                     import SkillMap
from chalkline.extraction.lexicons import LexiconRegistry
from chalkline.extraction.schemas  import PatternBundle


class SkillExtractor:
    """
    Aho-Corasick skill extractor with surface form augmentation.

    Generates lowercased, lemmatized, stemmed, and inverted bigram
    variants for each lexicon term, loads them into a
    `pyahocorasick` automaton, and provides a single `extract()`
    method that preprocesses posting text, lemmatizes, matches
    with word boundary enforcement via `iter_long()`, and returns
    deduplicated canonical skill names per document.
    """

    def __init__(self, registry: LexiconRegistry):
        """
        Build skill automaton from registry data.

        Args:
            registry: Lexicon registry with merged `lemma_index`
                      and `lemmatize` method.
        """
        self.registry  = registry
        self.stemmer   = PorterStemmer()
        self.bundle    = self._build_pattern_bundle()
        self.automaton = self._build_automaton()

        logger.info(
            f"Built automaton with {len(self.bundle.patterns)} patterns "
            f"from {len(self.vocabulary)} canonical skills"
        )

    @property
    def vocabulary(self) -> set[str]:
        """
        The set of all canonical skill names loadable by this
        extractor.

        Returns:
            Unique canonical names across all loaded patterns.
        """
        return set(self.bundle.canonicals)

    def _build_automaton(self) -> Automaton:
        """
        Load all surface form patterns into an Aho-Corasick automaton.

        Each pattern is keyed by its positional index into
        `self.bundle.patterns` and `self.bundle.metadata`,
        enabling O(1) metadata lookup during matching.

        Returns:
            A finalized automaton ready for `iter_long()` queries.
        """
        automaton = Automaton()
        for idx, p in enumerate(self.bundle.patterns):
            automaton.add_word(p, idx)

        automaton.make_automaton()
        return automaton

    def _build_pattern_bundle(self) -> PatternBundle:
        """
        Generate augmented surface forms with parallel canonicals.

        For each unique canonical name in the registry's
        `lemma_index`, produces lowercased canonical, all
        lemmatized keys, Porter-stemmed variants, and inverted
        bigrams for two-word skills. The character set is
        accumulated during iteration for use as a preprocessing
        filter.

        Returns:
            A `PatternBundle` of pattern strings, their canonical
            names, and the set of all characters in patterns.
        """
        canonicals = []
        chars      = set()
        patterns   = []
        seen       = set()

        canonical_to_lemmas = {}
        for lemma, canonical in self.registry.lemma_index.items():
            canonical_to_lemmas.setdefault(canonical, set()).add(lemma)

        for canonical, lemmas in sorted(canonical_to_lemmas.items()):
            canon = canonical.lower().replace("/", " ").strip()
            forms = {canon, self._stem(canon)} | {
                f.replace("/", " ").strip() for f in lemmas
            }

            if len(words := canon.split()) == 2:
                forms.add(f"{words[1]} {words[0]}")

            forms -= seen | {""}
            seen  |= forms
            for form in sorted(forms):
                chars.update(form)
                canonicals.append(canonical)
                patterns.append(form)

        return PatternBundle(
            canonicals = canonicals,
            chars      = chars,
            patterns   = patterns,
        )

    def _match(self, text: str) -> list[str]:
        """
        Run the automaton and return deduplicated canonical names.

        Matches are filtered for word boundaries and then
        deduplicated by canonical name. `iter_long()` already
        prefers longer matches at each position.

        Args:
            text: Lemmatized posting text.

        Returns:
            Sorted list of unique canonical skill names.
        """
        matched = set()
        text    = f" {text} "
        for end, idx in self.automaton.iter_long(text):
            start, stop = self.bundle.span_of(end, idx)
            if text[start - 1] == " " and text[stop] == " ":
                matched.add(self.bundle.canonicals[idx])

        return sorted(matched)

    def _preprocess(self, text: str) -> str:
        """
        Normalize raw posting text for skill matching.

        Drops preamble text before the first structural marker and
        EEO boilerplate after the first equal-opportunity marker,
        strips HTML tags, splits camelCase terms, lowercases,
        removes characters absent from lexicon patterns, and
        collapses whitespace. The allowed character set is derived
        at init time from the automaton patterns themselves, so the
        filter is always consistent with whatever the lexicons
        contain.

        Args:
            text: Raw posting description.

        Returns:
            Cleaned, lowercased text ready for matching.
        """
        if m := search(r"^[ \t]*(?:\*[\s*]|#{1,4}\s)", text, MULTILINE):
            text = text[m.start():]
        if (
            (m := search(
                r"equal\s+(?:opportunity|employment\s+opportunity)\s+employer|"
                r"without\s+regard\s+to\s+race",
                text,
                IGNORECASE,
            ))
            and m.start() > len(text) // 2
        ):
            text = text[:m.start()]
        text = sub(r"<[^>]+>", " ", text)
        text = unescape(text)
        text = sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = text.lower().strip()
        text = "".join(c if c in self.bundle.chars else " " for c in text)
        return sub(r"\s+", " ", text)

    def _stem(self, text: str) -> str:
        """
        Apply Porter stemming to each word in the text.

        Args:
            text: Canonical skill name or phrase.

        Returns:
            Space-joined stemmed form, lowercased.
        """
        return " ".join(
            self.stemmer.stem(word)
            for word in text.lower().split()
        )

    def extract(self, postings: dict[str, str]) -> SkillMap:
        """
        Extract canonical skill names from a corpus of posting
        texts.

        Each posting is preprocessed, lemmatized, and matched
        against the skill automaton. Postings with zero matched
        skills are excluded from the output.

        Args:
            postings: Mapping from document identifier to raw text.

        Returns:
            Mapping from document identifier to sorted canonical
            skill names, excluding documents with no matches.
        """
        results = {}

        for doc_id in sorted(postings):
            lemmatized = self.registry.lemmatize(self._preprocess(postings[doc_id]))
            if skills := self._match(lemmatized):
                results[doc_id] = skills

        excluded = len(postings) - len(results)
        if excluded:
            logger.info(f"Excluded {excluded} posting(s) with zero matched skills")

        logger.debug(f"Extracted skills from {len(results)} posting(s)")

        return results
