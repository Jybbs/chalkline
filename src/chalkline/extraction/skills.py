"""
Skill extraction from job posting text via Aho-Corasick matching.

Builds surface form variants for each lexicon term, loads them into an
`ahocorasick_rs` automaton with `LeftmostLongest` semantics, and runs a
single-pass match per posting with word boundary enforcement. The output
is a mapping from document identifiers to deduplicated, sorted canonical
skill names.
"""

from ahocorasick_rs import AhoCorasick, MatchKind
from html           import unescape
from logging        import getLogger
from nltk.stem      import PorterStemmer
from re             import IGNORECASE, MULTILINE, search, sub
from typing         import NamedTuple

from chalkline                     import SkillMap
from chalkline.extraction.lexicons import LexiconRegistry
from chalkline.extraction.schemas  import ConfidenceTier

logger = getLogger(__name__)

class PatternMeta(NamedTuple):
    """
    Metadata for a single surface form pattern in the automaton.

    Tracks which canonical skill name a pattern resolves to and
    its confidence tier for conflict resolution.
    """

    canonical : str
    tier      : ConfidenceTier

class SkillExtractor:
    """
    Aho-Corasick skill extractor with surface form augmentation.

    Generates lowercased, lemmatized, stemmed, and inverted bigram
    variants for each lexicon term, loads them into a
    `LeftmostLongest` automaton, and provides a single `extract()`
    method that preprocesses posting text, lemmatizes, matches with
    word boundary enforcement, and returns deduplicated canonical
    skill names per document.
    """

    def __init__(self, registry: LexiconRegistry):
        """
        Build skill automaton from registry data.

        Args:
            registry: Lexicon registry with merged `lemma_index`
                      and `lemmatize` method.
        """
        self.registry = registry
        self.stemmer  = PorterStemmer()

        patterns, self.metadata = self._build_patterns()
        self.pattern_chars      = frozenset(c for p in patterns for c in p)
        self.automaton          = AhoCorasick(patterns, MatchKind.LeftmostLongest)

    @property
    def vocabulary(self) -> set[str]:
        """
        The set of all canonical skill names loadable by this
        extractor.

        Returns:
            Unique canonical names across all loaded patterns.
        """
        return {m.canonical for m in self.metadata}

    def _build_patterns(self) -> tuple[list[str], list[PatternMeta]]:
        """
        Generate augmented surface forms with parallel metadata.

        For each unique canonical name in the registry's
        `lemma_index`, produces lowercased canonical, all
        lemmatized keys, Porter-stemmed variants, and inverted
        bigrams for two-word skills. Each pattern gets metadata
        tracking its canonical name and confidence tier.

        Returns:
            A parallel pair of pattern strings and their
            metadata.
        """
        patterns = []
        metadata = []
        seen     = set()

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
                patterns.append(form)
                metadata.append(PatternMeta(
                    canonical = canonical,
                    tier      = self._classify_tier(canonical, form)
                ))

        return patterns, metadata

    def _classify_tier(self, canonical: str, form: str) -> ConfidenceTier:
        """
        Assign a confidence tier based on form characteristics.

        Args:
            canonical : Original canonical name for abbreviation detection.
            form      : The lowercased surface form string.

        Returns:
            The appropriate confidence tier.
        """
        if " " in form:
            return ConfidenceTier.MULTI_WORD
        if canonical.isupper() and 2 <= len(canonical) <= 6:
            return ConfidenceTier.ABBREVIATION
        return ConfidenceTier.SINGLE_WORD

    def _is_word_boundary(self, end: int, start: int, text: str) -> bool:
        """
        Check whether a match span falls on word boundaries.

        After preprocessing, all non-pattern characters have been
        replaced with spaces, so a valid boundary is simply a
        space or string edge.

        Args:
            end   : End position (exclusive) of the match.
            start : Start position of the match.
            text  : The full text being searched.

        Returns:
            `True` if both boundaries are valid.
        """
        return (
            (start == 0 or text[start - 1] == " ")
            and (end == len(text) or text[end] == " ")
        )

    def _match(self, text: str) -> list[str]:
        """
        Run the automaton and return deduplicated canonical names.

        Matches are filtered for word boundaries and then
        deduplicated by canonical name. When multiple surface
        forms of the same canonical name match, the highest-
        confidence tier wins, though `LeftmostLongest` already
        prefers longer multi-word matches at each position.

        Args:
            text: Lemmatized posting text.

        Returns:
            Sorted list of unique canonical skill names.
        """
        hits = {}
        for pattern_idx, start, end in self.automaton.find_matches_as_indexes(text):
            if not self._is_word_boundary(end, start, text):
                continue
            canonical = (meta := self.metadata[pattern_idx]).canonical
            if canonical not in hits or meta.tier.value < hits[canonical].value:
                hits[canonical] = meta.tier

        return sorted(hits)

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
        text = "".join(c if c in self.pattern_chars else " " for c in text)
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
