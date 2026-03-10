"""
Skill extraction from job posting text via Aho-Corasick matching.

Builds surface form variants for each lexicon term, loads them into an
`ahocorasick_rs` automaton with `LeftmostLongest` semantics, and runs a
single-pass match per posting with word boundary enforcement and filler
phrase masking. The output is a mapping from document identifiers to
deduplicated, sorted canonical skill names.
"""

from ahocorasick_rs                  import AhoCorasick, MatchKind
from logging                         import getLogger
from nltk.stem                       import PorterStemmer
from re                              import sub
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from typing                          import NamedTuple

from chalkline.extraction.lexicons import LexiconRegistry
from chalkline.extraction.schemas  import ConfidenceTier


logger = getLogger(__name__)


# -------------------------------------------------------------------------
# Default filler phrases
# -------------------------------------------------------------------------

FILLER_PHRASES = (
    "ability to",
    "and or",
    "applicants must",
    "as needed",
    "as required",
    "at least",
    "candidates must",
    "candidates should",
    "demonstrated ability",
    "demonstrated experience",
    "equal opportunity employer",
    "experience in",
    "experience with",
    "familiarity with",
    "hands on experience",
    "ideal candidate",
    "in accordance with",
    "in addition to",
    "job description",
    "job duties",
    "job requirements",
    "knowledge of",
    "minimum qualifications",
    "minimum requirements",
    "must be able to",
    "must be willing to",
    "must have",
    "or equivalent",
    "or more years",
    "preferred qualifications",
    "prior experience",
    "proficiency in",
    "proficient in",
    "proven ability",
    "proven track record",
    "required qualifications",
    "required to",
    "responsible for",
    "skilled in",
    "strong understanding of",
    "understanding of",
    "willing to",
    "working knowledge of",
    "years of experience"
)


# -------------------------------------------------------------------------
# Pattern metadata
# -------------------------------------------------------------------------

class PatternMeta(NamedTuple):
    """
    Metadata for a single surface form pattern in the automaton.

    Tracks which canonical skill name a pattern resolves to and its
    confidence tier for conflict resolution.
    """

    canonical : str
    tier      : ConfidenceTier


# -------------------------------------------------------------------------
# SkillExtractor
# -------------------------------------------------------------------------

class SkillExtractor:
    """
    Aho-Corasick skill extractor with surface form augmentation.

    Generates lowercased, lemmatized, stemmed, and inverted bigram variants
    for each lexicon term, loads them into a `LeftmostLongest` automaton,
    and provides a single `extract()` method that preprocesses posting text,
    lemmatizes, matches with word boundary enforcement, and returns
    deduplicated canonical skill names per document.
    """

    def __init__(
        self,
        registry       : LexiconRegistry,
        filler_phrases : tuple[str, ...] = FILLER_PHRASES
    ):
        """
        Build filler and skill automatons from registry data.

        Args:
            registry       : Lexicon registry with merged `lemma_index`
                             and `lemmatize` method.
            filler_phrases : Phrases to blank before skill matching.
        """
        self.registry = registry
        self.stemmer  = PorterStemmer()

        self.filler_automaton = AhoCorasick(
            [p.lower() for p in filler_phrases],
            matchkind=MatchKind.LeftmostLongest
        )

        patterns, self.metadata = self._build_patterns()
        self.automaton = AhoCorasick(
            patterns,
            matchkind=MatchKind.LeftmostLongest
        )

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def vocabulary(self) -> set[str]:
        """
        The set of all canonical skill names loadable by this extractor.
        """
        return {m.canonical for m in self.metadata}

    # -----------------------------------------------------------------
    # Private methods
    # -----------------------------------------------------------------

    def _build_patterns(self) -> tuple[list[str], list[PatternMeta]]:
        """
        Generate augmented surface forms with parallel metadata.

        For each unique canonical name in the registry's `lemma_index`,
        produces lowercased canonical, all lemmatized keys, Porter-stemmed
        variants, and inverted bigrams for two-word skills. Each pattern
        gets metadata tracking its canonical name and confidence tier.

        Returns:
            A parallel pair of pattern strings and their metadata.
        """
        patterns = []
        metadata = []
        seen     = set()

        canonical_to_lemmas = {}
        for lemma, canonical in self.registry.lemma_index.items():
            canonical_to_lemmas.setdefault(canonical, set()).add(lemma)

        for canonical, lemmas in sorted(canonical_to_lemmas.items()):
            forms = {canonical.lower(), self._stem(canonical)} | lemmas

            if len(words := canonical.lower().split()) == 2:
                forms.add(f"{words[1]} {words[0]}")

            for form in sorted(forms):
                if not form or form in seen:
                    continue
                seen.add(form)
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

        Args:
            end   : End position (exclusive) of the match.
            start : Start position of the match.
            text  : The full text being searched.

        Returns:
            `True` if both boundaries are valid.
        """
        return (
            (start == 0 or not text[start - 1].isalnum())
            and (end == len(text) or not text[end].isalnum())
        )

    def _match(self, text: str) -> list[str]:
        """
        Run the skill automaton and return deduplicated canonical names.

        Matches are filtered for word boundaries and then deduplicated by
        canonical name. When multiple surface forms of the same canonical
        name match, the highest-confidence tier wins, though
        `LeftmostLongest` already prefers longer multi-word matches at
        each position.

        Args:
            text: Lemmatized posting text.

        Returns:
            Sorted list of unique canonical skill names.
        """
        hits = {}
        for pattern_idx, start, end in self.automaton.find_matches_as_indexes(text):
            if not self._is_word_boundary(end, start, text):
                continue
            meta      = self.metadata[pattern_idx]
            canonical = meta.canonical
            if canonical not in hits or meta.tier.value < hits[canonical].value:
                hits[canonical] = meta.tier

        return sorted(hits)

    def _preprocess(self, text: str) -> str:
        """
        Normalize raw posting text and mask filler phrases.

        Converts bullet characters and semicolons to periods, splits
        camelCase terms into separate words, collapses whitespace,
        lowercases, and blanks filler phrase spans with whitespace
        before skill matching.

        Args:
            text: Raw posting description.

        Returns:
            Cleaned, lowercased text with fillers replaced by spaces.
        """
        text = sub(r"[•●■◦▪–—;]", ". ", text)
        text = sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = sub(r"\s+", " ", text)
        text = text.lower().strip()

        chars = list(text)
        for _, start, end in self.filler_automaton.find_matches_as_indexes(text):
            if self._is_word_boundary(end, start, text):
                for i in range(start, end):
                    chars[i] = " "
        return "".join(chars)

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

    # -----------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------

    def extract(self, postings: dict[str, str]) -> dict[str, list[str]]:
        """
        Extract canonical skill names from a corpus of posting texts.

        Each posting is preprocessed, lemmatized, and matched against the
        skill automaton. Postings with zero matched skills are excluded
        from the output. Unmatched term frequencies are logged for lexicon
        coverage diagnostics.

        Args:
            postings: Mapping from document identifier to raw text.

        Returns:
            Mapping from document identifier to sorted canonical skill
            names, excluding documents with no matches.
        """
        results        = {}
        corpus_tokens  = set()
        matched_tokens = set()

        for doc_id in sorted(postings):
            lemmatized = self.registry.lemmatize(
                self._preprocess(postings[doc_id])
            )
            skills = self._match(lemmatized)

            if skills:
                results[doc_id] = skills

            corpus_tokens.update(
                t for t in lemmatized.split()
                if len(t) >= 3 and t not in ENGLISH_STOP_WORDS
            )
            matched_tokens.update(
                t for s in skills for t in s.lower().split()
            )

        excluded  = len(postings) - len(results)
        unmatched = corpus_tokens - matched_tokens

        if excluded:
            logger.info(
                f"Excluded {excluded} posting(s) with zero matched skills"
            )

        if corpus_tokens:
            unmatched_rate = len(unmatched) / len(corpus_tokens)
            if unmatched_rate > 0.15:
                logger.warning(
                    f"Unmatched term rate {unmatched_rate:.1%} exceeds "
                    f"threshold. Top unmatched: "
                    f"{sorted(unmatched)[:20]}"
                )

        logger.debug(
            f"Extracted skills from {len(results)} posting(s), "
            f"{excluded} excluded, "
            f"{len(unmatched)} unique unmatched terms"
        )

        return results
