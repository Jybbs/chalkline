"""
Skill normalization against OSHA, O*NET, and supplement lexicons.

Builds lemmatized lookup indices with OSHA > O*NET > supplement
priority from pre-decomposed O*NET sub-phrases, OSHA topic terms,
and domain supplement terms so that Aho-Corasick matching can find
fragments within posting text.
"""

from functools import cache
from nltk      import download
from nltk.data import find
from nltk.stem import WordNetLemmatizer

from chalkline.extraction.schemas import OnetOccupation


@cache
def _ensure_nltk_data():
    """
    Download NLTK data required for noun-default lemmatization.
    """
    for corpus in ("omw-1.4", "wordnet"):
        try:
            find(f"corpora/{corpus}")
        except LookupError:
            download(corpus, quiet=True)


class LexiconRegistry:
    """
    Normalization index merging OSHA, O*NET, and supplement lexicons.

    Builds lemmatized lookup indices from all lexicon sources using
    pre-decomposed sub-phrases for O*NET Tasks and DWAs, and exposes
    a merged `lemma_index` with OSHA > O*NET > supplement priority
    for pattern matching.
    """

    def __init__(
        self,
        occupations      : list[OnetOccupation],
        osha_terms       : list[str],
        supplement_terms : list[str] | None = None
    ):
        """
        Build normalization indices from loaded lexicon data.

        Args:
            occupations      : Validated O*NET occupation records.
            osha_terms       : Validated OSHA topic strings.
            supplement_terms : Domain supplement terms at lowest
                               priority, or `None` to skip.
        """
        _ensure_nltk_data()
        self.lemma_cache = {}
        self.lemmatizer  = WordNetLemmatizer()
        self.lemma_index = (
            self._build_lemma_index(supplement_terms or [])
            | self._build_onet_index(occupations)
            | self._build_lemma_index(osha_terms)
        )

    def _build_lemma_index(self, terms: list[str]) -> dict[str, str]:
        """
        Build a mapping from lemmatized forms to canonical terms.

        Each term is stored under both its lemmatized form and its
        lowercased original for robust lookup when lemmatization
        produces an unexpected variant.

        Args:
            terms: Canonical skill names to index.

        Returns:
            Mapping from lookup form to canonical form.
        """
        return {
            key: term
            for term in terms
            for key  in (self.lemmatize(term), term.lower())
        }

    def _build_onet_index(self, occupations: list[OnetOccupation]) -> dict[str, str]:
        """
        Build the O*NET normalization index from concrete types.

        Decomposable types read pre-computed sub-phrases from the
        `phrases` field populated at curation time, mapping each phrase
        to itself as a canonical name. Non-decomposable concrete types
        are indexed directly by `skill.name`.

        Args:
            occupations: O*NET occupation records to index.

        Returns:
            Mapping from lemmatized form to canonical skill name.
        """
        return {
            self.lemmatize(phrase): phrase
            for occupation in occupations
            for skill      in occupation.skills if skill.type.is_concrete
            for phrase     in (
                skill.phrases if skill.phrases is not None
                else [skill.name]
            )
        }

    def lemmatize(self, text: str) -> str:
        """
        Lowercase and lemmatize a term using noun-default WordNet.

        All tokens are lemmatized as nouns because construction skill
        terms are overwhelmingly nominal and the matching contract
        requires only that index construction and extraction-time
        lemmatization agree, not that they be linguistically exact.
        A word-level cache avoids redundant lemmatization across
        postings that share vocabulary.

        Args:
            text: Raw skill term or phrase.

        Returns:
            Space-joined lemmatized form.
        """
        return " ".join(
            self.lemma_cache.get(word)
            or self.lemma_cache.setdefault(
                word, self.lemmatizer.lemmatize(word, pos="n")
            )
            for word in text.lower().split()
        )

    def normalize(self, raw_term: str) -> str | None:
        """
        Resolve a raw term to its canonical lexicon form.

        Lemmatizes the input and looks it up in the merged index, where OSHA
        entries take priority over O*NET.

        Args:
            raw_term: Unprocessed skill term from posting text.

        Returns:
            The canonical form, or `None` if no match.
        """
        return self.lemma_index.get(self.lemmatize(raw_term))
