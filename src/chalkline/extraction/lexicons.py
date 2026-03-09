"""
Skill normalization against OSHA and O*NET lexicons.

Builds lemmatized lookup indices with OSHA > O*NET priority and decomposes
sentence-length O*NET Tasks and DWAs into matchable sub-phrases via
POS-based chunking so that Aho-Corasick matching in CL-06 can find fragments
within posting text.
"""

from functools import cache
from nltk      import download, pos_tag, RegexpParser, Tree, word_tokenize
from nltk.stem import WordNetLemmatizer

from chalkline.extraction.schemas import OnetOccupation


@cache
def _ensure_nltk_data():
    """
    Download required NLTK data packages if not already present.
    """
    for package in (
        "averaged_perceptron_tagger_eng",
        "punkt_tab",
        "wordnet"
    ):
        download(package, quiet=True)


class LexiconRegistry:
    """
    Normalization index merging OSHA and O*NET lexicons.

    Builds lemmatized lookup indices from both lexicon sources, decomposes
    sentence-length O*NET entries into matchable sub-phrases, and exposes a
    merged `lemma_index` with OSHA > O*NET priority for CL-06 pattern
    matching.
    """

    def __init__(self, occupations: list[OnetOccupation], osha_terms: list[str]):
        """
        Build normalization indices from loaded lexicon data.

        Args:
            occupations : Validated O*NET occupation records.
            osha_terms  : Validated OSHA topic strings.
        """
        _ensure_nltk_data()
        self.lemmatizer  = WordNetLemmatizer()
        self.lemma_index = {}
        self.lemma_index.update(self._build_onet_index(occupations))
        self.lemma_index.update(self._build_lemma_index(osha_terms))


    def _build_lemma_index(self, terms: list[str]) -> dict[str, str]:
        """
        Build a mapping from lemmatized forms to canonical terms.

        Each term is stored under both its lemmatized form and its
        lowercased original, because POS tagging of isolated terms is
        unreliable ("scaffolding" alone tags as VBG rather than NN,
        producing a different lemma than "scaffoldings").

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

        Tasks and DWAs are decomposed into sub-phrases, each mapping back to
        the parent canonical entry. Technology, tool, and alternate title
        entries are indexed directly.

        Args:
            occupations: O*NET occupation records to index.

        Returns:
            Mapping from lemmatized form to canonical skill name.
        """
        return {
            self.lemmatize(phrase): skill.name
            for occupation in occupations
            for skill      in occupation.skills if skill.type.is_concrete
            for phrase     in [
                skill.name,
                *(self._decompose(skill.name) if skill.type.is_decomposable else [])
            ]
        }

    def _decompose(self, text: str) -> list[str]:
        """
        Extract matchable sub-phrases from a sentence-length entry.

        Uses `nltk.RegexpParser` with NP and VP grammar rules to chunk
        POS-tagged tokens. Noun phrases and verb-object pairs are extracted
        from subtrees, while lone nouns not captured by NP/VP rules are
        collected separately to handle coordinate structures like "sinks,
        toilets, and bathtubs." All results are returned as lowercase
        strings.

        Args:
            text: A sentence-length O*NET Task or DWA.

        Returns:
            List of extracted sub-phrase strings.
        """
        tree = RegexpParser(r"""
            NP: {<DT>?<JJ>*<NN.*>+}
            VP: {<VB.*><NP>}
        """).parse(pos_tag(word_tokenize(text)))
        assert isinstance(tree, Tree)

        return [
            *(" ".join(word for word, _ in subtree.leaves()).lower()
              for subtree in tree.subtrees(lambda t: t.label() in ("NP", "VP"))),
            *(item[0].lower() for item in tree
              if isinstance(item, tuple) and item[1].startswith("NN"))
        ]

    def lemmatize(self, text: str) -> str:
        """
        Lowercase, tokenize, POS-tag, and lemmatize a term.

        Penn Treebank tags are mapped to WordNet POS constants by first
        letter (J → adjective, R → adverb, V → verb), defaulting to noun
        when the tagger produces an unexpected category because noun
        lemmatization is the safest fallback.

        Args:
            text: Raw skill term or phrase.

        Returns:
            Space-joined lemmatized form.
        """
        return " ".join(
            self.lemmatizer.lemmatize(
                pos  = {"J": "a", "R": "r", "V": "v"}.get(tag[0], "n"),
                word = word
            )
            for word, tag in pos_tag(word_tokenize(text.lower()))
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
