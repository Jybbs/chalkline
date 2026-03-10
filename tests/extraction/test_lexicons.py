"""
Tests for skill normalization and lexicon indexing.

Validates normalization priority, lemmatized matching, task decomposition,
and graceful handling of empty inputs using synthetic fixture data.
"""

from pytest import mark

from chalkline.extraction.lexicons import LexiconRegistry
from chalkline.extraction.schemas  import OnetOccupation, OnetSkill


class TestLexiconRegistry:
    """
    Validate normalization priority, decomposition, and lemma index merging.
    """

    # ---------------------------------------------------------
    # Decomposition
    # ---------------------------------------------------------

    def test_decompose_dwa_multi_phrase(self, registry: LexiconRegistry):
        """
        A single DWA entry can produce multiple matchable sub-phrases
        from distinct NP and VP chunks.
        """
        assert registry.normalize("spread concrete") == "spread concrete"

    def test_decompose_dwa_yields_sub_phrases(self, registry: LexiconRegistry):
        """
        DWA entries are decomposed alongside tasks, yielding matchable noun
        phrases from sentence-length descriptions.
        """
        assert registry.normalize("building foundations") == "building foundations"

    def test_decompose_filters_single_word_nouns(self, registry: LexiconRegistry):
        """
        Single-word nouns from coordinate structures are filtered by the
        two-token minimum, preventing generic terms like "equipment" from
        entering the index.
        """
        assert registry.normalize("equipment") is None

    def test_decompose_maps_to_self(self, registry: LexiconRegistry):
        """
        Sub-phrases from decomposed tasks map to themselves as canonical
        names rather than the parent sentence.
        """
        assert registry.normalize("electrical wiring") == "electrical wiring"

    def test_full_sentence_not_indexed(self, registry: LexiconRegistry):
        """
        The full text of decomposable entries is not indexed, preventing
        sentence-length matches from inflating the skill vocabulary.
        """
        assert registry.normalize(
            "Install electrical wiring, equipment, and fixtures"
        ) is None

    # ---------------------------------------------------------
    # Index construction
    # ---------------------------------------------------------

    def test_empty_inputs_returns_none(self):
        """
        With no lexicon data, `normalize` returns `None`.
        """
        assert LexiconRegistry([], []).normalize("anything") is None

    def test_explicit_none_supplement(
        self,
        occupations : list[OnetOccupation],
        osha_terms  : list[str]
    ):
        """
        Passing `None` for `supplement_terms` builds the index without
        supplement data, equivalent to omitting the argument.
        """
        registry = LexiconRegistry(occupations, osha_terms, None)
        assert registry.normalize("fall protection") == "fall protection"
        assert registry.normalize("rebar") is None

    @mark.parametrize("term", ["Autodesk AutoCAD", "fall protection"])
    def test_lemma_index_contains_terms(self, registry: LexiconRegistry, term: str):
        """
        Both O*NET and OSHA terms appear in the merged index.
        """
        assert term in registry.lemma_index.values()

    def test_lemma_index_osha_overwrites_onet(self, registry: LexiconRegistry):
        """
        When a lemmatized form exists in both indices, the `lemma_index`
        resolves to the OSHA canonical form.
        """
        # "welding" is in both OSHA (as "welding") and O*NET
        # (as technology "Welding"). The merged index should
        # have the OSHA lowercase version.
        assert registry.lemma_index[registry.lemmatize("welding")] == "welding"

    def test_lemmatize_caches_words(self, registry: LexiconRegistry):
        """
        The word-level cache is populated after lemmatization so that
        repeated tokens across postings skip the WordNet lookup.
        """
        registry.lemmatize("scaffoldings welding")
        assert "scaffoldings" in registry.lemma_cache
        assert "welding" in registry.lemma_cache

    def test_lemmatize_noun_default(self, registry: LexiconRegistry):
        """
        Noun-default lemmatization reduces plurals without POS tagging,
        leaving verb forms like "installing" unchanged.
        """
        assert registry.lemmatize("scaffoldings") == "scaffolding"
        assert registry.lemmatize("installing") == "installing"

    # ---------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------

    def test_normalize_case_insensitive(self, registry: LexiconRegistry):
        """
        Uppercase input normalizes to the canonical lowercase form via
        lemmatization.
        """
        assert registry.normalize("FALL PROTECTION") == "fall protection"

    def test_normalize_empty_string(self, registry: LexiconRegistry):
        """
        An empty string normalizes to `None`.
        """
        assert registry.normalize("") is None

    def test_normalize_excludes_ksa_types(self, registry: LexiconRegistry):
        """
        Abstract KSA types are excluded from the normalization index,
        returning `None` even though they exist in the occupation profile.
        """
        assert registry.normalize("Blueprint Reading") is None
        assert registry.normalize("Mathematics") is None

    def test_normalize_handles_plural(self, registry: LexiconRegistry):
        """
        Plural forms match their singular canonical entry via lemmatization.
        """
        assert registry.normalize("scaffoldings") == "scaffolding"

    def test_normalize_onet_term(self, registry: LexiconRegistry):
        """
        An O*NET technology term normalizes to its canonical form.
        """
        assert registry.normalize("Autodesk AutoCAD") == "Autodesk AutoCAD"

    def test_normalize_osha_exact_case(self, registry: LexiconRegistry):
        """
        OSHA terms match via the lowercased-original index entry independent
        of any POS-based lemmatization ambiguity.
        """
        assert registry.normalize("asbestos") == "asbestos"
        assert registry.normalize("Asbestos") == "asbestos"

    def test_normalize_osha_priority(self, registry: LexiconRegistry):
        """
        A term in both OSHA and O*NET resolves to the OSHA canonical form.
        """
        # "welding" is an OSHA term and "Welding" is an O*NET
        # technology. OSHA priority means the lowercase OSHA
        # form wins.
        assert registry.normalize("welding") == "welding"

    def test_normalize_osha_term(self, registry: LexiconRegistry):
        """
        An OSHA safety term normalizes to its canonical form.
        """
        assert registry.normalize("fall protection") == "fall protection"

    def test_normalize_unknown_returns_none(self, registry: LexiconRegistry):
        """
        An unrecognized term returns `None`.
        """
        assert registry.normalize("quantum computing") is None

    def test_normalize_vp_phrase(self, registry: LexiconRegistry):
        """
        VP chunks from decomposed tasks produce matchable phrases
        alongside NP chunks.
        """
        assert (
            registry.normalize("operate construction equipment")
            == "operate construction equipment"
        )

    def test_phrases_empty_skips_indexing(self):
        """
        A decomposable entry with an empty `phrases` list indexes
        nothing rather than falling back to the full sentence.
        """
        registry = LexiconRegistry([OnetOccupation(
            job_zone = 1,
            sector   = "Test",
            skills   = [OnetSkill(
                name    = "Some task sentence",
                phrases = [],
                type    = "task"
            )],
            soc_code = "99-0000.00",
            title    = "Test"
        )], [])
        assert registry.normalize("Some task sentence") is None

    def test_short_phrases_not_decomposed(self, registry: LexiconRegistry):
        """
        Tool entries are indexed directly without decomposition.
        """
        assert registry.normalize("Laptop computers") == "Laptop computers"

    # ---------------------------------------------------------
    # Supplement integration
    # ---------------------------------------------------------

    def test_supplement_onet_overwrites_supplement(self, registry: LexiconRegistry):
        """
        A term in both supplement and O*NET resolves to the O*NET
        canonical form, confirming O*NET > supplement priority.
        """
        # "concrete finishing" is both an O*NET technology on the
        # paving operator and a supplement term. O*NET wins.
        assert registry.normalize("concrete finishing") == "Concrete finishing"

    def test_supplement_only_term(self, registry: LexiconRegistry):
        """
        A term present only in the supplement lexicon normalizes
        to its canonical form.
        """
        assert registry.normalize("rebar") == "rebar"

    def test_supplement_term_not_in_primary(self, registry: LexiconRegistry):
        """
        Supplement terms that overlap with neither OSHA nor O*NET
        are still resolved by the registry.
        """
        assert registry.normalize("excavation") == "excavation"
