"""
Tests for skill normalization and lexicon indexing.

Validates normalization priority, lemmatized matching, task decomposition,
and graceful handling of empty inputs using synthetic fixture data.
"""

from pytest import mark

from chalkline.extraction.lexicons import LexiconRegistry


class TestLexiconRegistry:
    """
    Validate normalization priority, decomposition, and lemma index merging.
    """

    # ---------------------------------------------------------
    # Decomposition
    # ---------------------------------------------------------

    def test_decompose_dwa_yields_sub_phrases(self, registry: LexiconRegistry):
        """
        DWA entries are decomposed alongside tasks, yielding matchable noun
        phrases from sentence-length descriptions.
        """
        # "Pour and spread concrete for building foundations"
        # should decompose to include "building foundations"
        assert registry.normalize("building foundations") is not None

    def test_decompose_extracts_coordinate_nouns(self, registry: LexiconRegistry):
        """
        Nouns within coordinate structures from decomposed Tasks are
        individually matchable through the normalization index.
        """
        assert registry.normalize("equipment") is not None

    def test_decompose_maps_to_parent(self, registry: LexiconRegistry):
        """
        Sub-phrases from a decomposed task resolve to the parent canonical
        entry through `normalize`.
        """
        assert (result := registry.normalize("electrical wiring")) is not None
        assert "wiring" in result.lower()

    # ---------------------------------------------------------
    # Empty inputs
    # ---------------------------------------------------------

    def test_empty_inputs_returns_none(self):
        """
        With no lexicon data, `normalize` returns `None`.
        """
        assert LexiconRegistry([], []).normalize("anything") is None

    # ---------------------------------------------------------
    # Lemma index
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Normalization
    # ---------------------------------------------------------

    def test_normalize_case_insensitive(self, registry: LexiconRegistry):
        """
        Uppercase input normalizes to the canonical lowercase form via
        lemmatization.
        """
        assert registry.normalize("FALL PROTECTION") == "fall protection"

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

    def test_short_phrases_not_decomposed(self, registry: LexiconRegistry):
        """
        Tool entries are indexed directly without decomposition.
        """
        assert registry.normalize("Laptop computers") is not None
