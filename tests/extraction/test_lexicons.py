"""
Tests for skill normalization and lexicon indexing.

Validates normalization priority, lemmatized matching, task
decomposition, and graceful handling of empty inputs using synthetic
fixture data.
"""

from pytest import mark

from chalkline.extraction.lexicons import LexiconRegistry
from chalkline.extraction.schemas  import Certification, OnetOccupation, OnetSkill


class TestLexiconRegistry:
    """
    Validate normalization priority, decomposition, and lemma index
    merging.
    """

    # -------------------------------------------------------------------------
    # Decomposition
    # -------------------------------------------------------------------------

    def test_decompose_filters_single_words(self, registry: LexiconRegistry):
        """
        Single-word nouns from coordinate structures are filtered
        by the two-token minimum, preventing generic terms like
        "equipment" from entering the index.
        """
        assert registry.normalize("equipment") is None

    @mark.parametrize("phrase", [
        "building foundations",
        "electrical wiring",
        "operate construction equipment",
        "spread concrete"
    ])
    def test_decompose_sub_phrase(self, phrase: str, registry: LexiconRegistry):
        """
        Sub-phrases from decomposed tasks and DWAs normalize to
        themselves rather than the parent sentence, covering both NP
        chunks and VP chunks.
        """
        assert registry.normalize(phrase) == phrase

    def test_full_sentence_not_indexed(self, registry: LexiconRegistry):
        """
        The full text of decomposable entries is not indexed,
        preventing sentence-length matches from inflating the
        skill vocabulary.
        """
        assert registry.normalize(
            "Install electrical wiring, equipment, and fixtures"
        ) is None

    # -------------------------------------------------------------------------
    # Index construction
    # -------------------------------------------------------------------------

    def test_empty_inputs(self):
        """
        With no lexicon data, `normalize` returns `None`.
        """
        assert LexiconRegistry(
            occupations = [],
            osha_terms  = []
        ).normalize("anything") is None

    def test_lemmatize_noun_default(self, registry: LexiconRegistry):
        """
        Noun-default lemmatization reduces plurals without POS
        tagging, leaving verb forms like "installing" unchanged.
        """
        assert registry.lemmatize("scaffoldings") == "scaffolding"
        assert registry.lemmatize("installing") == "installing"

    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------

    def test_normalize_case_insensitive(self, registry: LexiconRegistry):
        """
        Uppercase input normalizes to the canonical lowercase form
        via lemmatization.
        """
        assert registry.normalize("FALL PROTECTION") == "fall protection"

    def test_normalize_empty_string(self, registry: LexiconRegistry):
        """
        An empty string normalizes to `None`.
        """
        assert registry.normalize("") is None

    @mark.parametrize("term", ["Blueprint Reading", "Mathematics"])
    def test_normalize_excludes_ksa(self, registry: LexiconRegistry, term: str):
        """
        Abstract KSA types are excluded from the normalization
        index, returning `None` even though they exist in the
        occupation profile.
        """
        assert registry.normalize(term) is None

    @mark.parametrize("term", ["Asbestos", "asbestos"])
    def test_normalize_osha_exact_case(self, registry: LexiconRegistry, term: str):
        """
        OSHA terms match via the lowercased-original index entry
        independent of any POS-based lemmatization ambiguity.
        """
        assert registry.normalize(term) == "asbestos"

    def test_normalize_osha_priority(self, registry: LexiconRegistry):
        """
        A term in both OSHA and O*NET resolves to the OSHA
        canonical form.
        """
        # "welding" is an OSHA term and "Welding" is an O*NET
        # technology. OSHA priority means the lowercase OSHA
        # form wins.
        assert registry.normalize("welding") == "welding"

    def test_normalize_plural(self, registry: LexiconRegistry):
        """
        Plural forms match their singular canonical entry via
        lemmatization.
        """
        assert registry.normalize("scaffoldings") == "scaffolding"

    def test_normalize_unknown(self, registry: LexiconRegistry):
        """
        An unrecognized term returns `None`.
        """
        assert registry.normalize("quantum computing") is None

    def test_onet_collision_last_wins(self):
        """
        When skills across occupations share a lemmatized form,
        the later occupation's canonical form wins via dict
        overwrite. A refactor changing occupation iteration order
        would silently alter every downstream TF-IDF vector.
        """
        occupations = [
            OnetOccupation(
                job_zone = 2,
                sector   = "Heavy Highway Construction",
                skills   = [OnetSkill(
                    name = "concrete finishing",
                    type = "technology"
                )],
                soc_code = "47-2071.00",
                title    = "Paving Equipment Operators"
            ),
            OnetOccupation(
                job_zone = 3,
                sector   = "Building Construction",
                skills   = [OnetSkill(
                    name = "Concrete Finishing",
                    type = "technology"
                )],
                soc_code = "47-2111.00",
                title    = "Electricians"
            )
        ]
        registry = LexiconRegistry(
            occupations = occupations,
            osha_terms  = []
        )
        assert registry.normalize("concrete finishing") == "Concrete Finishing"

    def test_phrases_empty(self):
        """
        A decomposable entry with an empty `phrases` list indexes
        nothing rather than falling back to the full sentence.
        """
        registry = LexiconRegistry(
            occupations = [OnetOccupation(
                job_zone = 1,
                sector   = "Test",
                skills   = [OnetSkill(
                    name    = "Some task sentence",
                    phrases = [],
                    type    = "task"
                )],
                soc_code = "99-0000.00",
                title    = "Test"
            )],
            osha_terms  = []
        )
        assert registry.normalize("Some task sentence") is None

    def test_short_phrases_intact(self, registry: LexiconRegistry):
        """
        Tool entries are indexed directly without decomposition.
        """
        assert registry.normalize("Laptop computers") == "Laptop computers"

    # -------------------------------------------------------------------------
    # Certification integration
    # -------------------------------------------------------------------------

    def test_certification_acronym(self, registry: LexiconRegistry):
        """
        Acronyms index as lookup keys pointing to the full certification
        name.
        """
        assert registry.normalize("CWI") == "Certified Welding Inspector"

    def test_certification_name(self, registry: LexiconRegistry):
        """
        A certification name resolves via both lemmatized and lowercased
        lookup forms.
        """
        assert registry.normalize("Rigging Qualification") == "Rigging Qualification"

    def test_certification_overwrites_supplement(self):
        """
        A term in both certifications and supplement resolves to the
        certification canonical form, confirming cert > supplement
        priority.
        """
        registry = LexiconRegistry(
            certifications   = [Certification(
                name      = "Excavation",
                soc_codes = ["47-2071.00"]
            )],
            occupations      = [],
            osha_terms       = [],
            supplement_terms = ["excavation"]
        )
        assert registry.normalize("excavation") == "Excavation"

    def test_certification_phrase(self, registry: LexiconRegistry):
        """
        Description sub-phrases from certification records are individually
        matchable.
        """
        assert registry.normalize("rigging hardware") == "rigging hardware"

    def test_onet_overwrites_certification(self, occupations: list[OnetOccupation]):
        """
        A term in both O*NET and certifications resolves to the O*NET
        canonical form, confirming O*NET > cert priority.
        """
        assert LexiconRegistry(
            certifications = [Certification(
                name      = "AutoCAD Cert",
                phrases   = ["autodesk autocad"],
                soc_codes = ["47-2111.00"]
            )],
            occupations    = occupations,
            osha_terms     = []
        ).normalize("autodesk autocad") == "Autodesk AutoCAD"

    # -------------------------------------------------------------------------
    # Supplement integration
    # -------------------------------------------------------------------------

    def test_supplement_onet_overwrites_supplement(self, registry: LexiconRegistry):
        """
        A term in both supplement and O*NET resolves to the O*NET canonical
        form, confirming O*NET > supplement priority.
        """
        # "concrete finishing" is both an O*NET technology on the
        # paving operator and a supplement term. O*NET wins.
        assert registry.normalize("concrete finishing") == "Concrete finishing"

    @mark.parametrize("term", ["excavation", "rebar"])
    def test_supplement_term(self, registry: LexiconRegistry, term: str):
        """
        Terms present only in the supplement lexicon normalize to their
        canonical forms.
        """
        assert registry.normalize(term) == term
