"""
Tests for Aho-Corasick skill extraction and surface form matching.

Validates multi-word extraction, case insensitivity, filler masking, word
boundary enforcement, OSHA priority, zero-skill exclusion, and unmatched
term logging using synthetic fixture data.
"""

from chalkline.extraction.skills import SkillExtractor


class TestSkillExtractor:
    """
    Validate extraction behavior, priority, and edge cases.
    """

    # ---------------------------------------------------------
    # Extraction
    # ---------------------------------------------------------

    def test_case_insensitive(self, extractor: SkillExtractor):
        """
        Identical text in different casings produces identical output.
        """
        text = "Fall protection and welding required"
        assert (
            extractor.extract({"a": text.lower()})["a"]
            == extractor.extract({"a": text.upper()})["a"]
        )

    def test_decomposed_phrase(self, extractor: SkillExtractor):
        """
        Sub-phrases from decomposed O*NET Tasks and DWAs are matchable
        through the automaton.
        """
        assert any(
            "wiring" in s.lower()
            for s in extractor.extract({
                "a": "Install electrical wiring in the building"
            }).get("a", [])
        )

    def test_inverted_bigram(self, extractor: SkillExtractor):
        """
        Two-word skills match even when their words appear in reverse
        order, because the automaton indexes inverted bigram variants.
        """
        assert "fall protection" in extractor.extract({
            "a": "protection fall is important"
        }).get("a", [])

    def test_multi_word(self, extractor: SkillExtractor):
        """
        Multi-word lexicon terms like "fall protection" are extracted as
        complete phrases rather than individual words.
        """
        assert "fall protection" in extractor.extract({
            "a": "Fall protection and welding are required"
        })["a"]

    def test_osha_priority(self, extractor: SkillExtractor):
        """
        A term present in both OSHA and O*NET resolves to the OSHA
        canonical form.
        """
        result = extractor.extract({
            "a": "welding certification required"
        })
        assert "welding" in result["a"]
        # OSHA "welding" is lowercase; O*NET is "Welding"
        assert "Welding" not in result["a"]

    def test_stemmed_form(self, extractor: SkillExtractor):
        """
        Porter-stemmed surface forms match their canonical entry,
        allowing shortened word forms to resolve correctly.
        """
        assert "scaffolding" in extractor.extract({
            "a": "scaffold inspection required"
        }).get("a", [])

    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------

    def test_high_unmatched_rate_warns(self, caplog, extractor: SkillExtractor):
        """
        A corpus with mostly unrecognized terms triggers a warning when the
        unmatched rate exceeds the 25% threshold.
        """
        with caplog.at_level("WARNING", logger = "chalkline.extraction.skills"):
            extractor.extract({
                "a": "quantum computing blockchain artificial intelligence "
                     "machine learning deep neural networks cryptocurrency "
                     "decentralized finance web3 metaverse fall protection"
            })
        assert any("unmatched" in r.message.lower() for r in caplog.records)

    def test_unmatched_terms_logged(self, caplog, extractor: SkillExtractor):
        """
        Unmatched term diagnostics are logged at debug level.
        """
        with caplog.at_level("DEBUG", logger = "chalkline.extraction.skills"):
            extractor.extract({
                "a": "Fall protection and quantum computing required"
            })
        assert any("unmatched" in r.message.lower() for r in caplog.records)

    # ---------------------------------------------------------
    # Output format
    # ---------------------------------------------------------

    def test_output_sorted(self, extractor: SkillExtractor):
        """
        Each posting's skill list is sorted alphabetically with no
        duplicates, even when the same term appears multiple times.
        """
        skills = extractor.extract({
            "a": "welding and welding, plus fall protection"
        })["a"]
        assert skills == sorted(set(skills))

    def test_output_type(self, extractor: SkillExtractor):
        """
        The return value is a `dict[str, list[str]]`.
        """
        result = extractor.extract({
            "a": "Fall protection required"
        })
        assert isinstance(result, dict)
        assert isinstance(result["a"], list)
        assert all(isinstance(s, str) for s in result["a"])

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

    def test_bullet_normalization(self, extractor: SkillExtractor):
        """
        Bullet characters are normalized to sentence boundaries so that
        skills separated by bullets are extracted individually.
        """
        assert "fall protection" in extractor.extract({
            "a": "• fall protection • welding"
        })["a"]

    def test_camelcase_splitting(self, extractor: SkillExtractor):
        """
        CamelCase terms are split into separate words before matching so
        that concatenated skill names resolve to their canonical form.
        """
        assert "fall protection" in extractor.extract({
            "a": "fallProtection is required"
        })["a"]

    def test_filler_no_false_match(self, extractor: SkillExtractor):
        """
        Filler phrases are blanked before matching so that terms within
        them do not produce false positives.
        """
        result = extractor.extract({
            "a": "Must have years of experience with safety procedures"
        })
        assert "a" not in result or "experience" not in result.get("a", [])

    def test_semicolon_normalization(self, extractor: SkillExtractor):
        """
        Semicolons are normalized to sentence boundaries so that skills
        separated by semicolons are extracted individually.
        """
        assert "fall protection" in extractor.extract({
            "a": "fall protection; welding"
        })["a"]

    # ---------------------------------------------------------
    # Vocabulary
    # ---------------------------------------------------------

    def test_vocabulary_osha(self, extractor: SkillExtractor):
        """
        OSHA lexicon terms appear in the extractor's vocabulary.
        """
        assert "fall protection" in extractor.vocabulary
        assert "welding" in extractor.vocabulary

    def test_vocabulary_supplement(self, extractor: SkillExtractor):
        """
        Supplement lexicon terms appear in the extractor's vocabulary
        alongside OSHA and O*NET terms.
        """
        assert "rebar" in extractor.vocabulary

    # ---------------------------------------------------------
    # Word boundaries
    # ---------------------------------------------------------

    def test_boundary_digit(self, extractor: SkillExtractor):
        """
        Digits not appearing in any lexicon pattern are treated as
        separators, so a match abutting such a digit is valid.
        """
        assert "welding" in extractor.extract(
            {"a": "3welding required on site"}
        ).get("a", [])

    def test_boundary_hyphen(self, extractor: SkillExtractor):
        """
        Hyphens are non-alphanumeric, so terms separated by hyphens
        are treated as valid word boundaries.
        """
        assert "welding" in extractor.extract(
            {"a": "welding-certified technician needed"}
        ).get("a", [])

    def test_word_boundary(self, extractor: SkillExtractor):
        """
        Partial-word matches are rejected, meaning "weld" inside
        "welder" does not match the "weld" surface form.
        """
        assert "weld" not in extractor.extract(
            {"a": "The welder inspected the scaffoldings"}
        ).get("a", [])

    # ---------------------------------------------------------
    # Zero-skill exclusion
    # ---------------------------------------------------------

    def test_empty_corpus(self, extractor: SkillExtractor):
        """
        An empty corpus returns an empty mapping without error.
        """
        assert extractor.extract({}) == {}

    def test_filler_only_excluded(self, extractor: SkillExtractor):
        """
        A posting consisting entirely of filler phrases produces zero
        skills and is excluded from output.
        """
        assert extractor.extract({
            "a": "Must have experience with knowledge of"
        }) == {}

    def test_zero_skills_excluded(self, extractor: SkillExtractor):
        """
        Postings with no matched skills are omitted from the output.
        """
        result = extractor.extract({
            "a" : "This posting has no construction skills mentioned",
            "b" : "Fall protection required"
        })
        assert "b" in result
        assert "a" not in result
