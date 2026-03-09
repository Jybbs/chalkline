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
    # Case insensitivity
    # ---------------------------------------------------------

    def test_case_insensitive_extraction(self, skill_extractor: SkillExtractor):
        """
        Identical text in different casings produces identical output.
        """
        text = "Fall protection and welding required"
        assert (
            skill_extractor.extract({"a": text.lower()})["a"]
            == skill_extractor.extract({"a": text.upper()})["a"]
        )

    # ---------------------------------------------------------
    # Filler masking
    # ---------------------------------------------------------

    def test_filler_masking_prevents_false_match(
        self, skill_extractor: SkillExtractor
    ):
        """
        Filler phrases are blanked before matching so that terms within
        them do not produce false positives.
        """
        assert "a" not in (result := skill_extractor.extract({
            "a": "Must have years of experience with safety procedures"
        })) or "experience" not in result.get("a", [])

    # ---------------------------------------------------------
    # Multi-word extraction
    # ---------------------------------------------------------

    def test_decomposed_sub_phrase_extraction(
        self, skill_extractor: SkillExtractor
    ):
        """
        Sub-phrases from decomposed O*NET Tasks and DWAs are matchable
        through the automaton.
        """
        assert any(
            "wiring" in s.lower()
            for s in skill_extractor.extract({
                "a": "Install electrical wiring in the building"
            }).get("a", [])
        )

    def test_inverted_bigram_extraction(
        self, skill_extractor: SkillExtractor
    ):
        """
        Two-word skills match even when their words appear in reverse
        order, because the automaton indexes inverted bigram variants.
        """
        assert "fall protection" in skill_extractor.extract({
            "a": "protection fall is important"
        }).get("a", [])

    def test_multi_word_extraction(self, skill_extractor: SkillExtractor):
        """
        Multi-word lexicon terms like "fall protection" are extracted as
        complete phrases rather than individual words.
        """
        assert "fall protection" in skill_extractor.extract({
            "a": "Fall protection and welding are required"
        })["a"]

    # ---------------------------------------------------------
    # Output format
    # ---------------------------------------------------------

    def test_output_is_sorted_and_deduplicated(
        self, skill_extractor: SkillExtractor
    ):
        """
        Each posting's skill list is sorted alphabetically with no
        duplicates, even when the same term appears multiple times.
        """
        skills = skill_extractor.extract({
            "a": "welding and welding, plus fall protection"
        })["a"]
        assert skills == sorted(set(skills))

    def test_output_type(self, skill_extractor: SkillExtractor):
        """
        The return value is a `dict[str, list[str]]`.
        """
        result = skill_extractor.extract({
            "a": "Fall protection required"
        })
        assert isinstance(result, dict)
        assert isinstance(result["a"], list)
        assert all(isinstance(s, str) for s in result["a"])

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

    def test_bullet_normalization(self, skill_extractor: SkillExtractor):
        """
        Bullet characters are normalized to sentence boundaries so that
        skills separated by bullets are extracted individually.
        """
        assert "fall protection" in skill_extractor.extract({
            "a": "• fall protection • welding"
        })["a"]

    def test_camelcase_splitting(self, skill_extractor: SkillExtractor):
        """
        CamelCase terms are split into separate words before matching so
        that concatenated skill names resolve to their canonical form.
        """
        assert "fall protection" in skill_extractor.extract({
            "a": "fallProtection is required"
        })["a"]

    def test_semicolon_normalization(self, skill_extractor: SkillExtractor):
        """
        Semicolons are normalized to sentence boundaries so that skills
        separated by semicolons are extracted individually.
        """
        assert "fall protection" in skill_extractor.extract({
            "a": "fall protection; welding"
        })["a"]

    # ---------------------------------------------------------
    # Priority
    # ---------------------------------------------------------

    def test_osha_priority_over_onet(self, skill_extractor: SkillExtractor):
        """
        A term present in both OSHA and O*NET resolves to the OSHA
        canonical form.
        """
        result = skill_extractor.extract({
            "a": "welding certification required"
        })
        assert "welding" in result["a"]
        # OSHA "welding" is lowercase; O*NET is "Welding"
        assert "Welding" not in result["a"]

    # ---------------------------------------------------------
    # Stemming
    # ---------------------------------------------------------

    def test_stemmed_form_matching(self, skill_extractor: SkillExtractor):
        """
        Porter-stemmed surface forms match their canonical entry,
        allowing shortened word forms to resolve correctly.
        """
        assert "scaffolding" in skill_extractor.extract({
            "a": "scaffold inspection required"
        }).get("a", [])

    # ---------------------------------------------------------
    # Unmatched logging
    # ---------------------------------------------------------

    def test_high_unmatched_rate_logs_warning(
        self, skill_extractor: SkillExtractor, caplog
    ):
        """
        A corpus with mostly unrecognized terms triggers a warning when the
        unmatched rate exceeds the 25% threshold.
        """
        with caplog.at_level("WARNING", logger="chalkline.extraction.skills"):
            skill_extractor.extract({
                "a": "quantum computing blockchain artificial intelligence "
                     "machine learning deep neural networks cryptocurrency "
                     "decentralized finance web3 metaverse fall protection"
            })
        assert any("unmatched" in r.message.lower() for r in caplog.records)

    def test_unmatched_terms_logged(
        self, skill_extractor: SkillExtractor, caplog
    ):
        """
        Unmatched term diagnostics are logged at debug level.
        """
        with caplog.at_level("DEBUG", logger="chalkline.extraction.skills"):
            skill_extractor.extract({
                "a": "Fall protection and quantum computing required"
            })
        assert any("unmatched" in r.message.lower() for r in caplog.records)

    # ---------------------------------------------------------
    # Vocabulary
    # ---------------------------------------------------------

    def test_vocabulary_nonempty(self, skill_extractor: SkillExtractor):
        """
        The extractor's vocabulary is non-empty when built from fixture data.
        """
        assert len(skill_extractor.vocabulary) > 0

    # ---------------------------------------------------------
    # Word boundaries
    # ---------------------------------------------------------

    def test_word_boundary_enforcement(
        self, skill_extractor: SkillExtractor
    ):
        """
        Partial-word matches are rejected, meaning "weld" inside
        "welder" does not match the "weld" surface form.
        """
        assert "weld" not in skill_extractor.extract(
            {"a": "The welder inspected the scaffoldings"}
        ).get("a", [])

    # ---------------------------------------------------------
    # Zero-skill exclusion
    # ---------------------------------------------------------

    def test_zero_skill_postings_excluded(
        self, skill_extractor: SkillExtractor
    ):
        """
        Postings with no matched skills are omitted from the output.
        """
        result = skill_extractor.extract({
            "a" : "This posting has no construction skills mentioned",
            "b" : "Fall protection required"
        })
        assert "b" in result
        assert "a" not in result
