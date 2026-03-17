"""
Tests for Aho-Corasick skill extraction and surface form matching.

Validates multi-word extraction, case insensitivity, preamble
stripping, word boundary enforcement, OSHA priority, zero-skill
exclusion, and unmatched term logging using synthetic fixture data.
"""

from chalkline.extraction.skills import SkillExtractor


class TestSkillExtractor:
    """
    Validate extraction behavior, priority, and edge cases.
    """

    def test_case_insensitive(self, extractor: SkillExtractor):
        """
        Identical text in different casings produces identical
        output.
        """
        text = "Fall protection and welding required"
        assert (
            extractor.extract({"a": text.lower()})["a"]
            == extractor.extract({"a": text.upper()})["a"]
        )

    def test_decomposed_phrase(self, extractor: SkillExtractor):
        """
        Sub-phrases from decomposed O*NET Tasks and DWAs are
        matchable through the automaton.
        """
        assert any(
            "wiring" in s.lower()
            for s in extractor.extract({
                "a": "Install electrical wiring in the building"
            }).get("a", [])
        )

    def test_inverted_bigram(self, extractor: SkillExtractor):
        """
        Two-word skills match even when their words appear in
        reverse order, because the automaton indexes inverted
        bigram variants.
        """
        assert "fall protection" in extractor.extract({
            "a": "protection fall is important"
        }).get("a", [])

    def test_multi_word(self, extractor: SkillExtractor):
        """
        Multi-word lexicon terms like "fall protection" are extracted
        as complete phrases rather than individual words.
        """
        assert "fall protection" in extractor.extract({
            "a": "Fall protection and welding are required"
        })["a"]

    def test_osha_priority(self, extractor: SkillExtractor):
        """
        A term present in both OSHA and O*NET resolves to the OSHA canonical
        form.
        """
        result = extractor.extract({"a": "welding certification required"})
        assert "welding" in result["a"]
        # OSHA "welding" is lowercase; O*NET is "Welding"
        assert "Welding" not in result["a"]

    def test_stemmed_form(self, extractor: SkillExtractor):
        """
        Porter-stemmed surface forms match their canonical entry, allowing
        shortened word forms to resolve correctly.
        """
        assert "scaffolding" in extractor.extract({
            "a": "scaffold inspection required"
        }).get("a", [])

    def test_output_sorted(self, extractor: SkillExtractor):
        """
        Each posting's skill list is sorted alphabetically with no
        duplicates, even when the same term appears multiple times.
        """
        skills = extractor.extract({
            "a": "welding and welding, plus fall protection"
        })["a"]
        assert skills == sorted(set(skills))

    def test_bullet_normalization(self, extractor: SkillExtractor):
        """
        Bullet characters are normalized to sentence boundaries so
        that skills separated by bullets are extracted individually.
        """
        assert "fall protection" in extractor.extract({
            "a": "• fall protection • welding"
        })["a"]

    def test_camelcase_splitting(self, extractor: SkillExtractor):
        """
        CamelCase terms are split into separate words before matching
        so that concatenated skill names resolve to their canonical
        form.
        """
        assert "fall protection" in extractor.extract({
            "a": "fallProtection is required"
        })["a"]

    def test_preamble_strip(self, extractor: SkillExtractor):
        """
        Introductory prose before the first structural marker is dropped
        so that company narrative does not produce matches.
        """
        text = (
            "We are a leading construction firm.\n"
            "* Fall protection required\n"
            "* Welding certification"
        )
        result = extractor.extract({"a": text})
        assert "fall protection" in result.get("a", [])

    def test_boundary_digit(self, extractor: SkillExtractor):
        """
        Digits not appearing in any lexicon pattern are treated as
        separators, so a match abutting such a digit is valid.
        """
        assert "welding" in extractor.extract(
            {"a": "3welding required on site"}
        ).get("a", [])

    def test_word_boundary(self, extractor: SkillExtractor):
        """
        Partial-word matches are rejected, so "weld" inside "welder"
        does not match the "weld" surface form.
        """
        assert "weld" not in extractor.extract(
            {"a": "The welder inspected the scaffoldings"}
        ).get("a", [])

    def test_empty_corpus(self, extractor: SkillExtractor):
        """
        An empty corpus returns an empty mapping without error.
        """
        assert extractor.extract({}) == {}

    def test_no_skills_excluded(self, extractor: SkillExtractor):
        """
        A posting with no matching skills is excluded from the
        output.
        """
        assert extractor.extract({"a": "Must have experience with knowledge of"}) == {}
