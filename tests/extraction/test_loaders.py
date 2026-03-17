"""
Tests for lexicon file loading.

Validates graceful handling of missing files and successful loading of
valid certifications, OSHA, O*NET, and supplement lexicon data.
"""

from logging import WARNING
from pathlib import Path
from pytest  import LogCaptureFixture

from chalkline.extraction.loaders import LexiconLoader


class TestLexiconLoader:
    """
    Validate lexicon loading and missing-file handling.
    """

    def test_load_certifications(self, lexicon_dir: Path):
        """
        Valid certifications JSON deserializes into certification
        records.
        """
        loader = LexiconLoader(lexicon_dir)
        assert len(loader.certifications) == 2
        assert loader.certifications[0].name == "Certified Welding Inspector"

    def test_load_onet(self, lexicon_dir: Path):
        """
        Valid O*NET JSON deserializes into occupation records.
        """
        loader = LexiconLoader(lexicon_dir)
        assert len(loader.occupations) == 2
        assert loader.occupations[0].soc_code == "47-2111.00"

    def test_load_osha(self, lexicon_dir: Path):
        """
        Valid OSHA JSON deserializes into a list of strings.
        """
        loader = LexiconLoader(lexicon_dir)
        assert len(loader.osha_terms) == 5
        assert "fall protection" in loader.osha_terms

    def test_load_supplement(self, lexicon_dir: Path):
        """
        Valid supplement JSON deserializes into a list of strings.
        """
        loader = LexiconLoader(lexicon_dir)
        assert len(loader.supplement_terms) == 3
        assert "rebar" in loader.supplement_terms

    def test_missing_file_warns(
        self,
        caplog   : LogCaptureFixture,
        tmp_path : Path
    ):
        """
        Missing lexicon files log warnings and produce empty lists.
        """
        with caplog.at_level(WARNING):
            loader = LexiconLoader(tmp_path)
        assert loader.certifications == []
        assert loader.occupations    == []
        assert loader.osha_terms     == []
        assert loader.supplement_terms == []
        assert "Certifications lexicon not found" in caplog.text
        assert "O*NET lexicon not found"          in caplog.text
        assert "OSHA lexicon not found"           in caplog.text
        assert "Supplement lexicon not found"     in caplog.text
