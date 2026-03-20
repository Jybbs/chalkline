"""
Tests for lexicon file loading.

Validates graceful handling of missing files and successful loading of
valid certifications and O*NET lexicon data.
"""

from pathlib import Path

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

    def test_missing_file_warns(self, caplog, tmp_path: Path):
        """
        Missing lexicon files log warnings and produce empty lists.
        """
        loader = LexiconLoader(tmp_path)
        assert loader.certifications == []
        assert loader.occupations    == []
        assert "Certifications lexicon not found" in caplog.text
        assert "O*NET lexicon not found"          in caplog.text
