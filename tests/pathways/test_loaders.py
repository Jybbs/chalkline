"""
Tests for lexicon file loading.

Validates graceful handling of missing files and successful loading of valid
O*NET lexicon data.
"""

from pathlib import Path

from chalkline.pathways.loaders import LexiconLoader


class TestLexiconLoader:
    """
    Validate lexicon loading and missing-file handling.
    """

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
        assert loader.occupations == []
        assert "O*NET lexicon not found" in caplog.text
