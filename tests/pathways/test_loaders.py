"""
Tests for lexicon file loading and stakeholder reference data.

Validates graceful handling of missing files, successful loading of valid
O*NET lexicon data, and stakeholder data filtering.
"""

from json    import dumps
from pathlib import Path

from chalkline.pathways.loaders import LexiconLoader, StakeholderReference


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
        assert len(loader.occupations) == 0
        assert "O*NET lexicon not found" in caplog.text

    def test_nearest_occupation(self, lexicon_dir: Path):
        """
        `nearest_occupation` returns the occupation at the argmax
        position in the similarity row.
        """
        import numpy as np
        loader = LexiconLoader(lexicon_dir)
        row    = np.array([0.3, 0.9])
        assert loader.nearest_occupation(row).soc_code == loader.occupations[1].soc_code


class TestStakeholderFilterBoards:
    """
    Validate keyword-based job board filtering.
    """

    def test_matches_keyword(self, tmp_path: Path):
        """
        Boards whose focus contains a search keyword are returned.
        """
        boards = {"Maine": [
            {"focus": "general construction", "best_for": "all trades", "name": "A"},
            {"focus": "technology", "best_for": "software", "name": "B"}
        ]}
        (tmp_path / "job_boards.json").write_text(dumps(boards))
        ref    = StakeholderReference(tmp_path)
        result = ref.filter_boards({"construction"})
        assert len(result) == 1
        assert result[0]["name"] == "A"

    def test_limit_caps_results(self, tmp_path: Path):
        """
        The `limit` parameter caps the total number of returned boards.
        """
        boards = {"Maine": [
            {"focus": f"construction {i}", "best_for": "", "name": f"B{i}"}
            for i in range(10)
        ]}
        (tmp_path / "job_boards.json").write_text(dumps(boards))
        ref = StakeholderReference(tmp_path)
        assert len(ref.filter_boards({"construction"}, limit=3)) == 3

    def test_no_match_empty(self, tmp_path: Path):
        """
        No matching keywords returns an empty list.
        """
        boards = {"Maine": [{"focus": "tech", "best_for": "IT", "name": "X"}]}
        (tmp_path / "job_boards.json").write_text(dumps(boards))
        ref = StakeholderReference(tmp_path)
        assert ref.filter_boards({"plumbing"}) == []
