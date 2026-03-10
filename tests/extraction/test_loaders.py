"""
Tests for lexicon file loading.

Validates graceful handling of missing files and successful loading of valid
OSHA and O*NET lexicon data.
"""

from collections.abc import Callable
from logging         import WARNING
from pathlib         import Path
from pytest          import LogCaptureFixture, mark

from chalkline.extraction.loaders import load_onet, load_osha, load_supplement


class TestLoaders:
    """
    Validate lexicon file loading and missing-file handling.
    """

    def test_load_onet_valid(self, lexicon_dir: Path):
        """
        Valid O*NET JSON deserializes into occupation records.
        """
        assert len(occupations := load_onet(lexicon_dir / "onet.json")) == 2
        assert occupations[0].soc_code == "47-2111.00"

    def test_load_osha_valid(self, lexicon_dir: Path):
        """
        Valid OSHA JSON deserializes into a list of strings.
        """
        assert len(terms := load_osha(lexicon_dir / "osha.json")) == 5
        assert "fall protection" in terms

    def test_load_supplement_valid(self, lexicon_dir: Path):
        """
        Valid supplement JSON deserializes into a list of strings.
        """
        assert len(terms := load_supplement(lexicon_dir / "supplement.json")) == 3
        assert "rebar" in terms

    @mark.parametrize(("loader", "label"), [
        (load_onet,       "O*NET"),
        (load_osha,       "OSHA"),
        (load_supplement, "Supplement")
    ])
    def test_missing_file_logs_warning(
        self,
        caplog   : LogCaptureFixture,
        label    : str,
        loader   : Callable,
        tmp_path : Path
    ):
        """
        A missing lexicon file logs a warning and returns an empty list.
        """
        with caplog.at_level(WARNING):
            assert loader(tmp_path / "missing.json") == []
        assert f"{label} lexicon not found" in caplog.text
