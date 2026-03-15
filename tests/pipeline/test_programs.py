"""
Tests for the shared educational program loader.
"""

from json    import dumps
from pathlib import Path
from pytest  import raises

from chalkline.pipeline.programs import load_programs


class TestLoadPrograms:
    """
    Verify field normalization and loading from both community college
    and university reference files.
    """

    def test_cc_normalization(self, tmp_path: Path):
        """
        Community college `college` field maps to `institution`.
        """
        (tmp_path / "cc_programs.json").write_text(dumps({
            "degrees": [{
                "college"    : "CMCC",
                "credential" : "AAS Degree",
                "program"    : "Building Construction Technology",
                "url"        : "https://example.com"
            }]
        }))
        programs = load_programs(tmp_path)
        assert len(programs) == 1
        assert programs[0].institution == "CMCC"
        assert programs[0].credential == "AAS Degree"

    def test_combined_count(self, tmp_path: Path):
        """
        Degrees, initiatives, and university programs merge into a
        single list.
        """
        (tmp_path / "cc_programs.json").write_text(dumps({
            "degrees": [
                {"college": "A", "credential": "C", "program": "P1", "url": "u"},
                {"college": "B", "credential": "D", "program": "P2", "url": "u"}
            ],
            "initiatives": [
                {"best_for": "BF", "description": "D", "initiative": "I1", "url": "u"}
            ]
        }))
        (tmp_path / "umaine_programs.json").write_text(dumps([
            {"campus": "X", "category": "C", "degree": "D", "program": "P3", "url": "u"}
        ]))
        assert len(load_programs(tmp_path)) == 4

    def test_initiative_missing_url(self, tmp_path: Path):
        """
        Initiatives without a URL default to an empty string.
        """
        (tmp_path / "cc_programs.json").write_text(dumps({
            "degrees"     : [],
            "initiatives" : [{
                "best_for"    : "Entry-level workers",
                "description" : "Training program.",
                "initiative"  : "Pre-Apprenticeship"
            }]
        }))
        programs = load_programs(tmp_path)
        assert len(programs) == 1
        assert programs[0].url == ""

    def test_initiative_normalization(self, tmp_path: Path):
        """
        Workforce initiatives map `initiative` to `program`, `best_for`
        to `credential`, and set institution to `"Statewide"`.
        """
        (tmp_path / "cc_programs.json").write_text(dumps({
            "degrees": [],
            "initiatives": [{
                "best_for"    : "Adults seeking entry into the trades",
                "description" : "4-week training.",
                "initiative"  : "Maine Construction Academy",
                "url"         : "https://example.com/mca"
            }]
        }))
        programs = load_programs(tmp_path)
        assert len(programs) == 1
        assert programs[0].program == "Maine Construction Academy"
        assert programs[0].institution == "Statewide"
        assert programs[0].credential == "Adults seeking entry into the trades"

    def test_malformed_entry(self, tmp_path: Path):
        """
        A record missing a required key raises `KeyError` rather than
        silently producing a corrupt program.
        """
        (tmp_path / "cc_programs.json").write_text(dumps({
            "degrees" : [{"college" : "CMCC"}]
        }))
        with raises(KeyError):
            load_programs(tmp_path)

    def test_missing_files(self, tmp_path: Path):
        """
        Missing reference files produce an empty list.
        """
        assert load_programs(tmp_path) == []

    def test_umaine_normalization(self, tmp_path: Path):
        """
        University `campus` maps to `institution` and `degree` maps to
        `credential`.
        """
        (tmp_path / "umaine_programs.json").write_text(dumps([{
            "campus"   : "UMaine",
            "category" : "Construction",
            "degree"   : "B.S.",
            "program"  : "Construction Engineering Technology",
            "url"      : "https://example.com"
        }]))
        programs = load_programs(tmp_path)
        assert len(programs) == 1
        assert programs[0].institution == "UMaine"
        assert programs[0].credential == "B.S."
