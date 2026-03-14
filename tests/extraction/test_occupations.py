"""
Tests for occupation-level queries and SOC matching.

Validates Job Zone lookups, sector queries, skill retrieval, bare-prefix SOC
resolution, and vectorized Jaccard matching against the synthetic
two-occupation fixture.
"""

from pytest import mark, raises

from chalkline.extraction.occupations import OccupationIndex
from chalkline.extraction.schemas     import OnetOccupation


class TestOccupationIndex:
    """
    Validate occupation lookups, SOC resolution, and Jaccard matching.
    """

    def test_ambiguous_prefix(self):
        """
        A bare prefix with multiple suffixes raises `KeyError` because the
        resolver cannot disambiguate.
        """
        with raises(KeyError):
            OccupationIndex([
                OnetOccupation(
                    job_zone = 3,
                    sector   = "Building Construction",
                    skills   = [],
                    soc_code = code,
                    title    = title
                )
                for code, title in [
                    ("17-2051.00", "Civil Engineers"),
                    ("17-2051.01", "Transportation Engineers")
                ]
            ]).get("17-2051")

    def test_empty_occupations(self):
        """
        An empty occupation list builds without raising.
        """
        assert OccupationIndex([]).occupation_map == {}

    def test_get(self, occupation_index: OccupationIndex, soc: str):
        """
        `get` accepts both suffixed and bare prefix formats.
        """
        assert occupation_index.get(soc).job_zone == 3

    def test_invalid_soc(self, occupation_index: OccupationIndex):
        """
        An unrecognized SOC code raises `KeyError`.
        """
        with raises(KeyError):
            occupation_index.get("99-9999.00")

    @mark.parametrize("skills", [set(), {"blockchain", "quantum computing"}])
    def test_nearest_degenerate_input(
        self,
        occupation_index : OccupationIndex,
        skills           : set[str]
    ):
        """
        Empty or fully unknown skill sets return a valid SOC code without
        raising.
        """
        assert occupation_index.nearest(skills) in {
            "47-2071.00",
            "47-2111.00"
        }

    @mark.parametrize("skills", [
        {"Autodesk AutoCAD"},
        {"Autodesk AutoCAD", "Laptop computers", "Mathematics"},
        {"Autodesk AutoCAD", "Backhoes", "Laptop computers", "Welding"}
    ])
    def test_nearest_electrician(
        self,
        occupation_index : OccupationIndex,
        skills           : set[str]
    ):
        """
        Skill sets with majority electrician overlap resolve to the
        electrician SOC code.
        """
        assert occupation_index.nearest(skills) == "47-2111.00"

    def test_nearest_mixed_overlap(self, occupation_index: OccupationIndex):
        """
        A skill set with majority paving operator overlap resolves to the
        paving operator SOC code even with shared terms.
        """
        assert occupation_index.nearest(
            {"Backhoes", "Concrete finishing", "Welding"}
        ) == "47-2071.00"

    def test_soc_skill_matrix_shape(self, occupation_index: OccupationIndex):
        """
        The precomputed matrix has one row per SOC code and one column per
        unique skill across all occupations.
        """
        assert occupation_index.soc_skill_matrix.shape == (2, 12)
