"""
Tests for display table builders.
"""

from typing import Callable

from pytest import mark

from chalkline.display.tables   import TableBuilder
from chalkline.matching.schemas import MatchResult
from chalkline.pathways.schemas import Reach


class TestBoardRows:
    """
    Validate sector keyword matching derived from cluster profile
    titles.
    """

    def test_empty_boards(
        self,
        match_result       : MatchResult,
        pipeline_namespace
    ):
        """
        Empty board dict returns empty tuples without error.
        """
        builder = TableBuilder(
            pipeline  = pipeline_namespace,
            reference = {
                "agc_members" : [],
                "career_urls" : [],
                "job_boards"  : {}
            },
            result    = match_result
        )
        maine, national = builder.board_rows()
        assert maine == []
        assert national == []

    def test_returns_rows(self, table_builder: TableBuilder):
        """
        Board rows returns formatted tuples for the matched sector.
        """
        maine, national = table_builder.board_rows()
        assert isinstance(maine, list)
        assert isinstance(national, list)


@mark.parametrize("kind, method, expected_key", [
    ("apprenticeship", "apprenticeship_rows", "RAPIDS Code"),
    ("program",        "program_rows",        "Institution")
])
class TestCredentialRows:
    """
    Validate deduplication and row formatting for credential types.

    Parameterized across apprenticeship and program rows, which share
    identical deduplication and empty-edge logic.
    """

    def test_deduplicates(
        self,
        edge_factory  : Callable,
        expected_key  : str,
        kind          : str,
        method        : str,
        table_builder : TableBuilder
    ):
        """
        Duplicate credentials on two edges produce one row.
        """
        edge  = edge_factory(kind)
        reach = Reach(advancement=[edge, edge])
        rows  = getattr(table_builder, method)(reach)
        assert len(rows) == 1
        assert expected_key in rows[0]

    def test_empty_edges(
        self,
        edge_factory  : Callable,
        expected_key  : str,
        kind          : str,
        method        : str,
        table_builder : TableBuilder
    ):
        """
        No credentials of this type returns empty list.
        """
        edge = edge_factory()
        assert getattr(table_builder, method)(Reach(advancement=[edge])) == []


class TestMatchMember:
    """
    Validate SequenceMatcher-based company name matching.
    """

    def test_below_threshold(
        self,
        member_names : tuple[list[dict], list[str]]
    ):
        """
        Unrelated company names return None.
        """
        members, names = member_names
        assert TableBuilder._match_member(
            "acme corp", members, names
        ) is None

    def test_exact_match(
        self,
        member_names : tuple[list[dict], list[str]]
    ):
        """
        Identical names produce a match.
        """
        members, names = member_names
        m = TableBuilder._match_member(
            "cianbro corporation", members, names
        )
        assert m is not None
        assert m["name"] == "Cianbro Corporation"

    def test_fuzzy_match(
        self,
        member_names : tuple[list[dict], list[str]]
    ):
        """
        Abbreviation and punctuation differences still match
        above the 0.7 threshold.
        """
        members, names = member_names
        m = TableBuilder._match_member(
            "rj grondin & sons", members, names
        )
        assert m is not None
        assert m["name"] == "R.J. Grondin and Sons"



