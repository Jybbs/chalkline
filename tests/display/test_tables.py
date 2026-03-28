"""
Tests for display-layer data builders.

Validates fuzzy company matching and credential deduplication
logic used by the career report.
"""

from typing import Callable

from pytest import mark

from chalkline.display.schemas  import _match_member
from chalkline.pathways.schemas import Reach


@mark.parametrize("kind", ["apprenticeship", "program"])
class TestCredentialDedup:
    """
    Validate per-kind deduplication in `Reach.credentials_by_kind`.
    """

    def test_deduplicates(self, edge_factory: Callable, kind: str):
        """
        Duplicate credentials on two edges produce one unique entry.
        """
        edge = edge_factory(kind)
        assert len(Reach(advancement=[edge, edge]).credentials_by_kind(kind)) == 1

    def test_empty_edges(self, edge_factory: Callable, kind: str):
        """
        No credentials of the requested type returns empty dict.
        """
        assert Reach(advancement=[edge_factory()]).credentials_by_kind(kind) == {}


class TestMatchMember:
    """
    Validate SequenceMatcher-based company name matching.
    """

    def test_below_threshold(
        self,
        member_names: tuple[list[dict], list[str]]
    ):
        """
        Unrelated company names return None.
        """
        members, names = member_names
        assert _match_member("acme corp", names, members) is None

    def test_exact_match(
        self,
        member_names: tuple[list[dict], list[str]]
    ):
        """
        Identical names produce a match.
        """
        members, names = member_names
        m = _match_member("cianbro corporation", names, members)
        assert m is not None
        assert m["name"] == "Cianbro Corporation"

    def test_fuzzy_match(
        self,
        member_names: tuple[list[dict], list[str]]
    ):
        """
        Abbreviation and punctuation differences still match
        above the 0.7 threshold.
        """
        members, names = member_names
        m = _match_member("rj grondin & sons", names, members)
        assert m is not None
        assert m["name"] == "R.J. Grondin and Sons"
