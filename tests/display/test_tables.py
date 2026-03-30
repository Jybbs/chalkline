"""
Tests for display-layer data builders.

Validates credential deduplication via `Credential.key` and employer
fuzzy matching via `StakeholderReference.match_employers`.
"""

from typing import Callable

from pytest import mark

from chalkline.pathways.loaders import StakeholderReference
from chalkline.pathways.schemas import Reach


@mark.parametrize("kind", ["apprenticeship", "program"])
class TestCredentialDedup:
    """
    Validate that `Credential.key` deduplicates correctly per kind.
    """

    def test_deduplicates(self, edge_factory: Callable, kind: str):
        """
        Duplicate credentials on two edges produce one unique entry.
        """
        edge  = edge_factory(kind)
        reach = Reach(advancement=[edge, edge])
        assert len(reach.credentials_by_kind.get(kind, [])) == 1

    def test_empty_edges(self, edge_factory: Callable, kind: str):
        """
        No credentials of the requested type returns empty list.
        """
        reach = Reach(advancement=[edge_factory()])
        assert reach.credentials_by_kind.get(kind, []) == []


class TestMatchEmployers:
    """
    Validate fuzzy company-to-member matching in
    `StakeholderReference.match_employers`.
    """

    def test_deduplicates_company(
        self,
        posting_factory : Callable,
        reference       : StakeholderReference
    ):
        """
        Multiple postings from the same company produce one row.
        """
        postings = [posting_factory("Cianbro Corporation")] * 2
        assert len(reference.match_employers(postings)) == 1

    def test_exact_match(
        self,
        posting_factory : Callable,
        reference       : StakeholderReference
    ):
        """
        Company names identical to a member produce a matched row.
        """
        rows = reference.match_employers([posting_factory("Cianbro Corporation")])
        assert len(rows) == 1
        assert rows[0]["name"] == "Cianbro Corporation"

    def test_fuzzy_match(
        self,
        posting_factory : Callable,
        reference       : StakeholderReference
    ):
        """
        Abbreviated or punctuated names still match above the 0.7
        threshold.
        """
        rows = reference.match_employers([posting_factory("rj grondin & sons")])
        assert len(rows) == 1
        assert rows[0]["name"] == "R.J. Grondin and Sons"

    def test_no_match(
        self,
        posting_factory : Callable,
        reference       : StakeholderReference
    ):
        """
        Unrelated company names produce no rows.
        """
        assert reference.match_employers([posting_factory("Acme Corp")]) == []
