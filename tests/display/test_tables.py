"""
Tests for display-layer data builders.

Validates employer fuzzy matching via `StakeholderReference.match_employers`.
"""

from typing import Callable

from pytest import mark

from chalkline.pathways.loaders import StakeholderReference


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

    @mark.parametrize(("company", "expected_name"), [
        ("Cianbro Corporation", "Cianbro Corporation"),
        ("rj grondin & sons",   "R.J. Grondin and Sons")
    ], ids=["exact", "fuzzy"])
    def test_match(
        self,
        company         : str,
        expected_name   : str,
        posting_factory : Callable,
        reference       : StakeholderReference
    ):
        """
        Company names matching a member exactly or above the 0.7
        fuzzy threshold produce a matched row.
        """
        rows = reference.match_employers([posting_factory(company)])
        assert len(rows) == 1
        assert rows[0]["name"] == expected_name

    def test_no_match(
        self,
        posting_factory : Callable,
        reference       : StakeholderReference
    ):
        """
        Unrelated company names produce no rows.
        """
        assert reference.match_employers([posting_factory("Acme Corp")]) == []
