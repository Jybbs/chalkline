"""
Tests for Apriori frequent itemset mining and PPMI comparison.

Validates rule generation, support threshold behavior, and Jaccard
overlap computation using the synthetic 20-posting fixture chain.
"""

from chalkline.association.apriori      import AprioriComparison
from chalkline.association.cooccurrence import CooccurrenceNetwork


class TestAprioriComparison:
    """
    Validate Apriori rule mining and PPMI overlap.
    """

    # ---------------------------------------------------------
    # Rules
    # ---------------------------------------------------------

    def test_mine_valid(self, apriori: AprioriComparison):
        """
        Mining produces an AprioriResult with non-negative counts.
        """
        result = apriori.mine()
        assert result.n_itemsets >= 0
        assert result.n_rules >= 0
        assert result.min_support > 0

    def test_rules_are_pairs(self, apriori: AprioriComparison):
        """
        Every rule involves at least two distinct skills, meaning
        single-item antecedent-consequent pairs are excluded by
        the confidence and lift filters.
        """
        result = apriori.mine()
        for rule in result.rules_summary:
            assert len(rule["antecedents"]) >= 1
            assert len(rule["consequents"]) >= 1
            assert len(set(rule["antecedents"]) | set(rule["consequents"])) >= 2

    def test_rules_have_metrics(self, apriori: AprioriComparison):
        """
        Each rule in the summary has support, confidence, and lift.
        """
        result = apriori.mine()
        for rule in result.rules_summary:
            assert "support" in rule
            assert "confidence" in rule
            assert "lift" in rule

    def test_support_monotonic(self, apriori: AprioriComparison):
        """
        Lowering the support threshold increases or maintains the
        number of itemsets.
        """
        high = apriori.mine(min_support = 0.30)
        low  = apriori.mine(min_support = 0.05)
        assert low.n_itemsets >= high.n_itemsets

    # ---------------------------------------------------------
    # Overlap
    # ---------------------------------------------------------

    def test_overlap_bounded(
        self,
        apriori : AprioriComparison,
        network : CooccurrenceNetwork
    ):
        """
        Jaccard overlap is bounded to [0, 1].
        """
        names      = network.feature_names
        rows, cols = network.ppmi_matrix.nonzero()
        pairs = {
            (names[r], names[c])
            for r, c in zip(rows, cols)
            if r < c
        }
        assert 0 <= apriori.overlap(pairs) <= 1
