"""
Tests for Apriori frequent itemset mining and PPMI comparison.

Validates rule generation, support threshold behavior, and Jaccard
overlap computation using the synthetic 20-posting fixture chain.
"""

from chalkline.association.apriori import AprioriComparison


class TestAprioriComparison:
    """
    Validate Apriori rule mining and PPMI overlap.
    """

    # ---------------------------------------------------------
    # Rules
    # ---------------------------------------------------------

    def test_rules_are_pairs(self, apriori: AprioriComparison):
        """
        Every rule involves at least two distinct skills, meaning
        single-item antecedent-consequent pairs are excluded by the
        confidence and lift filters.
        """
        result = apriori.mine()
        for rule in result.rules_summary:
            assert len(rule["antecedents"]) >= 1
            assert len(rule["consequents"]) >= 1
            assert len(set(rule["antecedents"]) | set(rule["consequents"])) >= 2

    def test_support_monotonic(self, apriori: AprioriComparison):
        """
        Lowering the support threshold increases or maintains the number
        of itemsets.
        """
        high = apriori.mine(min_support = 0.30)
        low  = apriori.mine(min_support = 0.05)
        assert low.n_itemsets >= high.n_itemsets
