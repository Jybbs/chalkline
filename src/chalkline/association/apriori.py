"""
Apriori frequent itemset mining for DS5230 comparison.

Converts the binary skill matrix to a boolean DataFrame, mines
frequent itemsets via mlxtend's Apriori implementation, and generates
association rules with support, confidence, and lift metrics. This
module exists for the DS5230 class deliverable only and is excluded
from the production pipeline.
"""

import pandas as pd

from mlxtend.frequent_patterns import apriori, association_rules
from scipy.sparse              import spmatrix

from chalkline.association.schemas import AprioriResult


class AprioriComparison:
    """
    Apriori frequent itemset mining and rule generation.

    Receives the same binary matrix consumed by `CooccurrenceNetwork`
    and produces association rules with support, confidence, and lift.
    The overlap between Apriori lift > 1.0 pairs and PPMI positive
    pairs is the primary comparison metric.
    """

    def __init__(
        self,
        binary_matrix : spmatrix,
        feature_names : list[str]
    ):
        """
        Convert the binary matrix to a boolean DataFrame.

        Args:
            binary_matrix : CSR binary presence/absence matrix
                            from `SkillVectorizer.binary_matrix`.
            feature_names : Vocabulary in column order, matching
                            matrix column indices.
        """
        self.boolean_df = pd.DataFrame(
            binary_matrix.toarray().astype(bool),
            columns = feature_names
        )

    # -----------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------

    def mine(self, min_support: float = 0.05) -> AprioriResult:
        """
        Mine frequent itemsets and generate association rules.

        If no rules are produced at the initial support threshold,
        iteratively halves the support down to a floor of 0.01.
        Rules are filtered to lift > 1.0 and confidence >= 0.3.

        Args:
            min_support: Minimum support threshold for frequent
                         itemsets.
        """
        support  = min_support
        itemsets = pd.DataFrame()
        rules    = pd.DataFrame()

        while support >= 0.01:
            itemsets = apriori(
                self.boolean_df,
                min_support    = support,
                use_colnames   = True
            )
            if not itemsets.empty:
                rules = association_rules(
                    itemsets,
                    metric        = "confidence",
                    min_threshold = 0.3
                )
                rules = rules[rules["lift"] > 1.0]
                if not rules.empty:
                    break
            support /= 2

        return AprioriResult(
            min_support   = support,
            n_itemsets    = len(itemsets),
            n_rules       = len(rules),
            rules_summary = [
                {
                    "antecedents" : sorted(row["antecedents"]),
                    "confidence"  : float(row["confidence"]),
                    "consequents" : sorted(row["consequents"]),
                    "lift"        : float(row["lift"]),
                    "support"     : float(row["support"])
                }
                for _, row in rules.head(20).iterrows()
            ]
        )

    def overlap(self, ppmi_pairs: set[tuple[str, str]]) -> float:
        """
        Jaccard overlap between Apriori and PPMI positive pairs.

            J = |P_apriori ∩ P_ppmi| / |P_apriori ∪ P_ppmi|

        where P_apriori is the set of skill pairs with lift > 1.0
        and P_ppmi is the set of pairs with positive PPMI from the
        co-occurrence network.

        Args:
            ppmi_pairs: Set of (skill_a, skill_b) tuples with
                        positive PPMI, canonically ordered.
        """
        apriori_pairs = {
            tuple(sorted([ant, con]))
            for rule in self.mine().rules_summary
            for ant in rule["antecedents"]
            for con in rule["consequents"]
        }

        if not apriori_pairs and not ppmi_pairs:
            return 0.0

        return len(apriori_pairs & ppmi_pairs) / len(apriori_pairs | ppmi_pairs)
