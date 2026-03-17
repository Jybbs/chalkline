"""
Apriori frequent itemset mining for DS5230 comparison.

Converts the binary skill matrix to a boolean DataFrame, mines frequent
itemsets via mlxtend's Apriori implementation, and generates association
rules with support, confidence, and lift metrics. This module exists for
the DS5230 class deliverable only and is excluded from the production
pipeline.
"""

import pandas as pd

from itertools                 import takewhile
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.sparse              import spmatrix

from chalkline.association.schemas import AprioriResult


class AprioriComparison:
    """
    Apriori frequent itemset mining and rule generation.

    Receives the same binary matrix consumed by `CooccurrenceNetwork`
    and produces association rules with support, confidence, and lift.
    The overlap between Apriori lift > 1.0 pairs and PPMI positive pairs
    is the primary comparison metric.
    """

    def __init__(
        self,
        binary_matrix : spmatrix,
        feature_names : list[str]
    ):
        """
        Convert the binary presence/absence matrix from
        `SkillVectorizer.binary_matrix` to a boolean DataFrame,
        using the feature vocabulary as column headers to
        preserve the matrix-to-skill-name alignment.

        Args:
            binary_matrix : CSR binary presence/absence matrix.
            feature_names : Vocabulary in column order.
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
        iteratively halves the support down to a floor of 0.01. Rules
        are filtered to lift > 1.0 and confidence >= 0.3.

        Args:
            min_support: Minimum support threshold for frequent
                         itemsets.

        Returns:
            `AprioriResult` with the final support threshold, itemset
            and rule counts, and a summary of the top 20 rules.
        """
        itemsets = pd.DataFrame()
        rules    = pd.DataFrame()
        for support in takewhile(
            lambda s: s >= 0.01,
            (min_support / 2**i for i in range(20))
        ):
            itemsets = apriori(
                self.boolean_df,
                min_support  = support,
                use_colnames = True
            )
            if itemsets.empty:
                continue
            rules = association_rules(
                itemsets,
                metric        = "confidence",
                min_threshold = 0.3
            )
            if not (rules := rules[rules["lift"] > 1.0]).empty:
                break

        return AprioriResult(
            min_support   = support,
            n_itemsets    = len(itemsets),
            n_rules       = len(rules),
            rules_summary = (top := rules.head(20)).assign(
                antecedents = top["antecedents"].map(sorted),
                consequents = top["consequents"].map(sorted)
            )[
                ["antecedents", "confidence", "consequents", "lift", "support"]
            ].to_dict(orient="records")
        )
