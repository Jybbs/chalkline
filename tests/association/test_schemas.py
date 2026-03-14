"""
Tests for association schemas.
"""

from pytest import mark, param, raises

from chalkline.association.schemas import AprioriResult, CommunityLabel
from chalkline.association.schemas import GraphDiagnostics, MeasureComparison


class TestAssociationSchemas:
    """
    Validate field constraints and extra-field rejection across
    association schema models.
    """

    # ---------------------------------------------------------
    # AprioriResult
    # ---------------------------------------------------------

    def test_apriori_extra_field(self):
        """
        Extra fields are rejected on AprioriResult.
        """
        with raises(ValueError, match = "Extra inputs"):
            AprioriResult(
                extra       = True,
                min_support = 0.05,
                n_itemsets  = 10,
                n_rules     = 5
            )

    def test_apriori_negative_itemsets(self):
        """
        Negative itemset counts are rejected.
        """
        with raises(ValueError, match = "greater than or equal"):
            AprioriResult(
                min_support = 0.05,
                n_itemsets  = -1,
                n_rules     = 0
            )

    # ---------------------------------------------------------
    # CommunityLabel
    # ---------------------------------------------------------

    @mark.parametrize("kwargs, match", [
        param(
            {
                "community_id"        : 0,
                "extra_field"         : True,
                "size"                : 5,
                "top_skills"          : ["welding"],
                "weighted_degree_sum" : 10.0
            },
            "Extra inputs",
            id = "extra_field"
        ),
        param(
            {
                "community_id"        : -1,
                "size"                : 5,
                "top_skills"          : ["welding"],
                "weighted_degree_sum" : 10.0
            },
            "greater than or equal",
            id = "negative_community_id"
        ),
        param(
            {
                "community_id"        : 0,
                "size"                : 0,
                "top_skills"          : ["welding"],
                "weighted_degree_sum" : 10.0
            },
            "greater than or equal",
            id = "zero_size"
        )
    ])
    def test_community_label_invalid(self, kwargs, match):
        """
        Invalid kwargs are rejected by Pydantic validation.
        """
        with raises(ValueError, match = match):
            CommunityLabel(**kwargs)

    # ---------------------------------------------------------
    # GraphDiagnostics
    # ---------------------------------------------------------

    def test_diagnostics_extra_field(self):
        """
        Extra fields are rejected on GraphDiagnostics.
        """
        with raises(ValueError, match = "Extra inputs"):
            GraphDiagnostics(
                connected_components = 1,
                coverage             = 0.8,
                edge_count           = 10,
                extra                = True,
                isolate_count        = 2,
                largest_component    = 8,
                node_count           = 10,
                performance          = 0.7
            )

    # ---------------------------------------------------------
    # MeasureComparison
    # ---------------------------------------------------------

    def test_measure_extra_field(self):
        """
        Extra fields are rejected on MeasureComparison.
        """
        with raises(ValueError, match = "Extra inputs"):
            MeasureComparison(
                density       = 0.05,
                edge_count    = 100,
                extra         = True,
                measure       = "npmi",
                n_communities = 5
            )
