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

    def test_apriori_valid(self):
        """
        A minimal AprioriResult with defaults validates.
        """
        result = AprioriResult(
            min_support = 0.05,
            n_itemsets  = 10,
            n_rules     = 3
        )
        assert result.overlap_jaccard is None
        assert result.rules_summary == []

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

    def test_community_label_valid(self):
        """
        A well-formed community label validates successfully.
        """
        label = CommunityLabel(
            community_id        = 0,
            size                = 10,
            top_skills          = ["welding", "scaffolding", "rigging"],
            weighted_degree_sum = 25.5
        )
        assert label.community_id == 0
        assert len(label.top_skills) == 3

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

    def test_diagnostics_no_modularity(self):
        """
        Modularity defaults to None for degenerate graphs.
        """
        diag = GraphDiagnostics(
            connected_components = 0,
            coverage             = 0.0,
            edge_count           = 0,
            isolate_count        = 0,
            largest_component    = 0,
            node_count           = 0,
            performance          = 0.0
        )
        assert diag.modularity is None

    def test_diagnostics_valid(self):
        """
        A complete diagnostics object with optional modularity
        validates.
        """
        diag = GraphDiagnostics(
            connected_components = 3,
            coverage             = 0.85,
            edge_count           = 50,
            isolate_count        = 5,
            largest_component    = 20,
            modularity           = 0.45,
            node_count           = 30,
            performance          = 0.72
        )
        assert diag.modularity == 0.45
        assert diag.edge_count == 50

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

    def test_measure_valid(self):
        """
        A well-formed measure comparison validates.
        """
        result = MeasureComparison(
            density       = 0.05,
            edge_count    = 100,
            measure       = "npmi",
            n_communities = 5
        )
        assert result.measure == "npmi"
