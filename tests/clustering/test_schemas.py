"""
Tests for clustering schemas.
"""

from pytest import mark, param, raises

from chalkline.clustering.schemas import ClusterLabel, ComparisonResult
from chalkline.clustering.schemas import CopheneticResult


class TestClusteringSchemas:
    """
    Validate field constraints and extra-field rejection across
    clustering schema models.
    """

    # ---------------------------------------------------------
    # ClusterLabel
    # ---------------------------------------------------------

    @mark.parametrize("kwargs, match", [
        param(
            {
                "cluster_id"     : 0,
                "extra_field"    : True,
                "leader_node_id" : 0,
                "size"           : 1,
                "terms"          : ["welding"],
                "weights"        : [0.5]
            },
            "Extra inputs",
            id = "extra_field"
        ),
        param(
            {
                "cluster_id"     : -1,
                "leader_node_id" : 0,
                "size"           : 1,
                "terms"          : ["welding"],
                "weights"        : [0.5]
            },
            "greater than or equal",
            id = "negative_cluster_id"
        ),
        param(
            {
                "cluster_id"     : 0,
                "leader_node_id" : 0,
                "size"           : 0,
                "terms"          : ["welding"],
                "weights"        : [0.5]
            },
            "greater than or equal",
            id = "zero_size"
        )
    ])
    def test_cluster_label_invalid(self, kwargs, match):
        """
        Invalid kwargs are rejected by Pydantic validation.
        """
        with raises(ValueError, match = match):
            ClusterLabel(**kwargs)

    # ---------------------------------------------------------
    # ComparisonResult
    # ---------------------------------------------------------

    def test_comparison_result_extra_field(self):
        """
        Extra fields are rejected on ComparisonResult.
        """
        with raises(ValueError, match = "Extra inputs"):
            ComparisonResult(
                assignments = [0, 1],
                extra       = True,
                method      = "kmeans",
                n_clusters  = 2
            )

    def test_comparison_result_negative_n_clusters(self):
        """
        Negative cluster counts are rejected by field validation.
        """
        with raises(ValueError, match = "greater than or equal"):
            ComparisonResult(
                assignments = [0, 1],
                method      = "kmeans",
                n_clusters  = -1
            )

    # ---------------------------------------------------------
    # CopheneticResult
    # ---------------------------------------------------------

    def test_cophenetic_result_extra_field(self):
        """
        Extra fields are rejected on CopheneticResult.
        """
        with raises(ValueError, match = "Extra inputs"):
            CopheneticResult(
                correlation = 0.85,
                extra       = True,
                method      = "ward"
            )
