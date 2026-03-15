"""
Tests for comparison clustering methods.

Validates K-Means, DBSCAN, Mean Shift, and HDBSCAN result structure,
validity metrics, silhouette details, and sector ARI computation using
the synthetic 20-posting fixture chain.
"""

import numpy as np

from pytest import mark

from chalkline.clustering.comparison import ClusterComparison


class TestClusterComparison:
    """
    Validate result structure and metrics from each comparison clustering
    method.
    """

    # ---------------------------------------------------------
    # Structural properties across all methods
    # ---------------------------------------------------------

    @mark.parametrize("method", ["dbscan", "hdbscan", "kmeans", "mean_shift"])
    def test_assignment_count(self, comparison: ClusterComparison, method: str):
        """
        Every method assigns one label per posting.
        """
        result = getattr(comparison, method)()
        assert len(result.assignments) == len(comparison.coordinates)

    # ---------------------------------------------------------
    # K-Means specifics
    # ---------------------------------------------------------

    def test_kmeans_at_least_two_clusters(self, comparison: ClusterComparison):
        """
        K-Means produces at least 2 clusters from the elbow method.
        """
        assert comparison.kmeans().n_clusters >= 2

    # ---------------------------------------------------------
    # Sector ARI
    # ---------------------------------------------------------

    def test_ari_none_without_sectors(self, comparison: ClusterComparison):
        """
        Results carry no ARI when the runner has no sector labels.
        """
        result = comparison.kmeans()
        assert result.ari_vs_sectors is None

    def test_ari_with_sectors(self, comparison_with_sectors: ClusterComparison):
        """
        ARI is computed when sector labels are provided.

        The sector-label masking in `_build_result` must align the mask
        across both `sector_labels` and `assignments`. A misalignment
        would produce a meaningless ARI that the comparison report presents
        as valid.
        """
        result = comparison_with_sectors.kmeans()
        assert result.ari_vs_sectors is not None
        assert -1.0 <= result.ari_vs_sectors <= 1.0

    # ---------------------------------------------------------
    # Silhouette details
    # ---------------------------------------------------------

    def test_silhouette_details_degenerate(self, comparison: ClusterComparison):
        """
        Degenerate clustering (single label) returns all zeros.
        """
        details = comparison.silhouette_details(
            np.zeros(len(comparison.coordinates), dtype = int)
        )
        assert (details == 0).all()
