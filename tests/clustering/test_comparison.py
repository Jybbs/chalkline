"""
Tests for comparison clustering methods.

Validates K-Means, DBSCAN, Mean Shift, and HDBSCAN result structure,
validity metrics, silhouette details, and sector ARI computation
using the synthetic 20-posting fixture chain.
"""

import numpy as np

from pytest import mark

from chalkline.clustering.comparison import ClusterComparison


class TestClusterComparison:
    """
    Validate result structure and metrics from each comparison
    clustering method.
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

    @mark.parametrize("method", ["kmeans", "mean_shift"])
    def test_no_noise(self, comparison: ClusterComparison, method: str):
        """
        Partitioning methods produce no noise points.
        """
        result = getattr(comparison, method)()
        assert result.noise_count == 0

    @mark.parametrize("method", ["dbscan", "hdbscan"])
    def test_noise_nonnegative(self, comparison: ClusterComparison, method: str):
        """
        Density methods report non-negative noise counts.
        """
        result = getattr(comparison, method)()
        assert result.noise_count >= 0

    # ---------------------------------------------------------
    # K-Means specifics
    # ---------------------------------------------------------

    def test_kmeans_at_least_two_clusters(self, comparison: ClusterComparison):
        """
        K-Means produces at least 2 clusters from the elbow method.
        """
        assert comparison.kmeans().n_clusters >= 2

    def test_kmeans_with_sectors(self, comparison_with_sectors: ClusterComparison):
        """
        K-Means ARI against sector labels falls within [-1, 1].
        """
        result = comparison_with_sectors.kmeans()
        if result.ari_vs_sectors is not None:
            assert -1 <= result.ari_vs_sectors <= 1

    # ---------------------------------------------------------
    # Sector ARI
    # ---------------------------------------------------------

    def test_ari_none_without_sectors(self, comparison: ClusterComparison):
        """
        Results carry no ARI when the runner has no sector labels.
        """
        result = comparison.kmeans()
        assert result.ari_vs_sectors is None

    @mark.parametrize("method", ["dbscan", "hdbscan"])
    def test_density_ari_bounded_with_sectors(
        self,
        comparison_with_sectors : ClusterComparison,
        method                  : str
    ):
        """
        Density methods produce bounded ARI when sector labels are
        present and at least 2 non-noise clusters exist.
        """
        result = getattr(comparison_with_sectors, method)()
        if result.ari_vs_sectors is not None:
            assert -1 <= result.ari_vs_sectors <= 1

    # ---------------------------------------------------------
    # Validity metrics
    # ---------------------------------------------------------

    def test_validity_metrics_bounded(self, comparison: ClusterComparison):
        """
        Internal validity metrics fall within expected bounds
        when present.
        """
        result = comparison.kmeans()
        if result.calinski_harabasz is not None:
            assert result.calinski_harabasz > 0
        if result.davies_bouldin is not None:
            assert result.davies_bouldin >= 0
        if result.silhouette is not None:
            assert -1 <= result.silhouette <= 1

    # ---------------------------------------------------------
    # Silhouette details
    # ---------------------------------------------------------

    def test_silhouette_details_bounded(self, comparison: ClusterComparison):
        """
        Per-posting silhouette coefficients fall within [-1, 1].
        """
        result  = comparison.kmeans()
        details = comparison.silhouette_details(np.array(result.assignments))
        assert (details >= -1).all() and (details <= 1).all()

    def test_silhouette_details_degenerate(self, comparison: ClusterComparison):
        """
        Degenerate clustering (single label) returns all zeros.
        """
        details = comparison.silhouette_details(
            np.zeros(len(comparison.coordinates), dtype = int)
        )
        assert (details == 0).all()

    def test_silhouette_details_length(self, comparison: ClusterComparison):
        """
        Per-posting silhouette array has one entry per posting.
        """
        result  = comparison.kmeans()
        details = comparison.silhouette_details(np.array(result.assignments))
        assert len(details) == len(comparison.coordinates)
