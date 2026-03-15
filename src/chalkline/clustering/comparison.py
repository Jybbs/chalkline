"""
DS5230 comparison clustering methods.

Fits K-Means, DBSCAN, Mean Shift, and HDBSCAN on PCA-reduced coordinates
with automatic parameter selection via KneeLocator. Each method returns a
`ComparisonResult` with cluster assignments and optional internal validity
metrics. This module is excluded from the production pipeline.
"""

import numpy as np

from kneed             import KneeLocator
from math              import floor
from sklearn.cluster   import DBSCAN, estimate_bandwidth, HDBSCAN
from sklearn.cluster   import KMeans, MeanShift
from sklearn.metrics   import adjusted_rand_score
from sklearn.metrics   import calinski_harabasz_score
from sklearn.metrics   import davies_bouldin_score
from sklearn.metrics   import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors

from chalkline.clustering.schemas import ComparisonResult


class ClusterComparison:
    """
    Comparison clustering methods for DS5230 evaluation.

    Stores PCA coordinates and provides methods that fit each algorithm
    with corpus-scaled parameter selection, returning structured results
    with validity metrics.
    """

    def __init__(
        self,
        coordinates   : np.ndarray,
        random_seed   : int,
        sector_labels : list[str] | None = None
    ):
        """
        Store PCA coordinates of shape `(n_postings, n_selected)`
        and derive matrix dimensions. When sector labels are
        provided, all comparison methods compute ARI against them.

        Args:
            coordinates   : PCA-reduced posting coordinates.
            random_seed   : Reproducibility seed for K-Means.
            sector_labels : Optional sector strings for ARI.
        """
        self.coordinates   = coordinates
        self.random_seed   = random_seed
        self.sector_labels = sector_labels
        self.n, self.d     = coordinates.shape

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    def _build_result(
        self,
        assignments : np.ndarray,
        method      : str,
        noise_count : int = 0
    ) -> ComparisonResult:
        """
        Wrap assignments and validity metrics into a structured result.

        Masks noise labels before computing ARI and internal validity scores,
        returning `None` for metrics when fewer than 2 non-noise clusters
        exist.

        Args:
            assignments : Cluster label per observation.
            method      : Algorithm name for the result record.
            noise_count : Number of observations assigned to noise.

        Returns:
            Structured result with cluster assignments, validity metrics,
            and optional ARI against sector labels.
        """
        mask       = assignments >= 0
        n_clusters = len(np.unique(assignments[mask]))

        metrics = self._validity_metrics(assignments)

        ari = None
        if self.sector_labels is not None and n_clusters >= 2:
            ari = float(adjusted_rand_score(
                np.array(self.sector_labels)[mask],
                assignments[mask]
            ))

        return ComparisonResult(
            assignments       = assignments.tolist(),
            ari_vs_sectors    = ari,
            calinski_harabasz = metrics["calinski_harabasz"],
            davies_bouldin    = metrics["davies_bouldin"],
            method            = method,
            n_clusters        = n_clusters,
            noise_count       = noise_count,
            silhouette        = metrics["silhouette"]
        )

    def _validity_metrics(self, assignments: np.ndarray) -> dict[str, float | None]:
        """
        Compute internal validity metrics for non-degenerate clusterings.

        Returns `None` for each metric when fewer than 2 non-noise clusters
        exist, because sklearn's scoring functions require at least 2
        distinct labels.

        Args:
            assignments: Cluster label per observation.

        Returns:
            Dictionary mapping metric names to scores, with `None` values
            for degenerate clusterings.
        """
        mask   = assignments >= 0
        coords = self.coordinates[mask]
        labels = assignments[mask]

        metrics = [
            ("calinski_harabasz", calinski_harabasz_score),
            ("davies_bouldin",    davies_bouldin_score),
            ("silhouette",        silhouette_score)
        ]

        if (n := len(np.unique(labels))) < 2 or n >= self.n:
            return {name: None for name, _ in metrics}

        return {
            name: float(func(coords, labels))
            for name, func in metrics
        }

    # -----------------------------------------------------------------
    # Clustering methods
    # -----------------------------------------------------------------

    def dbscan(self) -> ComparisonResult:
        """
        DBSCAN with KneeLocator epsilon from k-distance graph.

        Computes `min_samples` from PCA dimensionality and corpus size,
        builds the k-distance graph via `NearestNeighbors`, and selects
        epsilon at the knee of the sorted distances. Falls back to the
        median k-distance when no knee is found.

        Returns:
            Structured result with cluster assignments and noise count.
        """
        min_samples = max(2, min(floor(0.1 * self.n), 2 * self.d))

        k_distances = np.sort(
            NearestNeighbors(n_neighbors = min_samples)
            .fit(self.coordinates)
            .kneighbors(self.coordinates)[0][:, -1]
        )

        knee = KneeLocator(
            curve     = "convex",
            direction = "increasing",
            x         = range(len(k_distances)),
            y         = k_distances
        ).knee

        eps = (
            float(k_distances[knee])
            if knee is not None
            else float(np.median(k_distances))
        )
        if eps <= 0:
            eps = float(np.max(k_distances)) or 0.5

        assignments = DBSCAN(
            eps         = eps,
            min_samples = min_samples
        ).fit_predict(self.coordinates)

        return self._build_result(
            assignments = assignments,
            method      = "dbscan",
            noise_count = int((assignments == -1).sum())
        )

    def hdbscan(self) -> ComparisonResult:
        """
        HDBSCAN with corpus-scaled minimum cluster size.

        Returns:
            Structured result with cluster assignments and noise count.
        """
        assignments = HDBSCAN(
            copy             = False,
            min_cluster_size = max(3, floor(0.05 * self.n))
        ).fit_predict(self.coordinates)

        return self._build_result(
            assignments = assignments,
            method      = "hdbscan",
            noise_count = int((assignments == -1).sum())
        )

    def kmeans(self) -> ComparisonResult:
        """
        K-Means with KneeLocator K from inertia curve.

        Fits K-Means across a range of cluster counts and selects the
        elbow via KneeLocator. Falls back to K=3 when no knee is found.

        Returns:
            Structured result with cluster assignments and zero noise.
        """
        ks = range(2, min(self.n, 11))

        inertias = [
            KMeans(
                n_clusters   = k,
                n_init       = 10,
                random_state = self.random_seed
            ).fit(self.coordinates).inertia_
            for k in ks
        ]

        knee = KneeLocator(
            curve     = "convex",
            direction = "decreasing",
            x         = ks,
            y         = inertias
        ).knee

        assignments = KMeans(
            n_clusters   = knee if knee is not None else 3,
            n_init       = 10,
            random_state = self.random_seed
        ).fit_predict(self.coordinates)

        return self._build_result(assignments, "kmeans")

    def mean_shift(self) -> ComparisonResult:
        """
        Mean Shift with bandwidth estimated at quantile 0.3.

        Returns:
            Structured result with cluster assignments and zero noise.
        """
        assignments = MeanShift(
            bandwidth = estimate_bandwidth(self.coordinates, quantile = 0.3) or None
        ).fit_predict(self.coordinates)

        return self._build_result(assignments, "mean_shift")

    # -----------------------------------------------------------------
    # Silhouette
    # -----------------------------------------------------------------

    def silhouette_details(self, assignments: np.ndarray) -> np.ndarray:
        """
        Per-posting silhouette coefficients.

        Returns an array of length `n_postings` with silhouette scores in
        [-1, 1]. For degenerate clusterings (fewer than 2 labels), returns
        zeros.

        Args:
            assignments: Cluster label per observation.

        Returns:
            Array of per-posting silhouette scores, zeros when degenerate.
        """
        mask = assignments >= 0
        if len(np.unique(assignments[mask])) < 2:
            return np.zeros(self.n)

        scores = np.zeros(self.n)
        scores[mask] = silhouette_samples(self.coordinates[mask], assignments[mask])
        return scores
