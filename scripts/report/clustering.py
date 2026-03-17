"""
DS5230 clustering comparison and dendrogram analysis.

Compares HAC average linkage against K-Means, DBSCAN, HDBSCAN, and
Mean Shift with internal validity metrics. Includes cophenetic
validation across linkage methods and dendrogram visualization data.
Not part of the production pipeline.
"""

import numpy as np

from kneed             import KneeLocator
from math              import floor
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
from scipy.spatial.distance  import pdist
from sklearn.cluster   import DBSCAN, estimate_bandwidth, HDBSCAN
from sklearn.cluster   import KMeans, MeanShift
from sklearn.metrics   import adjusted_rand_score
from sklearn.metrics   import calinski_harabasz_score
from sklearn.metrics   import davies_bouldin_score
from sklearn.metrics   import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors


class ClusterComparison:
    """
    Fit K-Means, DBSCAN, HDBSCAN, and Mean Shift on PCA coordinates
    with automatic parameter selection and internal validity metrics.
    """

    def __init__(
        self,
        coordinates   : np.ndarray,
        random_seed   : int,
        sector_labels : list[str] | None = None
    ):
        self.coordinates   = coordinates
        self.random_seed   = random_seed
        self.sector_labels = sector_labels
        self.n, self.d     = coordinates.shape

    def _build_result(
        self,
        assignments,
        method,
        noise_count = 0
    ) -> dict:
        """
        Wrap assignments and validity metrics into a result dict.
        """
        mask       = assignments >= 0
        n_clusters = len(np.unique(assignments[mask]))
        metrics    = self._validity_metrics(assignments)

        ari = None
        if self.sector_labels is not None and n_clusters >= 2:
            ari = float(adjusted_rand_score(
                np.array(self.sector_labels)[mask],
                assignments[mask]
            ))

        return {
            "assignments"       : assignments.tolist(),
            "ari_vs_sectors"    : ari,
            "calinski_harabasz" : metrics["calinski_harabasz"],
            "davies_bouldin"    : metrics["davies_bouldin"],
            "method"            : method,
            "n_clusters"        : n_clusters,
            "noise_count"       : noise_count,
            "silhouette"        : metrics["silhouette"]
        }

    def _validity_metrics(self, assignments) -> dict:
        """
        Calinski-Harabasz, Davies-Bouldin, and silhouette scores,
        or None for each when fewer than 2 non-noise clusters.
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

    def dbscan(self) -> dict:
        """
        DBSCAN with KneeLocator epsilon from the k-distance graph.
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

    def hdbscan(self) -> dict:
        """
        HDBSCAN with corpus-scaled minimum cluster size.
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

    def kmeans(self) -> dict:
        """
        K-Means with KneeLocator K from the inertia curve.
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

    def mean_shift(self) -> dict:
        """
        Mean Shift with bandwidth estimated at quantile 0.3.
        """
        assignments = MeanShift(
            bandwidth = estimate_bandwidth(self.coordinates, quantile = 0.3) or None
        ).fit_predict(self.coordinates)

        return self._build_result(assignments, "mean_shift")

    def silhouette_details(self, assignments: np.ndarray) -> np.ndarray:
        """
        Per-posting silhouette coefficients, zeros when degenerate.
        """
        mask = assignments >= 0
        if len(np.unique(assignments[mask])) < 2:
            return np.zeros(self.n)

        scores = np.zeros(self.n)
        scores[mask] = silhouette_samples(self.coordinates[mask], assignments[mask])
        return scores


def cophenetic_comparison(
    coordinates    : np.ndarray,
    linkage_matrix : np.ndarray
) -> list[dict]:
    """
    Cophenetic correlations for Ward, complete, and average linkage
    on the same coordinates.
    """
    distances = pdist(coordinates)
    return [
        {"correlation": cophenet(z, distances)[0], "method": method}
        for method, z in [
            ("average",  linkage_matrix),
            ("complete", linkage(coordinates, method = "complete")),
            ("ward",     linkage(coordinates, method = "ward"))
        ]
    ]


def dendrogram_data(
    linkage_matrix : np.ndarray,
    document_ids   : list[str],
    title_map      : dict[str, str] | None = None
) -> dict:
    """
    Scipy dendrogram structure for rendering without plotting.
    Optionally maps document IDs to display titles via `title_map`.
    """
    return dendrogram(
        linkage_matrix,
        labels  = [
            title_map.get(doc, doc) for doc in document_ids
        ] if title_map else document_ids,
        no_plot = True
    )
