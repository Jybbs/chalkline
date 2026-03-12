"""
Average-linkage hierarchical agglomerative clustering with cophenetic
validation and TF-IDF centroid labeling.

Fits average linkage on PCA-reduced coordinates, selects a flat
partition via the merge-height acceleration criterion, and exposes
cophenetic comparison and internal validity metrics as on-demand
methods. Cluster labels are derived from TF-IDF centroid terms when
explicitly requested via `labels()`.
"""

import numpy as np

from scipy.cluster.hierarchy import cophenet, cut_tree, dendrogram
from scipy.cluster.hierarchy import fcluster, leaders, linkage
from scipy.sparse            import spmatrix
from scipy.spatial.distance  import pdist
from sklearn.metrics         import adjusted_rand_score
from sklearn.metrics         import calinski_harabasz_score
from sklearn.metrics         import davies_bouldin_score
from sklearn.metrics         import silhouette_samples, silhouette_score

from chalkline.clustering.schemas     import ClusterLabel
from chalkline.clustering.schemas     import CopheneticResult, ValidationMetrics
from chalkline.extraction.occupations import OccupationIndex


def compute_sector_labels(
    document_ids     : list[str],
    extracted_skills : dict[str, list[str]],
    occupation_index : OccupationIndex
) -> list[str]:
    """
    Map postings to SOC codes via Jaccard-nearest occupation.

    For each document in row order, finds the O*NET occupation
    whose skill profile has maximum Jaccard overlap with the
    posting's canonical skill set, then returns that occupation's
    SOC code, giving 21 distinct ground-truth classes for ARI
    evaluation rather than 3 broad sectors.

    Args:
        document_ids     : Posting identifiers in matrix row order.
        extracted_skills : Mapping from document identifier to
                           canonical skill names.
        occupation_index : O*NET occupation lookup with Jaccard
                           matching.
    """
    return [
        occupation_index.get(
            occupation_index.nearest(set(extracted_skills[doc]))
        ).soc_code
        for doc in document_ids
    ]


class HierarchicalClusterer:
    """
    Average-linkage HAC with multi-method cophenetic validation.

    Computes the average linkage matrix with optimal leaf ordering and
    selects a flat partition via the merge-height acceleration
    criterion. Cophenetic comparison and internal validity metrics
    are computed on demand via `cophenetic_comparison()` and
    `validation_metrics()`. Cluster labels are derived from TF-IDF
    centroid terms when explicitly requested via `labels()`.

    Requirement coverage: linkage (1), cophenetic (2, 7),
    dendrogram (3), acceleration cut (4), cut_tree (5),
    validate_at_k (6), cluster labels (8), ARI (9).
    """

    def __init__(
        self,
        coordinates  : np.ndarray,
        document_ids : list[str]
    ):
        """
        Fit Ward-linkage HAC and derive cluster assignments.

        Args:
            coordinates  : PCA output of shape
                           `(n_postings, n_selected)`.
            document_ids : Posting identifiers in row order.
        """
        self.coordinates  = coordinates
        self.document_ids = document_ids

        n = len(coordinates)

        self.linkage = linkage(
            coordinates,
            method           = "average",
            optimal_ordering = True
        )

        heights = self.linkage[:, 2]
        if n <= 3:
            k = 2
        else:
            acceleration = np.diff(heights, n = 2)
            k            = int(acceleration[::-1].argmax()) + 2
        self.cut_height  = float(heights[-k])
        self.assignments = fcluster(self.linkage, criterion = "maxclust", t = k)
        self.k           = len(np.unique(self.assignments))

        max_k         = min(n - 1, 20)
        self.cut_tree = (
            cut_tree(self.linkage, n_clusters = range(2, max_k + 1))
            if max_k >= 2
            else np.empty((n, 0), dtype = int)
        )

    # -----------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------

    def ari_vs_sectors(self, sector_labels: list[str]) -> float:
        """
        Adjusted Rand index between HAC assignments and sector
        labels.

        Args:
            sector_labels: Sector strings aligned with
                           `document_ids` row order.
        """
        return float(adjusted_rand_score(sector_labels, self.assignments))

    def cophenetic_comparison(self) -> list[CopheneticResult]:
        """
        Cophenetic correlations for Ward, complete, and average
        linkage on the same coordinates.

        Computes `pdist` and two additional linkage matrices on
        demand. Results are not cached, so repeated calls refit.
        """
        distances = pdist(self.coordinates)
        return [
            CopheneticResult(
                correlation = float(cophenet(z, distances)[0]),
                method      = method
            )
            for method, z in [
                ("average",  self.linkage),
                ("complete", linkage(self.coordinates, method = "complete")),
                ("ward",     linkage(self.coordinates, method = "ward"))
            ]
        ]

    def dendrogram_data(self, title_map: dict[str, str] | None = None) -> dict:
        """
        Dendrogram structure for rendering without plotting.

        Returns the scipy dendrogram dict with `icoord`, `dcoord`,
        `ivl`, and `color_list` keys. When `title_map` is provided,
        leaf labels are mapped from document identifiers to display
        titles.

        Args:
            title_map: Optional mapping from document identifier to
                       display title for leaf labeling.
        """
        return dendrogram(
            self.linkage,
            labels  = [
                title_map.get(doc, doc) for doc in self.document_ids
            ] if title_map else self.document_ids,
            no_plot = True
        )

    def labels(
        self,
        feature_names : list[str],
        tfidf_matrix  : spmatrix,
        top_n         : int = 5
    ) -> list[ClusterLabel]:
        """
        Human-readable labels from top TF-IDF centroid terms.

        Averages the TF-IDF vectors of each cluster's members and
        extracts the `top_n` highest-weighted terms. The dense
        conversion of `tfidf_matrix` is scoped to this call.

        Args:
            feature_names : TF-IDF vocabulary in column order.
            tfidf_matrix  : Sparse TF-IDF matrix for centroid
                            computation, aligned with `document_ids`.
            top_n         : Number of top terms per cluster label.
        """
        tfidf_dense                 = tfidf_matrix.toarray()
        leader_nodes, leader_labels = leaders(self.linkage, self.assignments)
        leader_map                  = dict(zip(leader_labels, leader_nodes))
        return [
            ClusterLabel(
                cluster_id     = int(cluster_id),
                leader_node_id = int(leader_map[cluster_id]),
                size           = int(mask.sum()),
                terms          = [feature_names[j] for j in indices],
                weights        = [float(centroid[j]) for j in indices]
            )
            for cluster_id in np.unique(self.assignments)
            for mask     in [self.assignments == cluster_id]
            for centroid in [tfidf_dense[mask].mean(0)]
            for indices  in [np.argsort(centroid)[::-1][:top_n]]
        ]

    def validate_at_k(self, k: int) -> np.ndarray:
        """
        Flat cluster assignments at a specified number of clusters.

        Cuts the Ward linkage tree at exactly `k` clusters via
        `fcluster` with the `maxclust` criterion, enabling
        post-hoc comparison against K-Means' elbow-selected K.

        Args:
            k: Target number of clusters.
        """
        return fcluster(self.linkage, criterion = "maxclust", t = k)

    def validation_metrics(self) -> ValidationMetrics:
        """
        Internal validity metrics for the current flat partition.

        Computes silhouette, Calinski-Harabasz, and Davies-Bouldin
        scores on demand. Returns a `ValidationMetrics` instance
        with all fields set to `None` when the partition is
        degenerate (fewer than 2 clusters).
        """
        n = len(self.coordinates)
        if not (2 <= self.k < n):
            return ValidationMetrics()
        args = (self.coordinates, self.assignments)
        return ValidationMetrics(
            calinski_harabasz  = float(calinski_harabasz_score(*args)),
            davies_bouldin     = float(davies_bouldin_score(*args)),
            silhouette         = float(silhouette_score(*args)),
            silhouette_samples = silhouette_samples(*args).tolist()
        )
