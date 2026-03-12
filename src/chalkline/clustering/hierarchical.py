"""
Ward-linkage hierarchical agglomerative clustering with cophenetic
validation and TF-IDF centroid labeling.

Fits Ward, complete, and average linkage matrices on PCA-reduced
coordinates, selects a flat partition via the inconsistency method,
and derives human-readable cluster labels from the top TF-IDF terms
of each cluster's centroid vector. The complete and average linkage
matrices exist only for cophenetic comparison and are not stored.
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

from chalkline.clustering.schemas     import ClusterLabel, CopheneticResult
from chalkline.extraction.occupations import OccupationIndex


def compute_sector_labels(
    document_ids     : list[str],
    extracted_skills : dict[str, list[str]],
    occupation_index : OccupationIndex
) -> list[str]:
    """
    Map postings to sector labels via Jaccard-nearest SOC code.

    For each document in row order, finds the O*NET occupation
    whose skill profile has maximum Jaccard overlap with the
    posting's canonical skill set, then returns that occupation's
    sector field.

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
        ).sector
        for doc in document_ids
    ]


class HierarchicalClusterer:
    """
    Ward-linkage HAC with multi-method cophenetic validation.

    Computes the Ward linkage matrix with optimal leaf ordering,
    selects a flat partition via the inconsistency method, and
    derives cluster labels from the top TF-IDF centroid terms.
    Complete and average linkage matrices are computed for
    cophenetic comparison but not retained.

    Requirement coverage: linkage (1), cophenetic (2, 7),
    dendrogram (3), inconsistency cut (4), cut_tree (5),
    validate_at_k (6), cluster labels (8), ARI (9).
    """

    def __init__(
        self,
        coordinates   : np.ndarray,
        document_ids  : list[str],
        feature_names : list[str],
        tfidf_matrix  : spmatrix,
        depth         : int   = 2,
        threshold     : float = 1.0
    ):
        """
        Fit Ward-linkage HAC and derive cluster assignments.

        Args:
            coordinates   : PCA output of shape
                            `(n_postings, n_selected)`.
            document_ids  : Posting identifiers in row order.
            feature_names : TF-IDF vocabulary in column order.
            tfidf_matrix  : Sparse TF-IDF matrix for cluster
                            centroid labeling.
            depth         : Inconsistency calculation depth.
            threshold     : Inconsistency cut threshold.
        """
        self.document_ids  = document_ids
        self.feature_names = feature_names

        n = len(coordinates)

        self.linkage = linkage(
            coordinates,
            method           = "ward",
            optimal_ordering = True
        )

        distances = pdist(coordinates)
        self.cophenetic = [
            CopheneticResult(
                correlation = float(cophenet(z, distances)[0]),
                method      = method
            )
            for method, z in [
                ("average",  linkage(coordinates, method = "average")),
                ("complete", linkage(coordinates, method = "complete")),
                ("ward",     self.linkage)
            ]
        ]

        self.cut_height  = threshold
        self.assignments = fcluster(
            self.linkage,
            criterion = "inconsistent",
            depth     = depth,
            t         = threshold
        )
        self.k = len(np.unique(self.assignments))

        max_k = min(n - 1, 20)
        if max_k >= 2:
            self.cut_tree = cut_tree(
                self.linkage,
                n_clusters = range(2, max_k + 1)
            )
        else:
            self.cut_tree = np.empty((n, 0), dtype = int)

        self.tfidf_dense = tfidf_matrix.toarray()
        leader_nodes, leader_labels = leaders(self.linkage, self.assignments)
        self.leader_map = dict(zip(leader_labels, leader_nodes))

        if 2 <= self.k < n:
            args = (coordinates, self.assignments)
            self.calinski_harabasz   = float(calinski_harabasz_score(*args))
            self.davies_bouldin      = float(davies_bouldin_score(*args))
            self.silhouette          = float(silhouette_score(*args))
            self.silhouette_samples_ = silhouette_samples(*args)
        else:
            self.calinski_harabasz   = None
            self.davies_bouldin      = None
            self.silhouette          = None
            self.silhouette_samples_ = np.zeros(n)

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

    def labels(self, top_n: int = 5) -> list[ClusterLabel]:
        """
        Human-readable labels from top TF-IDF centroid terms.

        Averages the original TF-IDF vectors of each cluster's
        members and extracts the `top_n` highest-weighted terms.

        Args:
            top_n: Number of top terms per cluster label.
        """
        return [
            ClusterLabel(
                cluster_id     = int(cluster_id),
                leader_node_id = int(self.leader_map[cluster_id]),
                size           = int(mask.sum()),
                terms          = [self.feature_names[j] for j in indices],
                weights        = [float(centroid[j]) for j in indices]
            )
            for cluster_id in np.unique(self.assignments)
            for mask     in [self.assignments == cluster_id]
            for centroid in [self.tfidf_dense[mask].mean(0)]
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
