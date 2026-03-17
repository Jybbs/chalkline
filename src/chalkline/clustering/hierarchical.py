"""
Average-linkage hierarchical agglomerative clustering with cophenetic
validation and TF-IDF centroid labeling.

Fits average linkage on PCA-reduced coordinates, selects a flat partition
via the merge-height acceleration criterion, and exposes cophenetic
comparison and internal validity metrics as on-demand methods. Cluster
labels are derived from TF-IDF centroid terms when explicitly requested
via `labels()`.
"""

import numpy  as np
import pandas as pd

from functools               import cached_property
from scipy.cluster.hierarchy import cophenet, dendrogram
from scipy.cluster.hierarchy import fcluster, leaders, linkage
from scipy.sparse            import spmatrix
from scipy.spatial.distance  import pdist

from chalkline.clustering.schemas import ClusterLabel, CopheneticResult


class HierarchicalClusterer:
    """
    Average-linkage HAC with multi-method cophenetic validation.

    Computes the average linkage matrix with optimal leaf ordering and
    selects a flat partition via the merge-height acceleration criterion,
    which finds the largest second derivative of the merge height
    sequence:

        k = argmax{Δ²hᵢ} + 2

    where Δ²hᵢ = hᵢ₊₂ - 2hᵢ₊₁ + hᵢ and k is the number of clusters.
    Cophenetic comparison and internal validity metrics are computed on
    demand via `cophenetic_comparison()` and `validation_metrics()`.
    Cluster labels are derived from TF-IDF centroid terms when
    explicitly requested via `labels()`.
    """

    def __init__(
        self,
        coordinates  : np.ndarray,
        document_ids : list[str]
    ):
        """
        Fit average-linkage HAC and derive cluster assignments.

        Computes the linkage matrix with optimal leaf ordering,
        selects k via merge-height acceleration, and cuts the
        tree at the selected partition.

        Args:
            coordinates  : PCA output, shape `(n_postings, n_selected)`.
            document_ids : Posting identifiers in row order.
        """
        self.coordinates  = coordinates
        self.document_ids = document_ids
        self.linkage      = linkage(
            coordinates,
            method           = "average",
            optimal_ordering = True
        )

        self.assignments = fcluster(
            self.linkage, 
            criterion = "maxclust", 
            t         = self._select_k()
        )

    @cached_property
    def centroids(self) -> pd.DataFrame:
        """
        Mean PCA coordinates per cluster, indexed by cluster ID.
        """
        return pd.DataFrame(
            self.coordinates
        ).groupby(self.assignments).mean()

    def _select_k(self) -> int:
        """
        Select the number of clusters via merge-height acceleration.

        Finds the largest second derivative of the merge height
        sequence, scanning from the right (fewest clusters) toward
        finer partitions. Falls back to k=2 when the tree has
        fewer than 4 leaves.

        Returns:
            Optimal cluster count.
        """
        if (accel := np.diff(self.linkage[:, 2], n=2)).size:
            return accel[::-1].argmax() + 2
        return 2

    def cophenetic_comparison(self) -> list[CopheneticResult]:
        """
        Cophenetic correlations for Ward, complete, and average linkage
        on the same coordinates.

            r = corr(Z_coph, pdist(X))

        where `Z_coph` is the cophenetic distance matrix derived from each
        linkage. Computes `pdist` and two additional linkage matrices on
        demand. Results are not cached, so repeated calls refit.

        Returns:
            One `CopheneticResult` per linkage method.
        """
        distances = pdist(self.coordinates)
        return [
            CopheneticResult(
                correlation = cophenet(z, distances)[0],
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

        Returns the scipy dendrogram dict with `icoord`, `dcoord`, `ivl`,
        and `color_list` keys. When `title_map` is provided, leaf labels
        are mapped from document identifiers to display titles.

        Args:
            title_map: Optional mapping from document identifier to
                       display title for leaf labeling.

        Returns:
            Scipy dendrogram dictionary.
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

        Averages the TF-IDF vectors of each cluster's members
        (aligned with `document_ids` row order) and extracts the
        `top_n` highest-weighted terms. Centroid computation
        stays sparse per cluster rather than materializing the
        full dense matrix.

        Args:
            feature_names : TF-IDF vocabulary in column order.
            tfidf_matrix  : Sparse TF-IDF matrix.
            top_n         : Number of top terms per label.

        Returns:
            One `ClusterLabel` per unique cluster assignment.
        """
        nodes, labels = leaders(self.linkage, self.assignments)
        names         = np.array(feature_names)
        return [
            ClusterLabel(
                cluster_id     = cid,
                leader_node_id = node,
                size           = mask.sum(),
                terms          = names[indices].tolist(),
                weights        = centroid[indices].tolist()
            )
            for cid, node in zip(labels, nodes)
            for mask      in [self.assignments == cid]
            for centroid  in [tfidf_matrix[mask].mean(axis=0).A1]
            for indices   in [np.argsort(centroid)[::-1][:top_n]]
        ]
