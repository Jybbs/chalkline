"""
Tests for Ward-linkage hierarchical agglomerative clustering.

Validates linkage structure, cophenetic correlations, cluster
assignments, label derivation, dendrogram output, ARI computation,
and post-hoc validation at arbitrary K values using the synthetic
20-posting fixture chain.
"""

import numpy as np

from pytest import mark

from chalkline.clustering.hierarchical import HierarchicalClusterer
from chalkline.clustering.schemas      import ClusterLabel
from chalkline.extraction.vectorize    import SkillVectorizer


class TestComputeSectorLabels:
    """
    Validate sector label derivation from Jaccard nearest SOC.
    """

    def test_labels_are_strings(self, sector_labels: list[str]):
        """
        Every sector label is a non-empty string.
        """
        assert all(isinstance(s, str) and s for s in sector_labels)

    def test_length_matches_documents(
        self,
        extracted_skills : dict[str, list[str]],
        sector_labels    : list[str]
    ):
        """
        One sector label per document in the PCA reducer's row
        order.
        """
        assert len(sector_labels) == len(extracted_skills)


class TestHierarchicalClusterer:
    """
    Validate linkage, assignments, labels, dendrogram, and
    validity metrics from the Ward-linkage HAC fit.
    """

    # ---------------------------------------------------------
    # Linkage
    # ---------------------------------------------------------

    def test_cophenetic_bounded(self, clusterer: HierarchicalClusterer):
        """
        Cophenetic correlations fall within [-1, 1].
        """
        for result in clusterer.cophenetic_comparison():
            assert -1 <= result.correlation <= 1

    def test_cophenetic_count(self, clusterer: HierarchicalClusterer):
        """
        Three cophenetic results for ward, complete, and average
        linkage methods.
        """
        assert len(clusterer.cophenetic_comparison()) == 3

    def test_cophenetic_methods(self, clusterer: HierarchicalClusterer):
        """
        Cophenetic results cover all three linkage methods.
        """
        methods = {result.method for result in clusterer.cophenetic_comparison()}
        assert methods == {"average", "complete", "ward"}

    def test_linkage_shape(self, clusterer: HierarchicalClusterer):
        """
        Ward linkage matrix has shape `(n - 1, 4)` where `n` is
        the number of postings.
        """
        n = len(clusterer.document_ids)
        assert clusterer.linkage.shape == (n - 1, 4)

    # ---------------------------------------------------------
    # Assignments
    # ---------------------------------------------------------

    def test_cut_height_positive(self, clusterer: HierarchicalClusterer):
        """
        The inconsistency cut threshold is positive.
        """
        assert clusterer.cut_height > 0

    def test_cut_tree_rows(self, clusterer: HierarchicalClusterer):
        """
        Cut tree has one row per posting.
        """
        n = len(clusterer.document_ids)
        assert clusterer.cut_tree.shape[0] == n

    def test_every_posting_assigned(self, clusterer: HierarchicalClusterer):
        """
        Every posting receives exactly one cluster assignment with
        no unassigned labels.
        """
        n = len(clusterer.document_ids)
        assert len(clusterer.assignments) == n
        assert (clusterer.assignments > 0).all()

    def test_k_positive(self, clusterer: HierarchicalClusterer):
        """
        At least one cluster is produced.
        """
        assert clusterer.k >= 1

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------

    def test_labels_aligned(self, cluster_labels: list[ClusterLabel]):
        """
        Each label has matching term and weight counts.
        """
        for label in cluster_labels:
            assert len(label.terms) == len(label.weights)

    def test_labels_count(
        self,
        cluster_labels : list[ClusterLabel],
        clusterer      : HierarchicalClusterer
    ):
        """
        One label per unique cluster.
        """
        assert len(cluster_labels) == clusterer.k

    def test_labels_readable(self, cluster_labels: list[ClusterLabel]):
        """
        Cluster label terms are human-readable strings, not
        numeric indices.
        """
        for label in cluster_labels:
            assert all(isinstance(t, str) for t in label.terms)
            assert all(not t.isdigit() for t in label.terms)

    def test_labels_size_sums(
        self,
        cluster_labels : list[ClusterLabel],
        clusterer      : HierarchicalClusterer
    ):
        """
        Cluster sizes sum to the total number of postings.
        """
        total = sum(label.size for label in cluster_labels)
        assert total == len(clusterer.document_ids)

    def test_labels_top_n(
        self,
        clusterer        : HierarchicalClusterer,
        skill_vectorizer : SkillVectorizer
    ):
        """
        Requesting fewer top terms limits the returned list
        length.
        """
        for label in clusterer.labels(
            feature_names = skill_vectorizer.feature_names,
            tfidf_matrix  = skill_vectorizer.tfidf_matrix,
            top_n         = 2
        ):
            assert len(label.terms) <= 2

    # ---------------------------------------------------------
    # Dendrogram
    # ---------------------------------------------------------

    def test_dendrogram_data_keys(self, clusterer: HierarchicalClusterer):
        """
        Dendrogram dict contains the expected scipy keys.
        """
        data = clusterer.dendrogram_data()
        assert {"icoord", "dcoord", "ivl", "color_list"} <= data.keys()

    def test_dendrogram_leaf_count(self, clusterer: HierarchicalClusterer):
        """
        Leaf labels match the number of postings.
        """
        data = clusterer.dendrogram_data()
        assert len(data["ivl"]) == len(clusterer.document_ids)

    def test_dendrogram_title_map(self, clusterer: HierarchicalClusterer):
        """
        Title map substitutes document identifiers in leaf labels.
        """
        title_map = {doc: f"Title {i}" for i, doc in enumerate(clusterer.document_ids)}
        data = clusterer.dendrogram_data(title_map = title_map)
        assert all(label.startswith("Title ") for label in data["ivl"])

    # ---------------------------------------------------------
    # ARI and validity
    # ---------------------------------------------------------

    def test_ari_vs_sectors_bounded(
        self,
        clusterer     : HierarchicalClusterer,
        sector_labels : list[str]
    ):
        """
        Adjusted Rand index falls within [-1, 1].
        """
        assert -1 <= clusterer.ari_vs_sectors(sector_labels) <= 1

    def test_silhouette_samples_length(self, clusterer: HierarchicalClusterer):
        """
        Per-posting silhouette array has one entry per posting.
        """
        n = len(clusterer.document_ids)
        assert len(clusterer.validation_metrics().silhouette_samples) == n

    def test_validity_metrics_populated(self, clusterer: HierarchicalClusterer):
        """
        Internal validity metrics are non-None when k >= 2.
        """
        metrics = clusterer.validation_metrics()
        assert metrics.calinski_harabasz is not None
        assert metrics.calinski_harabasz > 0
        assert metrics.davies_bouldin is not None
        assert metrics.davies_bouldin >= 0
        assert metrics.silhouette is not None
        assert -1 <= metrics.silhouette <= 1

    # ---------------------------------------------------------
    # Validate at K
    # ---------------------------------------------------------

    @mark.parametrize("k", [
        2,
        3
    ])
    def test_validate_at_k(self, clusterer: HierarchicalClusterer, k: int):
        """
        Assignments at a given K cover all postings and produce
        exactly K unique clusters.
        """
        result = clusterer.validate_at_k(k = k)
        assert len(result) == len(clusterer.document_ids)
        assert len(np.unique(result)) == k
