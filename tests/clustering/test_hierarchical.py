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

    def test_cophenetic_count(self, clusterer: HierarchicalClusterer):
        """
        Three cophenetic results for ward, complete, and average
        linkage methods.
        """
        assert len(clusterer.cophenetic_comparison()) == 3

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

    def test_every_posting_assigned(self, clusterer: HierarchicalClusterer):
        """
        Every posting receives exactly one cluster assignment with
        no unassigned labels.
        """
        n = len(clusterer.document_ids)
        assert len(clusterer.assignments) == n
        assert (clusterer.assignments > 0).all()

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
        clusterer  : HierarchicalClusterer,
        vectorizer : SkillVectorizer
    ):
        """
        Requesting fewer top terms limits the returned list
        length.
        """
        for label in clusterer.labels(
            feature_names = vectorizer.feature_names,
            tfidf_matrix  = vectorizer.tfidf_matrix,
            top_n         = 2
        ):
            assert len(label.terms) <= 2

    # ---------------------------------------------------------
    # Dendrogram
    # ---------------------------------------------------------

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
        title_map = {
            doc: f"Title {i}"
            for i, doc in enumerate(clusterer.document_ids)
        }
        data = clusterer.dendrogram_data(title_map = title_map)
        assert all(label.startswith("Title ") for label in data["ivl"])

    # ---------------------------------------------------------
    # Validate at K
    # ---------------------------------------------------------

    @mark.parametrize("k", [2, 3])
    def test_validate_at_k(self, clusterer: HierarchicalClusterer, k: int):
        """
        Assignments at a given K cover all postings and produce
        exactly K unique clusters.
        """
        result = clusterer.validate_at_k(k = k)
        assert len(result) == len(clusterer.document_ids)
        assert len(np.unique(result)) == k
