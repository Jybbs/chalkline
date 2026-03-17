"""
Tests for average-linkage hierarchical agglomerative clustering.

Validates cluster assignments and label derivation using the synthetic
20-posting fixture chain.
"""

import numpy as np

from chalkline.clustering.hierarchical import HierarchicalClusterer
from chalkline.clustering.schemas      import ClusterLabel
from chalkline.extraction.vectorize    import SkillVectorizer


class TestComputeSectorLabels:
    """
    Validate sector label derivation from Jaccard-nearest SOC code.
    """

    def test_length_matches_documents(
        self,
        extracted_skills : dict[str, list[str]],
        sector_labels    : list[str]
    ):
        """
        One sector label per document in the PCA reducer's row order.
        """
        assert len(sector_labels) == len(extracted_skills)


class TestHierarchicalClusterer:
    """
    Validate assignments and labels from the average-linkage HAC fit.
    """

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
        assert len(cluster_labels) == len(np.unique(clusterer.assignments))

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
        Requesting fewer top terms limits the returned list length.
        """
        for label in clusterer.labels(
            feature_names = vectorizer.feature_names,
            tfidf_matrix  = vectorizer.tfidf_matrix,
            top_n         = 2
        ):
            assert len(label.terms) <= 2
