"""
Tests for cluster entity and aggregate containers.

Validates derived indices, cached properties, and container protocol
methods on `Cluster` and `Clusters` that would silently corrupt
downstream graph construction or display rendering if broken.
"""

import numpy as np

from pytest import mark

from chalkline.pathways.clusters import Clusters


class TestCluster:
    """
    Validate per-cluster derived properties.
    """

    def test_display_label_format(self, clusters: Clusters):
        """
        Display label includes the cluster ID, SOC title, and 1-indexed
        wage tier for human-readable dropdown labels.
        """
        cluster = clusters[clusters.cluster_ids[0]]
        label   = cluster.display_label
        assert f"Cluster {cluster.cluster_id}" in label
        assert cluster.soc_title in label
        assert f"Tier {cluster.wage_tier + 1}" in label

    def test_sub_role_labels_k1_fallback(self, clusters: Clusters):
        """
        With k=1, all postings fall into one sub-cluster. The
        `doc_freq[w] < k` filter eliminates every word because
        doc_freq is always >= 1 == k, so the label falls back to
        a numbered default.
        """
        cluster     = clusters[clusters.cluster_ids[0]]
        assignments = np.zeros(len(cluster.postings), dtype=int)
        labels      = cluster.sub_role_labels(assignments, k=1)
        assert len(labels) == 1
        assert labels[0] == "Sub-role 1"


class TestClusters:
    """
    Validate container protocol and derived indices on `Clusters`.
    """

    @mark.parametrize("prop", [
        "centroid_cosine",
        "cluster_heatmap",
        "pairwise_distances",
        "soc_heatmap",
        "vector_map",
        "wage_tier_map"
    ])
    def test_cached_property_stable(self, clusters: Clusters, prop: str):
        """
        Cached properties return the same object on repeated access,
        confirming the cache is not recomputed per call.
        """
        assert getattr(clusters, prop) is getattr(clusters, prop)

    def test_cluster_index_maps_id_to_position(self, clusters: Clusters):
        """
        `cluster_index` maps each cluster ID to its row position in
        the centroids and vectors matrices, consistent with the sorted
        `cluster_ids` ordering.
        """
        for position, cluster_id in enumerate(clusters.cluster_ids):
            assert clusters.cluster_index[cluster_id] == position

    def test_contains(self, clusters: Clusters):
        """
        Membership check delegates to the cluster ID set.
        """
        assert clusters.cluster_ids[0] in clusters
        assert 9999 not in clusters

    def test_cosine_similarity_matrix_symmetric(self, clusters: Clusters):
        """
        The centroid cosine similarity matrix is square and symmetric
        because cosine similarity is commutative.
        """
        matrix = clusters.cosine_similarity_matrix
        n      = len(clusters.cluster_ids)
        assert len(matrix) == n
        for i in range(n):
            assert len(matrix[i]) == n
            for j in range(n):
                assert matrix[i][j] == matrix[j][i]

    def test_display_titles_disambiguate_duplicates(self, clusters: Clusters):
        """
        Clusters sharing a SOC title receive distinguishing TF-IDF
        suffixes while clusters with a unique SOC title keep the bare
        title unchanged.
        """
        titles_by_soc: dict[str, list[str]] = {}
        for cid, label in clusters.display_titles.items():
            titles_by_soc.setdefault(clusters[cid].soc_title, []).append(label)

        for soc, labels in titles_by_soc.items():
            if len(labels) == 1:
                assert labels[0] == soc
            else:
                assert all(label.startswith(f"{soc} (") for label in labels)
                assert len(set(labels)) == len(labels)

    def test_getitem(self, clusters: Clusters):
        """
        Bracket access returns the `Cluster` with the matching ID.
        """
        cid     = clusters.cluster_ids[0]
        cluster = clusters[cid]
        assert cluster.cluster_id == cid

    def test_iter_sorted(self, clusters: Clusters):
        """
        Iteration yields cluster IDs in sorted order.
        """
        assert list(clusters) == sorted(clusters.cluster_ids)

    def test_len(self, clusters: Clusters):
        """
        Length reflects the number of clusters.
        """
        assert len(clusters) == 4

    def test_sector_sizes_sum(self, clusters: Clusters):
        """
        Sector sizes across all sectors sum to the total posting count.
        """
        total = sum(c.size for c in clusters.values())
        assert sum(clusters.sector_sizes.values()) == total

    def test_sizes_order(self, clusters: Clusters):
        """
        `sizes` returns posting counts in the same sorted cluster-ID
        order as `cluster_ids`.
        """
        assert clusters.sizes == [
            clusters[cid].size for cid in clusters.cluster_ids
        ]

    def test_vector_map_keys(self, clusters: Clusters):
        """
        `vector_map` has an entry for every cluster ID with the
        correct embedding dimensionality.
        """
        for cid in clusters.cluster_ids:
            assert cid in clusters.vector_map
            assert clusters.vector_map[cid].shape == (clusters.vectors.shape[1],)
