"""
Tests for embedding-based resume matching with per-task gap analysis.
"""

import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from typing                import Any

from chalkline.matching.matcher import ResumeMatcher
from chalkline.pipeline.graph   import CareerPathwayGraph
from chalkline.pipeline.schemas import ClusterProfile
from tests.conftest             import EMBEDDING_DIM, MockEncoder


class TestResumeMatcher:
    """
    Validate embedding-based cluster assignment, task gap analysis, and
    neighborhood retrieval.
    """

    def _build_matcher(
        self,
        centroids     : np.ndarray,
        cluster_ids   : list[int],
        pathway_graph : CareerPathwayGraph,
        profiles      : dict[int, ClusterProfile],
        svd_model     : TruncatedSVD
    ) -> ResumeMatcher:
        """
        Build a matcher with mock encoder and synthetic task embeddings.
        """
        mock: Any = MockEncoder()
        return ResumeMatcher(
            centroids    = centroids,
            cluster_ids  = cluster_ids,
            graph        = pathway_graph,
            model        = mock,
            profiles     = profiles,
            svd          = svd_model,
            task_labels  = {
                cluster_id: [f"Task {cluster_id}-{i}" for i in range(5)]
                for cluster_id in cluster_ids
            },
            task_vectors = {
                cluster_id: normalize(
                    np.random.RandomState(cluster_id).randn(
                        5, EMBEDDING_DIM
                    ).astype(np.float32)
                )
                for cluster_id in cluster_ids
            }
        )

    def test_cluster_assigned(
        self, centroids, cluster_ids, pathway_graph,
        profiles, svd_model
    ):
        """
        Match result assigns a valid cluster ID from the profile set.
        """
        matcher = self._build_matcher(
            centroids, cluster_ids, pathway_graph,
            profiles, svd_model
        )
        result = matcher.match("Electrician with welding experience")
        assert result.cluster_id in profiles

    def test_cluster_distances(
        self, centroids, cluster_ids, pathway_graph,
        profiles, svd_model
    ):
        """
        Cluster distances are sorted ascending and cover all clusters.
        """
        matcher = self._build_matcher(
            centroids, cluster_ids, pathway_graph,
            profiles, svd_model
        )
        result    = matcher.match("Construction worker")
        distances = [d.distance for d in result.cluster_distances]
        assert distances == sorted(distances)
        assert len(result.cluster_distances) == len(cluster_ids)

    def test_gap_split(
        self, centroids, cluster_ids, pathway_graph,
        profiles, svd_model
    ):
        """
        Demonstrated and gaps partition the task set at the median.
        """
        matcher = self._build_matcher(
            centroids, cluster_ids, pathway_graph,
            profiles, svd_model
        )
        result = matcher.match("Electrical wiring specialist")
        total  = len(result.demonstrated) + len(result.gaps)
        assert total == 5

    def test_neighborhood_present(
        self, centroids, cluster_ids, pathway_graph,
        profiles, svd_model
    ):
        """
        Match result includes a neighborhood with advancement and lateral
        edge lists.
        """
        matcher = self._build_matcher(
            centroids, cluster_ids, pathway_graph,
            profiles, svd_model
        )
        result = matcher.match("Project manager in construction")
        assert hasattr(result.neighborhood, "advancement")
        assert hasattr(result.neighborhood, "lateral")

    def test_sector_assigned(
        self, centroids, cluster_ids, pathway_graph,
        profiles, svd_model
    ):
        """
        Match result carries a sector string from the assigned cluster
        profile.
        """
        matcher = self._build_matcher(
            centroids, cluster_ids, pathway_graph,
            profiles, svd_model
        )
        result = matcher.match("Heavy equipment operator")
        assert result.sector == profiles[result.cluster_id].sector
