"""
Tests for embedding-based resume matching with per-task gap analysis.
"""

from chalkline.matching.schemas import MatchResult


class TestResumeMatcher:
    """
    Validate embedding-based cluster assignment, task gap analysis, and
    reach retrieval.
    """

    def test_cluster_assigned(self, clusters, match_result: MatchResult):
        """
        Match result assigns a valid cluster ID from the profile set.
        """
        assert match_result.cluster_id in clusters

    def test_cluster_distances(self, cluster_ids, match_result: MatchResult):
        """
        Cluster distances are sorted ascending and cover all clusters.
        """
        distances = [d.distance for d in match_result.cluster_distances]
        assert distances == sorted(distances)
        assert len(distances) == len(cluster_ids)

    def test_confidence_range(self, match_result: MatchResult):
        """
        Confidence is a 0-100 integer percentage.
        """
        assert 0 <= match_result.confidence <= 100

    def test_sector_assigned(self, clusters, match_result: MatchResult):
        """
        Match result carries a sector string from the assigned cluster
        profile.
        """
        assert match_result.sector == clusters[match_result.cluster_id].sector

    def test_tasks_by_type(self, match_result: MatchResult):
        """
        Task grouping preserves total count across types.
        """
        by_type = match_result.tasks_by_type
        assert sum(len(v) for v in by_type.values()) == len(
            match_result.scored_tasks
        )
        assert len(by_type) >= 2
