"""
Tests for embedding-based resume matching with per-task gap analysis.
"""


class TestResumeMatcher:
    """
    Validate embedding-based cluster assignment, task gap analysis, and
    neighborhood retrieval.
    """

    def test_cluster_assigned(self, profiles, resume_matcher):
        """
        Match result assigns a valid cluster ID from the profile set.
        """
        assert resume_matcher.match(
            "Electrician with welding experience"
        ).cluster_id in profiles

    def test_cluster_distances(self, cluster_ids, resume_matcher):
        """
        Cluster distances are sorted ascending and cover all clusters.
        """
        distances = [
            d.distance
            for d in resume_matcher.match("Construction worker").cluster_distances
        ]
        assert distances == sorted(distances)
        assert len(distances) == len(cluster_ids)

    def test_gap_split(self, resume_matcher):
        """
        Demonstrated and gaps partition the task set at the median.
        """
        result = resume_matcher.match("Electrical wiring specialist")
        assert len(result.demonstrated) + len(result.gaps) == 5

    def test_neighborhood_present(self, resume_matcher):
        """
        Match result includes a neighborhood with advancement and lateral
        edge lists.
        """
        result = resume_matcher.match("Project manager in construction")
        assert hasattr(result.neighborhood, "advancement")
        assert hasattr(result.neighborhood, "lateral")

    def test_sector_assigned(self, profiles, resume_matcher):
        """
        Match result carries a sector string from the assigned cluster
        profile.
        """
        result = resume_matcher.match("Heavy equipment operator")
        assert result.sector == profiles[result.cluster_id].sector
