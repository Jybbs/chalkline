"""
Tests for embedding-based resume matching with per-task gap analysis.
"""

import numpy as np

from chalkline.matching.matcher  import ResumeMatcher
from chalkline.matching.schemas  import MatchResult, ScoredTask
from chalkline.pathways.clusters import Cluster, Clusters


class TestResumeMatcher:
    """
    Validate embedding-based cluster assignment, task gap analysis, and
    reach retrieval.
    """

    def test_cluster_assigned(self, clusters: Clusters, match_result: MatchResult):
        """
        Match result assigns a valid cluster ID from the profile set.
        """
        assert match_result.cluster_id in clusters

    def test_cluster_distances(
        self, 
        cluster_ids  : list[int], 
        match_result : MatchResult
    ):
        """
        Cluster distances cover every cluster as non-negative floats.
        """
        distances = match_result.cluster_distances
        assert len(distances) == len(cluster_ids)
        assert all(d >= 0 for d in distances)

    def test_confidence_range(self, match_result: MatchResult):
        """
        Confidence is a 0-100 integer percentage.
        """
        assert 0 <= match_result.confidence <= 100

    def test_score_destination(
        self,
        clusters       : Clusters,
        match_result   : MatchResult,
        resume_matcher : ResumeMatcher
    ):
        """
        Scoring against an arbitrary destination cluster returns one
        `ScoredTask` per task, sorted by descending similarity.
        """
        target = next(
            c for c in clusters.values()
            if c.cluster_id != match_result.cluster_id and c.tasks
        )
        scored = resume_matcher.score_destination(target)
        assert len(scored) == len(target.tasks)
        similarities = [t.similarity for t in scored]
        assert similarities == sorted(similarities, reverse=True)

    def test_score_destination_empty_tasks(
        self,
        match_result   : MatchResult,
        resume_matcher : ResumeMatcher
    ):
        """
        Scoring against a cluster with no tasks returns an empty list.
        """
        empty = Cluster(
            cluster_id  = 999,
            job_zone    = 1,
            modal_title = "Empty",
            postings    = [],
            sector      = "Test",
            size        = 0,
            soc_title   = "Empty",
            tasks       = []
        )
        assert resume_matcher.score_destination(empty) == []
