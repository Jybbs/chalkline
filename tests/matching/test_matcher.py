"""
Tests for embedding-based resume matching with per-task gap analysis.
"""

import numpy as np

from pytest import mark

from chalkline.matching.matcher  import ResumeMatcher
from chalkline.matching.schemas  import MatchResult, ScoredTask
from chalkline.pathways.clusters import Cluster, Clusters, Task


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
            embeddings  = np.empty((0, 4)),
            job_zone    = 1,
            modal_title = "Empty",
            postings    = [],
            sector      = "Test",
            size        = 0,
            soc_title   = "Empty",
            tasks       = []
        )
        assert resume_matcher.score_destination(empty) == []

    def test_single_task_all_demonstrated(
        self,
        match_result   : MatchResult,
        resume_matcher : ResumeMatcher
    ):
        """
        A cluster with one task makes the median equal to the only
        similarity, so the task is always >= median and flagged as
        demonstrated.
        """
        vec = np.random.RandomState(99).randn(1, 16).astype(np.float32)
        solo = Cluster(
            cluster_id  = 888,
            embeddings  = vec,
            job_zone    = 2,
            modal_title = "Solo",
            postings    = [],
            sector      = "Test",
            size        = 1,
            soc_title   = "Solo",
            tasks       = [Task(name="Only task", vector=vec[0])]
        )
        scored = resume_matcher.score_destination(solo)
        assert len(scored) == 1
        assert scored[0].demonstrated is True


class TestMatchResult:
    """
    Validate derived properties on match results.
    """

    def test_confidence_uniform_distances(self):
        """
        When all cluster distances are equal and nonzero, min/max
        ratio is 1.0, producing 0% confidence because the resume
        is equidistant to every cluster.
        """
        from chalkline.pathways.schemas import Reach
        result = MatchResult(
            cluster_distances = [0.5, 0.5, 0.5],
            cluster_id        = 0,
            reach             = Reach()
        )
        assert result.confidence == 0

    def test_confidence_perfect_match(self):
        """
        When the nearest cluster is at distance 0, confidence is 100%
        regardless of the farthest distance.
        """
        from chalkline.pathways.schemas import Reach
        result = MatchResult(
            cluster_distances = [0.0, 0.5, 1.0],
            cluster_id        = 0,
            reach             = Reach()
        )
        assert result.confidence == 100

    def test_coordinates_default_empty(self):
        """
        Coordinates default to an empty list when not provided.
        """
        from chalkline.pathways.schemas import Reach
        result = MatchResult(
            cluster_distances = [0.1, 0.5],
            cluster_id        = 0,
            reach             = Reach()
        )
        assert result.coordinates == []


class TestScoredTask:
    """
    Validate derived properties on scored task results.
    """

    @mark.parametrize(("similarity", "expected"), [
        (0.85,  85.0),
        (0.0,   0.0),
        (-0.5, -50.0),
        (1.0,  100.0)
    ])
    def test_pct(self, similarity: float, expected: float):
        """
        `pct` converts the -1..1 similarity to a percentage with
        one decimal.
        """
        task = ScoredTask(demonstrated=True, name="Test", similarity=similarity)
        assert task.pct == expected
