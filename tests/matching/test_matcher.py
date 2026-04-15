"""
Tests for embedding-based resume matching with per-task gap analysis.
"""

import numpy as np

from datetime import date
from pytest   import mark

from chalkline.collection.schemas import Posting
from chalkline.matching.matcher   import ResumeMatcher
from chalkline.matching.schemas   import MatchResult, ScoredTask
from chalkline.pathways.clusters  import Cluster, Clusters, Task
from chalkline.pathways.schemas   import Credential, Reach


class TestMatchResult:
    """
    Validate derived properties on match results.
    """

    def test_confidence_perfect_match(self):
        """
        When the nearest cluster is at distance 0, confidence is 100%
        regardless of the farthest distance.
        """
        result = MatchResult(
            cluster_distances = [0.0, 0.5, 1.0],
            cluster_id        = 0,
            reach             = Reach()
        )
        assert result.confidence == 100

    def test_confidence_uniform_distances(self):
        """
        When all cluster distances are equal and nonzero, min/max
        ratio is 1.0, producing 0% confidence because the resume
        is equidistant to every cluster.
        """
        result = MatchResult(
            cluster_distances = [0.5, 0.5, 0.5],
            cluster_id        = 0,
            reach             = Reach()
        )
        assert result.confidence == 0

    def test_coordinates_default_empty(self):
        """
        Coordinates default to an empty list when not provided.
        """
        result = MatchResult(
            cluster_distances = [0.1, 0.5],
            cluster_id        = 0,
            reach             = Reach()
        )
        assert result.coordinates == []


class TestResumeMatcher:
    """
    Validate embedding-based cluster assignment, task gap analysis, and
    reach retrieval.
    """

    def test_calibrate_cluster_means(
        self,
        clusters       : Clusters,
        match_result   : MatchResult,
        resume_matcher : ResumeMatcher
    ):
        """
        Calibration returns a mean similarity for every cluster,
        including zero for taskless clusters.
        """
        means = resume_matcher.calibrate()
        assert set(means.keys()) == set(clusters)
        assert all(-1.0 <= v <= 1.0 for v in means.values())

    def test_calibrate_threshold(
        self,
        match_result   : MatchResult,
        resume_matcher : ResumeMatcher
    ):
        """
        Calibration sets `global_threshold` to the median of all
        per-task similarities.
        """
        resume_matcher.calibrate()
        assert resume_matcher.global_threshold >= 0

    def test_chunk_shape(self, resume_matcher: ResumeMatcher):
        """
        Matching a multi-sentence resume stores `resume_chunks` with
        one row per sentence, distinct from the single-row
        `resume_embedding`.
        """
        resume_matcher.match("First sentence. Second sentence. Third sentence.")
        assert resume_matcher.resume_chunks.shape[0] == 3
        assert resume_matcher.resume_embedding.shape[0] == 1

    def test_chunk_task_max_pool(
        self,
        clusters       : Clusters,
        resume_matcher : ResumeMatcher
    ):
        """
        Task scoring via `_task_similarities` returns one similarity
        per task, each the max across resume chunks.
        """
        resume_matcher.match("First sentence. Second sentence. Third sentence.")
        target = next(c for c in clusters.values() if c.tasks)
        sims = resume_matcher._task_similarities(target)
        assert sims.shape == (len(target.tasks),)

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

    def test_credential_coverage_empty(
        self,
        clusters       : Clusters,
        match_result   : MatchResult,
        resume_matcher : ResumeMatcher
    ):
        """
        No credentials produces an empty coverage dict regardless
        of gap indices.
        """
        target = next(c for c in clusters.values() if c.tasks)
        assert resume_matcher.credential_coverage([], target, [0]) == {}

    def test_credential_coverage_varies_by_topicality(
        self,
        clusters       : Clusters,
        match_result   : MatchResult,
        resume_matcher : ResumeMatcher
    ):
        """
        A credential whose vector aligns with the gap tasks covers
        more positions than one pointed away, proving coverage sets
        vary with true topicality rather than a fixed slice.
        """
        target        = next(c for c in clusters.values() if c.tasks)
        gap_indices   = list(range(len(target.tasks)))
        aligned       = target.task_matrix.mean(axis=0)
        aligned       = aligned / np.linalg.norm(aligned)
        opposite      = -aligned

        credentials = [
            Credential(
                embedding_text = " ".join(t.name for t in target.tasks),
                kind           = "certification",
                label          = "Aligned",
                vector         = aligned.tolist()
            ),
            Credential(
                embedding_text = "irrelevant",
                kind           = "certification",
                label          = "Diffuse",
                vector         = opposite.tolist()
            )
        ]
        resume_matcher.global_threshold = 0.1
        coverage = resume_matcher.credential_coverage(
            credentials = credentials,
            destination = target,
            gap_indices = gap_indices
        )
        assert len(coverage["Aligned"]) > len(coverage["Diffuse"])

    def test_global_threshold(
        self,
        clusters       : Clusters,
        match_result   : MatchResult,
        resume_matcher : ResumeMatcher
    ):
        """
        Setting `global_threshold` to 1.0 forces every task below the
        bar, marking all as gaps rather than demonstrated.
        """
        resume_matcher.global_threshold = 1.0
        target = next(c for c in clusters.values() if c.tasks)
        scored = resume_matcher.score_destination(target)
        assert all(not t.demonstrated for t in scored)

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

    def test_score_postings_date_order(
        self,
        match_result   : MatchResult,
        resume_matcher : ResumeMatcher
    ):
        """
        Dated postings are returned in reverse chronological order,
        and undated postings are filtered out.
        """
        postings = [
            Posting(
                company     = "A",
                date_posted = date(2026, 1, 1),
                description = "x" * 50,
                source_url  = "https://example.com",
                title       = "Old"
            ),
            Posting(
                company     = "B",
                date_posted = None,
                description = "x" * 50,
                source_url  = "https://example.com",
                title       = "Undated"
            ),
            Posting(
                company     = "C",
                date_posted = date(2026, 6, 1),
                description = "x" * 50,
                source_url  = "https://example.com",
                title       = "Recent"
            )
        ]
        cluster = Cluster(
            cluster_id  = 777,
            embeddings  = np.random.RandomState(55).randn(3, 16).astype(np.float32),
            job_zone    = 2,
            modal_title = "Test",
            postings    = postings,
            sector      = "Test",
            size        = 3,
            soc_title   = "Test"
        )
        scored = resume_matcher.score_postings(cluster)
        assert len(scored) == 2
        assert scored[0][0].title == "Recent"
        assert scored[1][0].title == "Old"
        assert all(type(s) is float for _, s in scored)

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
