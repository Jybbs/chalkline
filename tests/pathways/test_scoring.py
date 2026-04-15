"""
Tests for ColBERTv2-style late-interaction SOC scoring.

Validates that `SOCScorer` produces the expected similarity matrix shape
and ranking behavior. A pooled-vector baseline would silently misrank
specialty SOCs against cluster centroids; these tests lock in the
per-posting per-task MaxSim semantics that the pipeline depends on.
"""

import numpy as np

from pytest                import fixture, raises
from sklearn.preprocessing import normalize

from chalkline.pathways.schemas import EncodedOccupation, Occupation, Skill, SkillType
from chalkline.pathways.scoring import SOCScorer


def make_occupation(title: str, task_names: list[str]) -> Occupation:
    """
    Construct an `Occupation` with only Task-type skills, enough to satisfy
    `EncodedOccupation` and exercise the scorer without loading the full
    curated lexicon.
    """
    return Occupation(
        job_zone = 3,
        sector   = "Building Construction",
        skills   = [Skill(name=n, type=SkillType.TASK) for n in task_names],
        title    = title
    )


@fixture
def encoded_occupations() -> list[EncodedOccupation]:
    """
    Three occupations with disjoint axis-aligned task vectors so each
    occupation's rows activate independently under MaxSim.
    """
    return [
        EncodedOccupation(
            occupation = make_occupation("Alpha",   ["a1", "a2"]),
            tasks      = normalize(np.array([[1.0, 0, 0, 0], [0.9, 0.1, 0, 0]]))
        ),
        EncodedOccupation(
            occupation = make_occupation("Beta",    ["b1", "b2", "b3"]),
            tasks      = normalize(np.array([[0, 1.0, 0, 0], [0, 0.8, 0.2, 0], [0, 0.5, 0.5, 0]]))
        ),
        EncodedOccupation(
            occupation = make_occupation("Gamma",   ["g1"]),
            tasks      = normalize(np.array([[0, 0, 1.0, 0]]))
        )
    ]


class TestSOCScorer:
    """
    Validate shape, ordering, and numerical correctness of MaxSim output.
    """

    def test_score_shape_and_dtype(
        self,
        encoded_occupations : list[EncodedOccupation]
    ):
        """
        Output is a `(n_clusters, n_occupations)` float32 matrix.
        """
        raw_vectors = np.random.default_rng(0).standard_normal((12, 4)).astype(np.float32)
        assignments = np.array([0] * 6 + [1] * 6)
        scores      = SOCScorer(occupations=encoded_occupations).score(
            assignments = assignments,
            raw_vectors = raw_vectors
        )
        assert scores.shape == (2, 3)
        assert scores.dtype == np.float32

    def test_score_ranks_axis_aligned_postings(
        self,
        encoded_occupations : list[EncodedOccupation]
    ):
        """
        Cluster 0's postings align with occupation Alpha's task axis and
        cluster 1's postings align with Gamma's. Each cluster should rank
        its matching occupation strictly above the other two.
        """
        raw_vectors = np.array([
            [1.0, 0, 0, 0], [0.95, 0.05, 0, 0], [0.9, 0.0, 0.1, 0],
            [0, 0, 1.0, 0], [0, 0.1, 0.95, 0], [0.1, 0, 0.9, 0]
        ], dtype=np.float32)
        assignments = np.array([0, 0, 0, 1, 1, 1])
        scores      = SOCScorer(occupations=encoded_occupations).score(
            assignments = assignments,
            raw_vectors = raw_vectors
        )
        assert np.argmax(scores[0]) == 0
        assert np.argmax(scores[1]) == 2

    def test_score_respects_occupation_boundaries(
        self,
        encoded_occupations : list[EncodedOccupation]
    ):
        """
        `np.maximum.reduceat` must segment by `owner_starts` so each column
        reflects only that occupation's tasks. A posting aligned with
        Beta's task axis should score higher against Beta than against
        Alpha or Gamma, whose task vectors are orthogonal to it.
        """
        raw_vectors = np.array([[0, 1.0, 0, 0]], dtype=np.float32)
        assignments = np.array([0])
        scores      = SOCScorer(occupations=encoded_occupations).score(
            assignments = assignments,
            raw_vectors = raw_vectors
        )[0]
        assert scores[1] > scores[0]
        assert scores[1] > scores[2]

    def test_score_cluster_order_matches_sorted_ids(
        self,
        encoded_occupations : list[EncodedOccupation]
    ):
        """
        Cluster rows appear in sorted assignment-id order regardless of
        the order in which posting labels arrive.
        """
        raw_vectors = np.array([
            [0, 0, 1.0, 0], [1.0, 0, 0, 0], [0, 0, 1.0, 0], [1.0, 0, 0, 0]
        ], dtype=np.float32)
        assignments = np.array([2, 0, 2, 0])
        scores      = SOCScorer(occupations=encoded_occupations).score(
            assignments = assignments,
            raw_vectors = raw_vectors
        )
        assert scores.shape == (2, 3)
        assert np.argmax(scores[0]) == 0
        assert np.argmax(scores[1]) == 2

    def test_score_rejects_empty_task_matrix(self):
        """
        `SOCScorer` refuses to construct when any occupation contributes
        zero task vectors, since `reduceat` has no defined behavior for
        empty segments and every SOC in the curated lexicon lists tasks.
        """
        occupations = [
            EncodedOccupation(
                occupation = make_occupation("Alpha", ["a"]),
                tasks      = normalize(np.array([[1.0, 0]]))
            ),
            EncodedOccupation(
                occupation = make_occupation("Empty", []),
                tasks      = np.empty((0, 2), dtype=np.float32)
            )
        ]
        with raises(ValueError, match="at least one task"):
            SOCScorer(occupations=occupations)
