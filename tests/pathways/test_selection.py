"""
Tests for candidate-set selection algorithms over the pathways domain.

Validates `SOCScorer` MaxSim semantics (a pooled-vector baseline would
silently misrank specialty SOCs against cluster centroids) and the
waste-aware Pareto-knee credential picker's coverage behavior, edge
cases, and determinism guarantees.
"""

import numpy as np

from pytest                import fixture, mark, raises
from sklearn.preprocessing import normalize

from chalkline.pathways.schemas   import EncodedOccupation, Occupation, Skill, SkillType
from chalkline.pathways.selection import CredentialSelector, SOCScorer


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
            tasks      = normalize(np.array([
                [0, 1.0, 0,   0],
                [0, 0.8, 0.2, 0],
                [0, 0.5, 0.5, 0]
            ]))
        ),
        EncodedOccupation(
            occupation = make_occupation("Gamma",   ["g1"]),
            tasks      = normalize(np.array([[0, 0, 1.0, 0]]))
        )
    ]


@fixture
def selector() -> CredentialSelector:
    """
    Default-config picker, reused across tests.
    """
    return CredentialSelector(coverage_floor=0.80)


class TestCredentialSelector:
    """
    Validate waste-aware Pareto-knee selection across edge cases that
    would silently misrank credential stacks.
    """

    @mark.parametrize(
        "coverage, gap_set, max_picks",
        [
            ({},                              frozenset({0}),    3),
            ({"A": {0: 0.5}},                 frozenset(),       3),
            ({"A": {0: 0.5}, "B": {1: 0.5}},  frozenset({0, 1}), 0)
        ],
        ids=["empty_coverage", "empty_gap_set", "zero_max_picks"]
    )
    def test_degenerate_inputs_return_empty(
        self,
        selector  : CredentialSelector,
        coverage  : dict[str, dict[int, float]],
        gap_set   : frozenset[int],
        max_picks : int
    ):
        """
        Empty pool, empty gap set, and a zero pick cap each short-circuit
        to an empty list rather than raising or producing a phantom stack.
        """
        assert selector.select_stack(
            coverage  = coverage,
            gap_set   = gap_set,
            max_picks = max_picks
        ) == []

    def test_determinism_across_runs(self, selector: CredentialSelector):
        """
        Repeated calls on identical input return identical picks. The
        Pareto-knee greedy must not introduce hidden randomness.
        """
        coverage  = {
            "A": {0: 0.5, 1: 0.5},
            "B": {1: 0.5, 2: 0.5},
            "C": {2: 0.5, 3: 0.5}
        }
        gap_set   = frozenset({0, 1, 2, 3})
        max_picks = 3
        first  = selector.select_stack(coverage=coverage, gap_set=gap_set, max_picks=max_picks)
        second = selector.select_stack(coverage=coverage, gap_set=gap_set, max_picks=max_picks)
        assert first == second

    def test_drops_picks_with_zero_unique_gain(self, selector: CredentialSelector):
        """
        A credential whose gap coverage is fully subsumed by an
        already-picked credential adds no unique gap and should not
        appear in the output, even if it sits on the alpha-sweep frontier
        for some intermediate setting.
        """
        picks = selector.select_stack(
            coverage  = {
                "A": {0: 0.5, 1: 0.5, 2: 0.5},
                "B": {0: 0.4, 1: 0.4}
            },
            gap_set   = frozenset({0, 1, 2, 3}),
            max_picks = 2
        )
        assert {pick.label for pick in picks} == {"A"}

    def test_high_floor_forces_full_coverage(self):
        """
        A near-100% coverage floor forces the picker past intermediate
        knee points to the highest-coverage frontier point. Two
        credentials together cover all four gaps; either alone covers
        only two, so the floor disqualifies single-pick frontier points.
        """
        picks = CredentialSelector(coverage_floor=0.99).select_stack(
            coverage  = {
                "A": {0: 0.5, 1: 0.5},
                "B": {2: 0.5, 3: 0.5}
            },
            gap_set   = frozenset({0, 1, 2, 3}),
            max_picks = 2
        )
        assert {pick.label for pick in picks} == {"A", "B"}

    def test_incremental_positions_are_non_overlapping(
        self,
        selector : CredentialSelector
    ):
        """
        Each pick's gap positions cover only the gaps it newly
        contributes, never double-counting a gap covered by an earlier
        pick in the same stack.
        """
        picks = selector.select_stack(
            coverage  = {"A": {0: 0.5, 1: 0.5}, "B": {1: 0.5, 2: 0.5}},
            gap_set   = frozenset({0, 1, 2}),
            max_picks = 2
        )
        seen = set()
        for pick in picks:
            assert not (pick.positions & seen)
            seen |= pick.positions

    def test_incremental_positions_restricted_to_gaps(
        self,
        selector : CredentialSelector
    ):
        """
        Each pick's gained positions report only gap tasks, never
        leaking onto already-demonstrated tasks the credential also
        happens to reach.
        """
        picks = selector.select_stack(
            coverage  = {"A": {0: 0.5, 5: 0.5}, "B": {1: 0.5, 6: 0.5}},
            gap_set   = frozenset({0, 1}),
            max_picks = 2
        )
        for pick in picks:
            assert pick.positions <= {0, 1}

    def test_low_floor_admits_knee_pick(self):
        """
        A relaxed coverage floor allows the picker to settle at the
        Pareto knee even when more credentials could push coverage
        higher. With a 0% floor and a single high-coverage credential
        whose alternative pair contributes only marginal extra reach,
        the knee selects the single credential.
        """
        picks = CredentialSelector(coverage_floor=0.0).select_stack(
            coverage  = {
                "A": {0: 0.5, 1: 0.5, 2: 0.5},
                "B": {3: 0.5}
            },
            gap_set   = frozenset({0, 1, 2, 3}),
            max_picks = 2
        )
        assert "A" in {pick.label for pick in picks}

    def test_picks_two_disjoint_credentials(self, selector: CredentialSelector):
        """
        Two non-overlapping credentials should both surface when together
        they cover a wider portion of the gap set than either alone.
        """
        picks = selector.select_stack(
            coverage  = {
                "A": {0: 0.6, 1: 0.5, 2: 0.4},
                "B": {3: 0.5, 4: 0.6}
            },
            gap_set   = frozenset({0, 1, 2, 3, 4}),
            max_picks = 3
        )
        assert {pick.label for pick in picks} == {"A", "B"}

    def test_scored_but_empty_dicts_excluded(self, selector: CredentialSelector):
        """
        Credentials with empty score dicts drop from the pool rather
        than appearing as zero-reach picks.
        """
        picks = selector.select_stack(
            coverage  = {"A": {0: 0.5}, "B": {}, "C": {1: 0.5}},
            gap_set   = frozenset({0, 1}),
            max_picks = 3
        )
        assert "B" not in {pick.label for pick in picks}

    def test_single_pick_breaks_ties_by_least_waste(self, selector: CredentialSelector):
        """
        When two credentials cover the same gap count, the single-pick
        path prefers the one reaching fewer non-gap tasks.
        """
        picks = selector.select_stack(
            coverage  = {
                "narrow": {0: 0.5, 1: 0.5},
                "broad":  {0: 0.5, 1: 0.5, 9: 0.5, 10: 0.5}
            },
            gap_set   = frozenset({0, 1}),
            max_picks = 1
        )
        assert [pick.label for pick in picks] == ["narrow"]

    def test_single_pick_picks_max_gap_credential(self, selector: CredentialSelector):
        """
        For `max_picks == 1` the picker returns the credential filling
        the most gaps, never a low-coverage low-waste alternative. The
        Pareto-knee machinery would otherwise settle at the curve's
        elbow, which on a single-pick decision lands at lower coverage
        than the user's "single best credential" intuition.
        """
        picks = selector.select_stack(
            coverage  = {
                "high":  {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5},
                "mid":   {0: 0.5, 1: 0.5},
                "low":   {0: 0.5}
            },
            gap_set   = frozenset({0, 1, 2, 3}),
            max_picks = 1
        )
        assert [pick.label for pick in picks] == ["high"]

    def test_unreachable_floor_falls_back_to_highest_coverage(self):
        """
        When the coverage floor exceeds what any frontier point can
        reach, the picker falls back to the unconstrained frontier
        rather than returning empty.
        """
        picks = CredentialSelector(coverage_floor=0.99).select_stack(
            coverage  = {
                "A": {0: 0.5, 1: 0.5},
                "B": {2: 0.5}
            },
            gap_set   = frozenset({0, 1, 2, 3, 4}),
            max_picks = 2
        )
        assert picks
        assert {pick.label for pick in picks} <= {"A", "B"}


class TestSOCScorer:
    """
    Validate shape, ordering, and numerical correctness of MaxSim output.
    """

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
