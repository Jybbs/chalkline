"""
Tests for candidate-set selection algorithms over the pathways domain.

Validates `SOCScorer` MaxSim semantics (a pooled-vector baseline would
silently misrank specialty SOCs against cluster centroids) and the
waste-aware Pareto-knee credential picker's coverage floor, fallback,
and determinism guarantees.
"""

import numpy as np

from pytest                import fixture, mark, raises
from sklearn.preprocessing import normalize

from chalkline.pathways.schemas   import EncodedOccupation, Occupation, Skill
from chalkline.pathways.schemas   import SelectorConfig, SelectorFrontier, SkillType
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


class TestSelectorFrontier:
    """
    Validate the `waste` property derives correctly from stored counts.
    """

    def test_waste_is_total_minus_filled(self):
        """
        `waste` equals `total_reach - gaps_filled` for any valid point.
        """
        point = SelectorFrontier(alpha=0.3, gaps_filled=7, picks=(0, 1), total_reach=12)
        assert point.waste == 5

    def test_waste_zero_when_perfectly_efficient(self):
        """
        A stack whose reach lands entirely on gaps has zero waste.
        """
        point = SelectorFrontier(alpha=0.0, gaps_filled=4, picks=(3,), total_reach=4)
        assert point.waste == 0


@fixture
def selector() -> CredentialSelector:
    """
    Default-config picker, reused across tests that do not exercise
    `coverage_floor` tuning.
    """
    return CredentialSelector()


class TestCredentialSelector:
    """
    Validate waste-aware selection under the Pareto-knee + coverage-floor
    rule across edge cases that would silently misrank credential stacks.
    """

    def test_floor_met_when_coverage_reachable(self, selector: CredentialSelector):
        """
        A pool that can reach 100% coverage satisfies the 80% floor and
        returns `floor_met=True`.
        """
        picks, floor_met = selector.select_stack(
            coverage  = {
                "A": {0: 0.6, 1: 0.5, 2: 0.4},
                "B": {3: 0.5, 4: 0.6}
            },
            gap_set   = frozenset({0, 1, 2, 3, 4}),
            max_picks = 3
        )
        assert floor_met
        assert {label for label, _ in picks} == {"A", "B"}

    def test_floor_missed_falls_back_to_best(self, selector: CredentialSelector):
        """
        When no stack can reach the floor, `floor_met=False` and the
        returned stack is whatever the frontier knee selects from the
        unconstrained Pareto curve.
        """
        picks, floor_met = selector.select_stack(
            coverage  = {"A": {0: 0.5}, "B": {1: 0.5}},
            gap_set   = frozenset(range(10)),
            max_picks = 3
        )
        assert floor_met is False
        assert len(picks) <= 3

    @mark.parametrize(
        "coverage, gap_set, max_picks",
        [
            ({},                              frozenset({0}),    3),
            ({"A": {0: 0.5}},                 frozenset(),       3),
            ({"A": {0: 0.5}, "B": {1: 0.5}},  frozenset({0, 1}), 0)
        ],
        ids = ["empty_coverage", "empty_gap_set", "zero_max_picks"]
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
        to `([], False)` rather than raising or producing a phantom stack.
        """
        assert selector.select_stack(
            coverage  = coverage,
            gap_set   = gap_set,
            max_picks = max_picks
        ) == ([], False)

    def test_scored_but_empty_dicts_excluded(self, selector: CredentialSelector):
        """
        Credentials with empty score dicts drop from the pool rather
        than appearing as zero-reach picks.
        """
        picks, _ = selector.select_stack(
            coverage  = {"A": {0: 0.5}, "B": {}, "C": {1: 0.5}},
            gap_set   = frozenset({0, 1}),
            max_picks = 3
        )
        assert "B" not in {label for label, _ in picks}

    def test_incremental_positions_are_non_overlapping(
        self,
        selector : CredentialSelector
    ):
        """
        Each pick's gap positions cover only the gaps it newly
        contributes, never double-counting a gap covered by an earlier
        pick in the same stack.
        """
        picks, _ = selector.select_stack(
            coverage  = {"A": {0: 0.5, 1: 0.5}, "B": {1: 0.5, 2: 0.5}},
            gap_set   = frozenset({0, 1, 2}),
            max_picks = 2
        )
        seen = set()
        for _, positions in picks:
            assert not (positions & seen)
            seen |= positions

    def test_incremental_positions_restricted_to_gaps(
        self,
        selector : CredentialSelector
    ):
        """
        Each pick's gained positions report only gap tasks, never
        leaking onto already-demonstrated tasks the credential also
        happens to reach.
        """
        picks, _ = selector.select_stack(
            coverage  = {"A": {0: 0.5, 5: 0.5}, "B": {1: 0.5, 6: 0.5}},
            gap_set   = frozenset({0, 1}),
            max_picks = 2
        )
        for _, positions in picks:
            assert positions <= {0, 1}

    def test_determinism_across_runs(self, selector: CredentialSelector):
        """
        Repeated calls on identical input return identical picks. The
        Kneedle algorithm and the α sweep must not introduce hidden
        randomness.
        """
        payload = dict(
            coverage  = {"A": {0: 0.5, 1: 0.5}, "B": {1: 0.5, 2: 0.5}, "C": {2: 0.5, 3: 0.5}},
            gap_set   = frozenset({0, 1, 2, 3}),
            max_picks = 3
        )
        assert selector.select_stack(**payload) == selector.select_stack(**payload)

    @mark.parametrize(
        "coverage_floor, expected_met",
        [(0.5, True), (1.0, False)],
        ids = ["lenient", "strict"]
    )
    def test_config_coverage_floor_threshold(
        self,
        coverage_floor : float,
        expected_met   : bool
    ):
        """
        Raising `coverage_floor` tightens the eligible region, so a pool
        that satisfies 0.5 may fail to satisfy 1.0.
        """
        picker = CredentialSelector(config=SelectorConfig(coverage_floor=coverage_floor))
        _, floor_met = picker.select_stack(
            coverage  = {"A": {0: 0.5}, "B": {1: 0.5}},
            gap_set   = frozenset({0, 1, 2, 3}),
            max_picks = 3
        )
        assert floor_met is expected_met
