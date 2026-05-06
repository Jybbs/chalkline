"""
Tests for the map widget tier assignment over the match-percent curve.

Covers the three branches of `_tier_assignments` (small corpora, flat
curves, kneed elbow) and the matched-always-tier-1 invariant that
drives hero card rendering.
"""

from chalkline.display.tabs.map.widget import _tier_assignments


class TestTierAssignments:
    """
    Branch coverage for the elbow-clamped tier picker.
    """

    def test_empty_input_returns_empty(self):
        """
        No clusters yields no tier assignments.
        """
        assert _tier_assignments({}, matched_id=0) == {}

    def test_few_candidates_promote_everyone(self):
        """
        With 10 or fewer non-matched candidates, every cluster lands
        in tier 1 because there is nothing to hide.
        """
        scores = {i: 0.9 - 0.05 * i for i in range(8)}
        tiers  = _tier_assignments(scores, matched_id=0)
        assert all(t == 1 for t in tiers.values())

    def test_flat_curve_caps_at_fifteen(self):
        """
        A flat match curve with more than 10 candidates bypasses
        kneed and uses the upper bound, capping tier 1 at 16
        (matched plus 15 others).
        """
        scores = {i: 0.5 for i in range(20)}
        tiers  = _tier_assignments(scores, matched_id=0)
        assert sum(t == 1 for t in tiers.values()) == 16

    def test_matched_always_tier_one(self):
        """
        The matched cluster is always tier 1 even when its score
        sits below the elbow over remaining candidates.
        """
        scores = {0: 0.05, 1: 0.9, 2: 0.85, 3: 0.4, 4: 0.1}
        tiers  = _tier_assignments(scores, matched_id=0)
        assert tiers[0] == 1

    def test_sharp_elbow_floors_at_ten(self):
        """
        A curve whose elbow falls below 10 still promotes 10
        non-matched neighbors so the canvas does not feel sparse.
        """
        scores = {0: 1.0}
        for i in range(1, 20):
            scores[i] = 0.9 if i <= 3 else 0.05
        tiers = _tier_assignments(scores, matched_id=0)
        assert sum(t == 1 for t in tiers.values()) == 11
