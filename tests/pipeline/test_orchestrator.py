"""
Tests for the `Chalkline` dataclass lifecycle.

Validates that `match()` returns correctly shaped results using
cached transforms from the fitted pipeline.
"""

from chalkline.matching.schemas import MatchResult
from chalkline.reduction.pca    import PcaReducer


class TestChalkline:
    """
    Tests for the `Chalkline` dataclass lifecycle.
    """

    def test_match_result_has_coords(
        self,
        match_result: MatchResult,
        pca_reducer : PcaReducer
    ):
        """
        A match result includes PCA coordinates with the expected
        number of components.
        """
        assert len(match_result.pca_coordinates) == pca_reducer.n_selected
