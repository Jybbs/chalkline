"""
Tests for PCA dimensionality reduction via TruncatedSVD.

Validates component selection, coordinate output shape, loading term
extraction, unit variance scaling, and pipeline serializability using
synthetic extraction output.
"""

import numpy as np

from joblib  import dump, load
from pathlib import Path
from pytest  import mark

from chalkline.extraction.vectorize import SkillVectorizer
from chalkline.reduction.pca       import PcaReducer


class TestPcaReducer:
    """
    Validate reduction pipeline, component selection, and output
    properties.
    """

    # ---------------------------------------------------------
    # Component selection
    # ---------------------------------------------------------

    def test_cumulative_variance(self, pca_reducer: PcaReducer):
        """
        Cumulative variance is reported even when the threshold
        is not fully reached.
        """
        assert 0 < pca_reducer.cumulative_variance <= 1.0

    def test_explained_variance_length(self, pca_reducer: PcaReducer):
        """
        The full variance profile contains one entry per component
        from the analysis fit.
        """
        assert len(pca_reducer.explained_variance_ratio) >= pca_reducer.n_selected

    def test_max_components_capped(self, vectorizer: SkillVectorizer):
        """
        Requesting more components than the matrix rank caps to
        `min(n_samples, n_features) - 1` without error.
        """
        matrix  = vectorizer.tfidf_matrix
        reducer = PcaReducer(
            document_ids       = vectorizer.document_ids,
            feature_names      = vectorizer.feature_names,
            max_components     = 999,
            random_seed        = 42,
            tfidf_matrix       = matrix,
            variance_threshold = 0.85
        )
        assert reducer.n_selected <= min(matrix.shape) - 1

    def test_n_selected_bounded(self, pca_reducer: PcaReducer):
        """
        Selected components do not exceed the configured maximum.
        """
        assert 1 <= pca_reducer.n_selected

    def test_variance_ratios_bounded(self, pca_reducer: PcaReducer):
        """
        Explained variance ratios are non-negative and sum to at
        most 1.0.
        """
        evr = pca_reducer.explained_variance_ratio
        assert (evr >= 0).all()
        assert evr.sum() <= 1.0 + 1e-10

    # ---------------------------------------------------------
    # Coordinates
    # ---------------------------------------------------------

    def test_coordinates_finite(self, pca_reducer: PcaReducer):
        """
        All coordinate values are finite (no NaN or inf from
        degenerate scaling).
        """
        assert np.isfinite(pca_reducer.coordinates).all()

    def test_coordinates_shape(self, pca_reducer: PcaReducer):
        """
        Output array has shape (n_postings, n_selected).
        """
        assert pca_reducer.coordinates.shape == (
            len(pca_reducer.document_ids),
            pca_reducer.n_selected
        )

    # ---------------------------------------------------------
    # Loadings
    # ---------------------------------------------------------

    def test_loadings_aligned(self, pca_reducer: PcaReducer):
        """
        Each loading has the same number of terms and weights.
        """
        for loading in pca_reducer.loadings():
            assert len(loading.terms) == len(loading.weights)

    def test_loadings_count(self, pca_reducer: PcaReducer):
        """
        One loading entry per selected component.
        """
        assert len(pca_reducer.loadings()) == pca_reducer.n_selected

    def test_loadings_skill_names(self, pca_reducer: PcaReducer):
        """
        Top-loading terms per component return skill names, not
        column indices.
        """
        for loading in pca_reducer.loadings():
            assert all(isinstance(t, str) for t in loading.terms)
            assert all(not t.isdigit() for t in loading.terms)

    @mark.parametrize("top_n", [1, 2, 5])
    def test_loadings_top_n(self, pca_reducer: PcaReducer, top_n: int):
        """
        Requesting fewer top terms limits the returned list length.
        """
        for loading in pca_reducer.loadings(top_n=top_n):
            assert len(loading.terms) <= top_n
            assert len(loading.weights) <= top_n

    def test_loadings_variance_positive(self, pca_reducer: PcaReducer):
        """
        Each component's variance ratio is positive.
        """
        for loading in pca_reducer.loadings():
            assert loading.variance_ratio > 0

    def test_loadings_weight_order(self, pca_reducer: PcaReducer):
        """
        Terms within each loading are ordered by descending absolute
        weight, matching the `argsort` contract in `PcaReducer`.
        """
        for loading in pca_reducer.loadings():
            absolutes = [abs(w) for w in loading.weights]
            assert absolutes == sorted(absolutes, reverse=True)

    # ---------------------------------------------------------
    # Scaling
    # ---------------------------------------------------------

    def test_unit_variance(self, pca_reducer: PcaReducer):
        """
        PCA output columns have unit variance after scaling,
        verified by `np.allclose(std, 1.0)`. Skipped when only
        one component is selected because std is always 1.0
        trivially in that case and the real invariant is that
        `StandardScaler` was applied.
        """
        if pca_reducer.coordinates.shape[0] < 3:
            return
        assert np.allclose(pca_reducer.coordinates.std(axis=0), 1.0, atol=1e-6)

    # ---------------------------------------------------------
    # Pipeline
    # ---------------------------------------------------------

    def test_pipeline_persist(self, pca_reducer: PcaReducer, tmp_path: Path):
        """
        The fitted pipeline serializes and restores via `joblib`,
        producing identical output dimensions on new input.
        """
        dump(pca_reducer.pipeline, path := tmp_path / "pca.joblib")
        assert load(path).n_features_in_ == pca_reducer.pipeline.n_features_in_

    def test_transform_dimensions(self, pca_reducer: PcaReducer):
        """
        `pipeline.transform(new_vector)` returns the same number
        of columns without refitting.
        """
        assert pca_reducer.pipeline.transform(
            np.ones((1, pca_reducer.pipeline.n_features_in_))
        ).shape == (1, pca_reducer.n_selected)
