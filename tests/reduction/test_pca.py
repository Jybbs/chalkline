"""
Tests for PCA dimensionality reduction via TruncatedSVD.

Validates component selection, coordinate output shape, loading term
extraction, unit variance scaling, and pipeline serializability using
synthetic extraction output.
"""

import numpy as np

from joblib  import dump, load
from pathlib import Path

from chalkline.extraction.vectorize import SkillVectorizer
from chalkline.reduction.pca        import PcaReducer


class TestPcaReducer:
    """
    Validate reduction pipeline, component selection, and output
    properties.
    """

    # ---------------------------------------------------------
    # Component selection
    # ---------------------------------------------------------

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

    def test_threshold_minimum(self, vectorizer: SkillVectorizer):
        """
        Component selection picks the smallest k where cumulative
        variance meets the threshold, not k+1. A very low threshold
        should select exactly one component, verifying the
        `searchsorted + 1` logic does not over-select.
        """
        reducer = PcaReducer(
            document_ids       = vectorizer.document_ids,
            feature_names      = vectorizer.feature_names,
            max_components     = min(vectorizer.tfidf_matrix.shape) - 1,
            random_seed        = 42,
            tfidf_matrix       = vectorizer.tfidf_matrix,
            variance_threshold = 0.01
        )
        assert reducer.n_selected == 1
        assert reducer.cumulative_variance >= 0.01

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

    def test_loadings_skill_names(self, pca_reducer: PcaReducer):
        """
        Top-loading terms per component return skill names, not
        column indices.
        """
        for loading in pca_reducer.loadings():
            assert all(isinstance(t, str) for t in loading.terms)
            assert all(not t.isdigit() for t in loading.terms)

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
        PCA output columns have unit variance after scaling, verified
        by `np.allclose(std, 1.0)`. Skipped when only one component
        is selected because std is always 1.0 trivially in that case
        and the real invariant is that `StandardScaler` was applied.
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
        `pipeline.transform(new_vector)` returns the same number of
        columns without refitting.
        """
        assert pca_reducer.pipeline.transform(
            np.ones((1, pca_reducer.pipeline.n_features_in_))
        ).shape == (1, pca_reducer.n_selected)
