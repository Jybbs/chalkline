"""
Tests for PCA dimensionality reduction via TruncatedSVD.

Validates component selection, coordinate output shape, and pipeline
serializability using synthetic extraction output.
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

    def test_max_components_capped(self, vectorizer: SkillVectorizer):
        """
        Requesting more components than the matrix rank caps to
        `min(n_samples, n_features) - 1` without error.
        """
        matrix  = vectorizer.tfidf_matrix
        reducer = PcaReducer(
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
            max_components     = min(vectorizer.tfidf_matrix.shape) - 1,
            random_seed        = 42,
            tfidf_matrix       = vectorizer.tfidf_matrix,
            variance_threshold = 0.01
        )
        assert reducer.n_selected == 1

    def test_coordinates_shape(
        self,
        pca_reducer : PcaReducer,
        vectorizer  : SkillVectorizer
    ):
        """
        Output array has shape (n_postings, n_selected).
        """
        assert pca_reducer.coordinates.shape == (
            len(vectorizer.document_ids),
            pca_reducer.n_selected
        )

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
