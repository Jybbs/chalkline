"""
Tests for skill vectorization into TF-IDF and binary matrices.

Validates matrix dimensions, binary constraints, vocabulary consistency,
joblib serialization, corpus statistics, and document identifier ordering
using synthetic extraction output.
"""

from joblib  import dump, load
from pathlib import Path

from chalkline.extraction.vectorize import SkillVectorizer


class TestSkillVectorizer:
    """
    Validate vectorization pipeline, matrices, and statistics.
    """

    # ---------------------------------------------------------
    # Document identifiers
    # ---------------------------------------------------------

    def test_document_ids_row_count(self, vectorizer: SkillVectorizer):
        """
        Document identifiers are returned in row order alongside
        both matrices.
        """
        assert (
            len(vectorizer.document_ids)
            == vectorizer.tfidf_matrix.shape[0]
            == vectorizer.binary_matrix.shape[0]
        )

    def test_document_ids_sorted(self, vectorizer: SkillVectorizer):
        """
        Document identifiers are in sorted order.
        """
        assert vectorizer.document_ids == sorted(
            vectorizer.document_ids
        )

    # ---------------------------------------------------------
    # Feature names
    # ---------------------------------------------------------

    def test_feature_names_columns(self, vectorizer: SkillVectorizer):
        """
        Feature names length matches matrix column count.
        """
        assert (
            len(vectorizer.feature_names)
            == vectorizer.tfidf_matrix.shape[1]
        )

    def test_feature_names_sorted(self, vectorizer: SkillVectorizer):
        """
        Feature names are in alphabetical order for stable column indices.
        """
        assert vectorizer.feature_names == sorted(
            vectorizer.feature_names
        )

    def test_transform_vocabulary(self, vectorizer: SkillVectorizer):
        """
        `DictVectorizer.transform(new_doc)` produces a vector with the
        same number of columns as `fit_transform()`.
        """
        assert vectorizer.pipeline.named_steps["vec"].transform(
            [{
                "unknown_skill" : 1,
                "welding"       : 1
            }]
        ).shape[1] == vectorizer.binary_matrix.shape[1]

    # ---------------------------------------------------------
    # Matrices
    # ---------------------------------------------------------

    def test_binary_matrix_nonempty(self, vectorizer: SkillVectorizer):
        """
        The binary matrix has at least one non-zero entry from the
        synthetic extraction fixture.
        """
        assert vectorizer.binary_matrix.nnz > 0

    def test_binary_matrix_values(self, vectorizer: SkillVectorizer):
        """
        The binary matrix contains only 0s and 1s.
        """
        assert (vectorizer.binary_matrix.data == 1).all()

    def test_matrices_dimensions(self, vectorizer: SkillVectorizer):
        """
        TF-IDF and binary matrices have the same shape.
        """
        assert (
            vectorizer.tfidf_matrix.shape
            == vectorizer.binary_matrix.shape
        )

    def test_tfidf_differs_from_binary(self, vectorizer: SkillVectorizer):
        """
        TF-IDF weighting and L2 normalization produce values distinct
        from the raw binary presence/absence matrix.
        """
        assert (
            vectorizer.tfidf_matrix.toarray()
            != vectorizer.binary_matrix.toarray()
        ).any()

    def test_tfidf_values_nonnegative(self, vectorizer: SkillVectorizer):
        """
        TF-IDF matrix values are non-negative after L2 normalization.
        """
        assert vectorizer.tfidf_matrix.min() >= 0

    # ---------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------

    def test_pipeline_persist(self, vectorizer: SkillVectorizer, tmp_path: Path):
        """
        The fitted pipeline serializes and restores via `joblib`,
        producing identical output on the same input.
        """
        dump(vectorizer.pipeline, path := tmp_path / "pipeline.joblib")
        test_dict = [{
            "scaffolding" : 1,
            "welding"     : 1
        }]
        assert (
            vectorizer.pipeline.transform(test_dict)
            - load(path).transform(test_dict)
        ).nnz == 0

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def test_single_posting(self):
        """
        A single-document corpus produces valid matrices where TF-IDF
        normalization degenerates to uniform weights.
        """
        vec = SkillVectorizer({"only": ["scaffolding", "welding"]})
        assert vec.tfidf_matrix.shape[0] == 1
        assert vec.binary_matrix.nnz == 2
        assert vec.statistics.mean_skills_per_posting == 2.0

    def test_statistics_fields(self, vectorizer: SkillVectorizer):
        """
        Corpus statistics report vocabulary size, sparsity, and
        per-posting skill counts.
        """
        stats = vectorizer.statistics
        assert stats.vocabulary_size > 0
        assert 0 <= stats.matrix_sparsity <= 1
        assert stats.mean_skills_per_posting > 0
        assert len(stats.skill_frequency) == stats.vocabulary_size

    def test_statistics_frequency(self, vectorizer: SkillVectorizer):
        """
        Per-skill frequency counts reflect actual document occurrences.
        """
        freq = vectorizer.statistics.skill_frequency
        assert all(v >= 1 for v in freq.values())
        assert sum(freq.values()) >= len(vectorizer.document_ids)
