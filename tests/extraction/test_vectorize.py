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
    # Binary matrix
    # ---------------------------------------------------------

    def test_binary_matrix_nonempty(
        self, skill_vectorizer: SkillVectorizer
    ):
        """
        The binary matrix has at least one non-zero entry from the
        synthetic extraction fixture.
        """
        assert skill_vectorizer.binary_matrix.nnz > 0

    def test_binary_matrix_values(self, skill_vectorizer: SkillVectorizer):
        """
        The binary matrix contains only 0s and 1s.
        """
        assert (skill_vectorizer.binary_matrix.data == 1).all()

    # ---------------------------------------------------------
    # Document identifiers
    # ---------------------------------------------------------

    def test_document_ids_match_row_count(
        self, skill_vectorizer: SkillVectorizer
    ):
        """
        Document identifiers are returned in row order alongside
        both matrices.
        """
        assert (
            len(skill_vectorizer.document_ids)
            == skill_vectorizer.tfidf_matrix.shape[0]
            == skill_vectorizer.binary_matrix.shape[0]
        )

    def test_document_ids_sorted(self, skill_vectorizer: SkillVectorizer):
        """
        Document identifiers are in sorted order.
        """
        assert skill_vectorizer.document_ids == sorted(
            skill_vectorizer.document_ids
        )

    # ---------------------------------------------------------
    # Feature names
    # ---------------------------------------------------------

    def test_feature_names_match_columns(
        self, skill_vectorizer: SkillVectorizer
    ):
        """
        Feature names length matches matrix column count.
        """
        assert (
            len(skill_vectorizer.feature_names)
            == skill_vectorizer.tfidf_matrix.shape[1]
        )

    def test_feature_names_sorted(self, skill_vectorizer: SkillVectorizer):
        """
        Feature names are in alphabetical order for stable column indices.
        """
        assert skill_vectorizer.feature_names == sorted(
            skill_vectorizer.feature_names
        )

    # ---------------------------------------------------------
    # Matrix dimensions
    # ---------------------------------------------------------

    def test_matrices_identical_dimensions(
        self, skill_vectorizer: SkillVectorizer
    ):
        """
        TF-IDF and binary matrices have the same shape.
        """
        assert (
            skill_vectorizer.tfidf_matrix.shape
            == skill_vectorizer.binary_matrix.shape
        )

    # ---------------------------------------------------------
    # Pipeline serialization
    # ---------------------------------------------------------

    def test_pipeline_serialization(
        self, skill_vectorizer: SkillVectorizer, tmp_path: Path
    ):
        """
        The fitted pipeline serializes and restores via `joblib`,
        producing identical output on the same input.
        """
        path = tmp_path / "pipeline.joblib"
        dump(skill_vectorizer.pipeline, path)
        test_dict = [{
            "scaffolding" : 1,
            "welding"     : 1
        }]
        assert (
            skill_vectorizer.pipeline.transform(test_dict)
            - load(path).transform(test_dict)
        ).nnz == 0

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def test_statistics_fields(self, skill_vectorizer: SkillVectorizer):
        """
        Corpus statistics report vocabulary size, sparsity, and
        per-posting skill counts.
        """
        stats = skill_vectorizer.statistics
        assert stats.vocabulary_size > 0
        assert 0 <= stats.matrix_sparsity <= 1
        assert stats.mean_skills_per_posting > 0
        assert len(stats.skill_frequency) == stats.vocabulary_size

    # ---------------------------------------------------------
    # TF-IDF values
    # ---------------------------------------------------------

    def test_tfidf_values_nonnegative(
        self, skill_vectorizer: SkillVectorizer
    ):
        """
        TF-IDF matrix values are non-negative after L2 normalization.
        """
        assert skill_vectorizer.tfidf_matrix.min() >= 0

    # ---------------------------------------------------------
    # Vocabulary consistency
    # ---------------------------------------------------------

    def test_transform_preserves_vocabulary(
        self, skill_vectorizer: SkillVectorizer
    ):
        """
        `DictVectorizer.transform(new_doc)` produces a vector with the
        same number of columns as `fit_transform()`.
        """
        assert skill_vectorizer.pipeline.named_steps["vec"].transform(
            [{
                "unknown_skill" : 1,
                "welding"       : 1
            }]
        ).shape[1] == skill_vectorizer.binary_matrix.shape[1]
