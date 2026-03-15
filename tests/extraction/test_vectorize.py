"""
Tests for skill vectorization into TF-IDF and binary matrices.

Validates matrix dimensions, binary constraints, vocabulary
consistency, joblib serialization, corpus statistics, and document
identifier ordering using synthetic extraction output.
"""

from joblib  import dump, load
from pathlib import Path
from pytest  import mark

from chalkline.extraction.vectorize import SkillVectorizer


class TestSkillVectorizer:
    """
    Validate vectorization pipeline, matrices, and statistics.
    """

    # ---------------------------------------------------------
    # Alignment
    # ---------------------------------------------------------

    @mark.parametrize("skills", [
        {
            "doc-a" : ["welding", "scaffolding"],
            "doc-b" : ["concrete finishing"]
        },
        {
            "z-doc" : ["welding"],
            "a-doc" : ["scaffolding", "welding"]
        }
    ])
    def test_matrix_alignment(self, skills: dict[str, list[str]]):
        """
        `document_ids[i]` and `feature_names[j]` correctly index the
        binary matrix regardless of insertion or sort order.
        """
        vec = SkillVectorizer(skills)
        for i, doc_id in enumerate(vec.document_ids):
            row     = vec.binary_matrix[i].toarray().flatten()
            present = {
                vec.feature_names[j]
                for j, v in enumerate(row) if v
            }
            assert present == set(skills[doc_id])

    # ---------------------------------------------------------
    # Feature names
    # ---------------------------------------------------------

    def test_transform_vocabulary(self, vectorizer: SkillVectorizer):
        """
        `DictVectorizer.transform(new_doc)` produces a vector with
        the same number of columns as `fit_transform()`.
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

    def test_binary_matrix_values(self, vectorizer: SkillVectorizer):
        """
        The binary matrix contains only 0s and 1s.
        """
        assert (vectorizer.binary_matrix.data == 1).all()

    def test_tfidf_differs_from_binary(self, vectorizer: SkillVectorizer):
        """
        TF-IDF weighting and L2 normalization produce values distinct from
        the raw binary presence/absence matrix.
        """
        assert (
            vectorizer.tfidf_matrix.toarray()
            != vectorizer.binary_matrix.toarray()
        ).any()

    # ---------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------

    def test_pipeline_persist(self, vectorizer: SkillVectorizer, tmp_path: Path):
        """
        The fitted pipeline serializes and restores via `joblib`, producing
        identical output on the same input.
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
