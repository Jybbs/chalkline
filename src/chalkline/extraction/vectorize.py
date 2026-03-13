"""
Skill vectorization into TF-IDF and binary matrices.

Chains `DictVectorizer`, `TfidfTransformer`, and `Normalizer` in an sklearn
`Pipeline` that fits on extracted skill lists and produces both a TF-IDF
matrix for the geometry track (PCA) and a binary presence/absence matrix for
the co-occurrence track (PMI). The fitted pipeline is serializable via
`joblib` for resume projection in CL-10.
"""

from collections                     import Counter
from functools                       import cached_property
from scipy.sparse                    import spmatrix
from sklearn.feature_extraction      import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline                import Pipeline
from sklearn.preprocessing           import Normalizer

from chalkline.extraction.schemas import CorpusStatistics


class SkillVectorizer:
    """
    TF-IDF and binary matrix builder from extracted skill lists.

    Receives the output of `SkillExtractor.extract()`, fits a three-stage
    sklearn `Pipeline`, and exposes both matrices alongside corpus-level
    statistics. Document identifiers are maintained in sorted order so
    that matrix rows map back to posting identities for downstream
    labeling and matching.
    """

    def __init__(self, skills: dict[str, list[str]]):
        """
        Fit the vectorization pipeline on extracted skill lists.

        Args:
            skills: Mapping from document identifier to sorted canonical
                    skill names, as returned by `SkillExtractor.extract`.
        """
        self.document_ids = sorted(skills)

        self._dicts = [
            dict.fromkeys(skills[doc], 1)
            for doc in self.document_ids
        ]

        self.pipeline = Pipeline([
            ("vec",   DictVectorizer()),
            ("tfidf", TfidfTransformer(norm = None)),
            ("norm",  Normalizer())
        ])

        self.tfidf_matrix = self.pipeline.fit_transform(self._dicts)

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @cached_property
    def binary_matrix(self) -> spmatrix:
        """
        Binary presence/absence matrix for PMI computation.

        Uses only the `DictVectorizer` step, bypassing TF-IDF weighting
        and L2 normalization. Values are strictly 0 or 1.
        """
        return self.pipeline.named_steps["vec"].transform(self._dicts)

    @cached_property
    def feature_names(self) -> list[str]:
        """
        Vocabulary in column order, matching matrix column indices.
        """
        return self.pipeline.named_steps["vec"].get_feature_names_out().tolist()

    @cached_property
    def statistics(self) -> CorpusStatistics:
        """
        Aggregate corpus statistics from the fitted vectorization.

        Reports vocabulary size, matrix sparsity, mean skills per posting,
        and per-skill frequency counts across the corpus.
        """
        binary     = self.binary_matrix
        rows, cols = binary.shape
        frequency  = Counter(skill for d in self._dicts for skill in d)

        return CorpusStatistics(
            matrix_sparsity         = 1 - binary.nnz / (rows * cols),
            mean_skills_per_posting = sum(map(len, self._dicts)) / len(self._dicts),
            skill_frequency         = dict(sorted(frequency.items())),
            vocabulary_size         = cols
        )
