"""
Skill vectorization into IDF-weighted and binary matrices.

Chains `DictVectorizer`, `TfidfTransformer`, and `Normalizer` in an
sklearn `Pipeline` that fits on extracted skill lists and produces both
an IDF-weighted matrix for the geometry track (PCA) and a binary
presence/absence matrix for the co-occurrence track (PMI).

Term frequency is always 1 because `SkillExtractor.extract()` returns
deduplicated canonical names, making the weighting effectively IDF with
L2 normalization rather than true TF-IDF. The fitted pipeline is
serializable via `joblib` for resume projection in CL-10, where the
resume must also use binary skill dicts to match the training
distribution.
"""

from functools                       import cached_property
from scipy.sparse                    import spmatrix
from sklearn.feature_extraction      import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline                import Pipeline
from sklearn.preprocessing           import Normalizer

from chalkline import SkillMap


class SkillVectorizer:
    """
    IDF-weighted and binary matrix builder from extracted skill
    lists.

    Receives the output of `SkillExtractor.extract()`, fits a
    three-stage sklearn `Pipeline`, and exposes both matrices
    alongside corpus-level statistics. Document identifiers are
    maintained in sorted order so that matrix rows map back to
    posting identities for downstream labeling and matching.

    Because the extractor deduplicates skills per posting, all term
    frequencies are 1 and `TfidfTransformer` applies only IDF
    weighting.
    """

    def __init__(self, skills: SkillMap):
        """
        Fit the vectorization pipeline on extracted skill lists.

        Args:
            skills: Mapping from document identifier to sorted canonical
                    skill names, as returned by `SkillExtractor.extract`.
        """
        self.document_ids = sorted(skills)

        self.dicts = [
            dict.fromkeys(skills[doc], 1)
            for doc in self.document_ids
        ]

        self.pipeline = Pipeline([
            ("vec",   DictVectorizer()),
            ("tfidf", TfidfTransformer(norm = None)),
            ("norm",  Normalizer())
        ])

        self.tfidf_matrix = self.pipeline.fit_transform(self.dicts)

    @cached_property
    def binary_matrix(self) -> spmatrix:
        """
        Binary presence/absence matrix for PMI computation.

        Uses only the `DictVectorizer` step, bypassing TF-IDF
        weighting and L2 normalization. Values are strictly 0 or
        1.

        Returns:
            Sparse matrix with binary skill presence per document.
        """
        return self.pipeline.named_steps["vec"].transform(self.dicts)

    @cached_property
    def feature_names(self) -> list[str]:
        """
        Vocabulary in column order, matching matrix column indices.

        Returns:
            Skill names ordered by their column position.
        """
        return self.pipeline.named_steps["vec"].get_feature_names_out().tolist()
