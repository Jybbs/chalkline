"""
Tests for Hamilton DAG node functions in `steps`.

Validates that `geometry_pipeline` produces a fitted sklearn
Pipeline, that `profiles` enriches clusters with sector, Job Zone,
and reference data annotations, and that `sector_labels` produces
valid SOC codes aligned with the vectorizer document order.
"""

from pytest import raises

from sklearn.pipeline         import Pipeline
from sklearn.utils.validation import check_is_fitted

from chalkline.extraction.occupations import OccupationIndex
from chalkline.extraction.vectorize   import SkillVectorizer
from chalkline.pipeline.schemas       import ClusterProfile, PipelineConfig
from chalkline.pipeline.steps         import corpus
from chalkline.reduction.pca          import PcaReducer


class TestCorpus:
    """
    Tests for the corpus loading node.
    """

    def test_empty_postings(self, tmp_path):
        """
        An empty postings directory raises `FileNotFoundError`.
        """
        config = PipelineConfig(
            lexicon_dir  = tmp_path,
            output_dir   = tmp_path,
            postings_dir = tmp_path
        )
        with raises(FileNotFoundError, match="No postings"):
            corpus(config)


class TestGeometryPipeline:
    """
    Tests for the five-step geometry pipeline composition.
    """

    def test_returns_pipeline(self, geometry_pipeline: Pipeline):
        """
        The composed pipeline is an sklearn `Pipeline` instance.
        """
        assert isinstance(geometry_pipeline, Pipeline)

    def test_fitted(self, geometry_pipeline: Pipeline):
        """
        The composed pipeline passes `check_is_fitted` without
        raising.
        """
        check_is_fitted(geometry_pipeline)

    def test_transform_shape(
        self,
        geometry_pipeline : Pipeline,
        pca_reducer       : PcaReducer
    ):
        """
        Transforming a single skill dict produces coordinates with
        the expected number of PCA components.
        """
        coords = geometry_pipeline.transform(
            [{"fall protection": 1, "welding": 1}]
        )
        assert coords.shape == (1, pca_reducer.n_selected)


class TestProfiles:
    """
    Tests for cluster profile enrichment.
    """

    def test_profiles_nonempty(self, profiles: dict[int, ClusterProfile]):
        """
        Every cluster has a non-empty skill set in its profile.
        """
        assert len(profiles) > 0
        for profile in profiles.values():
            assert len(profile.skills) > 0

    def test_job_zone_bounded(self, profiles: dict[int, ClusterProfile]):
        """
        Job Zone values are in [1, 5] per O*NET specification.
        """
        for profile in profiles.values():
            assert 1 <= profile.job_zone <= 5

    def test_sector_assigned(self, profiles: dict[int, ClusterProfile]):
        """
        Every profile has a non-empty sector string.
        """
        for profile in profiles.values():
            assert profile.sector


class TestSectorLabels:
    """
    Tests for SOC code assignment via Jaccard-nearest occupation.
    """

    def test_labels_aligned(
        self,
        sector_labels : list[str],
        vectorizer    : SkillVectorizer
    ):
        """
        One SOC code per document, aligned with vectorizer row order.
        """
        assert len(sector_labels) == len(vectorizer.document_ids)

    def test_codes_valid(
        self,
        occupation_index : OccupationIndex,
        sector_labels    : list[str]
    ):
        """
        Every returned SOC code exists in the occupation index.
        """
        for soc in sector_labels:
            assert occupation_index.get(soc) is not None
