"""
Tests for the pipeline orchestrator's composition helpers and
`Chalkline` dataclass.

Validates that `compose_geometry` produces a fitted sklearn Pipeline,
that `build_profiles` enriches clusters with sector, Job Zone, and
reference data annotations, and that `match()` returns correctly
shaped results.
"""

from sklearn.pipeline          import Pipeline as SklearnPipeline
from sklearn.utils.validation  import check_is_fitted

from chalkline.extraction.occupations import OccupationIndex
from chalkline.extraction.vectorize   import SkillVectorizer
from chalkline.matching.schemas       import MatchResult
from chalkline.pipeline.orchestrator  import compose_geometry
from chalkline.pipeline.schemas       import ClusterProfile
from chalkline.reduction.pca          import PcaReducer


class TestComposeGeometry:
    """
    Tests for the five-step geometry pipeline composition.
    """

    def test_returns_pipeline(
        self,
        pca_reducer : PcaReducer,
        vectorizer  : SkillVectorizer
    ):
        """
        The composed pipeline is an sklearn `Pipeline` instance.
        """
        result = compose_geometry(
            reducer    = pca_reducer,
            vectorizer = vectorizer
        )
        assert isinstance(result, SklearnPipeline)

    def test_fitted(
        self,
        pca_reducer : PcaReducer,
        vectorizer  : SkillVectorizer
    ):
        """
        The composed pipeline passes `check_is_fitted` without
        raising.
        """
        result = compose_geometry(
            reducer    = pca_reducer,
            vectorizer = vectorizer
        )
        check_is_fitted(result)

    def test_transform_shape(
        self,
        pca_reducer : PcaReducer,
        vectorizer  : SkillVectorizer
    ):
        """
        Transforming a single skill dict produces coordinates with
        the expected number of PCA components.
        """
        geo    = compose_geometry(
            reducer    = pca_reducer,
            vectorizer = vectorizer
        )
        coords = geo.transform([{"fall protection": 1, "welding": 1}])
        assert coords.shape == (1, pca_reducer.n_selected)


class TestBuildProfiles:
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


class TestComputeSectorLabels:
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
