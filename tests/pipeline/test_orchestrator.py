"""
Tests for the pipeline orchestrator's composition helpers and
Pipeline class.

Validates that `compose_geometry` produces a fitted sklearn Pipeline,
that `build_profiles` enriches clusters with sector, Job Zone, and
reference data annotations, and that the `Pipeline` class enforces
pre-fit guards on `match()`.
"""

from sklearn.pipeline         import Pipeline as SklearnPipeline
from sklearn.utils.validation import check_is_fitted

from chalkline.association.cooccurrence import CooccurrenceNetwork
from chalkline.clustering.hierarchical  import HierarchicalClusterer
from chalkline.clustering.schemas       import ClusterLabel
from chalkline.extraction.occupations   import OccupationIndex
from chalkline.extraction.vectorize     import SkillVectorizer
from chalkline.matching.schemas         import MatchResult
from chalkline.pipeline.enrichment      import EnrichmentContext
from chalkline.pipeline.orchestrator    import build_profiles, compose_geometry
from chalkline.pipeline.orchestrator    import Pipeline
from chalkline.pipeline.schemas         import ApprenticeshipContext
from chalkline.pipeline.schemas         import ClusterProfile, PipelineConfig
from chalkline.pipeline.schemas         import ProgramRecommendation
from chalkline.reduction.pca            import PcaReducer

import pytest


# ---------------------------------------------------------------------
# compose_geometry
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# build_profiles
# ---------------------------------------------------------------------

class TestBuildProfiles:
    """
    Tests for cluster profile enrichment.
    """

    def test_profiles_nonempty(
        self,
        cluster_labels   : list[ClusterLabel],
        clusterer        : HierarchicalClusterer,
        enrichment       : EnrichmentContext,
        extracted_skills : dict[str, list[str]],
        occupation_index : OccupationIndex,
        sector_labels    : list[str]
    ):
        """
        Every cluster has a non-empty skill set in its profile.
        """
        profiles = build_profiles(
            cluster_labels   = cluster_labels,
            clusterer        = clusterer,
            enrichment       = enrichment,
            extracted_skills = extracted_skills,
            occupation_index = occupation_index,
            sector_labels    = sector_labels
        )
        assert len(profiles) > 0
        for profile in profiles.values():
            assert len(profile.skills) > 0

    def test_job_zone_bounded(
        self,
        cluster_labels   : list[ClusterLabel],
        clusterer        : HierarchicalClusterer,
        enrichment       : EnrichmentContext,
        extracted_skills : dict[str, list[str]],
        occupation_index : OccupationIndex,
        sector_labels    : list[str]
    ):
        """
        Job Zone values are in [1, 5] per O*NET specification.
        """
        profiles = build_profiles(
            cluster_labels   = cluster_labels,
            clusterer        = clusterer,
            enrichment       = enrichment,
            extracted_skills = extracted_skills,
            occupation_index = occupation_index,
            sector_labels    = sector_labels
        )
        for profile in profiles.values():
            assert 1 <= profile.job_zone <= 5

    def test_sector_assigned(
        self,
        cluster_labels   : list[ClusterLabel],
        clusterer        : HierarchicalClusterer,
        enrichment       : EnrichmentContext,
        extracted_skills : dict[str, list[str]],
        occupation_index : OccupationIndex,
        sector_labels    : list[str]
    ):
        """
        Every profile has a non-empty sector string.
        """
        profiles = build_profiles(
            cluster_labels   = cluster_labels,
            clusterer        = clusterer,
            enrichment       = enrichment,
            extracted_skills = extracted_skills,
            occupation_index = occupation_index,
            sector_labels    = sector_labels
        )
        for profile in profiles.values():
            assert profile.sector


# ---------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------

class TestPipeline:
    """
    Tests for the Pipeline orchestrator lifecycle.
    """

    def test_unfitted_guard(self, tmp_path):
        """
        Calling `match()` before `fit()` raises `RuntimeError`.
        """
        config = PipelineConfig(
            lexicon_dir  = tmp_path,
            output_dir   = tmp_path,
            postings_dir = tmp_path
        )
        pipe = Pipeline(config)
        with pytest.raises(RuntimeError, match="not fitted"):
            pipe.match("some resume text")

    def test_fitted_property(self, tmp_path):
        """
        A fresh pipeline reports `fitted` as `False`.
        """
        config = PipelineConfig(
            lexicon_dir  = tmp_path,
            output_dir   = tmp_path,
            postings_dir = tmp_path
        )
        assert not Pipeline(config).fitted

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
