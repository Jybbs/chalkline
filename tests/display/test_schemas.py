"""
Tests for display-layer schemas and lazy-loading containers.
"""

from chalkline.display.schemas   import CareerTreemap, VarianceBreakdown
from chalkline.display.theme     import Theme
from chalkline.pathways.clusters import Clusters
from chalkline.pathways.loaders  import StakeholderReference


class TestCareerTreemap:
    """
    Validate treemap tile structure from cluster data.
    """

    def test_from_clusters(self, clusters: Clusters):
        """
        One header row per sector plus one tile per cluster, with
        empty parents on sector headers.
        """
        tm = CareerTreemap.from_clusters(clusters)
        assert len(tm.labels) == len(clusters.sectors) + len(clusters)
        assert all(p == "" for p in tm.parents[:len(clusters.sectors)])


class TestScoreColor:
    """
    Validate threshold-based color dispatch.
    """

    def test_score_bands(self, theme: Theme):
        """
        Scores below 40, between 40-70, and above 70 map to the
        error, primary, and success palette roles.
        """
        c = theme.colors
        assert theme.score_color(20) == c["error"]
        assert theme.score_color(55) == c["primary"]
        assert theme.score_color(85) == c["success"]


class TestStakeholderReference:
    """
    Validate lazy-loading and missing-file fallback.
    """

    def test_loads_json_on_access(self, tmp_path):
        """
        Attribute access deserializes the corresponding JSON file
        and caches the result.
        """
        (tmp_path / "trades.json").write_text('["electrician"]')
        ref = StakeholderReference(reference_dir=tmp_path)
        assert ref.trades == ["electrician"]
        assert ref.trades is ref.trades

    def test_missing_file_empty(self, tmp_path):
        """
        Accessing a name with no backing JSON file returns an
        empty list rather than raising.
        """
        ref = StakeholderReference(reference_dir=tmp_path)
        assert ref.nonexistent == []


class TestVarianceBreakdown:
    """
    Validate SVD variance percentage conversion.
    """

    def test_from_svd(self):
        """
        Ratios scale to percentages and accumulate into a
        cumulative trace.
        """
        vb = VarianceBreakdown.from_svd([0.35, 0.25, 0.15])
        assert vb.components == [35.0, 25.0, 15.0]
        assert vb.total == 75.0
        assert vb.labels == ["PC1", "PC2", "PC3"]
        assert vb.trace.y == [35.0, 60.0, 75.0]
