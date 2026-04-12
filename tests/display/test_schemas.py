"""
Tests for display-layer schemas and lazy-loading containers.
"""

from chalkline.display.schemas  import SectionContent, TabContent, VarianceBreakdown
from chalkline.display.theme    import Theme
from chalkline.pathways.loaders import StakeholderReference


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


class TestTabContent:
    """
    Validate section formatting and tuple ordering.
    """

    def test_section(self):
        """
        `section()` returns (description, title) to match `header()`'s
        alphabetized parameter order, with `{n}` substitution applied
        to both fields.
        """
        content = TabContent(sections={
            "overview": SectionContent(
                description = "Found {n} clusters",
                title       = "Overview of {n}"
            )
        })
        description, title = content.section("overview", n=21)
        assert description == "Found 21 clusters"
        assert title       == "Overview of 21"


class TestVarianceBreakdown:
    """
    Validate SVD variance percentage conversion.
    """

    def test_cumulative(self):
        """
        Cumulative percentages are a running sum of the per-component
        percentages, rounded to two decimals at each step.
        """
        vb = VarianceBreakdown.from_svd([0.35, 0.25, 0.15])
        assert vb.cumulative      == [35.0, 60.0, 75.0]
        assert vb.cumulative_dict == {"PC1": 35.0, "PC2": 60.0, "PC3": 75.0}
        assert vb.components_dict == {"PC1": 35.0, "PC2": 25.0, "PC3": 15.0}

    def test_from_svd(self):
        """
        Ratios scale to percentages with correct total and component
        labels.
        """
        vb = VarianceBreakdown.from_svd([0.35, 0.25, 0.15])
        assert vb.components == [35.0, 25.0, 15.0]
        assert vb.total == 75.0
        assert vb.labels == ["PC1", "PC2", "PC3"]
