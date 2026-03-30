"""
Tests for Plotly figure builders.

Validates that `Charts` produces figures with expected trace
counts, annotations, and structural invariants.
"""

from chalkline.display.charts   import Charts
from chalkline.pathways.schemas import Reach


class TestCharts:
    """
    Validate figure construction from synthetic pipeline fixtures.
    """

    def test_bar_horizontal(self, charts: Charts):
        """
        Horizontal bar sets x-axis title and reverses the y-axis.
        """
        layout = charts.bar(
            height     = 300,
            horizontal = True,
            title      = "Count",
            x          = [10, 20],
            y          = ["A", "B"]
        ).to_dict()["layout"]
        assert layout["xaxis"]["title"]["text"] == "Count"
        assert layout["yaxis"]["autorange"] == "reversed"

    def test_bar_vertical(self, charts: Charts):
        """
        Vertical bar sets y-axis title with no reversed axis.
        """
        layout = charts.bar(
            height = 300,
            title  = "Count",
            x      = ["A", "B"],
            y      = [10, 20]
        ).to_dict()["layout"]
        assert layout["yaxis"]["title"]["text"] == "Count"

    def test_dendrogram_annotation(self, charts: Charts):
        """
        Dendrogram annotation text appears in the figure layout.
        """
        layout = charts.dendrogram(
            annotation_text = "Here",
            title           = "D",
            x_title         = "X",
            y_title         = "Y"
        ).to_dict()["layout"]
        texts = [a["text"] for a in layout["annotations"]]
        assert "Here" in texts

    def test_dendrogram_traces(self, charts: Charts):
        """
        Dendrogram contains at least one line trace for the
        U-links.
        """
        fig = charts.dendrogram(
            annotation_text = "X",
            title           = "D",
            x_title         = "X",
            y_title         = "Y"
        )
        assert fig.to_dict()["data"][0]["mode"] == "lines"

    def test_landscape_resume_trace(self, charts: Charts):
        """
        Landscape with coordinates adds a second trace for the
        resume marker.
        """
        fig = charts.landscape(
            coordinates     = [0.1, 0.2],
            legend_families = "F",
            legend_resume   = "R",
            title           = "L",
            x_title         = "X",
            y_title         = "Y"
        )
        assert len(fig.to_dict()["data"]) == 2

    def test_pathways_edges(self, charts: Charts, reach: Reach):
        """
        Pathways figure contains edge and node traces.
        """
        fig = charts.pathways(
            reach      = reach,
            target_id  = charts.cluster_ids[0],
            title      = "P"
        )
        assert len(fig.to_dict()["data"]) >= 2

