"""
Tests for Plotly figure builders.

Validates that `Charts` produces figures with expected trace
counts, annotations, and structural invariants.
"""

from pytest import mark

from chalkline.display.charts import Charts


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

    def test_bar_with_line_overlay(self, charts: Charts):
        """
        Bar with `line` overlay adds a Scatter trace and a horizontal
        legend.
        """
        fig = charts.bar(
            data   = {"A": 10, "B": 20},
            height = 300,
            line   = {"A": 10, "B": 30},
            title  = "Count"
        ).to_dict()
        modes = [trace.get("mode") for trace in fig["data"]]
        assert len(fig["data"]) == 2
        assert "lines+markers" in modes
        assert fig["layout"]["legend"]["orientation"] == "h"

    def test_funnel_label_format(self, charts: Charts):
        """
        Funnel y-labels embed the stage count in parentheses with
        thousands separators.
        """
        y_labels = list(charts.funnel(
            height = 280,
            stages = {"Total": 12345, "Filtered": 678}
        ).to_dict()["data"][0]["y"])
        assert "Total (12,345)" in y_labels
        assert "Filtered (678)" in y_labels

    @mark.parametrize(("resume_coords", "expected_traces"), [
        ([],         1),
        ([0.1, 0.2], 2)
    ], ids=["no_resume", "with_resume"])
    def test_landscape_trace_count(
        self,
        charts          : Charts,
        resume_coords   : list,
        expected_traces : int
    ):
        """
        Landscape adds a resume marker trace only when coordinates
        are provided.
        """
        fig = charts.landscape(
            coordinates     = resume_coords,
            legend_families = "F",
            legend_resume   = "R",
            x_title         = "X",
            y_title         = "Y"
        )
        assert len(fig.to_dict()["data"]) == expected_traces

    def test_violin_filters_empty(self, charts: Charts):
        """
        Violin drops groups with no values, leaving only non-empty
        ones in the figure data.
        """
        fig = charts.violin(
            groups  = {"empty": [], "full": [1.0, 2.0, 3.0]},
            height  = 300,
            y_title = "Score"
        )
        names = [trace["name"] for trace in fig.to_dict()["data"]]
        assert names == ["full"]

