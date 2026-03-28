"""
Tests for Plotly figure builders.

Validates that `Charts` produces figures with expected trace
counts, annotations, and structural invariants.
"""

import plotly.graph_objects as go

from chalkline.display.charts   import Charts
from chalkline.pathways.schemas import Reach


class TestCharts:
    """
    Validate figure construction from synthetic pipeline fixtures.
    """

    def test_dendrogram_annotation(self, charts: Charts):
        """
        Dendrogram figure contains a "You" annotation for the
        matched cluster.
        """
        layout = charts.dendrogram().to_dict()["layout"]
        assert layout["annotations"]
        assert "You" in [a["text"] for a in layout["annotations"]]

    def test_dendrogram_traces(self, charts: Charts):
        """
        Dendrogram figure contains at least one line trace for the
        U-links.
        """
        traces = charts.dendrogram().to_dict()["data"]
        assert len(traces) >= 1
        assert traces[0]["mode"] == "lines"

    def test_landscape_matched(self, charts: Charts):
        """
        Landscape figure without coordinates contains the career
        families trace.
        """
        traces = charts.landscape([]).to_dict()["data"]
        assert len(traces) >= 1
        assert traces[0]["name"] == "Career Families"

    def test_landscape_resume(self, charts: Charts):
        """
        Landscape figure with coordinates adds a resume marker
        trace.
        """
        traces = charts.landscape([0.1, 0.2]).to_dict()["data"]
        assert len(traces) == 2
        assert traces[1]["name"] == "Your Resume"

    def test_pathways_edges(self, charts: Charts, reach: Reach):
        """
        Pathways figure contains edge and node traces.
        """
        target_id = charts.cluster_ids[0]
        traces = charts.pathways(reach, target_id).to_dict()["data"]
        assert len(traces) >= 2

    def test_returns_go_figure(self, charts: Charts):
        """
        All builder methods return `go.Figure` instances.
        """
        assert isinstance(charts.dendrogram(), go.Figure)
        assert isinstance(charts.landscape([]), go.Figure)
