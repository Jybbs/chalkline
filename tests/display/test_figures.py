"""
Tests for Plotly figure builders.

Validates that `FigureBuilder` produces figures with expected trace
counts, annotations, and structural invariants.
"""

import plotly.graph_objects as go

from chalkline.display.figures  import FigureBuilder
from chalkline.pathways.schemas import Neighborhood


class TestFigureBuilder:
    """
    Validate figure construction from synthetic pipeline fixtures.
    """

    def test_dendrogram_annotation(self, figure_builder: FigureBuilder):
        """
        Dendrogram figure contains a "You" annotation for the
        matched cluster.
        """
        layout = figure_builder.dendrogram().to_dict()["layout"]
        assert layout["annotations"]
        assert "You" in [a["text"] for a in layout["annotations"]]

    def test_dendrogram_traces(self, figure_builder: FigureBuilder):
        """
        Dendrogram figure contains at least one line trace for the
        U-links.
        """
        traces = figure_builder.dendrogram().to_dict()["data"]
        assert len(traces) >= 1
        assert traces[0]["mode"] == "lines"

    def test_landscape_matched(self, figure_builder: FigureBuilder):
        """
        Landscape figure without coordinates contains the career
        families trace.
        """
        traces = figure_builder.landscape([]).to_dict()["data"]
        assert len(traces) >= 1
        assert traces[0]["name"] == "Career Families"

    def test_landscape_resume(self, figure_builder: FigureBuilder):
        """
        Landscape figure with coordinates adds a resume marker
        trace.
        """
        traces = figure_builder.landscape([0.1, 0.2]).to_dict()["data"]
        assert len(traces) == 2
        assert traces[1]["name"] == "Your Resume"

    def test_pathways_edges(
        self,
        figure_builder : FigureBuilder,
        neighborhood   : Neighborhood
    ):
        """
        Pathways figure contains edge and node traces.
        """
        target_id = figure_builder.cluster_ids[0]
        traces    = figure_builder.pathways(
            neighborhood, target_id
        ).to_dict()["data"]
        assert len(traces) >= 2

    def test_returns_go_figure(self, figure_builder: FigureBuilder):
        """
        All builder methods return `go.Figure` instances.
        """
        assert isinstance(figure_builder.dendrogram(), go.Figure)
        assert isinstance(figure_builder.landscape([]), go.Figure)
