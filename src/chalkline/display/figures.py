"""
Plotly figure builders for the career report panels.

Provides a `FigureBuilder` that holds the fitted pathway graph,
matched cluster ID, and a theme callable, exposing `dendrogram`,
`landscape`, and `pathways` methods that each return a configured
`go.Figure`.
"""

import networkx as nx
import numpy    as np

import plotly.graph_objects as go

from typing import Callable

from chalkline.pathways.graph   import CareerPathwayGraph
from chalkline.pathways.schemas import Neighborhood


class FigureBuilder:
    """
    Stateful Plotly figure builder for the career report.

    Captures the pathway graph, matched cluster ID, and a theme
    callable at construction so that individual figure methods
    receive only per-panel arguments. The theme callable is
    evaluated at render time so dark/light toggles take effect.
    """

    def __init__(
        self,
        matched_id : int,
        pathway    : CareerPathwayGraph,
        theme      : Callable[[], str]
    ):
        """
        Args:
            matched_id : Matched cluster ID for highlighting.
            pathway    : Fitted career pathway graph.
            theme      : Callable returning a Plotly template name.
        """
        self.cluster_ids = sorted(pathway.clusters)
        self.matched_id  = matched_id
        self.pathway     = pathway
        self.theme       = theme

    def _node_colors(self, node_ids: np.ndarray, **highlights: int) -> np.ndarray:
        """
        Assign colors to nodes, with crimson for the matched
        cluster and optional additional highlights.

        Args:
            node_ids     : Array of cluster IDs.
            **highlights : Color name to cluster ID pairs
                           (e.g., `gold=target_id`).

        Returns:
            Array of color strings aligned with `node_ids`.
        """
        colors = np.where(node_ids == self.matched_id, "crimson", "steelblue")
        for color, cid in highlights.items():
            colors = np.where(node_ids == cid, color, colors)
        return colors

    def _node_hover(self, node_ids: list[int]) -> list[str]:
        """
        Build hover text for cluster nodes.

        Args:
            node_ids: Cluster IDs to generate hover text for.

        Returns:
            HTML-formatted hover strings.
        """
        return [
            f"Cluster {cid}<br>{self.pathway.clusters[cid].soc_title}<br>"
            f"JZ {self.pathway.clusters[cid].job_zone} · "
            f"{self.pathway.clusters[cid].size} postings"
            for cid in node_ids
        ]

    def dendrogram(self) -> go.Figure:
        """
        Ward-linkage dendrogram with the matched cluster annotated.

        Builds the dendrogram from scipy's `dendrogram` with
        `no_plot=True`, then renders the U-links as a single Plotly
        line trace. Normalizes centroids before linkage so that
        Ward distances reflect direction rather than magnitude.

        Returns:
            Configured dendrogram figure.
        """
        from scipy.cluster.hierarchy import dendrogram, linkage
        from sklearn.preprocessing   import normalize

        result = dendrogram(
            linkage(normalize(self.pathway.centroids), method="ward"),
            labels  = [f"C{cid}" for cid in self.cluster_ids],
            no_plot = True
        )

        fig = go.Figure(go.Scatter(
            hoverinfo  = "skip",
            line       = dict(color="steelblue", width=1.5),
            mode       = "lines",
            showlegend = False,
            x          = [v for ic in result["icoord"] for v in [*ic, None]],
            y          = [v for dc in result["dcoord"] for v in [*dc, None]]
        ))

        fig.add_annotation(
            arrowcolor = "crimson",
            arrowhead  = 2,
            arrowwidth = 2,
            font       = dict(color="crimson", size=12),
            showarrow  = True,
            text       = "You",
            x          = 5 + (ivl := result["ivl"]).index(f"C{self.matched_id}") * 10,
            y          = 0,
            yshift     = -20
        )

        fig.update_layout(
            height      = 500,
            template    = self.theme(),
            title       = "Hierarchical Clustering Dendrogram",
            xaxis       = dict(
                ticktext = ivl,
                tickvals = [5 + i * 10 for i in range(len(ivl))]
            ),
            xaxis_title = "Career Family",
            yaxis_title = "Ward Distance"
        )
        return fig

    def landscape(self, coordinates: list[float]) -> go.Figure:
        """
        Scatter plot of cluster centroids in SVD space with the
        resume position overlaid.

        Node sizes scale with betweenness centrality so that
        gateway clusters bridging multiple career families appear
        larger. The matched cluster is highlighted in crimson and
        the resume position rendered as a gold star.

        Args:
            coordinates: Resume SVD position, empty if unavailable.

        Returns:
            Configured landscape scatter figure.
        """
        betweenness = nx.betweenness_centrality(self.pathway.graph, weight="weight")
        cx = (c := self.pathway.centroids)[:, 0]
        cy = c[:, 1] if c.shape[1] > 1 else np.zeros(len(cx))

        traces = [go.Scatter(
            hovertext = self._node_hover(self.cluster_ids),
            marker    = dict(
                color = self._node_colors(np.array(self.cluster_ids)),
                line  = dict(color="white", width=1),
                size  = [10 + betweenness.get(cid, 0) * 80 for cid in self.cluster_ids]
            ),
            mode      = "markers",
            name      = "Career Families",
            x         = cx.tolist(),
            y         = cy.tolist()
        )]

        if coordinates:
            traces.append(go.Scatter(
                hovertext = ["Your Resume"],
                marker    = dict(
                    color  = "gold",
                    line   = dict(color="white", width=1.5),
                    size   = 14,
                    symbol = "star"
                ),
                mode      = "markers",
                name      = "Your Resume",
                x         = [coordinates[0]],
                y         = [coordinates[1]] if len(coordinates) > 1 else [0]
            ))

        return go.Figure(
            data   = traces,
            layout = dict(
                height      = 550,
                hovermode   = "closest",
                showlegend  = True,
                template    = self.theme(),
                title       = "Career Landscape (SVD Components 1-2)",
                xaxis_title = "SVD Component 1",
                yaxis_title = "SVD Component 2"
            )
        )

    def pathways(self, neighborhood: Neighborhood, target_id: int) -> go.Figure:
        """
        Spring-layout network of the target cluster's neighborhood
        with edge annotations showing apprenticeship hours.

        Builds a subgraph from the target and its advancement and
        lateral neighbors, positions nodes via spring layout, and
        annotates edges with up to two apprenticeship programs and
        their minimum hour requirements.

        Args:
            neighborhood : Advancement and lateral edges from the
                           target.
            target_id    : Center cluster for the neighborhood view.

        Returns:
            Configured pathways network figure.
        """
        sub = self.pathway.graph.subgraph(
            {target_id} |
            {e.cluster_id for e in neighborhood.all_edges}
        )
        pos = nx.spring_layout(sub, seed=42, weight="weight")

        apprenticeships = {
            e.cluster_id: [
                f"{c.label}: {c.metadata['min_hours']:,}h"
                for c in e.credentials if c.kind == "apprenticeship"
            ]
            for e in neighborhood.all_edges
        }

        edge_x = [v for u, w in sub.edges() for v in [pos[u][0], pos[w][0], None]]
        edge_y = [v for u, w in sub.edges() for v in [pos[u][1], pos[w][1], None]]

        edge_annotations = [
            dict(
                font      = dict(color="gray", size=9),
                showarrow = False,
                text      = "<br>".join(hours[:2]),
                x         = (pos[u][0] + pos[v][0]) / 2,
                y         = (pos[u][1] + pos[v][1]) / 2
            )
            for u, v in sub.edges()
            if (hours := apprenticeships.get(u, []) + apprenticeships.get(v, []))
        ]

        nodes       = list(sub.nodes())
        betweenness = nx.betweenness_centrality(sub, weight="weight")
        colors      = self._node_colors(np.array(nodes), gold=target_id)
        sizes       = [15 + betweenness.get(n, 0) * 60 for n in nodes]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            hoverinfo = "none",
            line      = dict(color="lightgray", width=1),
            mode      = "lines",
            x         = edge_x,
            y         = edge_y
        ))
        fig.add_trace(go.Scatter(
            hovertext    = self._node_hover(nodes),
            marker       = dict(
                color = colors,
                line  = dict(color="white", width=1),
                size  = sizes
            ),
            mode         = "markers+text",
            text         = [f"C{n}" for n in nodes],
            textposition = "top center",
            x            = [pos[n][0] for n in nodes],
            y            = [pos[n][1] for n in nodes]
        ))

        hidden_axis = dict(
            showgrid       = False,
            showticklabels = False,
            zeroline       = False
        )
        fig.update_layout(
            annotations = edge_annotations,
            height      = 550,
            hovermode   = "closest",
            showlegend  = False,
            template    = self.theme(),
            title       = f"Career Pathways from Cluster {target_id}",
            xaxis       = hidden_axis,
            yaxis       = hidden_axis
        )
        return fig
