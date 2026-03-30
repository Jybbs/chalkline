"""
Plotly chart factory for the Chalkline Marimo dashboard.

Provides a `Charts` class backed by a `Theme` and a fitted
`CareerPathwayGraph`. Every method reads the theme at call time for reactive
dark/light styling. Call sites pass pre-extracted data and receive a
configured `go.Figure`. All color resolution happens internally via theme.
"""

import numpy                as np
import plotly.graph_objects as go

from collections.abc       import Collection, Iterable, Sequence
from networkx              import betweenness_centrality, spring_layout
from sklearn.preprocessing import normalize

from chalkline.display.schemas   import GapScatterPoint, RadarTrace, Trace
from chalkline.display.theme     import Theme
from chalkline.pathways.clusters import Cluster
from chalkline.pathways.graph    import CareerPathwayGraph
from chalkline.pathways.schemas  import Reach


class Charts:
    """
    Plotly figure factory backed by a reactive `Theme`.

    Holds a fitted `CareerPathwayGraph` and `matched_id` for graph-aware
    methods (dendrogram, landscape, pathways). All other methods are pure:
    they accept pre-extracted data and return configured figures.
    """

    def __init__(
        self,
        matched_id : int,
        pathway    : CareerPathwayGraph,
        theme      : Theme
    ):
        """
        Args:
            matched_id : Cluster ID the resume matched to, used for highlighting in
                         graph visualizations.
            pathway    : Fitted career pathway graph.
            theme      : Reactive theme instance read at render time.
        """
        self.cluster_ids = pathway.clusters.cluster_ids
        self.matched_id  = matched_id
        self.pathway     = pathway
        self.theme       = theme

    def _apply_layout(
        self,
        fig    : go.Figure,
        height : int,
        **overrides
    ) -> go.Figure:
        """
        Apply standard Chalkline layout to a figure.

        Args:
            fig         : Figure to update in place.
            height      : Figure height in pixels.
            **overrides : Extra keys forwarded to `update_layout`.

        Returns:
            The same figure for chaining convenience.
        """
        fig.update_layout(**{
            "font"   : {
                "color"  : self.theme.colors["foreground"],
                "family" : "Inter, system-ui, sans-serif",
                "size"   : 12
            },
            "height" : height,
            "margin" : {"b" : 40, "l" : 20, "r" : 20, "t" : 10},
            "paper_bgcolor" : "rgba(0,0,0,0)",
            "plot_bgcolor"  : "rgba(0,0,0,0)",
            "template"      : self.theme.template,
            **overrides
        })
        return fig

    def _node_colors(self, node_ids: np.ndarray, **highlights: int) -> np.ndarray:
        """
        Assign colors to graph nodes with semantic highlights.

        Nodes matching `matched_id` are crimson. Additional keyword
        arguments map a color name to a cluster ID override.

        Args:
            node_ids     : Array of cluster IDs to color.
            **highlights : Color name to cluster ID pairs.

        Returns:
            String array of colors aligned with `node_ids`.
        """
        colors = np.where(node_ids == self.matched_id, "crimson", "steelblue")
        for color, cid in highlights.items():
            colors = np.where(node_ids == cid, color, colors)
        return colors

    def _node_hover(self, node_ids: Iterable[int]) -> list[str]:
        """
        Build HTML hover text for cluster nodes.

        Args:
            node_ids: Cluster IDs to generate hover text for.

        Returns:
            HTML-formatted hover strings.
        """
        return [
            f"Cluster {cid}<br>"
            f"{(c := self.pathway.clusters[cid]).soc_title}<br>"
            f"JZ {c.job_zone} · {c.size} postings"
            for cid in node_ids
        ]

    def _sector_color(self, sector: str) -> str:
        """
        Resolve one sector name to its theme hex color.
        """
        return self.theme.sector_colors.get(sector, self.theme.colors["accent"])

    def bar(
        self,
        height     : int,
        title      : str,
        color      : str | Sequence           = "accent",
        horizontal : bool                   = False,
        line       : Trace | None           = None,
        series     : Iterable[Trace] | None = None,
        x          : Iterable | None        = None,
        y          : Iterable | None        = None,
        **marker_kw
    ) -> go.Figure:
        """
        Bar chart with optional horizontal orientation, grouping, and line
        overlay.

        Pass `x`/`y` for a single-series bar, `series` for a grouped bar
        with per-series color roles, or `x`/`y` plus `line` for a
        bar-and-line combo.

        Args:
            color      : Single hex color or per-bar color list (single-series only).
            height     : Figure height in pixels.
            horizontal : Flip to horizontal bars.
            line       : Overlay scatter trace for combo charts.
            marker_kw  : Extra keywords for `go.Bar(marker=...)` (single-series only).
            series     : Typed traces for grouped bar mode.
            title      : Value-axis label.
            x          : Bar values (horizontal) or categories (vertical).
            y          : Categories (horizontal) or bar values (vertical).

        Returns:
            Configured bar figure.
        """
        layout = (
            dict(xaxis_title=title, yaxis=dict(autorange="reversed"))
            if horizontal
            else dict(yaxis_title=title)
        )

        if series:
            fig = go.Figure(data=[
                go.Bar(
                    marker      = dict(
                        color        = self.theme.colors[s.color_role],
                        cornerradius = 4
                    ),
                    name        = s.name,
                    orientation = "h" if horizontal else None,
                    x           = s.x,
                    y           = s.y
                )
                for s in series
            ])
            return self._apply_layout(
                fig, height,
                barmode = "group",
                legend  = dict(orientation="h", y=-0.15),
                **layout
            )

        color      = self.theme.resolve(color)
        bar_trace  = go.Bar(
            marker      = dict(color=color, cornerradius=4, **marker_kw),
            name        = "Individual" if line else None,
            orientation = "h" if horizontal else None,
            x           = x,
            y           = y
        )

        if line:
            return self._apply_layout(
                go.Figure(data=[bar_trace, go.Scatter(
                    line   = dict(color=self.theme.colors["primary"], width=2),
                    marker = dict(size=6),
                    mode   = "lines+markers",
                    name   = line.name or "Cumulative",
                    x      = line.x,
                    y      = line.y
                )]),
                height,
                legend = dict(orientation="h", y=-0.2),
                **layout
            )

        return self._apply_layout(go.Figure(bar_trace), height, **layout)

    def bubble_scatter(
        self,
        height  : int,
        points  : Iterable[GapScatterPoint],
        x_title : str,
        y_title : str
    ) -> go.Figure:
        """
        Gap-priority bubble scatter colored by magnitude.

        Extracts display lists from `GapScatterPoint` objects. Marker size
        scales with magnitude, color uses the OrRd palette, and axes show
        frequency vs magnitude.

        Args:
            height : Figure height in pixels.
            points : Gap scatter points with frequency, magnitude, and hover text.

        Returns:
            Configured scatter figure.
        """
        fig = go.Figure(go.Scatter(
            hovertext = [p.text for p in points],
            marker    = dict(
                color      = (magnitudes := [p.magnitude for p in points]),
                colorscale = "OrRd",
                size       = [max(8, m / 3) for m in magnitudes]
            ),
            mode      = "markers",
            x         = [p.frequency for p in points],
            y         = magnitudes
        ))
        return self._apply_layout(
            fig, height,
            xaxis_title = x_title,
            yaxis_title = y_title
        )

    def career_ladder(
        self,
        clusters    : Collection[Cluster],
        tick_labels : Iterable[str],
        x_title     : str,
        target_id   : int | None = None
    ) -> go.Figure:
        """
        Scatter of career families positioned by Job Zone.

        Extracts parallel display lists from `Cluster` objects and
        highlights the target cluster in primary color. All others are
        colored by their sector.

        Args:
            clusters  : Same-sector clusters sorted by Job Zone.
            target_id : Cluster ID to highlight.

        Returns:
            Configured scatter figure.
        """
        colors = [
            self.theme.colors["primary"]
            if target_id is not None and c.cluster_id == target_id
            else self._sector_color(c.sector)
            for c in clusters
        ]

        fig = go.Figure(go.Scatter(
            hovertext    = [f"{c.soc_title} ({c.size} postings)" for c in clusters],
            marker       = dict(
                color = colors,
                size  = [max(10, c.size / 3) for c in clusters]
            ),
            mode         = "markers+text",
            text         = [c.soc_title[:20] for c in clusters],
            textfont     = dict(size=10),
            textposition = "middle right",
            x            = [c.job_zone for c in clusters],
            y            = range(len(clusters))
        ))
        return self._apply_layout(
            fig, max(300, len(clusters) * 45),
            margin = dict(b=40, l=20, r=120, t=10),
            xaxis  = dict(
                dtick    = 1,
                range    = [0.5, 5.5],
                ticktext = tick_labels,
                tickvals = range(1, 6),
                title    = x_title
            ),
            yaxis  = dict(visible=False)
        )

    def dendrogram(
        self,
        annotation_text : str,
        title           : str,
        x_title         : str,
        y_title         : str
    ) -> go.Figure:
        """
        Ward-linkage dendrogram with the matched cluster annotated.

        Normalizes cluster centroids, computes Ward linkage via scipy,
        renders U-links as a Plotly line trace, and annotates the matched
        cluster with a crimson arrow.

        Returns:
            Configured dendrogram figure.
        """
        from scipy.cluster.hierarchy import dendrogram, linkage

        result = dendrogram(
            linkage(normalize(self.pathway.clusters.centroids), method="ward"),
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

        ivl = result["ivl"]
        fig.add_annotation(
            arrowcolor = "crimson",
            arrowhead  = 2,
            arrowwidth = 2,
            font       = dict(color="crimson", size=12),
            showarrow  = True,
            text       = annotation_text,
            x          = 5 + ivl.index(f"C{self.matched_id}") * 10,
            y          = 0,
            yshift     = -20
        )

        return self._apply_layout(
            fig, 500,
            title       = title,
            xaxis       = dict(
                ticktext = ivl,
                tickvals = [5 + i * 10 for i in range(len(ivl))]
            ),
            xaxis_title = x_title,
            yaxis_title = y_title
        )

    def gauge(
        self,
        title  : str,
        value  : float,
        height : int = 250
    ) -> go.Figure:
        """
        Semicircular gauge indicator with tri-color bands.

        Args:
            height : Figure height in pixels.
            title  : Label displayed below the gauge.
            value  : Integer percentage 0-100.

        Returns:
            Configured gauge figure.
        """
        sc = self.theme.score_color
        fig = go.Figure(go.Indicator(
            domain = dict(x=[0, 1], y=[0, 1]),
            gauge  = dict(
                axis  = dict(range=[0, 100], ticksuffix="%"),
                bar   = dict(color=sc(50)),
                steps = [
                    dict(color=sc(20), range=[0, 40]),
                    dict(color=sc(50), range=[40, 70]),
                    dict(color=sc(80), range=[70, 100])
                ]
            ),
            mode   = "gauge+number",
            number = dict(suffix="%"),
            title  = dict(font=dict(size=14), text=title),
            value  = value
        ))
        return self._apply_layout(fig, height, margin=dict(b=20, l=30, r=30, t=60))

    def histogram(
        self,
        height    : int,
        nbins     : int,
        x         : Iterable[float],
        x_title   : str,
        y_title   : str,
        threshold : float | None = None
    ) -> go.Figure:
        """
        Histogram with themed accent color and optional threshold line.

        Args:
            height    : Figure height in pixels.
            nbins     : Number of bins.
            threshold : X-position for a dashed vertical line.
            x         : Values to bin.
            x_title   : X-axis label.
            y_title   : Y-axis label.

        Returns:
            Configured histogram figure.
        """
        fig = go.Figure(go.Histogram(
            marker = dict(color=self.theme.colors["accent"], cornerradius=4),
            nbinsx = nbins,
            x      = x
        ))
        if threshold is not None:
            fig.add_vline(
                line_color = self.theme.colors["primary"],
                line_dash  = "dash",
                line_width = 2,
                x          = threshold
            )
        return self._apply_layout(
            fig, height,
            xaxis_title = x_title,
            yaxis_title = y_title
        )

    def landscape(
        self,
        coordinates     : Sequence[float],
        legend_families : str,
        legend_resume   : str,
        title           : str,
        x_title         : str,
        y_title         : str
    ) -> go.Figure:
        """
        SVD scatter of cluster centroids with resume overlay.

        Node sizes scale with brokerage centrality. The matched cluster is
        crimson and the resume position is a gold star.

        Args:
            coordinates: Resume SVD position, empty if unavailable.

        Returns:
            Configured landscape scatter figure.
        """
        brokerage = betweenness_centrality(self.pathway.graph, weight="weight")
        cx = (c := self.pathway.clusters.centroids)[:, 0]
        cy = c[:, 1] if c.shape[1] > 1 else np.zeros(len(cx))

        traces = [go.Scatter(
            hovertext = self._node_hover(self.cluster_ids),
            marker    = dict(
                color = self._node_colors(self.pathway.node_ids),
                line  = dict(color="white", width=1),
                size  = [10 + brokerage.get(c, 0) * 80 for c in self.cluster_ids]
            ),
            mode      = "markers",
            name      = legend_families,
            x         = cx.tolist(),
            y         = cy.tolist()
        )]

        if coordinates:
            traces.append(go.Scatter(
                hovertext = [legend_resume],
                marker    = dict(
                    color  = "gold",
                    line   = dict(color="white", width=1.5),
                    size   = 14,
                    symbol = "star"
                ),
                mode      = "markers",
                name      = legend_resume,
                x         = [coordinates[0]],
                y         = [coordinates[1]] if len(coordinates) > 1 else [0]
            ))

        return self._apply_layout(
            go.Figure(data=traces), 550,
            hovermode   = "closest",
            showlegend  = True,
            title       = title,
            xaxis_title = x_title,
            yaxis_title = y_title
        )

    def pathways(self, reach: Reach, target_id: int, title: str) -> go.Figure:
        """
        Spring-layout network of a cluster's reach neighborhood.

        Builds a subgraph from the target and its advancement and lateral
        neighbors, positions nodes via spring layout, and annotates edges
        with apprenticeship program hours.

        Args:
            reach     : Advancement and lateral edges from target.
            target_id : Center cluster for the reach view.

        Returns:
            Configured network figure.
        """
        sub = self.pathway.graph.subgraph(
            {target_id}
            | {e.cluster_id for e in reach.all_edges}
        )
        pos   = spring_layout(sub, seed=42, weight="weight")
        edges = list(sub.edges())

        apprenticeships = {
            e.cluster_id: [
                f"{c.label}: {c.metadata["min_hours"]:,}h"
                for c in e.credentials
                if c.kind == "apprenticeship"
            ]
            for e in reach.all_edges
        }

        edge_annotations = [
            dict(
                font      = dict(color="gray", size=9),
                showarrow = False,
                text      = "<br>".join(hours[:2]),
                x         = (pos[u][0] + pos[v][0]) / 2,
                y         = (pos[u][1] + pos[v][1]) / 2
            )
            for u, v in edges
            if (hours := apprenticeships.get(u, []) + apprenticeships.get(v, []))
        ]

        nodes       = list(sub.nodes())
        brokerage = betweenness_centrality(sub, weight="weight")

        hidden_axis = dict(
            showgrid       = False,
            showticklabels = False,
            zeroline       = False
        )

        fig = go.Figure(data=[
            go.Scatter(
                hoverinfo = "none",
                line      = dict(color="lightgray", width=1),
                mode      = "lines",
                x         = [v for u, w in edges for v in [pos[u][0], pos[w][0], None]],
                y         = [v for u, w in edges for v in [pos[u][1], pos[w][1], None]]
            ),
            go.Scatter(
                hovertext    = self._node_hover(nodes),
                marker       = dict(
                    color = self._node_colors(np.array(nodes), gold=target_id),
                    line  = dict(color="white", width=1),
                    size  = [15 + brokerage.get(n, 0) * 60 for n in nodes]
                ),
                mode         = "markers+text",
                text         = [f"C{n}" for n in nodes],
                textposition = "top center",
                x            = [pos[n][0] for n in nodes],
                y            = [pos[n][1] for n in nodes]
            )
        ])

        return self._apply_layout(
            fig, 550,
            annotations = edge_annotations,
            hovermode   = "closest",
            showlegend  = False,
            title       = title,
            xaxis       = hidden_axis,
            yaxis       = hidden_axis
        )

    def pie(self, height: int, **trace_kw) -> go.Figure:
        """
        Pie or donut chart with tight margins and no legend.

        Args:
            height     : Figure height in pixels.
            **trace_kw : Keywords forwarded to `go.Pie(...)`.

        Returns:
            Configured pie/donut figure.
        """
        return self._apply_layout(
            go.Figure(go.Pie(**trace_kw)), height,
            margin     = dict.fromkeys("blrt", 20),
            showlegend = False
        )

    def radar(
        self,
        labels : Sequence[str],
        traces : Iterable[RadarTrace],
        height : int = 400
    ) -> go.Figure:
        """
        Polar radar chart with one or more filled traces.

        The polygon is auto-closed by appending the first value.

        Args:
            height : Figure height in pixels.
            labels : Angular axis labels.
            traces : Typed traces with color role, name, values, and opacity.

        Returns:
            Configured radar figure.
        """
        fig = go.Figure()
        for t in traces:
            color = self.theme.colors[t.color_role]
            fig.add_trace(go.Scatterpolar(
                fill      = "toself",
                fillcolor = f"{color}{round(t.alpha * 255):02x}",
                line      = dict(color=color, dash=t.dash, width=2),
                name      = t.name,
                r         = [*t.values, t.values[0]],
                theta     = [*labels, labels[0]]
            ))

        return self._apply_layout(
            fig, height,
            legend = dict(orientation="h", y=-0.1),
            margin = dict(b=60, l=60, r=60, t=40),
            polar  = dict(
                bgcolor    = "rgba(0,0,0,0)",
                radialaxis = dict(range=[0, 100], visible=True)
            )
        )

    def sector_colors(self, sectors: Iterable[str]) -> list[str]:
        """
        Resolve sector names to theme hex colors.

        Falls back to the accent color for unrecognized sectors.
        """
        return [self._sector_color(s) for s in sectors]

    def timeline(
        self,
        dates  : Collection,
        height : int = 180,
        hover  : Iterable[str] | None = None
    ) -> go.Figure:
        """
        Strip scatter of dates along a hidden y-axis.

        Args:
            dates  : Date values for the x-axis.
            height : Figure height in pixels.
            hover  : Hover labels per point.

        Returns:
            Configured timeline scatter figure.
        """
        return self._apply_layout(
            go.Figure(go.Scatter(
                hovertext = hover,
                marker    = dict(color=self.theme.colors["accent"], size=8),
                mode      = "markers",
                x         = dates,
                y         = [1] * len(dates)
            )),
            height,
            yaxis = dict(visible=False)
        )

    def treemap(
        self,
        height        : int,
        labels        : Collection[str],
        values        : Collection[int | float],
        branch_values : str | None       = None,
        parents       : Collection[str] | None = None,
        sectors       : Collection[str] | None = None
    ) -> go.Figure:
        """
        Treemap with optional sector coloring and hierarchy.

        Args:
            branch_values : Plotly branch aggregation mode (e.g. "total").
            height        : Figure height in pixels.
            labels        : Tile labels.
            parents       : Parent labels (defaults to all empty).
            sectors       : Sector names for automatic color resolution.
            values        : Tile sizes.

        Returns:
            Configured treemap figure.
        """
        marker = {
            "colors": self.sector_colors(sectors)
        } if sectors else {"cornerradius": 4},
        fig    = go.Figure(go.Treemap(
            branchvalues = branch_values,
            labels       = labels,
            marker       = marker,
            parents      = parents or [""] * len(labels),
            textinfo     = None if sectors else "label+value",
            values       = values
        ))
        margin = {} if sectors else dict(b=10, l=10, r=10, t=10)
        return self._apply_layout(fig, height, **margin)

    def violin(
        self,
        groups  : dict[str, list[float]],
        height  : int,
        y_title : str
    ) -> go.Figure:
        """
        Violin plots grouped by sector name.

        Each group is colored by its sector color from the theme.

        Args:
            groups  : Sector name to list of values.
            height  : Figure height in pixels.
            y_title : Y-axis label.

        Returns:
            Configured violin figure.
        """
        fig = go.Figure(data=[
            go.Violin(
                box      = dict(visible=True),
                line     = dict(color=self._sector_color(sector)),
                meanline = dict(visible=True),
                name     = sector[:15],
                y        = values
            )
            for sector in sorted(groups)
            if (values := groups[sector])
        ])
        return self._apply_layout(
            fig, height,
            showlegend  = False,
            yaxis_title = y_title
        )
