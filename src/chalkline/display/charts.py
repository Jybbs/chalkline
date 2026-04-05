"""
Plotly chart factory for the Chalkline Marimo dashboard.

Provides a `Charts` class backed by a `Theme` and a fitted
`CareerPathwayGraph`. Every method reads the theme at call time for reactive
dark/light styling. Call sites pass pre-extracted data and receive a
configured `go.Figure`. All color resolution happens internally via theme.
"""

import numpy                as np
import plotly.graph_objects as go

from collections.abc       import Collection, Iterable, Mapping, Sequence
from itertools             import chain
from networkx              import betweenness_centrality
from plotly.basedatatypes  import BaseTraceType
from sklearn.preprocessing import normalize

from chalkline.display.schemas   import GapScatterPoint, HierarchyData, Trace
from chalkline.display.theme     import Theme
from chalkline.pathways.clusters import Cluster
from chalkline.pathways.graph    import CareerPathwayGraph


class Charts:
    """
    Plotly figure factory backed by a reactive `Theme`.

    Holds a fitted `CareerPathwayGraph` and `matched_id` for graph-aware
    methods (dendrogram, landscape). All other methods are pure: they accept
    pre-extracted data and return configured figures.
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
        trace_or_fig : go.Figure | BaseTraceType | list,
        height       : int,
        **overrides
    ) -> go.Figure:
        """
        Wrap a trace (or pre-built figure) and apply standard Chalkline
        layout.

        Accepts all `update_layout` keys via `**overrides` plus two friendly
        aliases: `x_title` / `y_title` map to Plotly's `xaxis_title` /
        `yaxis_title`, stripped when empty.

        Args:
            height       : Figure height in pixels.
            trace_or_fig : A trace, list of traces, or pre-built figure. Pre-built
                           figures are used as-is to preserve mutations like
                           `add_annotation` or `add_vline`.
            **overrides  : Layout keys forwarded to `update_layout`. Supports `x_title`
                           / `y_title` as aliases for Plotly's axis title keys.

        Returns:
            The configured figure.
        """
        fig = (
            trace_or_fig if isinstance(trace_or_fig, go.Figure)
            else go.Figure(trace_or_fig)
        )
        if x_title := overrides.pop("x_title", ""): overrides["xaxis_title"] = x_title
        if y_title := overrides.pop("y_title", ""): overrides["yaxis_title"] = y_title
        fig.update_layout(
            height   = height,
            template = self.theme.template,
            **overrides
        )
        return fig

    def _node_colors(self, node_ids: np.ndarray, **highlights: int) -> np.ndarray:
        """
        Assign colors to graph nodes with semantic highlights.

        Nodes matching `matched_id` render in the highlight palette color;
        all others render in the accent color. Additional keyword arguments
        map a palette key to a cluster ID override.

        Args:
            node_ids     : Array of cluster IDs to color.
            **highlights : Palette key to cluster ID pairs.

        Returns:
            String array of hex colors aligned with `node_ids`.
        """
        colors = np.where(
            node_ids == self.matched_id,
            self.theme.colors["highlight"],
            self.theme.colors["accent"],
        )
        for role, cid in highlights.items():
            colors = np.where(node_ids == cid, self.theme.colors[role], colors)
        return colors

    def _parallel(
        self,
        klass      : type,
        colorscale : str,
        dimensions : Sequence[dict],
        height     : int,
        color      : Sequence[float] | None,
        **extra
    ) -> go.Figure:
        """
        Shared builder for parallel-categories and parallel-coordinates
        diagrams, which share identical dimension and line structure.

        Args:
            colorscale : Plotly colorscale name for the line color mapping.
            klass      : `go.Parcats` or `go.Parcoords`.
            **extra    : Trace-specific keywords (e.g. `hoveron`).
        """
        return self._apply_layout(
            klass(
                dimensions = dimensions,
                line       = dict(
                    color      = color,
                    colorscale = colorscale
                ),
                **extra
            ),
            height = height
        )

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

    def bar(
        self,
        height     : int,
        title      : str,
        color      : str | Sequence         = "accent",
        horizontal : bool                   = False,
        line       : Trace | None           = None,
        series     : Iterable[Trace] | None = None,
        x          : Sequence               = (),
        y          : Sequence               = (),
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
            dict(x_title=title, yaxis=dict(autorange="reversed"))
            if horizontal
            else dict(y_title=title)
        )

        if series:
            fig = go.Figure(data=[
                go.Bar(
                    marker      = dict(color=self.theme.colors[s.color_role]),
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

        color = (
            self.theme.colors.get(color, color) if isinstance(color, str)
            else [
                self.theme.colors.get(c, c) 
                if isinstance(c, str) else c for c in color
            ]
        )
        bar_trace  = go.Bar(
            marker      = dict(color=color, **marker_kw),
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
        hovertext, magnitudes, frequencies = (
            zip(*((p.text, p.magnitude, p.frequency) for p in points))
            if points else ((), (), ())
        )
        return self._apply_layout(
            go.Scatter(
                hovertext = hovertext,
                marker    = dict(
                    color      = magnitudes,
                    colorscale = "OrRd",
                    size       = [max(8, m / 3) for m in magnitudes]
                ),
                mode      = "markers",
                x         = frequencies,
                y         = magnitudes
            ),
            height  = height,
            x_title = x_title,
            y_title = y_title
        )

    def career_ladder(
        self,
        clusters    : Collection[Cluster],
        tick_labels : Sequence[str],
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
            else self.theme.sectors[c.sector]
            for c in clusters
        ]

        fig = go.Figure(go.Scatter(
            hovertext    = [f"{c.soc_title} ({c.size} postings)" for c in clusters],
            marker       = dict(
                color = colors,
                size  = [max(10, c.size / 3) for c in clusters]
            ),
            mode         = "markers+text",
            text         = [c.soc_title for c in clusters],
            textfont     = dict(size=10),
            textposition = "middle right",
            x            = [c.job_zone for c in clusters],
            y            = list(range(len(clusters)))
        ))
        return self._apply_layout(
            fig, max(300, len(clusters) * 45),
            xaxis  = dict(
                dtick    = 1,
                range    = [0.5, 5.5],
                ticktext = tick_labels,
                tickvals = list(range(1, 6)),
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
            line = dict(color=self.theme.colors["accent"], width=1.5),
            mode = "lines",
            x    = list(chain.from_iterable([*ic, None] for ic in result["icoord"])),
            y    = list(chain.from_iterable([*dc, None] for dc in result["dcoord"]))
        ))

        ivl = result["ivl"]
        fig.add_annotation(
            arrowcolor = self.theme.colors["highlight"],
            arrowhead  = 2,
            arrowwidth = 2,
            font       = dict(color=self.theme.colors["highlight"], size=12),
            showarrow  = True,
            text       = annotation_text,
            x          = 5 + ivl.index(f"C{self.matched_id}") * 10,
            y          = 0,
            yshift     = -20
        )

        return self._apply_layout(
            fig, 500,
            title   = title,
            x_title = x_title,
            y_title = y_title,
            xaxis   = dict(
                ticktext = ivl,
                tickvals = [5 + i * 10 for i in range(len(ivl))]
            )
        )

    def funnel(
        self,
        height : int,
        labels : Sequence[str],
        values : Sequence[int | float],
        color  : str = "accent"
    ) -> go.Figure:
        """
        Horizontal funnel showing progressive narrowing.

        Args:
            color  : Theme role for funnel segments.
            height : Figure height in pixels.
            labels : Stage labels top to bottom.
            values : Stage values (decreasing).
        """
        fig = go.Figure(go.Funnel(
            marker    = dict(color=self.theme.colors.get(color, color)),
            textinfo  = "value+percent initial",
            x         = values,
            y         = labels
        ))
        return self._apply_layout(fig, height)

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
        fig = go.Figure(go.Indicator(
            domain = dict(x=[0, 1], y=[0, 1]),
            gauge  = dict(
                axis  = dict(range=[0, 100], ticksuffix="%"),
                bar   = dict(color=self.theme.score_color(50)),
                steps = [
                    dict(color=self.theme.score_color(mid), range=[lo, hi])
                    for lo, mid, hi in ((0, 20, 40), (40, 50, 70), (70, 80, 100))
                ]
            ),
            mode   = "gauge+number",
            number = dict(suffix="%"),
            title  = dict(font=dict(size=14), text=title),
            value  = value
        ))
        return self._apply_layout(fig, height)

    def heatmap(
        self,
        columns : Sequence[str],
        height  : int,
        labels  : Sequence[str],
        values  : Sequence[Sequence[float]],
        x_title : str = "",
        y_title : str = ""
    ) -> go.Figure:
        """
        Annotated heatmap with diverging color scale.

        Args:
            columns : Column labels along the x-axis.
            height  : Figure height in pixels.
            labels  : Row labels along the y-axis.
            values  : 2D array of values (rows x columns).
            x_title : X-axis title.
            y_title : Y-axis title.
        """
        return self._apply_layout(
            go.Heatmap(
                colorscale   = "Teal",
                texttemplate = "%{z:.2f}",
                x            = columns,
                y            = labels,
                z            = values
            ),
            height          = height,
            x_title         = x_title,
            y_title         = y_title,
            xaxis_side      = "top",
            yaxis_autorange = "reversed"
        )

    def histogram(
        self,
        height    : int,
        nbins     : int,
        x         : Sequence[float],
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
            marker = dict(color=self.theme.colors["accent"]),
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
            x_title = x_title,
            y_title = y_title
        )

    def indicator(
        self,
        height    : int,
        reference : float,
        title     : str,
        value     : float
    ) -> go.Figure:
        """
        Big-number indicator with delta from reference.

        Args:
            height    : Figure height in pixels.
            reference : Baseline for delta computation.
            title     : Indicator title text.
            value     : Current value.
        """
        fig = go.Figure(go.Indicator(
            delta  = dict(reference=reference),
            mode   = "number+delta",
            title  = dict(text=title),
            value  = value
        ))
        return self._apply_layout(fig, height)

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
        centroids = self.pathway.clusters.centroids

        traces = [go.Scatter(
            hovertext = self._node_hover(self.cluster_ids),
            marker    = dict(
                color = self._node_colors(self.pathway.node_ids),
                line  = dict(color=self.theme.colors["foreground"], width=1),
                size  = [10 + brokerage[c] * 80 for c in self.cluster_ids]
            ),
            mode      = "markers",
            name      = legend_families,
            x         = centroids[:, 0].tolist(),
            y         = centroids[:, 1].tolist()
        )]

        if coordinates:
            traces.append(go.Scatter(
                hovertext = [legend_resume],
                marker    = dict(
                    color  = self.theme.colors["primary"],
                    line   = dict(color=self.theme.colors["foreground"], width=1.5),
                    size   = 14,
                    symbol = "star"
                ),
                mode      = "markers",
                name      = legend_resume,
                x         = [coordinates[0]],
                y         = [coordinates[1]]
            ))

        return self._apply_layout(
            traces, 550,
            hovermode  = "closest",
            showlegend = True,
            title      = title,
            x_title    = x_title,
            y_title    = y_title
        )

    def parcats(
        self,
        dimensions : Sequence[dict],
        height     : int,
        color      : Sequence[float] | None = None,
        hoveron    : str = "color"
    ) -> go.Figure:
        """
        Parallel categories diagram showing flow across dimensions.

        Args:
            color      : Numeric array for ribbon coloring.
            dimensions : List of dicts with `label` and `values` keys.
            height     : Figure height in pixels.
            hoveron    : Hover target ("dimension", "color", or "category").
        """
        return self._parallel(
            go.Parcats, "RdYlGn", dimensions, height, color,
            hoveron = hoveron
        )

    def parcoords(
        self,
        dimensions : Sequence[dict],
        height     : int,
        color      : Sequence[float] | None = None
    ) -> go.Figure:
        """
        Parallel coordinates plot for multi-dimensional comparison.

        Args:
            color      : Numeric array for line coloring.
            dimensions : List of dicts with `label`, `values`, and optional `range`
                         keys.
            height     : Figure height in pixels.
        """
        return self._parallel(go.Parcoords, "Teal", dimensions, height, color)

    def pie(
        self,
        height : int,
        labels : Sequence,
        values : Sequence,
        **trace_kw
    ) -> go.Figure:
        """
        Pie or donut chart with no legend.

        Args:
            height     : Figure height in pixels.
            labels     : Slice labels.
            values     : Slice values.
            **trace_kw : Extra keywords forwarded to `go.Pie(...)`.

        Returns:
            Configured pie/donut figure.
        """
        return self._apply_layout(
            go.Figure(go.Pie(labels=labels, values=values, **trace_kw)),
            height,
            showlegend = False
        )

    def sankey(
        self,
        height : int,
        labels : Sequence[str],
        links  : dict,
        colors : Sequence[str] | None = None
    ) -> go.Figure:
        """
        Sankey flow diagram.

        Args:
            colors : Per-node hex colors.
            height : Figure height in pixels.
            labels : Node labels.
            links  : Dict with `source`, `target`, `value`, and optional `label` and
                     `color` keys.
        """
        fig = go.Figure(go.Sankey(
            node = dict(
                color = colors,
                label = labels,
                pad   = 20
            ),
            link = links
        ))
        return self._apply_layout(fig, height)

    def sunburst(
        self,
        data          : HierarchyData,
        height        : int,
        branch_values : str | None = None
    ) -> go.Figure:
        """
        Sunburst chart for hierarchical decomposition.

        Args:
            branch_values : Plotly branch aggregation mode (e.g. "total").
            data          : Typed hierarchy bundle with labels, values, and optional
                            ids/parents/colors.
            height        : Figure height in pixels.
        """
        fig = go.Figure(go.Sunburst(
            branchvalues = branch_values,
            ids          = data.ids,
            labels       = data.labels,
            marker       = dict(colors=data.colors) if data.colors else {},
            parents      = data.parents,
            values       = data.values
        ))
        return self._apply_layout(fig, height)

    def timeline(
        self,
        dates  : Sequence,
        height : int = 180,
        hover  : Sequence[str] | None = None
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
        data          : HierarchyData,
        height        : int,
        branch_values : str | None = None
    ) -> go.Figure:
        """
        Treemap with optional sector coloring and hierarchy.

        Args:
            branch_values : Plotly branch aggregation mode (e.g. "total").
            data          : Typed hierarchy bundle with labels, values, and optional
                            parents/sectors for color resolution.
            height        : Figure height in pixels.

        Returns:
            Configured treemap figure.
        """
        parents = data.parents if data.parents else [""] * len(data.labels)
        marker  = (
            {"colors": [self.theme.sectors[s] for s in data.sectors]}
            if data.sectors else {"cornerradius": 4}
        )
        fig = go.Figure(go.Treemap(
            branchvalues = branch_values,
            labels       = data.labels,
            marker       = marker,
            parents      = parents,
            textinfo     = None if data.sectors else "label+value",
            values       = data.values
        ))
        return self._apply_layout(fig, height)

    def violin(
        self,
        groups  : dict[str, list[float]],
        height  : int,
        y_title : str,
        colors  : Mapping[str, str] = {}
    ) -> go.Figure:
        """
        Violin plots grouped by label.

        Keys present in `colors` get their mapped line color; unmatched keys
        fall through to Plotly's colorway, which assigns distinct palette
        colors per violin in rendering order.

        Args:
            colors  : Label to hex color mapping for explicit per-group line coloring.
                      Empty by default.
            groups  : Label to list of values.
            height  : Figure height in pixels.
            y_title : Y-axis label.

        Returns:
            Configured violin figure.
        """
        fig = go.Figure(data=[
            go.Violin(
                box      = dict(visible=True),
                line     = dict(color=colors[key]) if key in colors else None,
                meanline = dict(visible=True),
                name     = key,
                y        = values
            )
            for key in sorted(groups)
            if (values := groups[key])
        ])
        return self._apply_layout(
            fig, height,
            showlegend = False,
            y_title    = y_title
        )

    def waterfall(
        self,
        height   : int,
        measures : Sequence[str],
        text     : Sequence[str],
        x        : Sequence[str],
        y        : Sequence[float]
    ) -> go.Figure:
        """
        Waterfall chart showing cumulative gain or loss.

        Args:
            height   : Figure height in pixels.
            measures : "relative", "total", or "absolute" per bar.
            text     : Annotation text per bar.
            x        : Category labels.
            y        : Delta values (positive or negative).
        """
        marker = lambda role: dict(marker=dict(color=self.theme.colors[role]))
        fig    = go.Figure(go.Waterfall(
            connector    = dict(line=dict(color=self.theme.colors["muted"])),
            decreasing   = marker("error"),
            increasing   = marker("success"),
            measure      = measures,
            text         = text,
            textposition = "outside",
            totals       = marker("primary"),
            x            = x,
            y            = y
        ))
        return self._apply_layout(fig, height)
