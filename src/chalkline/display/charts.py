"""
Plotly chart factory for the Chalkline Marimo dashboard.

Provides a `Charts` class backed by a `Theme` and a fitted
`CareerPathwayGraph`. Every method reads the theme at call time for reactive
dark/light styling. Call sites pass pre-extracted data and receive a
configured `go.Figure`. All color resolution happens internally via theme.
"""

import numpy                as np
import plotly.graph_objects as go

from collections.abc      import Iterable, Mapping, Sequence
from datetime             import date
from plotly.basedatatypes import BaseTraceType
from typing               import Any

from chalkline.display.schemas import ScatterSeries, SectorRanking
from chalkline.display.theme   import Theme
from chalkline.pathways.graph  import CareerPathwayGraph


class Charts:
    """
    Plotly figure factory backed by a reactive `Theme`.

    Holds a fitted `CareerPathwayGraph` and `matched_id` for `landscape`,
    the only graph-aware method. All other methods are pure: they accept
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
        height       : int,
        trace_or_fig : go.Figure | BaseTraceType | list[BaseTraceType],
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
                           figures (e.g. from `make_subplots`) pass through unchanged so
                           subplot domains survive layout application.
            **overrides  : Layout keys forwarded to `update_layout`. Supports `x_title`
                           / `y_title` as aliases for Plotly's axis title keys.

        Returns:
            The configured figure.
        """
        fig = (
            trace_or_fig if isinstance(trace_or_fig, go.Figure)
            else go.Figure(trace_or_fig)
        )
        for short, long in (("x_title", "xaxis_title"), ("y_title", "yaxis_title")):
            if value := overrides.pop(short, ""):
                overrides[long] = value
        fig.update_layout(
            height   = height,
            template = self.theme.template,
            **overrides
        )
        return fig

    def _node_colors(self, node_ids: np.ndarray) -> np.ndarray:
        """
        Assign colors to graph nodes with the matched cluster highlighted.

        Args:
            node_ids: Array of cluster IDs to color.

        Returns:
            String array of hex colors aligned with `node_ids`.
        """
        return np.where(
            node_ids == self.matched_id,
            self.theme.colors["highlight"],
            self.theme.colors["accent"]
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
            f"Job Zone {c.job_zone} · {c.size} postings"
            for cid in node_ids
        ]

    def bar(
        self,
        height     : int,
        title      : str,
        color      : str | Sequence[str]              = "accent",
        data       : Mapping[str, int | float] | None = None,
        horizontal : bool                             = False,
        line       : Mapping[str, int | float] | None = None,
        x          : Sequence[str | int | float]      = (),
        y          : Sequence[str | int | float]      = ()
    ) -> go.Figure:
        """
        Bar chart with optional horizontal orientation and line overlay.

        Pass `x`/`y` for explicit sequences, or `data` (a label-to-value
        mapping) to derive both axes from a dict — labels become categories
        and values become bar lengths, swapped for `horizontal=True`. Pass
        `line` for a bar-and-line combo.

        Args:
            color      : Single hex color or per-bar color list.
            data       : Label-to-value mapping; overrides `x`/`y` when set.
            height     : Figure height in pixels.
            horizontal : Flip to horizontal bars.
            line       : Overlay scatter trace for combo charts.
            title      : Value-axis label.
            x          : Bar values (horizontal) or categories (vertical).
            y          : Categories (horizontal) or bar values (vertical).

        Returns:
            Configured bar figure.
        """
        if data is not None:
            labels, values = list(data), list(data.values())
            x, y           = (values, labels) if horizontal else (labels, values)

        layout: dict[str, Any] = (
            dict(x_title=title, yaxis=dict(autorange="reversed"))
            if horizontal
            else dict(y_title=title)
        )

        color = (
            self.theme.resolve_color(color) if isinstance(color, str)
            else [self.theme.resolve_color(c) for c in color]
        )

        traces: list[BaseTraceType] = [go.Bar(
            marker      = dict(color=color),
            name        = "Individual" if line else None,
            orientation = "h" if horizontal else None,
            x           = x,
            y           = y
        )]
        if line:
            traces.append(go.Scatter(
                line   = dict(color=self.theme.colors["primary"], width=2),
                marker = dict(size=6),
                mode   = "lines+markers",
                name   = "Cumulative",
                x      = list(line),
                y      = list(line.values())
            ))
            layout["legend"] = dict(orientation="h", y=-0.2)

        return self._apply_layout(height, traces, **layout)

    def bubble_scatter(
        self,
        brokerage  : SectorRanking,
        height     : int,
        silhouette : SectorRanking,
        x_title    : str,
        y_title    : str
    ) -> go.Figure:
        """
        Bubble scatter of brokerage centrality vs silhouette coefficient.

        Brokerage values become integer-percentage x-coordinates (min 1 for
        marker visibility), silhouette values become one-decimal
        y-coordinates and marker color/size, and brokerage labels carry the
        hover text.

        Args:
            brokerage  : Per-cluster brokerage centrality ranking.
            height     : Figure height in pixels.
            silhouette : Per-cluster silhouette coefficient ranking.

        Returns:
            Configured scatter figure.
        """
        magnitudes = [round(s * 100, 1) for s in silhouette.values]
        return self._apply_layout(
            height,
            go.Scatter(
                hovertext = brokerage.labels,
                marker    = dict(
                    color      = magnitudes,
                    colorscale = "OrRd",
                    size       = [max(8, m / 3) for m in magnitudes]
                ),
                mode      = "markers",
                x         = [max(1, round(b * 100)) for b in brokerage.values],
                y         = magnitudes
            ),
            x_title = x_title,
            y_title = y_title
        )

    def category_scatter(
        self,
        data    : Mapping[str, ScatterSeries],
        height  : int,
        x_title : str,
        y_title : str
    ) -> go.Figure:
        """
        Scatter of `(x, y)` points grouped into one Plotly trace per
        category, so each category draws its own legend entry and distinct
        color from the theme colorway.

        Mirrors `bar.data` in shape, with a single mapping the call site
        builds directly instead of four parallel sequences the chart has to
        bucket internally. Each value is a `{"hover", "x", "y"}` dict of
        parallel lists for one trace.

        Args:
            data    : Category name to a `{"hover", "x", "y"}` mapping of parallel
                      sequences, one entry per point in that category.
            height  : Figure height in pixels.
            x_title : X-axis title.
            y_title : Y-axis title.

        Returns:
            Configured multi-trace scatter figure.
        """
        traces = [
            go.Scatter(
                hoverinfo = "text",
                hovertext = points["hover"],
                marker    = dict(opacity=0.85, size=12),
                mode      = "markers",
                name      = name,
                x         = points["x"],
                y         = points["y"]
            )
            for name, points in data.items()
        ]
        return self._apply_layout(
            height, traces,
            legend  = dict(orientation="h", y=-0.2),
            x_title = x_title,
            y_title = y_title
        )

    def faceted_treemap(
        self,
        facets       : Mapping[str, Mapping[str, int | float]],
        height       : int,
        descriptions : Mapping[str, str] | None = None
    ) -> go.Figure:
        """
        One Plotly figure containing N independently-scaled treemaps in a
        horizontal row, one per facet.

        Each facet gets its own subplot domain, so tile area is meaningful
        within a facet but never compared across facets. This sidesteps the
        crowding problem of a single hierarchical treemap when one branch's
        totals dwarf another. Subplot titles label each facet directly under
        the chart's main title, and optional `descriptions` render a smaller
        muted second line beneath each facet title to clarify how the facets
        differ from one another.

        Args:
            facets       : Facet title to label-value mapping per facet, in
                           left-to-right display order.
            height       : Figure height in pixels.
            descriptions : Optional facet title to short-description map rendered as a
                           second muted line beneath each facet title.

        Returns:
            Configured multi-domain treemap figure.
        """
        from plotly.subplots import make_subplots

        titles = list(facets)
        muted  = self.theme.colors["muted"]
        if descriptions:
            subplot_titles = [
                (
                    f"<b>{title}</b><br>"
                    f"<span style='font-size:11px;color:{muted}'>"
                    f"{descriptions.get(title, '')}"
                    f"</span>"
                )
                for title in titles
            ]
        else:
            subplot_titles = titles

        fig = make_subplots(
            cols           = len(titles),
            rows           = 1,
            specs          = [[{"type": "domain"} for _ in titles]],
            subplot_titles = subplot_titles
        )
        fig.add_traces(
            [
                go.Treemap(
                    labels   = list(items),
                    marker   = {"cornerradius": 4},
                    parents  = [""] * len(items),
                    textinfo = "label+value",
                    values   = list(items.values())
                )
                for items in facets.values()
            ],
            cols = list(range(1, len(facets) + 1)),
            rows = [1] * len(facets)
        )
        return self._apply_layout(height, fig)

    def funnel(
        self,
        height : int,
        stages : Mapping[str, int | float]
    ) -> go.Figure:
        """
        Horizontal funnel showing progressive narrowing.

        Stage names are decorated with their values in parentheses so the
        rendered label includes both name and count.

        Args:
            height : Figure height in pixels.
            stages : Stage name to value mapping in display order.
        """
        return self._apply_layout(
            height,
            go.Funnel(
                marker   = dict(color=self.theme.colors["accent"]),
                textinfo = "value+percent initial",
                x        = list(stages.values()),
                y        = [f"{name} ({count:,})" for name, count in stages.items()]
            )
        )

    def heatmap(
        self,
        data    : Mapping[str, Sequence[float]],
        columns : Sequence[str] | None = None,
        height  : int           | None = None,
        x_title : str                  = "",
        y_title : str                  = ""
    ) -> go.Figure:
        """
        Annotated heatmap with diverging color scale.

        Rows come from `data.keys()` and cell values from `data.values()`.
        `columns` defaults to the row labels (square heatmaps) and `height`
        auto-scales by row count.

        Args:
            columns : Column axis labels, defaults to row labels.
            data    : Row label to row values mapping.
            height  : Figure height in pixels, auto-scaled when omitted.
            x_title : X-axis title.
            y_title : Y-axis title.
        """
        labels = list(data)
        return self._apply_layout(
            height or max(400, len(labels) * 28),
            go.Heatmap(
                colorscale   = "Teal",
                texttemplate = "%{z:.2f}",
                x            = columns if columns is not None else labels,
                y            = labels,
                z            = list(data.values())
            ),
            x_title         = x_title,
            xaxis_side      = "top",
            y_title         = y_title,
            yaxis_autorange = "reversed"
        )

    def histogram(
        self,
        height  : int,
        nbins   : int,
        x       : Sequence[float],
        x_title : str,
        y_title : str
    ) -> go.Figure:
        """
        Histogram with themed accent color.

        Args:
            height  : Figure height in pixels.
            nbins   : Number of bins.
            x       : Values to bin.
            x_title : X-axis label.
            y_title : Y-axis label.

        Returns:
            Configured histogram figure.
        """
        return self._apply_layout(
            height,
            go.Histogram(
                marker = dict(color=self.theme.colors["accent"]),
                nbinsx = nbins,
                x      = x
            ),
            x_title = x_title,
            y_title = y_title
        )

    def landscape(
        self,
        coordinates     : Sequence[float],
        legend_families : str,
        legend_resume   : str,
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
        brokerage = self.pathway.centrality
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
            550, traces,
            hovermode  = "closest",
            showlegend = True,
            x_title    = x_title,
            y_title    = y_title
        )

    def ranking_bar(self, ranking: SectorRanking, title: str) -> go.Figure:
        """
        Horizontal bar chart of a `SectorRanking`, colored per-cluster by
        parent sector and auto-scaled by cluster count.
        """
        return self.bar(
            color      = self.theme.sector_colors(ranking.sectors),
            data       = ranking.value_map,
            height     = max(300, len(ranking.labels) * 26),
            horizontal = True,
            title      = title
        )

    def timeline(
        self,
        dates  : Sequence[date],
        height : int,
        hover  : Sequence[str]
    ) -> go.Figure:
        """
        Strip scatter plotting one dot per posting along a date axis.

        The y-axis is hidden so the chart reads as a one-dimensional
        timeline, with date clustering visible by horizontal density and the
        company or title revealed on hover. Marker size and alpha are kept
        low so dense regions remain legible.

        Args:
            dates  : Posting dates, one per dot.
            height : Figure height in pixels.
            hover  : Hover label per dot, parallel to `dates`.

        Returns:
            Configured timeline scatter figure.
        """
        return self._apply_layout(
            height,
            go.Scatter(
                hoverinfo = "text+x",
                hovertext = hover,
                marker    = dict(
                    color   = self.theme.colors["accent"],
                    opacity = 0.65,
                    size    = 9
                ),
                mode      = "markers",
                x         = dates,
                y         = [1] * len(dates)
            ),
            yaxis = dict(visible=False)
        )

    def violin(
        self,
        groups  : dict[str, list[float]],
        height  : int,
        y_title : str
    ) -> go.Figure:
        """
        Violin plots grouped by label, with Plotly's default colorway
        assigning distinct palette colors per violin in rendering order.

        Args:
            groups  : Label to list of values.
            height  : Figure height in pixels.
            y_title : Y-axis label.

        Returns:
            Configured violin figure.
        """
        return self._apply_layout(
            height,
            [
                go.Violin(
                    box      = dict(visible=True),
                    meanline = dict(visible=True),
                    name     = key,
                    y        = values
                )
                for key, values in sorted(groups.items())
                if values
            ],
            showlegend = False,
            y_title    = y_title
        )

