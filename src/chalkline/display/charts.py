"""
Plotly chart factory for the Chalkline Marimo dashboard.

Provides a `Charts` class backed by a `Theme` and a fitted
`CareerPathwayGraph`. Every method reads the theme at call time for reactive
dark/light styling. Call sites pass pre-extracted data and receive a
configured `go.Figure`. All color resolution happens internally via theme.
"""

import numpy                as np
import plotly.graph_objects as go

from collections.abc       import Iterable, Mapping, Sequence
from itertools             import chain
from plotly.basedatatypes  import BaseTraceType
from sklearn.preprocessing import normalize

from chalkline.display.schemas   import SectorRanking
from chalkline.display.theme     import Theme
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
            f"JZ {c.job_zone} · {c.size} postings"
            for cid in node_ids
        ]

    def bar(
        self,
        height     : int,
        title      : str,
        color      : str | Sequence = "accent",
        data       : Mapping | None = None,
        horizontal : bool           = False,
        line       : Mapping | None = None,
        x          : Sequence       = (),
        y          : Sequence       = ()
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
            labels = list(data)
            values = list(data.values())
            x      = values if horizontal else labels
            y      = labels if horizontal else values

        layout = (
            dict(x_title=title, yaxis=dict(autorange="reversed"))
            if horizontal
            else dict(y_title=title)
        )

        color     = (
            self.theme.resolve_color(color) if isinstance(color, str)
            else [self.theme.resolve_color(c) for c in color]
        )
        bar_trace = go.Bar(
            marker      = dict(color=color),
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
                    name   = "Cumulative",
                    x      = list(line),
                    y      = list(line.values())
                )]),
                height,
                legend = dict(orientation="h", y=-0.2),
                **layout
            )

        return self._apply_layout(go.Figure(bar_trace), height, **layout)

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

        Brokerage values become integer-percentage x-coordinates (min 1
        for marker visibility), silhouette values become one-decimal
        y-coordinates and marker color/size, and brokerage labels carry
        the hover text.

        Args:
            brokerage  : Per-cluster brokerage centrality ranking.
            height     : Figure height in pixels.
            silhouette : Per-cluster silhouette coefficient ranking.

        Returns:
            Configured scatter figure.
        """
        magnitudes = [round(s * 100, 1) for s in silhouette.values]
        return self._apply_layout(
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
            height  = height,
            x_title = x_title,
            y_title = y_title
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
        stages : Mapping[str, int | float]
    ) -> go.Figure:
        """
        Horizontal funnel showing progressive narrowing.

        Stage names are decorated with their values in parentheses so
        the rendered label includes both name and count.

        Args:
            height : Figure height in pixels.
            stages : Stage name to value mapping in display order.
        """
        fig = go.Figure(go.Funnel(
            marker    = dict(color=self.theme.colors["accent"]),
            textinfo  = "value+percent initial",
            x         = list(stages.values()),
            y         = [f"{name} ({count:,})" for name, count in stages.items()]
        ))
        return self._apply_layout(fig, height)

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
            go.Heatmap(
                colorscale   = "Teal",
                texttemplate = "%{z:.2f}",
                x            = columns if columns is not None else labels,
                y            = labels,
                z            = list(data.values())
            ),
            height          = height or max(400, len(labels) * 28),
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
        fig = go.Figure(go.Histogram(
            marker = dict(color=self.theme.colors["accent"]),
            nbinsx = nbins,
            x      = x
        ))
        return self._apply_layout(
            fig, height,
            x_title = x_title,
            y_title = y_title
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
            traces, 550,
            hovermode  = "closest",
            showlegend = True,
            title      = title,
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

    def treemap(self, data: Mapping[str, int | float], height: int) -> go.Figure:
        """
        Treemap of label/value tiles.

        Args:
            data   : Label-to-value mapping rendered as tiles.
            height : Figure height in pixels.

        Returns:
            Configured treemap figure.
        """
        fig = go.Figure(go.Treemap(
            labels   = list(data),
            marker   = {"cornerradius": 4},
            textinfo = "label+value",
            values   = list(data.values())
        ))
        return self._apply_layout(fig, height)

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
        fig = go.Figure(data=[
            go.Violin(
                box      = dict(visible=True),
                meanline = dict(visible=True),
                name     = key,
                y        = values
            )
            for key, values in sorted(groups.items())
            if values
        ])
        return self._apply_layout(
            fig, height,
            showlegend = False,
            y_title    = y_title
        )

