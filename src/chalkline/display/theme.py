"""
Plotly theme for the Chalkline Marimo dashboard.

Provides a `Theme` class with color and label accessors, and registers a
custom `chalkline_dark` Plotly template on import. The template overlays
`plotly_dark` with explicit font colors for every text-bearing component
so that chart labels, gauge ticks, polar axis text, and pie slice labels
are always legible.
"""

import plotly.graph_objects as go
import plotly.io            as pio

from bisect          import bisect
from collections.abc import Sequence
from types           import MappingProxyType
from typing          import Callable


def _register_template() -> None:
    """
    Build and register the Chalkline dark Plotly template.

    Overlays `plotly_dark` with explicit font colors on cartesian axes,
    polar axes, indicator gauges, and pie slices. The base template
    supplies backgrounds, gridlines, and colorways while the overlay
    controls text legibility.
    """
    fg   = "#ebebeb"
    grid = "#333333"

    t = go.layout.Template()

    t.layout.font = dict(color=fg, family="DM Sans, system-ui, sans-serif")
    t.layout.paper_bgcolor = "rgba(0,0,0,0)"
    t.layout.plot_bgcolor = "rgba(0,0,0,0)"

    axis_shared = dict(
        griddash      = "dot",
        gridcolor     = grid,
        tickfont      = dict(color=fg),
        title         = dict(font=dict(color=fg), standoff=12),
        zerolinecolor = grid,
    )
    t.layout.xaxis = dict(**axis_shared)
    t.layout.yaxis = dict(**axis_shared, ticklabelstandoff=8)

    t.layout.polar = dict(
        angularaxis = dict(
            griddash  = "dot",
            gridcolor = grid,
            linecolor = grid,
            tickfont  = dict(color=fg),
        ),
        bgcolor = "rgba(0,0,0,0)",
        radialaxis = dict(
            griddash  = "dot",
            gridcolor = grid,
            linecolor = grid,
            tickfont  = dict(color=fg),
        ),
    )

    t.layout.colorway = [
        "#7db3e0",
        "#e0854a",
        "#8cc5a3",
        "#E8C840",
        "#e8876f",
        "#5a9fd4",
        "#5cb878",
        "#c49bdb",
        "#d4a574",
        "#7ecfc0",
    ]
    t.layout.legend = dict(font=dict(color=fg))

    t.data.indicator = [go.Indicator(
        gauge  = dict(axis=dict(tickfont=dict(color=fg))),
        number = dict(font=dict(color=fg)),
        title  = dict(font=dict(color=fg)),
    )]

    t.data.pie = [go.Pie(
        insidetextfont  = dict(color=fg),
        outsidetextfont = dict(color=fg),
        textfont        = dict(color=fg),
        textposition    = "outside",
    )]

    base = pio.templates["plotly_dark"]
    merged = go.layout.Template(base)
    merged.update(t)
    pio.templates["chalkline_dark"] = merged


_register_template()


class Theme:
    """
    Single source of truth for color and label state across the
    Chalkline dashboard.
    """

    COLORS: MappingProxyType[str, str] = MappingProxyType({
        "accent"     : "#7db3e0",
        "error"      : "#e8876f",
        "foreground" : "#ebebeb",
        "muted"      : "#999999",
        "primary"    : "#E8C840",
        "success"    : "#8cc5a3",
    })

    SECTOR_COLORS: MappingProxyType[str, str] = MappingProxyType({
        "Building Construction" : "#5a9fd4",
        "Heavy Civil"           : "#e0854a",
        "Specialty Trade"       : "#5cb878",
    })

    TEMPLATE: str = "chalkline_dark"

    def __init__(
        self,
        dark_fn     : Callable[[], bool],
        jz_labels   : dict[str, str],
        type_labels : dict[str, str]
    ):
        """
        Args:
            dark_fn     : Retained for API compatibility but always True.
            jz_labels   : Job Zone int to display label.
            type_labels : O*NET skill type key to display label.
        """
        self.dark_fn     = dark_fn
        self.jz_labels   = jz_labels
        self.type_labels = type_labels

    @property
    def colors(self) -> MappingProxyType[str, str]:
        """
        Semantic color palette keyed by role name.

        Returns:
            Immutable mapping with keys `accent`, `error`, `foreground`,
            `muted`, `primary`, `success`.
        """
        return self.COLORS

    @property
    def sector_colors(self) -> MappingProxyType[str, str]:
        """
        Construction sector to hex color mapping.

        Returns:
            Immutable mapping of sector name to hex color.
        """
        return self.SECTOR_COLORS

    @property
    def template(self) -> str:
        """
        Active Plotly template name.
        """
        return self.TEMPLATE

    def jz_label(self, job_zone: int) -> str:
        """
        Human-readable label for an O*NET Job Zone level.

        Args:
            job_zone: Integer 1 through 5.

        Returns:
            Label such as `"Entry Level"` or `"Advanced"`.
        """
        return self.jz_labels.get(str(job_zone), str(job_zone))

    def resolve(self, color: str | Sequence[str]) -> str | list[str]:
        """
        Resolve color role names to hex values from the palette.

        Strings not found in the palette pass through unchanged, so
        pre-resolved hex values and numeric colorscale inputs are safe.

        Args:
            color: Single role name or hex string, or a sequence of them.

        Returns:
            Resolved hex string or list of hex strings.
        """
        c = self.colors
        if isinstance(color, str):
            return c.get(color, color)
        return [c.get(v, v) for v in color]

    def score_color(self, score: float) -> str:
        """
        Map a 0-100 percentage score to a semantic hex color.

        Args:
            score: Numeric score between 0 and 100.

        Returns:
            Hex color string from the palette.
        """
        return self.colors[("error", "primary", "success")[bisect([40, 70], score)]]

    def score_tier(self, score: float) -> str:
        """
        Map a 0-100 percentage score to a CSS tier class suffix.

        Uses the same 40/70 thresholds as `score_color` so the HTML
        skill tree and Plotly gauge stay in sync.

        Args:
            score: Numeric score between 0 and 100.

        Returns:
            `"low"`, `"mid"`, or `"high"`.
        """
        return ("low", "mid", "high")[bisect([40, 70], score)]

    def type_label(self, skill_type: str) -> str:
        """
        Human-readable display label for an O*NET skill type.

        Args:
            skill_type: Raw type key such as `"dwa"` or `"skill"`.

        Returns:
            Display label such as `"Detailed Work Activities"`.
        """
        return self.type_labels.get(skill_type, skill_type.title())
