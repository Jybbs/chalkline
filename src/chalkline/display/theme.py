"""
Plotly theme for the Chalkline Marimo dashboard.

Provides a `Theme` class that owns the unified color palette, sector color
mapping, and the registered Plotly template. Each instance registers its
template with Plotly on construction, making `template="chalkline_dark"`
resolve globally.
"""

import plotly.graph_objects as go
import plotly.io            as pio

from bisect          import bisect
from collections.abc import Iterable
from types           import MappingProxyType


class Theme:
    """
    Single source of truth for color and palette state across the
    Chalkline dashboard.
    """

    def __init__(self):
        self.colors = MappingProxyType({
            "accent"     : "#7db3e0",
            "building"   : "#5a9fd4",
            "cream"      : "#d4a574",
            "error"      : "#e8876f",
            "foreground" : "#ebebeb",
            "grid"       : "#333333",
            "heavy"      : "#e0854a",
            "highlight"  : "#dc143c",
            "lavender"   : "#c49bdb",
            "muted"      : "#999999",
            "primary"    : "#E8C840",
            "specialty"  : "#5cb878",
            "success"    : "#8cc5a3",
            "teal"       : "#7ecfc0"
        })
        self.sectors = MappingProxyType({
            "Building Construction"      : self.colors["building"],
            "Heavy Highway Construction" : self.colors["heavy"],
            "Specialty Trade"            : self.colors["specialty"]
        })
        self.template = "chalkline_dark"
        self._register_template()

    def _register_template(self) -> None:
        """
        Build and register the Chalkline dark Plotly template.

        Overlays `plotly_dark` with the palette colorway, transparent
        backgrounds, dotted palette gridlines, and the Chalkline font. Text
        colors propagate to every axis tick, title, legend, indicator, and
        pie slice via Plotly's `layout.font` inheritance, which
        `plotly_dark` itself relies on and sets no descendant-level font
        overrides against.
        """
        axis_shared = dict(
            griddash      = "dot",
            gridcolor     = self.colors["grid"],
            title         = dict(standoff=12),
            zerolinecolor = self.colors["grid"]
        )

        polar_axis = dict(
            griddash  = "dot",
            gridcolor = self.colors["grid"],
            linecolor = self.colors["grid"]
        )

        overlay = {
            "data"   : {
                "bar"       : [go.Bar(marker=dict(cornerradius=4))],
                "histogram" : [go.Histogram(marker=dict(cornerradius=4))],
                "pie"       : [go.Pie(textposition="outside")]
            },
            "layout" : {
                "colorway"      : [self.colors[k] for k in (
                    "accent", "heavy", "success", "primary", "error",
                    "building", "specialty", "lavender", "cream", "teal"
                )],
                "font"          : dict(
                    color  = self.colors["foreground"],
                    family = "DM Sans, system-ui, sans-serif"
                ),
                "margin"        : dict(b=50, l=30, r=30, t=20),
                "paper_bgcolor" : "rgba(0,0,0,0)",
                "plot_bgcolor"  : "rgba(0,0,0,0)",
                "polar"         : dict(
                    angularaxis = polar_axis,
                    bgcolor     = "rgba(0,0,0,0)",
                    radialaxis  = polar_axis
                ),
                "xaxis"         : axis_shared,
                "yaxis"         : dict(**axis_shared, ticklabelstandoff=8)
            }
        }

        merged = go.layout.Template(pio.templates["plotly_dark"])
        merged.update(overlay)
        pio.templates[self.template] = merged

    def credential_color(self, kind: str) -> str:
        """
        Accent color for a credential kind.

        Args:
            kind: Credential kind such as `"apprenticeship"` or
                  `"program"`.

        Returns:
            Hex color from the palette.
        """
        match kind:
            case "apprenticeship" : return self.colors["cream"]
            case "certification"  : return self.colors["lavender"]
            case "program"        : return self.colors["accent"]
            case _                : return self.colors["muted"]

    def resolve_color(self, color: str) -> str:
        """
        Hex color from a palette key, falling back to the input string
        when no palette entry matches.

        Centralizes the `colors.get(key, key)` pattern used by chart
        builders that accept either a theme key or a literal hex value.
        """
        return self.colors.get(color, color)

    def score_color(self, score: float) -> str:
        """
        Map a 0-100 percentage score to a semantic hex color.

        Args:
            score: Numeric score between 0 and 100.

        Returns:
            Hex color string from the palette.
        """
        return self.colors[("error", "primary", "success")[bisect((40, 70), score)]]

    def sector_colors(self, sectors: Iterable[str]) -> list[str]:
        """
        Per-bar sector colors aligned with an iterable of sector names.

        Centralizes the `theme.sectors[s]` lookup so chart call sites
        across the methods tab share one resolution path instead of
        repeating the same comprehension.
        """
        return [self.sectors[s] for s in sectors]

    def wage_color(self, delta: float) -> str:
        """
        Hex color for a signed wage delta: success for non-negative,
        error for negative.
        """
        return self.colors["success" if delta >= 0 else "error"]

