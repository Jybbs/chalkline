"""
Plotly theme for the Chalkline Marimo dashboard.

Provides a `Theme` class that owns the unified color palette, sector color
mapping, and the registered Plotly template. Each instance registers its
template with Plotly on construction, making `template="chalkline_dark"`
resolve globally.
"""

import plotly.colors        as pc
import plotly.graph_objects as go
import plotly.io            as pio

from types import MappingProxyType


class Theme:
    """
    Single source of truth for color and palette state across the Chalkline
    dashboard.
    """

    def __init__(self):
        self.colors: MappingProxyType[str, str] = MappingProxyType({
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
            "management" : "#5cb878",
            "success"    : "#8cc5a3",
            "teal"       : "#7ecfc0"
        })
        self.sectors: MappingProxyType[str, str] = MappingProxyType({
            "Building Construction"      : self.colors["building"],
            "Heavy Highway Construction" : self.colors["heavy"],
            "Construction Managers"      : self.colors["management"]
        })
        self.template = "chalkline_dark"
        self._register_template()

    def _register_template(self):
        """
        Build and register the Chalkline dark Plotly template.

        Overlays `plotly_dark` with the palette colorway, transparent
        backgrounds, dotted palette gridlines, and the Chalkline font. Text
        colors propagate to every axis tick, title, legend, indicator, and
        pie slice via Plotly's `layout.font` inheritance, which
        `plotly_dark` itself relies on and sets no descendant-level font
        overrides against.
        """
        grid        = self.colors["grid"]
        transparent = "rgba(0,0,0,0)"
        grid_base   = {"griddash": "dot", "gridcolor": grid}
        axis        = {
            **grid_base,
            "title"         : {"standoff": 12},
            "zerolinecolor" : grid
        }
        polar   = {**grid_base, "linecolor": grid}
        rounded = {"cornerradius": 4}

        pio.templates[self.template] = go.layout.Template(
            pio.templates["plotly_dark"]
        ).update({
            "data"   : {
                "bar"       : [go.Bar(marker=rounded)],
                "histogram" : [go.Histogram(marker=rounded)],
                "pie"       : [go.Pie(textposition="outside")]
            },
            "layout" : {
                "colorway"      : [self.colors[k] for k in (
                    "accent", "heavy", "success", "primary", "error",
                    "building", "management", "lavender", "cream", "teal"
                )],
                "font"          : {
                    "color"  : self.colors["foreground"],
                    "family" : "DM Sans, system-ui, sans-serif"
                },
                "margin"        : {"b": 50, "l": 30, "r": 30, "t": 20},
                "paper_bgcolor" : transparent,
                "plot_bgcolor"  : transparent,
                "polar"         : {
                    "angularaxis" : polar,
                    "bgcolor"     : transparent,
                    "radialaxis"  : polar
                },
                "xaxis"         : axis,
                "yaxis"         : {**axis, "ticklabelstandoff": 8}
            }
        })

    def credential_color(self, kind: str) -> str:
        """
        Accent color for a credential kind.

        Args:
            kind: Credential kind such as `"apprenticeship"` or `"program"`.

        Returns:
            Hex color from the palette.
        """
        match kind:
            case "apprenticeship" : return self.colors["cream"]
            case "career"         : return self.colors["highlight"]
            case "certification"  : return self.colors["lavender"]
            case "program"        : return self.colors["accent"]
            case _                : return self.colors["muted"]

    def resolve_color(self, color: str) -> str:
        """
        Hex color from a palette key, falling back to the input string when
        no palette entry matches.

        Centralizes the `colors.get(key, key)` pattern used by chart
        builders that accept either a theme key or a literal hex value.
        """
        return self.colors.get(color, color)

    def score_color(self, score: float) -> str:
        """
        Map a 0-100 percentage to a smooth red→gold→green gradient via
        Plotly's colorscale sampling.

        Interpolates between three anchor colors derived from the theme
        palette so every percentage gets a unique shade:

            0% → #e8876f  salmon (error)
           50% → #E8C840  gold   (primary)
          100% → #8cc5a3  green  (success)

        Args:
            score: Numeric score between 0 and 100.

        Returns:
            RGB color string.
        """
        t = max(0.0, min(score, 100.0)) / 100.0
        return str(pc.sample_colorscale(
            [
                [0,   self.colors["error"]],
                [0.5, self.colors["primary"]],
                [1,   self.colors["success"]]
            ],
            t
        )[0])

    def sector_background(self, sector: str) -> str:
        """
        Hex color for a sector name, falling back to muted when the sector
        is not in the three-sector palette.
        """
        return self.sectors.get(sector, self.colors["muted"])

    def sector_colors(self, sectors: list[str]) -> list[str]:
        """
        Per-bar sector colors aligned with an iterable of sector names.

        Centralizes the `theme.sectors[s]` lookup so chart call sites across
        the methods tab share one resolution path instead of repeating the
        same comprehension.
        """
        return [self.sectors[s] for s in sectors]

    def wage_color(self, delta: float) -> str:
        """
        Hex color for a signed wage delta: success for non-negative, error
        for negative.
        """
        return self.colors["success" if delta >= 0 else "error"]

