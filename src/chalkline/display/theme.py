"""
Plotly theme for the Chalkline Marimo dashboard.

Provides a `Theme` class that owns the unified color palette, sector color
mapping, skill type labels, Job Zone labels, and the registered Plotly
template. Each instance registers its template with Plotly on construction,
making `template="chalkline_dark"` resolve globally.
"""

import plotly.graph_objects as go
import plotly.io            as pio

from bisect     import bisect
from types      import MappingProxyType


class Theme:
    """
    Single source of truth for color and label state across the Chalkline
    dashboard.
    """

    def __init__(
        self,
        jz_labels   : dict[str, str],
        type_labels : dict[str, str]
    ):
        """
        Args:
            jz_labels   : Job Zone integer key to display label.
            type_labels : O*NET skill type key to display label.
        """
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
            "teal"       : "#7ecfc0",
        })
        self.sectors = MappingProxyType({
            "Building Construction"      : self.colors["building"],
            "Heavy Highway Construction" : self.colors["heavy"],
            "Specialty Trade"            : self.colors["specialty"],
        })
        self.jz_labels   = jz_labels
        self.type_labels = type_labels
        self.template    = "chalkline_dark"
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
            zerolinecolor = self.colors["grid"],
        )

        polar_axis = dict(
            griddash  = "dot",
            gridcolor = self.colors["grid"],
            linecolor = self.colors["grid"],
        )

        overlay = {
            "layout": {
                "colorway": [self.colors[k] for k in (
                    "accent", "heavy", "success", "primary", "error",
                    "building", "specialty", "lavender", "cream", "teal",
                )],
                "font"          : dict(
                    color  = self.colors["foreground"],
                    family = "DM Sans, system-ui, sans-serif",
                ),
                "margin"        : dict(b=50, l=30, r=30, t=20),
                "paper_bgcolor" : "rgba(0,0,0,0)",
                "plot_bgcolor"  : "rgba(0,0,0,0)",
                "polar"         : dict(
                    angularaxis = polar_axis,
                    bgcolor     = "rgba(0,0,0,0)",
                    radialaxis  = polar_axis,
                ),
                "xaxis"         : axis_shared,
                "yaxis"         : dict(**axis_shared, ticklabelstandoff=8),
            },
            "data": {
                "bar"       : [go.Bar(marker=dict(cornerradius=4))],
                "histogram" : [go.Histogram(marker=dict(cornerradius=4))],
                "pie"       : [go.Pie(textposition="outside")],
            },
        }

        merged = go.layout.Template(pio.templates["plotly_dark"])
        merged.update(overlay)
        pio.templates[self.template] = merged

    def jz_label(self, job_zone: int) -> str:
        """
        Human-readable label for an O*NET Job Zone level.

        Args:
            job_zone: Integer 1 through 5.

        Returns:
            Label such as `"Entry Level"` or `"Advanced"`.
        """
        return self.jz_labels.get(str(job_zone), str(job_zone))

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

        Uses the same 40/70 thresholds as `score_color` so the HTML skill
        tree and Plotly gauge stay in sync.

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
