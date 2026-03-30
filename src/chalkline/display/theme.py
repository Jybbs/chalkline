"""
Reactive Plotly theme for the Chalkline Marimo dashboard.

Wraps Marimo's dark/light toggle so that every color and label accessor
evaluates lazily at render time. Chart builders and layout helpers receive a
single `Theme` instance rather than a bag of closures and constants.
"""

from bisect           import bisect
from collections.abc  import Sequence
from types            import MappingProxyType
from typing           import Callable


class Theme:
    """
    Single source of truth for reactive color and label state across the
    Chalkline dashboard.

    Properties re-evaluate on every access so that Marimo's lazy tab
    rendering always reads the current theme mode. Do not cache any property
    value across renders.
    """

    def __init__(
        self,
        dark_fn     : Callable[[], bool],
        jz_labels   : dict[str, str],
        type_labels : dict[str, str]
    ):
        """
        Args:
            dark_fn     : Returns `True` when the active Marimo theme is dark.
            jz_labels   : Job Zone int to display label.
            type_labels : O*NET skill type key to display label.
        """
        self.dark_fn     = dark_fn
        self.jz_labels   = jz_labels
        self.type_labels = type_labels

    @property
    def colors(self) -> MappingProxyType[str, str]:
        """
        Active semantic color palette keyed by role name.

        Returns:
            Immutable mapping with keys `accent`, `error`, `foreground`, `muted`,
            `primary`, `success`.
        """
        return MappingProxyType({
            "accent"     : "#7db3e0",
            "error"      : "#e8876f",
            "foreground" : "#ebebeb",
            "muted"      : "#999999",
            "primary"    : "#E8C840",
            "success"    : "#8cc5a3"
        } if self.dark_fn() else {
            "accent"     : "#3b6298",
            "error"      : "#b33a2b",
            "foreground" : "#1a1a1a",
            "muted"      : "#737373",
            "primary"    : "#8a6b1e",
            "success"    : "#2d6e38"
        })

    @property
    def sector_colors(self) -> MappingProxyType[str, str]:
        """
        Construction sector to hex color mapping.

        Returns:
            Immutable mapping of sector name to hex color.
        """
        return MappingProxyType({
            "Building Construction" : "#5a9fd4",
            "Heavy Civil"           : "#e0854a",
            "Specialty Trade"       : "#5cb878"
        } if self.dark_fn() else {
            "Building Construction" : "#2e6da4",
            "Heavy Civil"           : "#b05a20",
            "Specialty Trade"       : "#267a4b"
        })

    @property
    def template(self) -> str:
        """
        Active Plotly template name.

        Returns:
            `"plotly_dark"` or `"plotly_white"`.
        """
        return "plotly_dark" if self.dark_fn() else "plotly_white"

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
        Resolve color role names to hex values from the active palette.

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
            Hex color string from the active palette.
        """
        return self.colors[("error", "primary", "success")[bisect([40, 70], score)]]

    def type_label(self, skill_type: str) -> str:
        """
        Human-readable display label for an O*NET skill type.

        Args:
            skill_type: Raw type key such as `"dwa"` or `"skill"`.

        Returns:
            Display label such as `"Detailed Work Activities"`.
        """
        return self.type_labels.get(skill_type, skill_type.title())
