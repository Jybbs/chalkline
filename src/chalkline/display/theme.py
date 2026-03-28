"""
Reactive Plotly theme for the Chalkline Marimo dashboard.

Wraps Marimo's dark/light toggle so that every color and label
accessor evaluates lazily at render time. Chart builders and
layout helpers receive a single `Theme` instance rather than a
bag of closures and constants.
"""

from bisect import bisect
from types  import MappingProxyType
from typing import Callable


class Theme:
    """
    Single source of truth for reactive color and label state
    across the Chalkline dashboard.

    Properties re-evaluate on every access so that Marimo's lazy
    tab rendering always reads the current theme mode. Do not
    cache any property value across renders.
    """

    def __init__(self, dark_fn: Callable[[], bool]):
        """
        Args:
            dark_fn: Zero-argument callable returning `True` when
                     the active Marimo theme is dark.
        """
        self.dark_fn = dark_fn

    @property
    def colors(self) -> MappingProxyType[str, str]:
        """
        Active semantic color palette keyed by role name.

        Returns:
            Immutable mapping with keys `accent`, `error`,
            `foreground`, `muted`, `primary`, `success`.
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
        return {
            1 : "Entry Level",
            2 : "Some Preparation",
            3 : "Mid-Career",
            4 : "Experienced",
            5 : "Advanced"
        }.get(job_zone, str(job_zone))

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
        return {
            "ability"    : "Abilities",
            "dwa"        : "Detailed Work Activities",
            "knowledge"  : "Knowledge",
            "skill"      : "Skills",
            "task"       : "Tasks",
            "technology" : "Technology Skills",
            "tool"       : "Tools & Equipment"
        }.get(skill_type, skill_type.title())
