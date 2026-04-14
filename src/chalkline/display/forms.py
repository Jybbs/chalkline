"""
Marimo UI primitive composer for interactive widgets.

Owns the runtime Marimo API so cells that need sliders and reactive
state can request composed widgets through factory methods without
threading `mo` through every downstream layer. Parallels how
`Routes` and `Charts` own their domains of HTML and figure rendering.
"""

from htpy       import span
from markupsafe import Markup
from math       import ceil, floor
from types      import ModuleType

from chalkline.display.loaders   import Layout
from chalkline.display.schemas   import WageFilter
from chalkline.pathways.clusters import Clusters
from chalkline.pathways.loaders  import LaborLoader


class Forms:
    """
    Marimo UI composer constructed once per session.

    The `mo` module is injected at construction so factory methods
    can build sliders and reactive state without taking `mo` as a
    per-call argument. Returned widget composers (`WageFilter`,
    future siblings) expose only the reactive surface their callers
    need.
    """

    def __init__(
        self,
        layout : Layout,
        mo     : ModuleType
    ):
        self.layout = layout
        self.mo     = mo

    def wage_filter(
        self,
        clusters : Clusters,
        labor    : LaborLoader
    ) -> WageFilter:
        """
        Build the minimum-salary filter rendered between the map and
        the route card.

        The slider covers the corpus wage range rounded outward to
        the nearest thousand so every cluster falls inside its
        bounds. Its default setting is the corpus floor so every
        cluster is visible on first render. Dragging lifts the
        minimum wage, pruning tier-2 cards that fall below it while
        leaving the matched cluster pinned. Debounce mode defers the
        update until the user releases the thumb, so the map
        re-renders once per gesture rather than on every tick.
        """
        wages  = sorted(
            w for c in clusters.values()
            if (w := labor[c.soc_title].annual_median)
        )
        slider = self.mo.ui.slider(
            debounce   = True,
            full_width = True,
            start      = (start := floor(wages[0] / 1000) * 1000),
            step       = 1000,
            stop       = ceil(wages[-1] / 1000) * 1000,
            value      = start,
        )

        return WageFilter(
            row = self.layout.to_html(
                span(".cl-wage-label")["Minimum Salary"],
                span(".cl-wage-row")[
                    span(".cl-wage-bound")[f"${slider.start // 1000}K"],
                    span(".cl-wage-track")[Markup(slider.text)],
                    span(".cl-wage-bound")[f"${slider.stop // 1000}K"],
                ],
                cls = "cl-wage-filter"
            ),
            slider = slider
        )
