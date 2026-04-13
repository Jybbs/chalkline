"""
Map tab renderer composing the force-directed career pathway map,
route verdict with skill evidence, and collapsible resource drawers.

The force-directed map widget spans the full width with the matched
career rendered as an enriched hero card within the SVG itself. Below
the map, the default state surfaces the matched career's skill
profile, credentials, and postings. When the user clicks a destination
node, the view switches to a transition route.
"""

from marimo import Html, accordion

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import RouteDetail


def map_tab(
    ctx    : TabContext,
    route  : RouteDetail,
    widget : Html
) -> Html:
    """
    Compose the full Map tab.

    Default state shows the matched career's skill profile,
    credentials, and postings. Clicking a different node switches to
    a transition route between the matched career and the selected
    destination.
    """
    tab = ctx.content.tab("map")

    return ctx.layout.stack(
        widget,

        ctx.routes.card(
            ctx.routes.verdict(route, tab),
            ctx.routes.recipe(ctx.pipeline, ctx.result, route, tab),
            ctx.routes.postings(route, tab)
        ),

        accordion({
            tab.sections["evidence"].title : ctx.routes.evidence(
                route, tab
            ),
            tab.sections["resources"].title : ctx.routes.resources(
                ctx.pipeline, ctx.reference, route, tab
            )
        })
    )
