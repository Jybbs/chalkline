"""
Map tab renderer composing the career pathway map, sidebar identity
card, route verdict with skill evidence, and collapsible resource
drawers.

The "You are here" sidebar card occupies the left 30% with the map
widget on the right 70%. Below the header, the default
state surfaces the matched career's skill profile, credentials, and
postings. When the user clicks a destination node, the view switches
to a transition route answering: is this worth it, what credentials
bridge the gap, and who's hiring.
"""

from marimo import Html, accordion

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import RouteDetail


def map_tab(
    ctx     : TabContext,
    route   : RouteDetail,
    sidebar : Html,
    widget  : Html
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
        ctx.layout.stack(
            sidebar, widget,
            align     = "start",
            direction = "h",
            widths    = [0.3, 0.7]
        ),

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
