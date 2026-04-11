"""
Map tab renderer composing the career pathway map, sidebar identity
card, route panel, and collapsible evidence and action drawers.

The map widget occupies the left 70% of the layout with a persistent
"You are here" sidebar card on the right. Below the map, the route
card updates reactively when the user clicks a destination node, and
the evidence and action drawers provide drill-down detail for the
selected career move.
"""

from htpy       import div, span, strong
from markupsafe import Markup
from marimo     import accordion, Html

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import RouteDetail, TabContent


def _build_actions(
    ctx   : TabContext,
    route : RouteDetail,
    tab   : TabContent
) -> Html:
    """
    Action drawer content with credential, employer, and job board
    cards filtered to the selected destination.

    Args:
        ctx   : Shared tab dependencies.
        route : Joined route data for the selected career move.
        tab   : Map tab content for fallback text.

    Returns:
        Card grid or a fallback callout.
    """
    dst   = route.destination
    ref   = ctx.reference
    cards = [
        *(ctx.layout.credential_card(c, ctx.theme) for c in route.credentials[:12]),
        *(
            ctx.layout.employer_card(**emp)
            for emp in ref.match_employers(dst.postings)[:8]
        ),
        *(
            ctx.layout.board_card(**b)
            for b in ref.filter_boards({dst.sector.lower()})
        )
    ]
    return (
        ctx.layout.grid(cards) 
        if cards else ctx.layout.callout(tab.fallbacks["no_resources"])
    )


def _build_evidence(
    ctx   : TabContext,
    route : RouteDetail,
    tab   : TabContent
) -> Html:
    """
    Evidence drawer content with ranked skill lists for strengths and
    gaps contextualized to the selected destination.

    Args:
        ctx   : Shared tab dependencies.
        route : Joined route data carrying `scored_tasks`.
        tab   : Map tab content for fallback text.

    Returns:
        Stacked skill panels or a fallback callout.
    """
    if parts := [
        ctx.layout.ranked_list(tab.chart_labels[key], skills, ctx.theme)
        for key, skills in (
            ("strengths_heading", route.top_strengths),
            ("gaps_heading",      route.top_gaps)
        )
        if skills
    ]:
        return ctx.layout.stack(*parts)
    return ctx.layout.callout(tab.fallbacks["no_gaps"])


def _build_route_card(
    ctx   : TabContext,
    route : RouteDetail,
    tab   : TabContent
) -> Html:
    """
    Career route panel showing destination, wage delta, skill transfers,
    skill gaps, and bridging credentials.

    Args:
        ctx   : Shared tab dependencies for `to_html` and `score_color`.
        route : Flattened route data from `RouteDetail.from_edges`.
        tab   : Map tab content for skill panel headings.

    Returns:
        Styled route card element.
    """
    dst     = route.destination
    section = lambda label, *body: div(".cl-route-section")[
        div(".cl-route-label")[label], *body
    ]
    sections = [
        section(tab.chart_labels[key], div[Markup(", ".join(
            str(span(style=f"color:{ctx.theme.score_color(t.pct)}")[
                f"{t.name} ({t.pct:.0f}%)"
            ])
            for t in tasks
        ))])
        for key, tasks in [
            ("strengths_heading", route.top_strengths),
            ("gaps_heading",      route.top_gaps)
        ]
        if tasks
    ]

    if top := route.top_credential:
        sections.append(
            section("Fastest path", strong[top.label], f" {top.description}")
        )

    if route.is_multi_hop:
        sections.append(section(
            f"Route ({route.step_count} steps)",
            span(".secondary")[route.transition_summary]
        ))

    return ctx.layout.to_html(
        div(".cl-route-header")[
            strong[dst.soc_title],
            " \u00b7 ",
            span(".secondary")[dst.sector],
            " \u00b7 ",
            span(".secondary")[ctx.content.labels.job_zones[dst.job_zone]]
        ],
        div(".cl-route-wage")[ctx.layout.wage_display(route.wage_delta, ctx.theme)],
        *sections,
        cls = "cl-route-card"
    )


def map_tab(
    ctx     : TabContext,
    route   : RouteDetail | None,
    sidebar : Html,
    widget  : Html
) -> Html:
    """
    Compose the full Map tab including the map widget, sidebar, route
    card, and collapsible evidence and action drawers.

    Args:
        ctx     : Shared tab dependencies.
        route   : Flattened route data for the selected destination
                  (None when no destination is selected).
        sidebar : Pre-rendered "You are here" identity card.
        widget  : Wrapped AnyWidget career pathway map.

    Returns:
        Vertically stacked map layout with route panels.
    """
    tab    = ctx.content.tab("map")
    header = ctx.layout.stack(
        widget, sidebar,
        align     = "start",
        direction = "h",
        widths    = [0.7, 0.3]
    )
    if not route:
        return ctx.layout.stack(
            header, 
            ctx.layout.callout(tab.fallbacks["no_selection"])
        )
    
    return ctx.layout.stack(
        header,
        _build_route_card(ctx, route, tab),
        accordion({
            tab.sections["evidence"].title : _build_evidence(ctx, route, tab),
            tab.sections["actions"].title  : _build_actions(ctx, route, tab)
        })
    )
