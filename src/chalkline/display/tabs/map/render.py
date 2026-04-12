"""
Map tab renderer composing the career pathway map, sidebar identity
card, neighborhood exploration view, recipe-style route panel, and
collapsible evidence and resource drawers.

The map widget occupies the left 70% of the layout with a persistent
"You are here" sidebar card on the right. Below the map, the default
state surfaces the CL-13 neighborhood exploration view from the
matched cluster's Reach (advancement paths and lateral pivots with
per-edge credentials). When the user clicks a destination node, the
neighborhood view is replaced by a route card answering: is this
worth it, what credentials bridge the gap, and who's hiring.
"""

import numpy as np

from htpy       import div, Element, p, span, strong
from markupsafe import Markup
from marimo     import accordion, Html

from chalkline.display.loaders  import TabContext
from chalkline.display.schemas  import RelevantJobBoards
from chalkline.display.schemas  import RouteDetail, TabContent
from chalkline.matching.schemas import ScoredTask
from chalkline.pathways.schemas import Credential


# ── Helpers ─────────────────────────────────────────────────────


def _credential_percentiles(
    ctx   : TabContext,
    route : RouteDetail
) -> dict[str, int]:
    """
    Compute each route credential's percentile rank among ALL
    credentials for this destination.

    A "Top 3%" label means only 3% of the 325 credentials in the
    catalog score higher against this destination cluster. This is
    a credible, interpretable metric that justifies the ranking
    without exposing an opaque cosine float.
    """
    col_idx        = ctx.pipeline.clusters.cluster_index[route.destination.cluster_id]
    similarity_col = ctx.pipeline.graph.credential_similarity[:, col_idx]
    n_total        = len(similarity_col)
    cred_list, _   = ctx.pipeline.graph.credential_matrix

    score_by_label = {
        c.label: float(similarity_col[i])
        for i, c in enumerate(cred_list)
    }

    percentiles = {}
    for credential in route.credentials:
        score = score_by_label.get(credential.label, 0)
        rank  = int(np.sum(similarity_col > score))
        percentiles[credential.label] = max(1, round((rank / max(n_total, 1)) * 100))
    return percentiles


def _percentile_calibrate(
    tasks     : list[ScoredTask],
    all_tasks : list[ScoredTask]
) -> list[ScoredTask]:
    """
    Replace raw cosine similarities with percentile ranks against
    the full scored-task set.

    For each task, the calibrated similarity equals the fraction of
    all tasks with strictly lower similarity. The top strength
    approaches 100%, the bottom gap approaches 0%, and everything
    in between reflects its relative position regardless of how
    narrow the raw cosine band is.
    """
    sorted_sims = sorted(t.similarity for t in all_tasks)
    n           = max(len(sorted_sims) - 1, 1)

    def percentile(sim: float) -> float:
        return sum(1 for s in sorted_sims if s < sim) / n

    return [
        t.model_copy(update={"similarity": percentile(t.similarity)})
        for t in tasks
    ]


# ── Route card sections ─────────────────────────────────────────


def _build_verdict(
    ctx   : TabContext,
    route : RouteDetail,
    tab   : TabContent
) -> Element:
    """
    Section 1: fit meter + wage bars + bold verdict + extras.

    Key values (percentage, counts, destination name) are wrapped in
    HTML strong tags via Markup so they render bold in the htpy tree.
    """
    fit_pct     = route.fit_percentage
    source_wage = route.source_wage      or 0
    dest_wage   = route.destination_wage or 0
    max_wage    = max(source_wage, dest_wage, 1)
    dst         = route.destination

    wage_bars = div(".cl-wage-bars")[
        div(".cl-wage-bar-row")[
            span(".cl-wage-bar-label")[
                f"${source_wage / 1000:.0f}k" if source_wage else "\u2014"
            ],
            div(".cl-wage-bar")[
                div(
                    ".cl-wage-bar-fill.cl-wage-bar-source",
                    style = f"width:{source_wage / max_wage * 100:.0f}%"
                )
            ]
        ],
        div(".cl-wage-bar-row")[
            span(".cl-wage-bar-label")[
                f"${dest_wage / 1000:.0f}k" if dest_wage else "\u2014"
            ],
            div(".cl-wage-bar")[
                div(
                    ".cl-wage-bar-fill.cl-wage-bar-dest",
                    style = f"width:{dest_wage / max_wage * 100:.0f}%"
                )
            ]
        ]
    ]

    fit_meter = div(".cl-fit-meter")[
        span[f"{fit_pct}%"],
        span(".cl-fit-meter-label")[tab.chart_labels["fit_meter_label"]]
    ]

    verdict = Markup(
        f"You're a <strong>{fit_pct}%</strong> match. "
        f"You already have <strong>{route.demonstrated_count}</strong> of "
        f"<strong>{route.total_tasks}</strong> skills common to "
        f"<strong>{dst.soc_title}</strong> roles."
    )

    extras: list = []
    if route.wage_delta is not None:
        sign = "+" if route.wage_delta >= 0 else ""
        extras.append(strong[f"{sign}${route.wage_delta:,.0f}/yr"])
        extras.append(" \u00b7 ")
    extras.extend([strong[f"{dst.size}"], " open positions"])

    bright_record = ctx.labor.items.get(dst.soc_title)
    if bright_record and bright_record.bright_outlook:
        extras.append(" \u00b7 ")
        extras.append(span(".cl-bright-outlook")[
            f"\u2605 {tab.chart_labels['bright_outlook']}"
        ])

    return div(".cl-route-hero")[
        div(".cl-route-hero-row")[fit_meter, wage_bars],
        p(".cl-verdict")[verdict],
        div(".cl-route-hero-extras")[extras]
    ]


def _build_recipe(
    ctx   : TabContext,
    route : RouteDetail,
    tab   : TabContent
) -> Element:
    """
    Section 2: top gaps as context, then credentials grouped by kind
    using the same credential_card styling as the Data tab.

    Opens with the user's top 3 gap skills so the user understands
    WHY these credentials are recommended. Credentials are rendered
    via `Layout.credential_card` for visual consistency with the Data
    tab, annotated with a percentile relevance label ("Top X%")
    computed from their rank among all 325 credentials for this
    destination. Grouped by kind in a horizontal stack matching the
    Data tab's three-column pattern.
    """
    top_gaps    = route.top_gaps[:3]
    percentiles = _credential_percentiles(ctx, route)

    gap_names = [
        strong[t.name[:80] + ("\u2026" if len(t.name) > 80 else "")]
        for t in top_gaps
    ]
    gap_context = div(".cl-recipe-context")[
        div(".cl-route-label")["Your biggest gaps for this role"],
        *(div(".cl-recipe-gap")[name] for name in gap_names)
    ] if gap_names else ""

    by_kind: dict[str, list[Credential]] = {
        "apprenticeship" : [],
        "certification"  : [],
        "program"        : []
    }
    for credential in route.credentials:
        if credential.kind in by_kind:
            by_kind[credential.kind].append(credential)

    nonempty = {k: v[:3] for k, v in by_kind.items() if v}
    if not nonempty:
        return div(".cl-recipe")[
            gap_context,
            div(".secondary")[tab.fallbacks["no_credentials"]]
        ]

    def annotated_card(credential: Credential) -> Html:
        pct  = percentiles.get(credential.label, 50)
        card = ctx.layout.credential_card(credential, ctx.theme)
        return ctx.layout.to_html(
            Markup(card.text),
            span(".cl-recipe-relevance")[f"Top {pct}% for this transition"],
            cls = "cl-recipe-annotated"
        )

    credential_columns = ctx.layout.stack(
        *(
            ctx.layout.grid(annotated_card(c) for c in cards)
            for cards in nonempty.values()
        ),
        direction = "h",
        widths    = [1] * len(nonempty)
    )

    return div(".cl-recipe")[
        gap_context,
        div(".cl-route-label")[tab.chart_labels["recipe_heading"]],
        Markup(credential_columns.text)
    ]


def _build_postings(
    ctx   : TabContext,
    route : RouteDetail,
    tab   : TabContent
) -> Element:
    """
    Section 3: destination posting cards in a 5-column grid,
    up to 2 rows (10 cards max).
    """
    postings = route.destination.postings[:10]
    if not postings:
        return div(".secondary")[tab.fallbacks["no_postings"]]

    heading = tab.chart_labels["postings_heading"].format(
        count = route.destination.size
    )
    return div[
        div(".cl-route-label")[heading],
        Markup(ctx.layout.to_html(
            [Markup(ctx.layout.posting_card(posting).text) for posting in postings],
            cls = "cl-card-grid.cl-card-grid-5"
        ).text)
    ]


def _build_route_card(
    ctx   : TabContext,
    route : RouteDetail,
    tab   : TabContent
) -> Html:
    """
    Compose the route card: verdict → credentials → postings.
    """
    return ctx.layout.to_html(
        _build_verdict(ctx, route, tab),
        _build_recipe(ctx, route, tab),
        _build_postings(ctx, route, tab),
        cls = "cl-route-card"
    )


# ── Evidence and Resource drawers ───────────────────────────────


def _build_evidence(
    ctx   : TabContext,
    route : RouteDetail,
    tab   : TabContent
) -> Html:
    """
    Evidence drawer with percentile-calibrated skill lists (10
    per side).
    """
    all_tasks = route.scored_tasks
    if parts := [
        ctx.layout.ranked_list(
            tab.chart_labels[key].format(count=count),
            _percentile_calibrate(skills, all_tasks),
            ctx.theme
        )
        for key, skills, count in (
            ("strengths_heading", route.top_strengths, route.demonstrated_count),
            ("gaps_heading",      route.top_gaps,      route.gap_count)
        )
        if skills
    ]:
        return ctx.layout.stack(*parts)
    return ctx.layout.callout(tab.fallbacks["no_gaps"])


def _build_resources(
    ctx   : TabContext,
    route : RouteDetail,
    tab   : TabContent
) -> Html:
    """
    Resource drawer with the full credential catalog, employers,
    and semantically-ranked job boards.

    Differs from the recipe section: the recipe shows the TOP picks
    with gap context and relevance annotations; this drawer is the
    complete reference catalog for deeper exploration.
    """
    destination = route.destination

    by_kind: dict[str, list[Credential]] = {
        "apprenticeship" : [],
        "certification"  : [],
        "program"        : []
    }
    for credential in route.credentials:
        if credential.kind in by_kind:
            by_kind[credential.kind].append(credential)

    nonempty = {k: v for k, v in by_kind.items() if v}
    if nonempty:
        credential_columns = ctx.layout.stack(
            *(
                ctx.layout.grid(
                    ctx.layout.credential_card(c, ctx.theme) for c in cards[:6]
                )
                for cards in nonempty.values()
            ),
            direction = "h",
            widths    = [1] * len(nonempty)
        )
    else:
        credential_columns = ctx.layout.callout(tab.fallbacks["no_credentials"])

    employers = ctx.reference.match_employers(destination.postings)[:8]
    employer_grid = (
        ctx.layout.grid(ctx.layout.employer_card(**emp) for emp in employers)
        if employers else ctx.layout.callout(tab.fallbacks["no_employers"])
    )

    boards = RelevantJobBoards.from_cluster(
        cluster   = destination,
        clusters  = ctx.pipeline.clusters,
        encoder   = ctx.pipeline.matcher.encoder,
        limit     = 6,
        reference = ctx.reference
    ).boards
    board_grid = (
        ctx.layout.grid(ctx.layout.board_chip(**b) for b in boards)
        if boards else ctx.layout.callout(tab.fallbacks["no_boards"])
    )

    return ctx.layout.stack(
        ctx.layout.header(tab, "credentials"),
        credential_columns,
        ctx.layout.header(tab, "employers"),
        employer_grid,
        ctx.layout.header(tab, "boards"),
        board_grid
    )


# ── Map tab composition ─────────────────────────────────────────


def map_tab(
    ctx     : TabContext,
    route   : RouteDetail | None,
    sidebar : Html,
    widget  : Html
) -> Html:
    """
    Compose the full Map tab.

    Default state shows a callout (checkpoint 2 will add neighborhood
    exploration). Clicked state shows the route card with evidence
    and resource drawers.
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
            tab.sections["evidence"].title  : _build_evidence(ctx, route, tab),
            tab.sections["resources"].title : _build_resources(ctx, route, tab)
        })
    )
