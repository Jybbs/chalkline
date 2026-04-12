"""
Map tab renderer composing the career pathway map, sidebar identity
card, route panel, and collapsible evidence and action drawers.

The map widget occupies the left 70% of the layout with a persistent
"You are here" sidebar card on the right. Below the map, the route
card updates reactively when the user clicks a destination node, and
the evidence and action drawers provide drill-down detail for the
selected career move.
"""

from htpy       import div, Element, p, span, strong
from markupsafe import Markup
from marimo     import accordion, Html

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import ProcessStep, RelevantJobBoards
from chalkline.display.schemas import RouteDetail, TabContent
from chalkline.pathways.schemas import Credential


def _hop_label(weight: float, tab: TabContent) -> str:
    """
    Map a similarity weight to a natural-language hop label.

    Buckets the cosine similarity into three bands so the user reads
    "natural move" / "real stretch" / "big jump" instead of an opaque
    decimal. Thresholds are tuned for the construction corpus.
    """
    if weight >= 0.85:
        return tab.chart_labels["hop_natural"]
    if weight >= 0.70:
        return tab.chart_labels["hop_stretch"]
    return tab.chart_labels["hop_jump"]


def _time_label(credentials: list[Credential], tab: TabContent) -> str:
    """
    Humanize the top credential's training time for the verdict sentence.

    Returns an empty string when no credentials are attached or the
    leading credential has no recorded hours, so the verdict template
    falls back to the strengths-only variant.
    """
    if not credentials:
        return ""
    top = credentials[0]
    if not top.hours:
        return ""
    template_key = {
        "apprenticeship" : "time_apprenticeship",
        "certification"  : "time_certification",
        "program"        : "time_program"
    }.get(top.kind, "time_apprenticeship")
    return tab.chart_labels[template_key].format(hours=top.hours)


def _verdict_text(route: RouteDetail, tab: TabContent) -> str:
    """
    Build the second-person verdict sentence opening the route card.

    Picks one of three TOML templates based on the available data:
    a sentence naming both transferable strengths and a credential
    bridge, a strengths-only variant when no credential carries hours,
    and a fit-percentage fallback when neither signal is meaningful.
    """
    strengths_count = sum(1 for t in route.scored_tasks if t.demonstrated)
    soc_title       = route.destination.soc_title
    time_label      = _time_label(route.credentials, tab)
    if strengths_count and time_label:
        return tab.chart_labels["verdict_with_credential"].format(
            soc_title       = soc_title,
            strengths_count = strengths_count,
            time_label      = time_label
        )
    if strengths_count:
        return tab.chart_labels["verdict_strengths_only"].format(
            soc_title       = soc_title,
            strengths_count = strengths_count
        )
    return tab.chart_labels["verdict_fallback"].format(
        fit_pct   = route.fit_percentage,
        soc_title = soc_title
    )


def _build_verdict(
    ctx   : TabContext,
    route : RouteDetail,
    tab   : TabContent
) -> Element:
    """
    Section A hero: verdict sentence + animated fit meter + before/after
    wage bars + posting count + optional Bright Outlook badge + sample
    posting card.

    The fit meter is a circular badge that pops in via CSS keyframes
    each time the section re-renders. The wage bars are two horizontal
    fills on a shared scale so the user pre-attentively sees the
    delta. The Bright Outlook badge is sourced from the labor JSON
    record for the destination SOC title and only renders when the
    record carries the designation. The sample posting is the first
    real Maine posting from the destination cluster.
    """
    fit_pct     = route.fit_percentage
    source_wage = route.source_wage      or 0
    dest_wage   = route.destination_wage or 0
    max_wage    = max(source_wage, dest_wage, 1)

    wage_bars = div(".cl-wage-bars")[
        div(".cl-wage-bar-row")[
            span(".cl-wage-bar-label")[
                f"${source_wage / 1000:.0f}k" if source_wage else "—"
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
                f"${dest_wage / 1000:.0f}k" if dest_wage else "—"
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

    extras: list = []
    if route.wage_delta is not None:
        sign = "+" if route.wage_delta >= 0 else ""
        extras.append(strong[f"{sign}${route.wage_delta:,.0f}/yr"])
        extras.append(" \u00b7 ")
    extras.append(span[f"{route.destination.size} open positions"])

    bright_record = ctx.labor.items.get(route.destination.soc_title)
    if bright_record and bright_record.bright_outlook:
        extras.append(" \u00b7 ")
        extras.append(span(".cl-bright-outlook")[
            f"\u2605 {tab.chart_labels['bright_outlook']}"
        ])

    children: list = [
        p(".cl-verdict")[_verdict_text(route, tab)],
        div(".cl-route-hero-row")[fit_meter, wage_bars],
        div(".cl-route-hero-extras")[extras]
    ]

    if route.destination.postings:
        children.append(div[
            div(".cl-route-label")[tab.chart_labels["recent_posting"]],
            Markup(ctx.layout.posting_card(route.destination.postings[0]).text)
        ])

    return div(".cl-route-hero")[children]


def _build_bridge(
    ctx   : TabContext,
    route : RouteDetail,
    tab   : TabContent
) -> Element:
    """
    Section B skill bridge: three-column htpy layout showing transferable
    strengths on the left, the top bridging credentials in the middle,
    and destination-relative gaps on the right.

    Caps each side at three skills (down from five) to convert "five
    failures in a row" into a finishable list. The middle column carries
    a single encouragement line tuned to whether the user already brings
    most of the skills or only a few.
    """
    strengths = [t for t in route.scored_tasks if t.demonstrated][:3]
    gaps      = sorted(
        (t for t in route.scored_tasks if not t.demonstrated),
        key = lambda t: t.similarity
    )[:3]

    def skill_row(task) -> Element:
        color = ctx.theme.score_color(task.pct)
        return div(".cl-skill-row", style=f"--row-color:{color}")[
            span(".cl-skill-name")[task.name],
            div(".cl-skill-bar")[
                div(".cl-skill-fill", style=f"width:{task.pct}%")
            ],
            span(".cl-skill-pct")[f"{task.pct:.0f}%"]
        ]

    left_column = div(".cl-bridge-side")[
        div(".cl-route-label")[tab.chart_labels["bridge_strengths"]],
        *(skill_row(t) for t in strengths)
    ] if strengths else div(".cl-bridge-side")[
        div(".cl-route-label")[tab.chart_labels["bridge_strengths"]],
        div(".secondary")[tab.fallbacks["no_strengths"]]
    ]

    right_column = div(".cl-bridge-side")[
        div(".cl-route-label")[tab.chart_labels["bridge_gaps"]],
        *(skill_row(t) for t in gaps)
    ] if gaps else div(".cl-bridge-side")[
        div(".cl-route-label")[tab.chart_labels["bridge_gaps"]],
        div(".secondary")[tab.fallbacks["no_gaps"]]
    ]

    top_creds = route.credentials[:2]
    if top_creds:
        cred_children: list = []
        for credential in top_creds:
            cred_children.append(div(".cl-bridge-credential")[credential.label])
            meta_parts = [credential.kind.title()]
            if credential.hours:
                meta_parts.append(f"{credential.hours:,} hours")
            cred_children.append(div(".cl-bridge-credential-meta")[
                " \u00b7 ".join(meta_parts)
            ])
        encouragement = (
            tab.chart_labels["encouragement_majority"]
            if len(strengths) >= len(gaps)
            else tab.chart_labels["encouragement_minority"]
        )
        cred_children.append(div(".cl-bridge-encouragement")[encouragement])
        middle_column = div(".cl-bridge-mid")[
            div(".cl-route-label")[tab.chart_labels["bridge_credentials"]],
            *cred_children
        ]
    else:
        middle_column = div(".cl-bridge-mid")[
            div(".cl-route-label")[tab.chart_labels["bridge_credentials"]],
            div(".secondary")[tab.fallbacks["no_credentials"]]
        ]

    return div(".cl-bridge")[left_column, middle_column, right_column]


def _build_path(
    ctx   : TabContext,
    route : RouteDetail,
    tab   : TabContent
) -> Element:
    """
    Section C path flow: extends `Layout.process_flow` with sector-color
    accents and natural-language arrow labels.

    Each path stop renders as a `ProcessStep` whose `accent` carries the
    cluster's sector color (applied via CSS as the left border), and the
    second through last steps carry an `arrow_label` bucketed from edge
    weight into "natural move" / "real stretch" / "big jump". The number
    field is left empty so the existing process_flow CSS hides it via
    `:empty`, keeping the path cards focused on SOC title and detail.
    """
    path_clusters = [ctx.pipeline.clusters[cid] for cid in route.path]
    edges         = ctx.pipeline.graph.path_edges(route.path)

    steps = []
    for i, cluster in enumerate(path_clusters):
        accent     = ctx.theme.sectors.get(cluster.sector, ctx.theme.colors["muted"])
        wage       = ctx.labor.wage(cluster.soc_title)
        zone_label = ctx.content.labels.job_zones[cluster.job_zone]
        wage_part  = f" \u00b7 ${wage / 1000:.0f}k" if wage else ""
        steps.append(ProcessStep(
            accent      = accent,
            arrow_label = _hop_label(edges[i - 1].weight, tab) if i > 0 else "",
            detail      = f"{cluster.sector} \u00b7 {zone_label}{wage_part} \u00b7 {cluster.size} jobs",
            label       = cluster.soc_title,
            number      = ""
        ))

    return div(".cl-route-path")[Markup(ctx.layout.process_flow(steps).text)]


def _build_alternative(
    ctx   : TabContext,
    route : RouteDetail,
    tab   : TabContent
) -> list:
    """
    Single-line callout describing the alternative direct route.

    Returns a one-element list when `route.alternative_path` exists so
    the route card composer can spread it cleanly with `*`. The
    alternative path is the harder direct edge when the primary is the
    smoother widest-path multi-hop route.
    """
    if not route.alternative_path:
        return []
    via = " \u2192 ".join(
        ctx.pipeline.clusters[cid].soc_title
        for cid in route.alternative_path[1:-1]
    ) or ctx.pipeline.clusters[route.alternative_path[-1]].soc_title
    return [div(".cl-route-alt")[
        tab.chart_labels["alternative_template"].format(
            step_count = len(route.alternative_path) - 1,
            via        = via
        )
    ]]


def _build_stats_and_cta(
    ctx   : TabContext,
    route : RouteDetail,
    tab   : TabContent
) -> Element:
    """
    Section D bottom: four-stat strip via the public `Layout.stats`
    method plus a single-line CTA pointing the user to the action plan
    drawer below.

    Stats restate the verdict and fit meter as hard numbers (deliberate
    visual rhyme, not redundancy) and add total credential time as a
    fourth signal. Routes that lack a wage delta or training time fall
    back to em-dashes so the strip stays four-wide.
    """
    total_hours  = sum(c.hours or 0 for c in route.credentials)
    wage_display = (
        f"+${route.wage_delta:,.0f}"
        if route.wage_delta is not None else "—"
    )
    pairs = [
        (tab.chart_labels["stat_wage"],     wage_display),
        (tab.chart_labels["stat_time"],     f"{total_hours:,} hrs" if total_hours else "—"),
        (tab.chart_labels["stat_fit"],      f"{route.fit_percentage}%"),
        (tab.chart_labels["stat_postings"], str(route.destination.size))
    ]
    return div(".cl-route-bottom")[
        Markup(ctx.layout.stats(pairs, rows=1).text),
        div(".cl-route-cta")[tab.chart_labels["explore_actions"]]
    ]


def _build_route_card(
    ctx   : TabContext,
    route : RouteDetail,
    tab   : TabContent
) -> Html:
    """
    Compose the four-section verdict-led route card.

    Section order is verdict (Section A) → skill bridge (Section B) →
    path flow (Section C) → optional alternative path callout → stats
    and CTA (Section D). The card answers the user's questions in the
    order they have them on click: "can I do this?" → "what would
    transfer?" → "how do I get there?" → "what are the hard numbers?".
    """
    return ctx.layout.to_html(
        _build_verdict(ctx, route, tab),
        _build_bridge(ctx, route, tab),
        _build_path(ctx, route, tab),
        *_build_alternative(ctx, route, tab),
        _build_stats_and_cta(ctx, route, tab),
        cls = "cl-route-card"
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


def _build_actions(
    ctx   : TabContext,
    route : RouteDetail,
    tab   : TabContent
) -> Html:
    """
    Action plan drawer reorganized into three numbered steps.

    Step 1 ranks edge-specific credentials in three columns by kind
    (apprenticeships / programs / certifications), mirroring the Data
    tab's credential pathway layout. Step 2 surfaces AGC member
    employers fuzzy-matched against the destination cluster's posting
    companies. Step 3 ranks Maine and national job boards by semantic
    similarity to the destination cluster's vector via the
    `RelevantJobBoards` schema, replacing the previous keyword filter.
    """
    destination = route.destination

    by_kind: dict[str, list[Credential]] = {
        "apprenticeship" : [],
        "program"        : [],
        "certification"  : []
    }
    for credential in route.credentials:
        if credential.kind in by_kind:
            by_kind[credential.kind].append(credential)

    credential_columns = ctx.layout.stack(
        *(
            ctx.layout.grid(
                ctx.layout.credential_card(c, ctx.theme) for c in cards[:4]
            ) if cards else ctx.layout.callout(tab.fallbacks["no_credentials"])
            for cards in by_kind.values()
        ),
        direction = "h",
        widths    = [1, 1, 1]
    )

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
        ctx.layout.callout(tab.chart_labels["action_intro"]),
        ctx.layout.header(tab, "action_credentials"),
        credential_columns,
        ctx.layout.header(tab, "action_employers"),
        employer_grid,
        ctx.layout.header(tab, "action_boards"),
        board_grid
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
