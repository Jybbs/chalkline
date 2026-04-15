"""
Career route rendering for the Map tab.

Parallels how `Charts` renders Plotly figures from theme state, composing
Marimo HTML elements for career transition routes via htpy typed elements.
"""

from htpy       import details, div, Element, p, span, strong, summary
from marimo     import Html
from markupsafe import Markup
from operator   import attrgetter
from typing     import Literal, TYPE_CHECKING

from chalkline.display.schemas       import GapCoverage, RelevantJobBoards, RouteDetail
from chalkline.display.schemas       import TabContent
from chalkline.display.theme         import Theme
from chalkline.matching.schemas      import ScoredTask
from chalkline.pathways.loaders      import StakeholderReference
from chalkline.pipeline.orchestrator import Chalkline

if TYPE_CHECKING:
    from chalkline.display.loaders import Layout


class Routes:
    """
    Map tab card builders for career transition routes.

    Delegates to `Layout` for generic primitives (`grid`, `callout`,
    `posting_card`) while owning the htpy construction of route-specific
    components like wage bars, fit meters, and credential path cards.
    """

    def __init__(
        self,
        layout : Layout,
        theme  : Theme
    ):
        self.layout = layout
        self.theme  = theme

    def _skill_cards(
        self,
        tasks   : list[ScoredTask],
        covered : set[str] | None = None
    ) -> list[Element]:
        """
        Render pre-calibrated scored tasks as individual card rows with the
        skill name on top and a full-width bar below.

        Each row reserves a fixed-width check column to its left so cards
        stay aligned whether or not the checkmark is rendered. When
        `covered` is provided, the glyph appears only for tasks whose names
        are in the set, marking which gaps the enclosing credential closes.
        """
        covered = covered or set()
        return [
            div(".cl-skill-row")[
                span(".cl-skill-row-check", aria_hidden="true")[
                    "\u2713" if t.name in covered else ""
                ],
                div(
                    ".cl-skill-card",
                    style=f"--row-color:{self.theme.score_color(t.pct)}"
                )[
                    div(".cl-skill-card-header")[
                        span(".cl-skill-card-name")[t.name],
                        span(".cl-skill-card-pct")[f"{t.pct:.0f}%"]
                    ],
                    div(".cl-skill-bar")[
                        div(".cl-skill-fill", style=f"width:{t.pct}%")
                    ]
                ]
            ]
            for t in tasks
        ]

    def _wage_bar(
        self,
        label   : str,
        pct     : int,
        variant : Literal["dest", "source"]
    ) -> Element:
        """
        Single wage comparison bar row with label and filled bar.
        """
        return div(".cl-wage-bar-row")[
            span(".cl-wage-bar-label")[label],
            div(".cl-wage-bar")[
                div(
                    f".cl-wage-bar-fill.cl-wage-bar-{variant}",
                    style=f"width:{pct}%"
                )
            ]
        ]

    def card(self, *sections: Html) -> Html:
        """
        Combine route sections (verdict, recipe, postings) into a single
        card container.
        """
        return self.layout.to_html(
            *(Markup(s.text) for s in sections),
            cls="cl-route-card"
        )

    def evidence(
        self,
        route : RouteDetail,
        tab   : TabContent
    ) -> Html:
        """
        Evidence drawer with percentile-calibrated skill cards (8 per
        section).

        Each skill renders as an individual card with name and percentage on
        top and a full-width bar below.
        """
        sections = [
            self.layout.to_html(
                div(".cl-route-label")[
                    tab.chart_labels[key].format(count=count)
                ],
                *self._skill_cards(skills),
                cls="cl-skill-card-list"
            )
            for key, skills, count in (
                ("strengths_heading", route.top_strengths, route.demonstrated_count),
                ("gaps_heading",      route.top_gaps,      route.gap_count)
            )
            if skills
        ]
        return (
            self.layout.stack(*sections) if sections
            else self.layout.callout(tab.fallbacks["no_gaps"])
        )

    def postings(
        self,
        route : RouteDetail,
        tab   : TabContent
    ) -> Html:
        """
        Destination posting cards in a 5-column grid, up to 10 cards.
        """
        if not (postings := route.destination.postings[:10]):
            return self.layout.callout(tab.fallbacks["no_postings"])
        return self.layout.to_html(
            div(".cl-route-label")[
                tab.chart_labels["postings_heading"].format(
                    count=route.destination.size
                )
            ],
            Markup(self.layout.grid(
                map(self.layout.posting_card, postings),
                columns=5
            ).text),
            cls="cl-route-postings"
        )

    def recipe(self, route: RouteDetail, tab: TabContent) -> Html:
        """
        Gap-coverage credential paths as stacked cards.

        Proposes 2-3 alternative credential combinations via exhaustive
        min-overlap set cover. Each path shows per-item gap coverage counts,
        hours or institution, and combined unique coverage.
        """
        labels = tab.chart_labels

        if not (paths := GapCoverage.from_route(route).paths):
            return self.layout.callout(tab.fallbacks["no_credentials"])

        path_cards = [
            div(".cl-path-card")[
                details(".cl-path-details", open=True)[
                    summary(".cl-path-header")[labels["path_header"].format(
                        n_items    = len(rows),
                        s          = "s" if len(rows) != 1 else "",
                        strategy   = labels.get(
                            f"strategy_{path.strategy}", path.strategy
                        ),
                        total      = route.gap_count,
                        unique_cov = path.unique_coverage
                    )],
                    *rows
                ]
            ]
            for path in paths
            for rows in [[
                details(
                    ".cl-gap-shelf",
                    name  = "gap-drawer",
                    style = f"border-inline-start-color:"
                            f"{self.theme.credential_color(item.kind)}"
                )[
                    summary(".cl-gap-shelf-summary")[
                        div(".cl-path-row-body")[
                            strong[item.label],
                            span(".secondary")[
                                f"{item.kind.title()} \u00b7 {item.detail}"
                            ]
                        ],
                        span(".cl-path-gaps")[labels["gap_badge"].format(
                            cov = item.coverage,
                            s   = "s" if item.coverage != 1 else ""
                        )]
                    ],
                    div(".cl-gap-shelf-body")[
                        div(".cl-gap-shelf-inner")[
                            *self._skill_cards(
                                covered = {
                                    route.gap_tasks[p].name for p in item.positions
                                },
                                tasks = sorted(
                                    route.gap_tasks,
                                    key = attrgetter("similarity")
                                )
                            )
                        ]
                    ]
                ]
                for item in path.items
            ]]
        ]

        return self.layout.to_html(
            div(".cl-recipe-intro")[Markup(labels["verdict_intro"].format(
                demonstrated = route.demonstrated_count,
                gaps         = route.gap_count,
                total        = route.total_tasks
            ))],
            *path_cards,
            cls="cl-recipe"
        )

    def resources(
        self,
        pipeline  : Chalkline,
        reference : StakeholderReference,
        route     : RouteDetail,
        tab       : TabContent
    ) -> Html:
        """
        Resource drawer with the full credential catalog, employers, and
        semantically-ranked job boards for deeper exploration.
        """
        return self.layout.stack(
            self.layout.header("credentials", tab),
            self.layout.credential_columns(by_kind, self.theme, per_kind=6)
            if (by_kind := route.credentials_by_kind)
            else self.layout.callout(tab.fallbacks["no_credentials"]),
            self.layout.header("employers", tab),
            self.layout.grid(
                self.layout.employer_card(**emp) for emp in employers
            )
            if (employers := reference.match_employers(
                route.destination.postings
            )[:8])
            else self.layout.callout(tab.fallbacks["no_employers"]),
            self.layout.header("boards", tab),
            self.layout.grid((self.layout.board_chip(**b) for b in boards), 5)
            if (boards := RelevantJobBoards.from_cluster(
                cluster   = route.destination,
                clusters  = pipeline.clusters,
                encoder   = pipeline.matcher.encoder,
                limit     = 5,
                reference = reference
            ).boards)
            else self.layout.callout(tab.fallbacks["no_boards"])
        )

    def verdict(
        self,
        route : RouteDetail,
        tab   : TabContent
    ) -> Html:
        """
        Fit meter, wage comparison bars, bold verdict, and extras.

        Key values are wrapped in HTML strong tags via Markup so they render
        bold in the htpy tree. Self-routes show one wage bar and skip the
        delta. Transition routes show source and destination bars with a
        signed delta.
        """
        dst    = route.destination
        wages  = route.wage_comparison
        labels = tab.chart_labels

        bars = [
            *(
                [self._wage_bar(
                    wages.source_label, wages.source_percentage, "source"
                )]
                if not route.is_self else []
            ),
            *(
                [self._wage_bar(
                    wages.destination_label,
                    wages.destination_percentage,
                    "dest"
                )]
                if wages.destination_wage else []
            )
        ]

        extras = [
            *(
                [span(
                    style=f"color:{self.theme.wage_color(wages.delta)}"
                          f";font-weight:bold"
                )[wages.delta_display], " \u00b7 "]
                if not route.is_self and wages.delta is not None else []
            ),
            strong[f"{dst.size}"],
            f" {labels['open_positions']}",
            *(
                [" \u00b7 ", span(".cl-bright-outlook")[
                    f"\u2605 {labels['bright_outlook']}"
                ]]
                if route.bright_outlook else []
            )
        ]

        return self.layout.to_html(
            div(".cl-route-hero-row")[
                div(".cl-fit-meter")[
                    span[f"{route.fit_percentage}%"],
                    span(".cl-fit-meter-label")[labels["fit_meter_label"]]
                ],
                div(".cl-wage-bars")[bars] if bars else None
            ],
            p(".cl-verdict")[Markup(labels["verdict_match"].format(
                demonstrated = route.demonstrated_count,
                fit_pct      = route.fit_percentage,
                soc_title    = route.display_title,
                total        = route.total_tasks
            ))],
            div(".cl-route-hero-extras")[extras],
            cls="cl-route-hero"
        )
