"""
Content loading, layout rendering, and dependency wiring for the display
layer.

`ContentLoader` mirrors the domain-layer loader pattern where a class owns a
directory path and provides cached access to validated data. `Layout`
composes Marimo HTML elements using htpy, paralleling how `Charts` renders
Plotly figures from theme state. `TabContext` bundles the shared
dependencies that every tab renderer needs, constructed once in the Marimo
notebook.
"""

import re

from bisect               import bisect
from collections.abc      import Iterable
from functools            import cache, cached_property
from htpy                 import a, br, div, Element, h1, hr, p, span, strong
from marimo               import hstack, Html, icon, md, ui, vstack
from markdown_it          import MarkdownIt
from markupsafe           import Markup
from pathlib              import Path
from plotly.graph_objects import Figure
from tomllib              import load
from typing               import Literal, NamedTuple

from chalkline.collection.schemas    import Posting
from chalkline.display.charts        import Charts
from chalkline.display.schemas       import Labels, ProcessStep, TabContent
from chalkline.display.theme         import Theme
from chalkline.matching.schemas      import MatchResult, ScoredTask
from chalkline.pathways.clusters     import Cluster
from chalkline.pathways.loaders      import LaborLoader, StakeholderReference
from chalkline.pathways.schemas      import Credential, Occupation
from chalkline.pipeline.orchestrator import Chalkline


class ContentLoader:
    """
    Centralized loader for display-layer TOML content.

    Provides cached access to shared labels and per-tab content from the
    display package directory. Constructed once in the Marimo notebook and
    threaded through `TabContext`.
    """

    def __init__(self, display_dir: Path | None = None):
        """
        Args:
            display_dir: Root of the `chalkline.display` package, containing
                         `tabs/shared/` and `tabs/<name>/content.toml`. Defaults to the
                         package directory when omitted.
        """
        self.display_dir = display_dir or Path(__file__).resolve().parent

    @cached_property
    def glossary(self) -> tuple[re.Pattern, dict[str, tuple]]:
        """
        Single alternation regex and definition lookup for all glossary
        terms, sorted longest-first so multi-word terms match before their
        substrings.

        Each entry's `term` plus any entries in its optional `aliases` list
        all resolve to the same tooltip content, with the canonical `term`
        shown as the title. This lets the content layer mention an acronym
        or plural anywhere and still surface the detailed definition on
        first occurrence.

        Each lookup value is a tuple of (title, definition, url, url_label)
        supporting rich tooltip rendering with optional links.
        """
        with (self.display_dir / "tabs/shared/glossary.toml").open("rb") as f:
            entries = load(f)["terms"]
        pairs = sorted(
            (
                (key, entry)
                for entry in entries
                for key in (entry["term"], *entry.get("aliases", ()))
            ),
            key = lambda p: -len(p[0])
        )
        alts = "|".join(re.escape(key) for key, _ in pairs)
        return (
            re.compile(rf"\b({alts})\b", re.IGNORECASE),
            {
                key.lower(): (
                    entry["term"],
                    entry["definition"],
                    entry.get("url", ""),
                    entry.get("url_label", "")
                )
                for key, entry in pairs
            }
        )

    @cached_property
    def labels(self) -> Labels:
        """
        Shared display labels validated from `tabs/shared/content.toml`.
        """
        with (self.display_dir / "tabs" / "shared" / "content.toml").open("rb") as f:
            return Labels.model_validate(load(f))

    @cache
    def tab(self, name: str) -> TabContent:
        """
        Load and validate a tab's `content.toml`, cached per name.

        Args:
            name: Tab directory name (e.g. `"your_match"`, `"splash"`).
        """
        with (self.display_dir / "tabs" / name / "content.toml").open("rb") as f:
            return TabContent.model_validate(load(f))


class Layout:
    """
    Marimo HTML composition using htpy typed elements.

    Each method handles data preprocessing (markdown rendering, description
    truncation, average computation) then builds HTML using htpy's
    Python-native element constructors. Parallels how `Charts` renders
    Plotly figures from theme state.
    """

    def __init__(
        self,
        content       : ContentLoader,
        substitutions : dict[str, str] | None = None
    ):
        """
        Args:
            content       : Loader providing shared labels and tab content.
            substitutions : Corpus-level values (e.g. `n_postings`, `n_clusters`)
                            substituted into TOML template strings at render time.
        """
        self.content       = content
        self.substitutions = substitutions or {}

    @cached_property
    def external_icon(self) -> Markup:
        """
        External-link icon SVG, rendered once per Layout and reused
        across every card builder that calls `_link`.
        """
        return Markup(icon("lucide:external-link", size=14).text)

    @cached_property
    def markdown_it(self) -> MarkdownIt:
        """
        Inline-emphasis Markdown parser, instantiated once per Layout.

        `annotate` is called dozens of times per render (once per callout,
        header, overview, sidebar, etc.), so the parser is cached on the
        Layout instance instead of rebuilt per call.
        """
        return MarkdownIt("zero").enable(["emphasis"])

    def _link(self, url: str) -> Element:
        """
        External link icon anchored to the bottom-right of a card.
        """
        return a(".cl-card-link", href=url, target="_blank")[self.external_icon]

    def _section_html(
        self,
        tab : TabContent,
        key : str,
        **fmt
    ) -> tuple[str, Markup]:
        """
        Render a section's title and description through the glossary
        annotator, returning `(title, body_markup)` for header/overview
        wrappers to compose into their own shapes.
        """
        description, title = tab.section(key, **self.substitutions, **fmt)
        return title, Markup(self.annotate(md(description).text))

    def _stat(self, label: str, value: str) -> Element:
        """
        Single branded stat with a gold value and muted label.
        """
        return div[
            div(".cl-stat-value")[value],
            div(".cl-stat-label")[label]
        ]

    def _stat_row(self, pairs: Iterable[tuple[str, str]], rows: int) -> Element:
        """
        Stat tile grid as a raw htpy `div`, used by both `stats` (which
        wraps the result in a Marimo `Html`) and `splash` (which nests
        the row inside a larger htpy tree). Returning htpy here keeps
        the marimo-`Html` boundary at exactly one place per call site.
        """
        tiles = [self._stat(label, value) for label, value in pairs]
        cols  = -(-len(tiles) // rows)
        return div(
            ".cl-stat-row",
            style = f"grid-template-columns:repeat({cols},1fr)"
        )[tiles]

    def annotate(self, text: str) -> str:
        """
        Annotate the first occurrence of each glossary term with an
        interactive tooltip popover via a single `re.sub` pass.

        Each popover contains a bold title, a description with inline
        emphasis rendered by `markdown-it-py`, and an optional outbound
        link. Terms are matched case-insensitively and longest-first to
        prevent partial overlap.

        Args:
            text: Rendered HTML to scan for glossary matches.
        """
        pattern, lookup = self.content.glossary
        seen: set[str]  = set()

        def render(raw: str) -> Markup:
            """
            Apply template substitution and inline emphasis to a glossary
            definition, returning safe HTML.
            """
            return Markup(
                self.markdown_it.renderInline(raw.format_map(self.substitutions))
            )

        def replace(m: re.Match) -> str:
            """
            Build a tooltip popover for a matched glossary term, skipping
            duplicates within the same text block.

            Dedup keys on the canonical title (not the raw match text), so
            an acronym and its full form in the same paragraph produce
            exactly one tooltip.
            """
            title, definition, url, url_label = lookup[m.group().lower()]
            if (canonical := title.lower()) in seen:
                return m.group()
            seen.add(canonical)
            children: list = [
                strong(".cl-tip-title")[title],
                hr(".cl-tip-rule"),
                span(".cl-tip-body")[render(definition)]
            ]
            if url:
                children.append(hr(".cl-tip-rule"))
                children.append(a(
                    ".cl-tip-link",
                    href   = url,
                    target = "_blank"
                )[url_label])

            return str(span(".cl-term")[m.group(), span(".cl-tip")[children]])

        return pattern.sub(replace, text)

    def board_card(self, **kwargs) -> Html:
        """
        Job board card with name, focus area, best-for description, and
        category tag.
        """
        return self.to_html(
            strong[kwargs["name"]],
            span(".badge")[kwargs["category"]], br,
            span(".secondary")[kwargs["focus"]], br,
            span(".meta")[kwargs["best_for"]],
            cls = "cl-card"
        )

    def board_chip(self, **kwargs) -> Html:
        """
        Compact job board ribbon tile with a semantic-similarity headline.

        Mirrors the splash stat tile aesthetic by leading with a large
        primary-color percentage (`match_score`) above the board name,
        category badge, and focus blurb. The percentage is the cosine
        similarity between the board's encoded focus text and the
        matched cluster's vector, rounded to the nearest integer.
        """
        return self.to_html(
            div(".cl-stat-value")[f"{kwargs['match_score']}%"],
            strong[kwargs["name"]], br,
            span(".badge")[kwargs["category"]], br,
            span(".meta")[kwargs["focus"]],
            cls = "cl-card"
        )

    def callout(self, text: str, kind: str = "info") -> Html:
        """
        Branded callout with left-border accent.

        Args:
            kind : Semantic variant ("info", "success", "warn").
            text : Markdown string rendered inside the callout.

        Returns:
            Styled callout element.
        """
        return self.to_html(
            Markup(self.annotate(md(text.format_map(self.substitutions)).text)),
            cls       = "cl-callout",
            data_kind = kind
        )

    def credential_card(self, credential: Credential, theme: Theme) -> Html:
        """
        Credential card with kind-aware accent color and metadata.

        The card's left border matches the kind's accent color (cream
        for apprenticeship, lavender for certification, accent for
        program), the kind badge sits alone on its own line at the top,
        and the title wraps onto a new line below the badge.

        The `.cl-credential` modifier class enforces a minimum height so
        cards in a grid stay visually uniform regardless of how much
        their individual content varies.

        Args:
            credential : Pathways credential with kind and metadata.
            theme      : For accent color lookup via `credential_color`.

        Returns:
            Styled card element with kind badge and detail line.
        """
        color  = theme.credential_color(credential.kind)
        detail = " \u00b7 ".join(filter(None, (
            f"{credential.hours:,} hours" if credential.hours else "",
            credential.metadata.get("institution", "")
        )))
        return self.to_html(
            span(".cl-badge", style=f"background:{color}")[credential.type_label],
            strong[credential.label],
            *self.stack_if(detail, span(".secondary")[detail]),
            *self.stack_if(
                url := credential.metadata.get("url", ""), self._link(url)
            ),
            cls   = "cl-card.cl-credential",
            style = f"border-inline-start-color:{color}"
        )

    def employer_card(self, **kwargs) -> Html:
        """
        AGC employer card with company name, member type, and posting link.

        Kwargs:
            member_type, name, posting_url.
        """
        return self.to_html(
            strong[kwargs["name"]],
            span(".badge")[kwargs["member_type"]], br,
            self._link(kwargs["posting_url"]),
            cls = "cl-card"
        )

    def grid(self, cards: Iterable[Html]) -> Html:
        """
        Arrange cards in a responsive two-column CSS grid.

        Args:
            cards: Card elements to arrange.

        Returns:
            Grid container wrapping all cards.
        """
        return self.to_html(
            [Markup(c.text) for c in cards],
            cls = "cl-card-grid"
        )

    def header(
        self,
        tab : TabContent,
        key : str,
        **fmt
    ) -> Html:
        """
        Section header with a bold serif title and muted description.

        Absorbs the `tab.section()` lookup so call sites pass the tab and
        key directly instead of a pre-resolved tuple.

        Args:
            tab  : Tab content holding section definitions.
            key  : Section key to look up.
            **fmt : Format kwargs forwarded to `tab.section()`.

        Returns:
            Vertically stacked title and description.
        """
        title, body = self._section_html(tab, key, **fmt)
        return self.stack(
            md(f"#### {title}"),
            self.to_html(body, cls="cl-section-desc"),
            gap = 0.25
        )

    def match_bar(self, profile: Cluster) -> Html:
        """
        Compact breadcrumb bar summarizing the matched career family.

        Displayed between the upload gate and the tabbed report so the
        user always sees which family they matched to.

        Args:
            profile: Matched career cluster.

        Returns:
            Single-line bar element with `.cl-match-bar` styling.
        """
        return self.to_html(
            strong[profile.soc_title],
            f" \u00b7 {profile.sector}"
            f" \u00b7 {self.content.labels.job_zones[profile.job_zone]}"
            f" \u00b7 {profile.size} postings",
            cls = "cl-match-bar"
        )

    def overview(
        self,
        tab : TabContent,
        key : str,
        **fmt
    ) -> Html:
        """
        Tab overview rendered as a branded banner.

        Uses a dedicated `.cl-overview` style with a gold top accent and
        serif title to visually distinguish explanatory tab openers from
        both chart headers and standard callouts.

        Args:
            tab  : Tab content holding section definitions.
            key  : Section key to look up.
            **fmt : Format kwargs forwarded to `tab.section()`.

        Returns:
            Styled overview banner with serif title and body description.
        """
        title, body = self._section_html(tab, key, **fmt)
        return self.to_html(
            div(".cl-overview-title")[title],
            div(".cl-overview-body")[body],
            cls = "cl-overview"
        )

    def panel(
        self,
        tab   : TabContent,
        key   : str,
        chart : Figure,
        **fmt
    ) -> Html:
        """
        Stack a section header above a Plotly chart as one logical unit.

        The core tab-renderer primitive: every chart panel is a header
        followed by a Plotly figure, grouped so rows inside `two_col`
        stay aligned and top-level charts share the same grouping.
        """
        return self.stack(
            self.header(tab, key, **fmt),
            ui.plotly(chart)
        )

    def posting_card(self, posting: Posting) -> Html:
        """
        Job posting card with title, company, location, date, and a link to
        the original listing.

        Args:
            posting: Corpus posting record.

        Returns:
            Styled card element.
        """
        return self.to_html(
            strong[posting.title], br,
            span(".secondary")[posting.company], br,
            span(".meta")[
                posting.location or self.content.labels.fallback_location,
                f" \u00b7 {posting.date_posted:%b %d, %Y}" 
                if posting.date_posted else ""
            ],
            self._link(posting.source_url),
            cls = "cl-card"
        )

    def process_flow(self, steps: Iterable[ProcessStep]) -> Html:
        """
        Pipeline process flow diagram as a CSS flexbox strip.

        Each step renders as a numbered card with a label and detail
        line, connected by arrow separators.

        Args:
            steps: Process steps with pre-formatted detail lines.

        Returns:
            Horizontal flow diagram element.
        """
        arrow = div(".cl-flow-arrow")["\u2192"]
        cards = []
        for step in steps:
            if cards:
                cards.append(arrow)
            cards.append(div(".cl-flow-step")[
                div(".cl-flow-num")[step.number],
                div(".cl-flow-label")[step.label],
                div(".cl-flow-detail")[step.detail]
            ])
        return self.to_html(*cards, cls="cl-flow")

    def ranked_list(
        self,
        heading : str,
        skills  : list[ScoredTask],
        theme   : Theme
    ) -> Html:
        """
        Compact heading + percentage-bar list for scored skills.

        Each row sets a `--row-color` CSS custom property from the score
        color, and the stylesheet applies it to both the fill bar's
        background and the percentage text's color. The Python layer
        only emits the value once per row.
        """
        return self.to_html(
            div(".cl-ranked-heading")[heading],
            *(
                div(".cl-skill-row", 
                    style=f"--row-color:{theme.score_color(task.pct)}")[
                        span(".cl-skill-name")[task.name],
                        div(".cl-skill-bar")[
                            div(".cl-skill-fill", style=f"width:{task.pct}%")
                        ],
                        span(".cl-skill-pct")[f"{task.pct:.0f}%"]
                    ]
                for task in skills
            ),
            cls = "cl-ranked-list"
        )

    def section_if(
        self,
        condition : object,
        tab       : TabContent,
        key       : str,
        chart     : Figure,
        **fmt
    ) -> list:
        """
        Conditionally render a headed chart panel.

        Returns a header followed by the wrapped Plotly chart when
        `condition` is truthy, or an empty list for unpacking into a
        `stack` call. Bakes the `ui.plotly` wrap so call sites match
        `panel`'s signature.

        Args:
            condition : Truthy value gating the section.
            tab       : Tab content with section definitions.
            key       : Section key to look up.
            chart     : Plotly figure to render below the header.
            **fmt     : Format kwargs for `tab.section()`.
        """
        return self.stack_if(
            condition, self.header(tab, key, **fmt), ui.plotly(chart)
        )

    def splash(
        self,
        logo_src    : str,
        stat_values : list[str],
        tab         : TabContent,
        stat_rows   : int = 1
    ) -> Html:
        """
        Pre-upload splash page with branding and corpus statistics.

        The logo uses a CSS mask-image so the splash works without a static
        file server. Stats render through the same `cl-stat-row` grid that
        the in-tab `Layout.stats` builds, so the splash and tabs share one
        stat-tile layout.

        Args:
            logo_src    : Base64 data URI for the logo image.
            stat_values : Pre-formatted stat strings aligned with `tab.stat_labels`.
            tab         : Splash tab content with stat labels, tagline, and title.
            stat_rows   : Number of rows to lay the stat tiles out in. The
                          splash overrides this to 2.

        Returns:
            Full-width splash element with logo, tagline, and stats.
        """
        mask = f"mask-image:url({logo_src});-webkit-mask-image:url({logo_src})"
        return self.to_html(
            div(".cl-brand")[
                span(".cl-logo", style=mask),
                h1[tab.title]
            ],
            p(".cl-tagline")[tab.tagline],
            self._stat_row(zip(tab.stat_labels, stat_values), rows=stat_rows),
            cls = "cl-splash"
        )

    @staticmethod
    def stack(
        *items    : object,
        align     : Literal["start", "end", "center", "stretch"] = "stretch",
        direction : Literal["v", "h"] = "v",
        gap       : float             = 1.5,
        **kw
    ) -> Html:
        """
        Stack items vertically or horizontally with consistent spacing.

        Args:
            *items    : Elements to stack (charts, callouts, headers).
            align     : Cross-axis alignment forwarded to the Marimo stack.
            direction : `"v"` for vertical, `"h"` for horizontal.
            gap       : Spacing between items in rem.
            **kw      : Forwarded to the Marimo stack function (e.g. `widths`,
                        `justify`).

        Returns:
            Composed Marimo HTML element.
        """
        fn = vstack if direction == "v" else hstack
        return fn(items=list(items), gap=gap, align=align, **kw)

    @staticmethod
    def stack_if(condition: object, *items: object) -> list:
        """
        Return `items` as a list when `condition` is truthy, else `[]`.

        Pairs with `*` spread inside `stack` calls so a tab renderer can
        drop a whole section based on data presence without wrapping a
        conditional expression and an empty-list fallback at the call
        site. `section_if` is a thin wrapper that pre-builds a header
        and chart pair on top of this primitive.
        """
        return list(items) if condition else []

    def stats(
        self,
        pairs : Iterable[tuple[str, str]],
        rows  : int = 1
    ) -> Html:
        """
        Branded stat tile grid laid out in exactly `rows` rows.

        Each pair renders as a gold value over a muted label, using the same
        `.cl-stat-value` / `.cl-stat-label` classes as the splash. The grid
        column count is computed as `ceil(len(pairs) / rows)` so tiles
        always distribute evenly across the full width, defaulting to a
        single row that stretches the strip end-to-end.

        Args:
            pairs : (label, value) tuples in display order.
            rows  : Number of rows to lay the tiles out in. The splash
                    overrides this to 2.

        Returns:
            Stat row element.
        """
        return Html(str(self._stat_row(pairs, rows).__html__()))

    def to_html(self, *children, cls: str, **attrs) -> Html:
        """
        Wrap children in a classed div and convert to `Html`.

        Single boundary between htpy's typed element tree and Marimo's
        `Html` wrapper. Every public method that returns `Html` from htpy
        elements goes through this.

        Args:
            *children : htpy elements, strings, or `Markup` values.
            cls       : CSS class for the wrapper div.
            **attrs   : Additional HTML attributes on the wrapper.
        """
        return Html(str(div(f".{cls}", **attrs)[children].__html__()))

    @staticmethod
    def two_col(left: object, right: object) -> Html:
        """
        Two-column horizontal stack with equal widths.

        Shorthand for the most common chart-pair layout in the methods
        and data tabs.
        """
        return Layout.stack(left, right, direction="h", widths=[1, 1])

    def wage_display(self, delta: float | None, theme: Theme) -> Element:
        """
        Wage delta element: bold signed dollar amount in success/error
        color, or a muted "unavailable" message when `delta` is `None`.
        """
        if delta is None:
            return span(".secondary")["Wage data unavailable"]
        return span(
            style=f"color:{theme.wage_color(delta)};font-size:1.2em;font-weight:bold"
        )[f"{'+' if delta >= 0 else ''}${delta:,.0f}/yr"]

    def you_are_here(
        self,
        confidence : int,
        profile    : Cluster,
        theme      : Theme,
        wage       : float | None = None
    ) -> Html:
        """
        Persistent sidebar identity card for the matched career
        family.

        Red left-accent card showing the user's matched position in
        the career landscape. Confidence renders as a verdict label
        rather than a gauge, with the underlying percentage in small
        text.

        Args:
            confidence : Match confidence 0-100.
            profile    : Matched career cluster.
            theme      : For confidence color and Job Zone label.
            wage       : Annual median wage (optional).

        Returns:
            Styled sidebar identity card.
        """
        verdict = ("Exploratory", "Multiple good fits", "Strong match")[
            bisect((40, 70), confidence)
        ]
        sector_bg     = theme.sectors.get(profile.sector, theme.colors["muted"])
        verdict_style = f"color:{theme.score_color(confidence)};font-weight:bold"

        return self.to_html(
            div(".cl-yah-title")[profile.soc_title],
            div(".cl-yah-meta")[
                span(".cl-badge", style=f"background:{sector_bg}")[profile.sector],
                " \u00b7 ",
                span[self.content.labels.job_zones[profile.job_zone]]
            ],
            div(".cl-yah-stats")[
                span(style=verdict_style)[verdict],
                span(".secondary")[f" ({confidence}%)"],
                " \u00b7 ",
                span[f"{profile.size} postings"],
                f" \u00b7 ${wage:,.0f} median" if wage else ""
            ],
            cls = "cl-you-are-here"
        )


class TabContext(NamedTuple):
    """
    Shared dependencies available to every tab function.

    Holds the raw pipeline inputs so each tab can construct the metrics it
    needs via factory classmethods on the metric models. Marimo's cell
    caching ensures each tab's factories run at most once.
    """

    charts      : Charts
    content     : ContentLoader
    labor       : LaborLoader
    layout      : Layout
    occupations : list[Occupation]
    pipeline    : Chalkline
    profile     : Cluster
    reference   : StakeholderReference
    result      : MatchResult
    theme       : Theme
