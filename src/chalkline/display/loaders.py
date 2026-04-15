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
from chalkline.display.routes        import Routes
from chalkline.display.schemas       import Labels, ProcessStep, TabContent
from chalkline.display.theme         import Theme
from chalkline.matching.schemas      import MatchResult
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
    def glossary(self) -> tuple[re.Pattern[str], dict[str, tuple[str, str, str, str]]]:
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
            key=lambda p: -len(p[0])
        )
        return (
            re.compile(
                rf"\b({'|'.join(re.escape(key) for key, _ in pairs)})\b",
                re.IGNORECASE
            ),
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
        External-link icon SVG, rendered once per Layout and reused across
        every card builder that calls `_link`.
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
        key : str,
        tab : TabContent,
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
        wraps the result in a Marimo `Html`) and `splash` (which nests the
        row inside a larger htpy tree). Returning htpy here keeps the
        marimo-`Html` boundary at exactly one place per call site.
        """
        tiles = [self._stat(label, value) for label, value in pairs]
        return div(
            ".cl-stat-row",
            style=f"grid-template-columns:repeat({-(-len(tiles) // rows)},1fr)"
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
            children = [
                strong(".cl-tip-title")[title],
                hr(".cl-tip-rule"),
                span(".cl-tip-body")[Markup(
                    self.markdown_it.renderInline(
                        definition.format_map(self.substitutions)
                    )
                )],
                hr(".cl-tip-rule") if url else None,
                a(".cl-tip-link", href=url, target="_blank")[url_label]
                if url else None
            ]
            return str(span(".cl-term")[m.group(), span(".cl-tip")[children]])

        return pattern.sub(replace, text)

    def board_chip(self, **kwargs) -> Html:
        """
        Compact job board ribbon tile with a semantic-similarity headline.

        Mirrors the splash stat tile aesthetic by leading with a large
        primary-color percentage (`match_score`) above the board name,
        category badge, and focus blurb. The percentage is the cosine
        similarity between the board's encoded focus text and the matched
        cluster's vector, rounded to the nearest integer.
        """
        return self.to_html(
            div(".cl-stat-value")[f"{kwargs['match_score']}%"],
            strong[kwargs["name"]], br,
            span(".badge")[kwargs["category"]], br,
            span(".meta")[kwargs["focus"]],
            cls="cl-card"
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

        The card's left border matches the kind's accent color (cream for
        apprenticeship, lavender for certification, accent for program), the
        kind badge sits alone on its own line at the top, and the title
        wraps onto a new line below the badge.

        The `.cl-credential` modifier class enforces a minimum height so
        cards in a grid stay visually uniform regardless of how much their
        individual content varies.

        Args:
            credential : Pathways credential with kind and metadata.
            theme      : For accent color lookup via `credential_color`.

        Returns:
            Styled card element with kind badge and detail line.
        """
        color = theme.credential_color(credential.kind)
        return self.to_html(
            span(".cl-badge", style=f"background:{color}")[credential.type_label],
            strong[credential.label],
            span(".secondary")[detail] if (detail := credential.card_detail) else None,
            self._link(url) if (url := credential.url) else None,
            cls   = "cl-card.cl-credential",
            style = f"border-inline-start-color:{color}"
        )

    def credential_columns(
        self,
        by_kind  : dict[str, list[Credential]],
        theme    : Theme,
        per_kind : int = 4
    ) -> Html:
        """
        Render credentials grouped by kind in a horizontal multi-column
        grid, one column per kind.

        Shared by the Data tab's credential pathways section, the Map tab's
        resources drawer, and any future tab that needs the same layout.
        Each column renders up to `per_kind` cards via `credential_card`,
        and the columns stack horizontally with equal widths.

        Args:
            by_kind  : Credential label to list mapping, keyed by kind.
            theme    : For credential color lookup.
            per_kind : Max cards per kind column.
        """
        nonempty = {k: v for k, v in by_kind.items() if v}
        if not nonempty:
            return self.callout("No credentials available.")
        return self.stack(
            *(
                self.grid(
                    self.credential_card(c, theme) for c in cards[:per_kind]
                )
                for cards in nonempty.values()
            ),
            direction = "h",
            widths    = [1] * len(nonempty)
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
            cls="cl-card"
        )

    def grid(self, cards: Iterable[Html], columns: int = 2) -> Html:
        """
        Arrange cards in a responsive CSS grid.

        Args:
            cards   : Card elements to arrange.
            columns : Number of grid columns.

        Returns:
            Grid container wrapping all cards.
        """
        cls = f"cl-card-grid.cl-card-grid-{columns}" if columns != 2 else "cl-card-grid"
        return self.to_html(
            [Markup(c.text) for c in cards],
            cls = cls
        )

    def header(
        self,
        key : str,
        tab : TabContent,
        **fmt
    ) -> Html:
        """
        Section header with a bold serif title and muted description.

        Absorbs the `tab.section()` lookup so call sites pass the tab and
        key directly instead of a pre-resolved tuple.

        Args:
            key  : Section key to look up.
            tab  : Tab content holding section definitions.
            **fmt : Format kwargs forwarded to `tab.section()`.

        Returns:
            Vertically stacked title and description.
        """
        title, body = self._section_html(key, tab, **fmt)
        return self.stack(
            md(f"#### {title}"),
            self.to_html(body, cls="cl-section-desc"),
            gap=0.25
        )

    def overview(
        self,
        key : str,
        tab : TabContent,
        **fmt
    ) -> Html:
        """
        Tab overview rendered as a branded banner.

        Uses a dedicated `.cl-overview` style with a gold top accent and
        serif title to visually distinguish explanatory tab openers from
        both chart headers and standard callouts.

        Args:
            key  : Section key to look up.
            tab  : Tab content holding section definitions.
            **fmt : Format kwargs forwarded to `tab.section()`.

        Returns:
            Styled overview banner with serif title and body description.
        """
        title, body = self._section_html(key, tab, **fmt)
        return self.to_html(
            div(".cl-overview-title")[title],
            div(".cl-overview-body")[body],
            cls="cl-overview"
        )

    def panel(
        self,
        chart : Figure,
        key   : str,
        tab   : TabContent,
        **fmt
    ) -> Html:
        """
        Stack a section header above a Plotly chart as one logical unit.

        The core tab-renderer primitive: every chart panel is a header
        followed by a Plotly figure, grouped so rows inside `two_col` stay
        aligned and top-level charts share the same grouping.
        """
        return self.stack(
            self.header(key, tab, **fmt),
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
                if posting.date_posted else None
            ],
            self._link(posting.source_url),
            cls="cl-card"
        )

    def posting_ribbon(
        self,
        color      : str,
        posting    : Posting,
        similarity : float
    ) -> Html:
        """
        Single-line posting row with cosine similarity score, title in
        serif, company, location, days since posted, and an external link on
        the right.

        Args:
            color      : CSS color string for the similarity score.
            posting    : Corpus posting record.
            similarity : Raw cosine similarity to the resume (0-1).
        """
        from datetime import date

        meta_parts = [posting.company]
        if posting.location:
            meta_parts.append(posting.location)
        if posting.date_posted:
            days = (date.today() - posting.date_posted).days
            meta_parts.append(f"{days}d ago")

        return self.to_html(
            span(
                ".cl-posting-ribbon-pct",
                style=f"color:{color}"
            )[f"{round(similarity * 100)}%"],
            div(".cl-posting-ribbon-line")[
                span(".cl-posting-ribbon-title")[posting.title],
                span(".cl-posting-ribbon-meta")[
                    " \u00b7 ".join(meta_parts)
                ]
            ],
            a(
                ".cl-posting-ribbon-link",
                href   = posting.source_url,
                target = "_blank"
            )[self.external_icon],
            cls = "cl-posting-ribbon"
        )

    def process_flow(self, steps: Iterable[ProcessStep]) -> Html:
        """
        Horizontal process flow diagram as a CSS flexbox strip.

        Each step renders as a numbered card with a label and detail line,
        connected by arrow separators. Optional `accent` and `arrow_label`
        fields on `ProcessStep` enable sector-color left borders and
        natural-language labels above the incoming arrow, used by the Map
        tab career path flow to convey sector identity and per-hop
        transition difficulty.

        Args:
            steps: Process steps with pre-formatted detail lines.

        Returns:
            Horizontal flow diagram element.
        """
        cards = []
        for step in steps:
            if cards:
                arrow = div(".cl-flow-arrow")["\u2192"]
                cards.append(
                    div(".cl-flow-arrow-wrap")[
                        div(".cl-flow-arrow-label")[step.arrow_label], arrow
                    ] if step.arrow_label else arrow
                )
            cards.append(div(
                ".cl-flow-step",
                **(
                    {"style": f"border-inline-start-color:{step.accent}"}
                    if step.accent else {}
                )
            )[
                div(".cl-flow-num")[step.number],
                div(".cl-flow-label")[step.label],
                div(".cl-flow-detail")[step.detail]
            ])
        return self.to_html(*cards, cls="cl-flow")

    def section_if(
        self,
        chart     : Figure,
        condition : object,
        key       : str,
        tab       : TabContent,
        **fmt
    ) -> list:
        """
        Conditionally render a headed chart panel.

        Returns a header followed by the wrapped Plotly chart when
        `condition` is truthy, or an empty list for unpacking into a `stack`
        call. Bakes the `ui.plotly` wrap so call sites match `panel`'s
        signature.

        Args:
            chart     : Plotly figure to render below the header.
            condition : Truthy value gating the section.
            key       : Section key to look up.
            tab       : Tab content with section definitions.
            **fmt     : Format kwargs for `tab.section()`.
        """
        return self.stack_if(
            condition, self.header(key, tab, **fmt), ui.plotly(chart)
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
            stat_rows   : Number of rows to lay the stat tiles out in. The splash
                          overrides this to 2.

        Returns:
            Full-width splash element with logo, tagline, and stats.
        """
        return self.to_html(
            div(".cl-brand")[
                span(
                    ".cl-logo",
                    style=(
                        f"mask-image:url({logo_src});"
                        f"-webkit-mask-image:url({logo_src})"
                    )
                ),
                h1[tab.title]
            ],
            p(".cl-tagline")[tab.tagline],
            self._stat_row(zip(tab.stat_labels, stat_values), rows=stat_rows),
            cls="cl-splash"
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
        conditional expression and an empty-list fallback at the call site.
        `section_if` is a thin wrapper that pre-builds a header and chart
        pair on top of this primitive.
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
            rows  : Number of rows to lay the tiles out in. The splash overrides this to
                    2.

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

        Shorthand for the most common chart-pair layout in the methods and
        data tabs.
        """
        return Layout.stack(left, right, direction="h", widths=[1, 1])


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
    routes      : Routes
    theme       : Theme
