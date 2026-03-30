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

import marimo as mo

from collections.abc import Iterable
from functools       import cached_property
from htpy            import a, br, details, div, h1, p, span, strong, summary
from markupsafe      import Markup
from pathlib         import Path
from statistics      import fmean
from tomllib         import load
from typing          import NamedTuple

from chalkline.collection.schemas    import Posting
from chalkline.display.charts        import Charts
from chalkline.display.schemas       import Labels, SplashMetrics, TabContent
from chalkline.display.theme         import Theme
from chalkline.matching.schemas      import MatchResult, ScoredTask
from chalkline.pathways.clusters     import Cluster
from chalkline.pathways.loaders      import LaborLoader, StakeholderReference
from chalkline.pathways.schemas      import Occupations
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
    def labels(self) -> Labels:
        """
        Shared display labels validated from `tabs/shared/content.toml`.
        """
        with (self.display_dir / "tabs" / "shared" / "content.toml").open("rb") as f:
            return Labels.model_validate(load(f))

    def tab(self, name: str) -> TabContent:
        """
        Load and validate a tab's `content.toml`.

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

    def __init__(self, content: ContentLoader):
        """
        Args:
            content: Loader providing shared labels and tab content.
        """
        self.content = content

    def _link(self, key: str, url: str):
        """
        Anchor element with link text from shared card labels.
        """
        return a(href=url)[self.content.labels.card_links[key]]

    def _stat(self, label: str, value: str):
        """
        Single branded stat with a gold value and muted label.
        """
        return div[
            div(".cl-stat-value")[value],
            div(".cl-stat-label")[label]
        ]

    def _to_html(self, *children, cls: str, **attrs) -> mo.Html:
        """
        Wrap children in a classed div and convert to `mo.Html`.

        Single boundary between htpy's typed element tree and Marimo's
        `Html` wrapper. Every public method that returns `mo.Html` from htpy
        elements goes through this.

        Args:
            *children : htpy elements, strings, or `Markup` values.
            cls       : CSS class for the wrapper div.
            **attrs   : Additional HTML attributes on the wrapper.
        """
        return mo.Html(str(div(f".{cls}", **attrs)[children]))

    def board_card(self, **kwargs) -> mo.Html:
        """
        Job board card with name, focus area, best-for description, and
        category tag.
        """
        return self._to_html(
            strong[kwargs["name"]],
            span(".badge")[kwargs["category"]], br,
            span(".secondary")[kwargs["focus"]], br,
            span(".meta")[kwargs["best_for"]],
            cls = "cl-card"
        )

    def callout(self, text: str, kind: str = "info") -> mo.Html:
        """
        Branded callout with left-border accent matching the splash theme.

        Uses `.cl-callout` CSS with `data-kind` variants instead of Marimo's
        default `mo.callout`, so callouts inherit the dashboard's Lora serif
        typography and design tokens.

        Args:
            kind : Semantic variant ("info", "success", "warn").
            text : Markdown string.

        Returns:
            Styled callout element.
        """
        return self._to_html(
            Markup(mo.md(text).text),
            cls       = "cl-callout",
            data_kind = kind
        )

    def card_grid(self, cards: Iterable[mo.Html]) -> mo.Html:
        """
        Arrange cards in a responsive two-column grid.

        Args:
            cards: Card elements to arrange.

        Returns:
            Grid container wrapping all cards.
        """
        return self._to_html(
            [Markup(c.text) for c in cards],
            cls = "cl-card-grid"
        )

    def employer_card(self, **kwargs) -> mo.Html:
        """
        AGC employer card with company name, member type, and links.

        Kwargs:
            career_url, member_type, name, posting_url.
        """
        career_url = kwargs.get("career_url", "")
        return self._to_html(
            strong[kwargs["name"]],
            span(".badge")[kwargs["member_type"]], br,
            self._link("view_posting", kwargs["posting_url"]),
            [
                " \u00b7 ",
                self._link("career_page", career_url)
            ] if career_url else None,
            cls = "cl-card"
        )

    def header(self, section: tuple[str, str]) -> mo.Html:
        """
        Chart section with a bold title and a one-sentence explanation.

        Every visualization in the dashboard gets a header that says what it
        shows and why it matters, with technical terms contextualized
        inline.

        Args:
            section: (description, title) pair from `TabContent.section()`.

        Returns:
            Vertically stacked title and description.
        """
        description, title = section
        return mo.vstack([
            mo.md(f"#### {title}"),
            mo.Html(str(span(style="color: var(--muted-foreground);")[
                description
            ]))
        ])

    def match_bar(
        self,
        jz_label  : str,
        postings  : int,
        sector    : str,
        soc_title : str
    ) -> mo.Html:
        """
        Compact breadcrumb bar summarizing the matched career family.

        Displayed between the upload gate and the tabbed report so the user
        always sees which family they matched to.

        Args:
            jz_label  : Human-readable Job Zone label.
            postings  : Number of postings in the matched family.
            sector    : Construction sector name.
            soc_title : O*NET occupation title.

        Returns:
            Single-line bar element with `.cl-match-bar` styling.
        """
        return self._to_html(
            strong[soc_title],
            f" \u00b7 {sector} \u00b7 {jz_label} \u00b7 {postings} postings",
            cls = "cl-match-bar"
        )

    def posting_card(self, posting: Posting) -> mo.Html:
        """
        Job posting card with title, company, location, date, truncated
        description, and a link to the original listing.

        Args:
            posting: Corpus posting record.

        Returns:
            Styled card element.
        """
        if len(description := posting.description) > 200:
            description = description[:200].rsplit(" ", 1)[0] + "..."

        location = posting.location or self.content.labels.fallback_location
        date_str = (
            f" \u00b7 {posting.date_posted:%b %d, %Y}"
            if posting.date_posted else ""
        )

        return self._to_html(
            strong[posting.title], br,
            span(".secondary")[posting.company], br,
            span(".meta")[location, date_str],
            p[description],
            self._link("view_posting", posting.source_url),
            cls = "cl-card"
        )

    def process_flow(self, steps: Iterable[tuple[str, str, str]]) -> mo.Html:
        """
        Pipeline process flow diagram as a CSS flexbox strip.

        Each step renders as a numbered card with a label and detail line,
        connected by arrow separators.

        Args:
            steps: Tuples of (number, label, detail) per stage.

        Returns:
            Horizontal flow diagram element.
        """
        arrow = div(".cl-flow-arrow")["\u2192"]
        cards = []
        for num, label, detail in steps:
            if cards:
                cards.append(arrow)
            cards.append(div(".cl-flow-step")[
                div(".cl-flow-num")[num],
                div(".cl-flow-label")[label],
                div(".cl-flow-detail")[detail]
            ])
        return self._to_html(*cards, cls="cl-flow")

    def program_card(self, **kwargs) -> mo.Html:
        """
        Educational program card with program name, institution, credential
        type, and enrollment link.

        Kwargs:
            credential, institution, name, url.
        """
        return self._to_html(
            strong[kwargs["name"]], br,
            span(".secondary")[kwargs["institution"]],
            f" \u00b7 ", span[kwargs["credential"]], br,
            self._link("program_details", kwargs["url"]),
            cls = "cl-card"
        )

    def section_if(self, condition, tab, key, *body, **fmt) -> list:
        """
        Conditionally render a headed section.

        Returns a header followed by body elements when `condition` is
        truthy, or an empty list for unpacking into `mo.vstack`.

        Args:
            condition : Truthy value gating the section.
            **fmt     : Format kwargs for `tab.section()`.
            key       : Section key in the tab's content.
            tab       : TabContent with section definitions.
        """
        if not condition:
            return []
        return [self.header(tab.section(key, **fmt)), *body]

    def skill_tree(
        self,
        demonstrated : bool,
        groups       : dict[str, list[ScoredTask]],
        theme        : Theme
    ) -> mo.Html:
        """
        Collapsible skill explorer tree grouped by O*NET category, filtered
        to either demonstrated or gap skills.

        Each category group is a `<details>/<summary>` element showing the
        category name, skill count, and average score on the summary row.
        Individual skills inside show the full O*NET name with a
        conditional-colored percentage score.

        Args:
            demonstrated : True for strengths, False for gaps.
            groups       : Skill type to scored skill lists.
            theme        : For `score_color` and `type_label` access.

        Returns:
            Styled HTML tree element.
        """
        score = lambda v: span(
            ".cl-skill-score",
            style=f"color:{theme.score_color(v)}"
        )[f"{v}%"]

        return self._to_html(
            *[
                details(".cl-skill-group")[
                    summary[
                        span(".cl-group-label")[
                            theme.type_label(stype)
                        ],
                        span(".cl-group-count")[str(len(filtered))],
                        score(round(fmean(s.pct for s in filtered), 1))
                    ],
                    [
                        div(".cl-skill-item")[
                            span(".cl-skill-name")[s.name],
                            score(s.pct)
                        ]
                        for s in filtered
                    ]
                ]
                for stype in (
                    "skill", "ability", "knowledge",
                    "task", "dwa", "technology", "tool", "other"
                )
                if (all_skills := groups.get(stype))
                if (filtered   := [
                    s for s in all_skills
                    if s.demonstrated == demonstrated
                ])
            ],
            cls = "cl-skill-tree"
        )

    def splash(
        self,
        logo_src : str,
        metrics  : SplashMetrics,
        tab      : TabContent
    ) -> mo.Html:
        """
        Pre-upload splash page with branding and corpus statistics.

        The logo uses a CSS mask-image so the splash works without a static
        file server.

        Args:
            logo_src : Base64 data URI for the logo image.
            metrics  : Pre-computed corpus and labor statistics.
            tab      : Splash tab content with stat labels, tagline, and title.

        Returns:
            Full-width splash element with logo, tagline, and stats.
        """
        mask = (
            f"mask-image:url({logo_src});"
            f"-webkit-mask-image:url({logo_src})"
        )
        return self._to_html(
            div(".cl-brand")[
                span(".cl-logo", style=mask),
                h1[tab.title]
            ],
            p(".cl-tagline")[tab.tagline],
            div(".cl-stats")[
                [
                    self._stat(label, value)
                    for label, value in zip(tab.stat_labels, metrics.stat_values)
                ]
            ],
            cls = "cl-splash"
        )

    def stat_strip(
        self,
        stats : Iterable[tuple[str, str]],
        cls   : str = "cl-stat-row"
    ) -> mo.Html:
        """
        Responsive grid of branded stats.

        Uses Lora serif, gold primary values, and the same `.cl-stat-value`
        / `.cl-stat-label` classes as the splash page. The grid adapts from
        2 to 6 columns based on the number of stats and viewport width.

        Args:
            cls   : Wrapper CSS class. Defaults to `"cl-stat-row"` for tab stat strips;
                    splash passes `"cl-stats"`.
            stats : (label, value) pairs in display order.

        Returns:
            HTML grid element with responsive column sizing.
        """
        return self._to_html(
            *[self._stat(label, value) for label, value in stats],
            cls = cls
        )

    def target_dropdown(
        self,
        cluster_options : dict[str, int],
        default_label   : str
    ) -> mo.ui.dropdown:
        """
        Searchable dropdown for exploring career families.

        Args:
            cluster_options : Display label to cluster ID mapping.
            default_label   : Initially selected label.

        Returns:
            Configured dropdown widget.
        """
        return mo.ui.dropdown(
            label      = self.content.labels.dropdown_label,
            options    = cluster_options,
            searchable = True,
            value      = default_label
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
    occupations : Occupations
    pipeline    : Chalkline
    profile     : Cluster
    reference   : StakeholderReference
    result      : MatchResult
    theme       : Theme
