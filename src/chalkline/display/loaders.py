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
import re

from collections.abc import Iterable
from functools       import cached_property
from htpy            import a, br, details, div, h1, hr, p, span, strong, summary
from markdown_it     import MarkdownIt
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
    def glossary(self) -> tuple[re.Pattern, dict[str, tuple]]:
        """
        Single alternation regex and definition lookup for all glossary
        terms, sorted longest-first so multi-word terms match before
        their substrings.

        Each lookup value is a tuple of (title, definition, url, url_label)
        supporting rich tooltip rendering with optional links.
        """
        with (self.display_dir / "tabs/shared/glossary.toml").open("rb") as f:
            terms = sorted(load(f)["terms"], key=lambda e: -len(e["term"]))
        alts = "|".join(re.escape(e["term"]) for e in terms)
        return (
            re.compile(rf"\b({alts})\b", re.IGNORECASE),
            {
                e["term"].lower(): (
                    e["term"],
                    e["definition"],
                    e.get("url", ""),
                    e.get("url_label", ""),
                )
                for e in terms
            }
        )

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

    def __init__(
        self,
        content       : ContentLoader,
        substitutions : dict[str, str] | None = None
    ):
        """
        Args:
            content       : Loader providing shared labels and tab content.
            substitutions : Corpus-level values (e.g. `n_postings`,
                            `n_clusters`) substituted into TOML template
                            strings at render time.
        """
        self.content       = content
        self.substitutions = substitutions or {}

    def _link(self, url: str):
        """
        External link icon anchored to the bottom-right of a card.
        """
        return a(".cl-card-link", href=url, target="_blank")[
            Markup(mo.icon("lucide:external-link", size=14).text)
        ]

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
        return mo.Html(str(div(f".{cls}", **attrs)[children].__html__()))

    def annotate(self, text: str) -> str:
        """
        Annotate the first occurrence of each glossary term with
        an interactive tooltip popover via a single `re.sub` pass.

        Each popover contains a bold title, a description with
        inline emphasis rendered by `markdown-it-py`, and an
        optional outbound link. Terms are matched
        case-insensitively and longest-first to prevent partial
        overlap.

        Args:
            text: Rendered HTML to scan for glossary matches.
        """
        pattern, lookup = self.content.glossary
        markdown_it     = MarkdownIt("zero").enable(["emphasis"])
        seen: set[str]  = set()

        def render(raw: str) -> Markup:
            """
            Apply template substitution and inline emphasis to a
            glossary definition, returning safe HTML.
            """
            return Markup(
                markdown_it.renderInline(raw.format_map(self.substitutions))
            )

        def replace(m: re.Match) -> str:
            """
            Build a tooltip popover for a matched glossary term,
            skipping duplicates within the same text block.
            """
            if (key := m.group().lower()) in seen:
                return m.group()
            seen.add(key)

            title, definition, url, url_label = lookup[key]
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
        Branded callout with left-border accent.

        Args:
            kind : Semantic variant ("info", "success", "warn").
            text : Markdown string rendered inside the callout.

        Returns:
            Styled callout element.
        """
        return self._to_html(
            Markup(self.annotate(mo.md(text.format_map(self.substitutions)).text)),
            cls       = "cl-callout",
            data_kind = kind
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
            self._link(kwargs["posting_url"]),
            cls = "cl-card"
        )

    def grid(self, cards: Iterable[mo.Html]) -> mo.Html:
        """
        Arrange cards in a responsive two-column CSS grid.

        Args:
            cards: Card elements to arrange.

        Returns:
            Grid container wrapping all cards.
        """
        return self._to_html(
            [Markup(c.text) for c in cards],
            cls = "cl-card-grid"
        )

    def header(
        self,
        tab : TabContent,
        key : str,
        **fmt
    ) -> mo.Html:
        """
        Section header with a bold serif title and muted description.

        Absorbs the `tab.section()` lookup so call sites pass the tab
        and key directly instead of a pre-resolved tuple.

        Args:
            tab  : Tab content holding section definitions.
            key  : Section key to look up.
            **fmt : Format kwargs forwarded to `tab.section()`.

        Returns:
            Vertically stacked title and description.
        """
        description, title = tab.section(key, **self.substitutions, **fmt)
        return self.stack(
            mo.md(f"#### {title}"),
            self._to_html(
                Markup(self.annotate(mo.md(description).text)),
                cls = "cl-section-desc"
            ),
            gap = 0.25
        )

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
        Job posting card with title, company, location, date, and a
        link to the original listing.

        Args:
            posting: Corpus posting record.

        Returns:
            Styled card element.
        """
        location = posting.location or self.content.labels.fallback_location
        date_str = (
            f" \u00b7 {posting.date_posted:%b %d, %Y}"
            if posting.date_posted else ""
        )

        return self._to_html(
            strong[posting.title], br,
            span(".secondary")[posting.company], br,
            span(".meta")[location, date_str],
            self._link(posting.source_url),
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
            self._link(kwargs["url"]),
            cls = "cl-card"
        )

    def section_if(
        self,
        condition : object,
        tab       : TabContent,
        key       : str,
        *body,
        **fmt
    ) -> list:
        """
        Conditionally render a headed section.

        Returns a header followed by body elements when `condition` is
        truthy, or an empty list for unpacking into a `stack` call.

        Args:
            condition : Truthy value gating the section.
            tab       : Tab content with section definitions.
            key       : Section key to look up.
            *body     : Elements to render below the header.
            **fmt     : Format kwargs for `tab.section()`.
        """
        if not condition:
            return []
        return [self.header(tab, key, **fmt), *body]

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
            theme        : For `type_label` access.

        Returns:
            Styled HTML tree element.
        """
        score = lambda v: span(
            f".cl-skill-score.cl-score-{theme.score_tier(v)}"
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

    def stack(
        self,
        *items    : object,
        direction : str   = "v",
        gap       : float = 1.5,
        **kw
    ) -> mo.Html:
        """
        Stack items vertically or horizontally with consistent spacing.

        Args:
            *items    : Elements to stack (charts, callouts, headers).
            direction : `"v"` for vertical, `"h"` for horizontal.
            gap       : Spacing between items in rem.
            **kw      : Forwarded to the Marimo stack function
                        (e.g. `widths`, `align`, `justify`).

        Returns:
            Composed Marimo HTML element.
        """
        fn = mo.vstack if direction == "v" else mo.hstack
        return fn(items=list(items), gap=gap, **kw)

    def stats(self, pairs: Iterable[tuple[str, str]]) -> mo.Html:
        """
        Responsive grid of branded stat tiles.

        Each pair renders as a gold value over a muted label, using the
        same `.cl-stat-value` / `.cl-stat-label` classes as the splash.

        Args:
            pairs: (label, value) tuples in display order.

        Returns:
            Responsive grid element.
        """
        return self._to_html(
            *[self._stat(label, value) for label, value in pairs],
            cls = "cl-stat-row"
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
