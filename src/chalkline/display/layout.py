"""
Marimo layout helpers for the career report dashboard.

Provides reusable composition patterns that reduce boilerplate across
notebook cells, including branded stat strips, callouts, section headers
with accessible descriptions, collapsible ML explainer panels, filtered
accordions, and toolbar composition.
"""

import marimo as mo


def callout(content: str, kind: str = "info") -> mo.Html:
    """
    Branded callout with left-border accent matching the splash theme.

    Uses `.cl-callout` CSS with `data-kind` variants instead of Marimo's
    default `mo.callout`, so callouts inherit the dashboard's Lora serif
    typography and design tokens.

    Args:
        content : Markdown string rendered inside the callout.
        kind    : Semantic variant ("info", "success", "warn").

    Returns:
        Styled callout element.
    """
    inner = mo.md(content).text
    return mo.Html(
        f'<div class="cl-callout" data-kind="{kind}">'
        f"{inner}</div>"
    )


def filtered_accordion(
    sections : dict[str, list[dict]],
    fallback : str = "No data available."
) -> mo.Html:
    """
    Accordion from label-to-rows pairs, skipping empty sections.

    Builds an accordion where each section renders as a table. Sections
    with empty row lists are excluded. When all sections are empty,
    renders the fallback message instead.

    Args:
        sections : Label to row dicts. Empty lists are filtered out.
        fallback : Markdown shown when every section is empty.

    Returns:
        Accordion element or fallback markdown.
    """
    nonempty: dict[str, object] = {
        f"{label} ({len(rows)})": mo.ui.table(rows)
        for label, rows in sections.items()
        if rows
    }
    return mo.accordion(nonempty, multiple=True) if nonempty else mo.md(fallback)


def header(title: str, description: str) -> mo.Html:
    """
    Chart section with a bold title and a one-sentence explanation.

    Every visualization in the dashboard gets a header that says what
    it shows and why it matters, with technical terms contextualized
    inline.

    Args:
        description : Plain-English explanation of the visualization.
        title       : Bold heading for the section.

    Returns:
        Vertically stacked title and description.
    """
    return mo.vstack([
        mo.md(f"#### {title}"),
        mo.md(f'<span style="color: var(--muted-foreground);">'
              f"{description}</span>")
    ])



def stat_html(label: str, value: str) -> str:
    """
    Single branded stat with a gold value and muted label.

    Returns raw HTML (not `mo.Html`) so multiple stats can be composed
    inside a `.cl-stat-row` grid before wrapping. Reuses the splash
    page's `.cl-stat-value` / `.cl-stat-label` classes for visual
    consistency across the dashboard.

    Args:
        label : Descriptive label displayed below the value.
        value : Display-ready string (formatted number, percentage).

    Returns:
        HTML div with `.cl-stat-value` and `.cl-stat-label` children.
    """
    return (
        "<div>"
        f'<div class="cl-stat-value">{value}</div>'
        f'<div class="cl-stat-label">{label}</div>'
        "</div>"
    )


def stat_strip(stats: dict[str, str]) -> mo.Html:
    """
    Responsive grid of branded stats matching the splash page layout.

    Replaces `mo.hstack([mo.stat(...), ...])` with custom HTML that
    uses Lora serif, gold primary values, and the same `.cl-stat-value`
    / `.cl-stat-label` classes as the splash page. The grid adapts from
    2 to 6 columns based on the number of stats and viewport width.

    Args:
        stats: Label to display value, rendered in insertion order.

    Returns:
        HTML grid element with responsive column sizing.
    """
    cells = "".join(stat_html(label=k, value=v) for k, v in stats.items())
    return mo.Html(f'<div class="cl-stat-row">{cells}</div>')


def toolbar(
    download_data : bytes,
    reupload      : mo.ui.file
) -> mo.Html:
    """
    Compact icon toolbar for the post-upload header bar.

    Provides download and re-upload controls. The light/dark toggle is
    handled by Marimo's native theme, so only actionable controls are
    included.

    Args:
        download_data : Report text encoded as bytes for download.
        reupload      : File upload widget for re-uploading a resume.

    Returns:
        Horizontally stacked toolbar elements.
    """
    return mo.hstack(
        [
            mo.download(
                data     = download_data,
                filename = "chalkline_report.txt",
                label    = "Download Report"
            ),
            reupload
        ],
        gap     = 0.5,
        justify = "end"
    )
