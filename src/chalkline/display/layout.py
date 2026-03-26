"""
Marimo layout helpers for the career report dashboard.

Provides reusable composition patterns that reduce boilerplate across
notebook cells, including Job Zone translation, horizontal stat strips,
section headers with accessible descriptions, filtered accordions, and
toolbar composition.
"""

import marimo as mo


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


def stat_row(stats: dict[str, str], **kwargs) -> mo.Html:
    """
    Horizontal strip of `mo.stat` elements from a label-value dict.

    Args:
        stats    : Label to display value. Rendered in insertion order.
        **kwargs : Forwarded to each `mo.stat` call.

    Returns:
        Horizontally stacked stat elements with wrapping.
    """
    return mo.hstack(
        [mo.stat(v, k, **kwargs) for k, v in stats.items()],
        gap  = 1,
        wrap = True
    )


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
