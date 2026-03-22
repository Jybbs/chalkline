"""
Marimo layout helpers for the career report notebook.

Provides reusable composition patterns that reduce boilerplate across
notebook cells, including horizontal stat strips, conditional tables,
and filtered accordions.
"""

import marimo as mo


def filtered_accordion(
    sections : dict[str, list[dict]],
    fallback : str = "No data available."
) -> mo.Html:
    """
    Accordion from label-to-rows pairs, skipping empty sections.

    Builds an accordion where each section renders as a table.
    Sections with empty row lists are excluded. When all sections
    are empty, renders the fallback message instead.

    Args:
        sections : Label to row dicts. Empty lists are filtered out.
        fallback : Markdown shown when every section is empty.

    Returns:
        Accordion element or fallback markdown.
    """
    nonempty = {
        f"{label} ({len(rows)})": mo.ui.table(rows)
        for label, rows in sections.items()
        if rows
    }
    if nonempty:
        return mo.accordion(nonempty, multiple=True)
    return mo.md(fallback)


def stat_row(stats: dict[str, str], **kwargs) -> mo.Html:
    """
    Horizontal strip of `mo.stat` elements from a label-value dict.

    Args:
        stats    : Label to display value. Rendered in insertion order.
        **kwargs : Additional keyword arguments forwarded to each
                   `mo.stat` call (e.g., `direction="decrease"`).

    Returns:
        Horizontally stacked stat elements with wrapping.
    """
    return mo.hstack(
        [mo.stat(label=label, value=value, **kwargs) for label, value in stats.items()],
        gap  = 1,
        wrap = True
    )


def table_or_empty(
    rows     : list[dict],
    fallback : str = "No results found."
) -> mo.Html:
    """
    Render a table when rows exist, or a fallback message.

    Args:
        rows     : Row dicts for `mo.ui.table`.
        fallback : Markdown shown when `rows` is empty.

    Returns:
        Table element or fallback markdown.
    """
    if rows:
        return mo.ui.table(rows)
    return mo.md(fallback)
