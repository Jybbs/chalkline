"""
Job Postings tab for the Chalkline career report.
"""

import marimo as mo

from functools import partial

from chalkline.display.layout       import callout, card_grid, header
from chalkline.display.layout       import posting_card, stat_strip
from chalkline.display.tabs.context import TabContext, load_content

content = load_content(__file__)


def job_postings_tab(ctx: TabContext) -> mo.Html:
    """
    Render the Job Postings tab showing real postings from the
    matched career family.
    """
    postings = ctx.data.postings
    section  = partial(content.section, soc_title=ctx.data.soc_title)

    return mo.vstack([
        header(*section("overview")),
        stat_strip({
            "Postings in Family" : str(postings.posting_count),
            "Companies Hiring"   : str(postings.unique_companies),
            **({"Locations": str(postings.unique_locations)}
               if postings.location_counts else {})
        }),

        header(*section("whos_hiring")),
        mo.ui.plotly(ctx.charts.hbar(
            color  = ctx.theme.colors["accent"],
            height = max(300, len(postings.company_counts) * 28),
            title  = "Number of Postings",
            x      = [cnt for _, cnt in postings.company_counts],
            y      = [name[:30] for name, _ in postings.company_counts]
        ))
        if postings.company_counts else mo.md("No company data."),

        *([
            header(*section("company_types")),
            mo.ui.plotly(ctx.charts.pie(
                300,
                hole     = 0.4,
                labels   = list(postings.member_types),
                textinfo = "label+percent",
                textfont = dict(size=11),
                values   = list(postings.member_types.values())
            ))
        ] if postings.member_types else []),

        *([
            header(*section("locations")),
            mo.ui.plotly(ctx.charts.hbar(
                color  = ctx.theme.colors["success"],
                height = max(250, len(postings.location_counts) * 28),
                title  = "Number of Postings",
                x      = [cnt for _, cnt in postings.location_counts],
                y      = [loc[:30] for loc, _ in postings.location_counts]
            ))
        ] if postings.location_counts else []),

        *([
            header(*section("timeline")),
            mo.ui.plotly(ctx.charts.timeline(
                dates = postings.dated_dates,
                hover = postings.dated_labels
            ))
        ] if postings.dated_dates else []),

        *([
            header(*section("common_words")),
            mo.ui.plotly(ctx.charts.treemap(
                height = 350,
                labels = [w   for w, _   in postings.title_words],
                values = [cnt for _, cnt in postings.title_words]
            ))
        ] if postings.title_words else []),

        header(*section("recent")),
        card_grid([posting_card(p) for p in postings.recent]),
        callout(content.info)
    ])
