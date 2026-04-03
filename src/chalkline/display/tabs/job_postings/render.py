"""
Job Postings tab for the Chalkline career report.
"""

import marimo as mo

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import JobPostingMetrics


def job_postings_tab(ctx: TabContext) -> mo.Html:
    """
    Render the Job Postings tab showing real postings from the matched
    career family.
    """
    tab = ctx.content.tab("job_postings")

    postings   = JobPostingMetrics.from_postings(ctx.profile.postings, ctx.reference)
    section_kw = {"soc_title": ctx.profile.soc_title}

    return ctx.layout.stack(
        ctx.layout.header(tab, "overview", **section_kw),
        ctx.layout.stats(zip(tab.stat_labels, postings.stat_values)),

        ctx.layout.header(tab, "whos_hiring", **section_kw),
        mo.ui.plotly(ctx.charts.bar(
            height     = max(300, len(postings.companies) * 28),
            horizontal = True,
            title      = tab.chart_labels["postings_title"],
            x          = postings.companies.values(),
            y          = postings.companies
        ))
        if postings.companies else mo.md(tab.fallbacks["no_company_data"]),

        *ctx.layout.section_if(postings.members, tab, "company_types",
            mo.ui.plotly(ctx.charts.pie(
                height   = 300,
                hole     = 0.4,
                labels   = postings.members,
                textfont = {"size": 11},
                textinfo = "label+percent",
                values   = postings.members.values()
            )), **section_kw),

        *ctx.layout.section_if(postings.locations, tab, "locations",
            mo.ui.plotly(ctx.charts.bar(
                color      = "success",
                height     = max(250, len(postings.locations) * 28),
                horizontal = True,
                title      = tab.chart_labels["postings_title"],
                x          = postings.locations.values(),
                y          = postings.locations
            )), **section_kw),

        *ctx.layout.section_if(postings.dated, tab, "timeline",
            mo.ui.plotly(ctx.charts.timeline(
                dates = postings.dated.values(),
                hover = postings.dated
            )), **section_kw),

        *ctx.layout.section_if(postings.titles, tab, "common_words",
            mo.ui.plotly(ctx.charts.treemap(
                height = 350,
                labels = postings.titles,
                values = postings.titles.values()
            )), **section_kw),

        ctx.layout.header(tab, "recent", **section_kw),
        ctx.layout.grid(ctx.layout.posting_card(p) for p in postings.recent),
        ctx.layout.callout(tab.info)
    )
