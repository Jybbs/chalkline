"""
Job Postings tab for the Chalkline career report.
"""

import marimo as mo

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import HierarchyData, JobPostingMetrics


def job_postings_tab(ctx: TabContext) -> mo.Html:
    """
    Render the Job Postings tab showing real postings from the matched
    career family.
    """
    tab = ctx.content.tab("job_postings")

    postings   = JobPostingMetrics.from_postings(ctx.profile.postings, ctx.reference)
    section_kw = {"soc_title": ctx.profile.soc_title}

    return ctx.layout.stack(
        ctx.layout.overview(tab, "overview", **section_kw),
        ctx.layout.stats(zip(tab.stat_labels, postings.stat_values)),

        ctx.layout.stack(
            ctx.layout.stack(
                ctx.layout.header(tab, "whos_hiring", **section_kw),
                mo.ui.plotly(ctx.charts.bar(
                    height     = max(300, len(postings.companies) * 28),
                    horizontal = True,
                    title      = tab.chart_labels["postings_title"],
                    x          = [*postings.companies.values()],
                    y          = [*postings.companies]
                ))
            )
            if postings.companies else mo.md(tab.fallbacks["no_company_data"]),
            ctx.layout.stack(
                ctx.layout.header(tab, "locations", **section_kw),
                mo.ui.plotly(ctx.charts.bar(
                    color      = "success",
                    height     = max(300, len(postings.locations) * 28),
                    horizontal = True,
                    title      = tab.chart_labels["postings_title"],
                    x          = [*postings.locations.values()],
                    y          = [*postings.locations]
                ))
            )
            if postings.locations else mo.md(""),
            direction = "h",
            widths    = [1, 1]
        ),

        ctx.layout.stack(
            ctx.layout.stack(
                ctx.layout.header(tab, "company_types", **section_kw),
                mo.ui.plotly(ctx.charts.pie(
                    height   = 350,
                    hole     = 0.4,
                    labels   = [*postings.members],
                    textfont = {"size": 11},
                    textinfo = "label+percent",
                    values   = [*postings.members.values()]
                ))
            ) if postings.members else mo.md(""),
            ctx.layout.stack(
                ctx.layout.header(tab, "common_words", **section_kw),
                mo.ui.plotly(ctx.charts.treemap(
                    data   = HierarchyData(
                        labels = [*postings.titles],
                        values = [*postings.titles.values()]
                    ),
                    height = 350
                ))
            ) if postings.titles else mo.md(""),
            ctx.layout.stack(
                ctx.layout.header(tab, "descriptions", **section_kw),
                mo.ui.plotly(ctx.charts.bar(
                    color      = "success",
                    height     = 350,
                    horizontal = True,
                    title      = "",
                    x          = [*postings.descriptions.values()],
                    y          = [*postings.descriptions]
                ))
            ) if postings.descriptions else mo.md(""),
            direction = "h",
            widths    = [1, 1, 1]
        ),

        ctx.layout.stack(
            ctx.layout.stack(
                ctx.layout.header(tab, "timeline", **section_kw),
                mo.ui.plotly(ctx.charts.timeline(
                    dates = [*postings.dated.values()],
                    hover = [*postings.dated]
                ))
            )
            if postings.dated else mo.md(""),
            ctx.layout.stack(
                ctx.layout.header(tab, "freshness", **section_kw),
                mo.ui.plotly(ctx.charts.histogram(
                    height  = 250,
                    nbins   = 15,
                    x       = postings.freshness,
                    x_title = "Days Since Posted",
                    y_title = "Count"
                ))
            )
            if postings.freshness else mo.md(""),
            direction = "h",
            widths    = [1, 1]
        ),

        ctx.layout.stack(
            ctx.layout.stack(
                ctx.layout.header(tab, "freshness_funnel", **section_kw),
                mo.ui.plotly(ctx.charts.funnel(
                    height = 250,
                    labels = [
                        f"All Dated ({len(postings.freshness)})",
                        f"Under 60 days ({sum(1 for d in postings.freshness if d < 60)})",
                        f"Under 30 days ({sum(1 for d in postings.freshness if d < 30)})",
                        f"Under 7 days ({sum(1 for d in postings.freshness if d < 7)})"
                    ],
                    values = [
                        len(postings.freshness),
                        sum(1 for d in postings.freshness if d < 60),
                        sum(1 for d in postings.freshness if d < 30),
                        sum(1 for d in postings.freshness if d < 7)
                    ]
                ))
            ) if postings.freshness else mo.md(""),
            ctx.layout.stack(
                ctx.layout.header(tab, "location_share", **section_kw),
                mo.ui.plotly(ctx.charts.pie(
                    height   = 280,
                    hole     = 0.4,
                    labels   = list(postings.locations),
                    textfont = {"size": 10},
                    textinfo = "label+percent",
                    values   = list(postings.locations.values())
                ))
            ) if postings.locations else mo.md(""),
            direction = "h",
            widths    = [1, 1]
        ),

        ctx.layout.header(tab, "recent", **section_kw),
        ctx.layout.grid(ctx.layout.posting_card(p) for p in postings.recent),
        ctx.layout.callout(tab.info)
    )
