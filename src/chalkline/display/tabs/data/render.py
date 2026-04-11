"""
Data tab for the Chalkline career report.

Trimmed view of the matched career family's job postings, promoting sample
posting cards to hero position and consolidating charts into companies,
locations, and two word treemaps. Timeline and company type breakdowns are
demoted to stats in the header strip rather than standalone charts.
"""

from marimo import Html

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import JobPostingMetrics


def data_tab(ctx: TabContext) -> Html:
    """
    Render the Data tab with posting cards first, followed by hiring
    companies, locations, and word frequency treemaps.

    Sample postings are promoted to hero position so the user sees
    real listings before aggregate charts. The freshness histogram,
    timeline scatter, company types pie, location pie, and freshness
    funnel are all removed. Timeline recency and member type counts
    are folded into the stat strip instead.
    """
    tab        = ctx.content.tab("data")
    postings   = JobPostingMetrics.from_postings(ctx.profile.postings, ctx.reference)
    section_kw = {"soc_title": ctx.profile.soc_title}

    return ctx.layout.stack(
        ctx.layout.overview(tab, "overview", **section_kw),
        ctx.layout.stats(zip(tab.stat_labels, postings.stat_values)),

        ctx.layout.header(tab, "recent", **section_kw),
        ctx.layout.grid(ctx.layout.posting_card(p) for p in postings.recent),

        ctx.layout.two_col(*(
            ctx.layout.panel(tab, key, ctx.charts.bar(
                color      = color,
                data       = data,
                height     = max(300, len(data) * 28),
                horizontal = True,
                title      = tab.chart_labels["postings_title"]
            ), **section_kw)
            for key, color, data in [
                ("whos_hiring", "accent",  postings.companies),
                ("locations",   "success", postings.locations)
            ]
        )),

        ctx.layout.two_col(*(
            ctx.layout.panel(tab, key,
                ctx.charts.treemap(data=data, height=350), **section_kw)
            for key, data in [
                ("common_titles",       postings.titles),
                ("common_descriptions", postings.descriptions)
            ]
        )),

        *ctx.layout.section_if(
            postings.by_month, tab, "freshness",
            ctx.charts.bar(
                color  = "accent",
                data   = postings.by_month,
                height = 280,
                title  = tab.chart_labels["postings_title"]
            )
        ),

        ctx.layout.callout(tab.info)
    )
