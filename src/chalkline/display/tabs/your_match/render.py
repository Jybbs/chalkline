"""
Your Match tab for the Chalkline career report.
"""

import marimo as mo

from chalkline.display.layout       import callout, header, stat_strip
from chalkline.display.tabs.context import TabContext, load_content

content = load_content(__file__)


def your_match_tab(ctx: TabContext) -> mo.Html:
    """
    Render the Your Match tab showing where the resume fits in
    Maine's construction landscape.
    """
    zones, counts = zip(*sorted(ctx.data.match.jz_counts.items()))

    return mo.vstack([
        callout(*content.hero.render(
            jz_label     = ctx.theme.jz_label(ctx.data.profile.job_zone).lower(),
            outlook_text = ctx.data.labor_metrics.outlook_text,
            salary_text  = ctx.data.labor_metrics.salary_text,
            sector       = ctx.data.profile.sector,
            size         = ctx.data.profile.size,
            soc_title    = ctx.data.soc_title
        )),
        stat_strip({
            "Career Family"    : ctx.data.soc_title,
            "Sector"           : ctx.data.profile.sector,
            "Experience Level" : ctx.theme.jz_label(ctx.data.profile.job_zone),
            "Postings"         : str(ctx.data.profile.size),
            "Growth Areas"     : str(len(ctx.data.result.gaps)),
            "Strengths"        : str(len(ctx.data.result.demonstrated))
        }),
        callout(content.info),

        mo.hstack(
            [
                mo.ui.plotly(ctx.charts.gauge(ctx.data.match.confidence)),
                mo.ui.plotly(ctx.charts.radar(
                    height = 380,
                    labels = ctx.data.match.top5_labels,
                    traces = [{
                        "alpha"      : 0.2,
                        "color_role" : "accent",
                        "name"       : "Top 5 Matches",
                        "values"     : ctx.data.match.top5_values
                    }]
                ))
            ],
            widths=[1, 2]
        ),

        *([
            header(*content.section("labor_snapshot")),
            stat_strip(ctx.data.labor_metrics.stat_strip)
        ] if ctx.data.labor_metrics.stat_strip else []),

        *([
            header(*content.section("maine_wages")),
            mo.ui.plotly(ctx.charts.hbar(
                color  = [
                    ctx.theme.colors[r] for r in
                    ("muted", "accent", "primary", "accent", "muted")
                ],
                height = 220,
                title  = "Annual Salary ($)",
                x      = [
                    ctx.data.labor_metrics.wage_10,
                    ctx.data.labor_metrics.wage_25,
                    ctx.data.labor_metrics.wage_median,
                    ctx.data.labor_metrics.wage_75,
                    ctx.data.labor_metrics.wage_90
                ],
                y      = ["10th", "25th", "Median", "75th", "90th"]
            ))
        ] if ctx.data.labor_metrics.wage_median else []),

        header(*content.section("sector_affinity")),
        mo.ui.plotly(ctx.charts.sector_donut(
            height = 320,
            labels = ctx.data.sectors.labels,
            values = ctx.data.sectors.values
        )),

        header(*content.section("proximity")),
        mo.ui.plotly(ctx.charts.proximity_bar(
            cluster_ids = ctx.data.match.cluster_ids,
            height      = max(400, len(ctx.data.match.proximity_labels) * 26),
            labels      = ctx.data.match.proximity_labels,
            matched_id  = ctx.data.match.matched_id,
            values      = ctx.data.match.proximity_values
        )),

        header(*content.section("jz_distribution")),
        mo.ui.plotly(ctx.charts.vbar(
            color  = ctx.theme.colors["accent"],
            height = 280,
            title  = "Career Families in Top 10",
            x      = [ctx.theme.jz_label(z) for z in zones],
            y      = list(counts)
        ))
    ])
