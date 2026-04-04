"""
Your Match tab for the Chalkline career report.
"""

import marimo as mo

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import LaborMetrics, MatchMetrics
from chalkline.display.schemas import RadarTrace, SectorMetrics


def your_match_tab(ctx: TabContext) -> mo.Html:
    """
    Render the Your Match tab showing where the resume fits in Maine's
    construction landscape.
    """
    tab     = ctx.content.tab("your_match")
    profile = ctx.profile
    match   = MatchMetrics.from_result(ctx.result)
    sectors = SectorMetrics.from_result(ctx.result)
    labor   = LaborMetrics.from_record(
        labor     = ctx.labor,
        soc_title = profile.soc_title,
        stat_keys = tab.labor_stats,
        templates = tab.labor_templates
    )

    return ctx.layout.stack(
        ctx.layout.callout(*tab.hero.render(
            jz_label     = ctx.theme.jz_label(profile.job_zone).lower(),
            outlook_text = labor.outlook_text,
            salary_text  = labor.salary_text,
            sector       = profile.sector,
            size         = profile.size,
            soc_title    = profile.soc_title
        )),
        ctx.layout.stats(zip(
            tab.stat_labels,
            [
                profile.soc_title,
                profile.sector,
                ctx.theme.jz_label(profile.job_zone),
                str(profile.size),
                str(ctx.result.gap_count),
                str(ctx.result.demonstrated_count)
            ]
        )),
        ctx.layout.callout(tab.info),

        ctx.layout.stack(
            mo.ui.plotly(ctx.charts.gauge(
                title = tab.chart_labels["gauge_title"],
                value = match.confidence
            )),
            mo.ui.plotly(ctx.charts.radar(
                height = 380,
                labels = list(match.top5),
                traces = [RadarTrace(
                    alpha      = 0.2,
                    color_role = "accent",
                    name       = tab.chart_labels["radar_trace"],
                    values     = list(match.top5.values())
                )]
            )),
            direction = "h",
            widths    = [1, 2]
        ),

        *ctx.layout.section_if(
            labor.stat_strip,
            tab,
            "labor_snapshot",
            ctx.layout.stats(labor.stat_strip.items())
        ),

        *ctx.layout.section_if(labor.wages, tab, "maine_wages",
            mo.ui.plotly(ctx.charts.bar(
                color      = ("muted", "accent", "primary", "accent", "muted"),
                height     = 220,
                horizontal = True,
                title      = tab.chart_labels["salary_title"],
                x          = labor.wages,
                y          = tab.chart_lists["wage_ticks"]
            ))),

        ctx.layout.stack(
            ctx.layout.stack(
                ctx.layout.header(tab, "sector_affinity"),
                mo.ui.plotly(ctx.charts.pie(
                    height   = 320,
                    hole     = 0.5,
                    labels   = sectors.scores,
                    marker   = dict(
                        colors=ctx.charts.sector_colors(sectors.scores)
                    ),
                    textfont = dict(size=12),
                    textinfo = "label+percent",
                    values   = sectors.scores.values()
                ))
            ),
            ctx.layout.stack(
                ctx.layout.header(tab, "jz_distribution"),
                mo.ui.plotly(ctx.charts.bar(
                    color  = "primary",
                    height = 320,
                    title  = tab.chart_labels["jz_title"],
                    x      = [
                        ctx.theme.jz_label(z) for z in match.job_zones
                    ],
                    y      = match.job_zones.values()
                ))
            ),
            direction = "h",
            widths    = [1, 1]
        ),

        ctx.layout.header(tab, "proximity"),
        mo.ui.plotly(ctx.charts.bar(
            color      = ctx.charts.sector_colors(match.sectors.values()),
            height     = max(400, len(match.proximity) * 26),
            horizontal = True,
            title      = tab.chart_labels["distance_title"],
            x          = match.proximity.values(),
            y          = match.proximity
        ))
    )
