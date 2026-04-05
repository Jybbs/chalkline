"""
Your Match tab for the Chalkline career report.
"""

import marimo as mo

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import HierarchyData, LaborMetrics, MatchMetrics
from chalkline.display.schemas import SectorMetrics


def _build_reach_sankey(ctx: TabContext) -> dict:
    """
    Build Sankey node/link data from the matched cluster's reach view, with
    advancement flowing right and lateral flowing left.
    """
    reach   = ctx.result.reach
    profile = ctx.profile
    source_label = profile.soc_title
    labels  = [source_label]
    colors  = [ctx.theme.colors["primary"]]
    sources, targets, values, link_colors = [], [], [], []

    for edge in reach.advancement:
        idx = len(labels)
        labels.append(edge.soc_title)
        colors.append(ctx.theme.colors["success"])
        sources.append(0)
        targets.append(idx)
        values.append(max(1, round(edge.weight * 100)))
        link_colors.append("rgba(140, 197, 163, 0.4)")

    for edge in reach.lateral:
        idx = len(labels)
        labels.append(edge.soc_title)
        colors.append(ctx.theme.colors["accent"])
        sources.append(0)
        targets.append(idx)
        values.append(max(1, round(edge.weight * 100)))
        link_colors.append("rgba(125, 179, 224, 0.4)")

    return {
        "colors" : colors,
        "labels" : labels,
        "links"  : {
            "color"  : link_colors,
            "source" : sources,
            "target" : targets,
            "value"  : values
        }
    }


def _build_skill_sunburst(ctx: TabContext) -> HierarchyData:
    """
    Build sunburst data decomposing the skill profile by type, then by
    demonstrated vs gap, with individual skills as leaves.
    """
    ids, labels, parents, values, colors = [], [], [], [], []
    ids.append("all")
    labels.append("All Skills")
    parents.append("")
    values.append(0)
    colors.append(ctx.theme.colors["primary"])

    for stype, tasks in ctx.result.tasks_by_type.items():
        type_label = ctx.theme.type_label(stype)
        type_id    = f"type_{stype}"
        ids.append(type_id)
        labels.append(type_label)
        parents.append("all")
        values.append(0)
        colors.append(ctx.theme.colors["accent"])

        for idx, task in enumerate(tasks):
            task_id = f"{type_id}_{idx}"
            ids.append(task_id)
            labels.append(task.name)
            parents.append(type_id)
            values.append(max(1, round(task.pct)))
            colors.append(
                ctx.theme.colors["success"] if task.demonstrated
                else ctx.theme.colors["error"]
            )

    return HierarchyData(
        colors  = colors,
        ids     = ids,
        labels  = labels,
        parents = parents,
        values  = values
    )


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

        ctx.layout.stack(
            mo.ui.plotly(ctx.charts.gauge(
                title = tab.chart_labels["gauge_title"],
                value = match.confidence
            )),
            mo.ui.plotly(ctx.charts.indicator(
                height    = 250,
                reference = 0,
                title     = "Runner-Up Gap",
                value     = match.runner_up_gap
            )),
            mo.ui.plotly(ctx.charts.funnel(
                color  = "accent",
                height = 250,
                labels = list(sectors.scores),
                values = list(sectors.scores.values())
            )),
            direction = "h",
            widths    = [1, 1, 1]
        ),

        ctx.layout.callout(tab.info),

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
                ctx.layout.header(tab, "jz_distribution"),
                mo.ui.plotly(ctx.charts.bar(
                    color  = "primary",
                    height = 320,
                    title  = tab.chart_labels["jz_title"],
                    x      = [
                        ctx.theme.jz_label(z) for z in match.job_zones
                    ],
                    y      = [*match.job_zones.values()]
                ))
            ),
            ctx.layout.stack(
                ctx.layout.header(tab, "sector_affinity"),
                mo.ui.plotly(ctx.charts.pie(
                    height   = 320,
                    hole     = 0.5,
                    labels   = [*sectors.scores],
                    marker   = dict(
                        colors=[ctx.theme.sectors[s] for s in sectors.scores]
                    ),
                    textfont = dict(size=12),
                    textinfo = "label+percent",
                    values   = [*sectors.scores.values()]
                ))
            ),
            direction = "h",
            widths    = [1, 1]
        ),

        *ctx.layout.section_if(
            ctx.result.reach.all_edges, tab, "career_transitions",
            mo.ui.plotly(ctx.charts.sankey(
                height = 400,
                **_build_reach_sankey(ctx),
            ))
        ),

        *ctx.layout.section_if(
            ctx.result.tasks_by_type, tab, "skill_landscape",
            mo.ui.plotly(ctx.charts.sunburst(
                data   = _build_skill_sunburst(ctx),
                height = 450
            ))
        ),

        ctx.layout.stack(
            ctx.layout.stack(
                ctx.layout.header(tab, "skill_waterfall"),
                mo.ui.plotly(ctx.charts.waterfall(
                    height   = 300,
                    measures = ["absolute"] + ["relative"] * len(ctx.result.gap_type_counts) + ["total"],
                    text     = [str(ctx.result.demonstrated_count)] + [
                        str(-v) for v in ctx.result.gap_type_counts.values()
                    ] + [str(ctx.result.demonstrated_count - ctx.result.gap_count)],
                    x        = ["Demonstrated"] + [
                        ctx.theme.type_label(t) for t in ctx.result.gap_type_counts
                    ] + ["Net"],
                    y        = [ctx.result.demonstrated_count] + [
                        -v for v in ctx.result.gap_type_counts.values()
                    ] + [0]
                ))
            ),
            ctx.layout.stack(
                ctx.layout.header(tab, "gap_concentration"),
                mo.ui.plotly(ctx.charts.treemap(
                    data   = HierarchyData(
                        labels = [
                            f"{ctx.theme.type_label(t)} ({v})"
                            for t, v in ctx.result.gap_type_counts.items()
                        ],
                        values = list(ctx.result.gap_type_counts.values())
                    ),
                    height = 300
                ))
            ) if ctx.result.gap_type_counts else mo.md(""),
            direction = "h",
            widths    = [1, 1]
        ),

        ctx.layout.stack(
            ctx.layout.stack(
                ctx.layout.header(tab, "similarity_overview"),
                mo.ui.plotly(ctx.charts.histogram(
                    height    = 320,
                    nbins     = 20,
                    threshold = ctx.result.mean_similarity,
                    x         = ctx.result.all_similarities,
                    x_title   = "Similarity (%)",
                    y_title   = "Skills"
                )) if ctx.result.all_similarities else mo.md("")
            ),
            ctx.layout.stack(
                ctx.layout.header(tab, "distance_by_sector"),
                mo.ui.plotly(ctx.charts.violin(
                    colors  = ctx.theme.sectors,
                    groups  = ctx.result.distances_by_sector,
                    height  = 320,
                    y_title = "Distance"
                ))
            ),
            direction = "h",
            widths    = [1, 1]
        ),

        ctx.layout.header(tab, "proximity"),
        mo.ui.plotly(ctx.charts.bar(
            color      = [ctx.theme.sectors[s] for s in match.sectors.values()],
            height     = max(400, len(match.proximity) * 26),
            horizontal = True,
            title      = tab.chart_labels["distance_title"],
            x          = [*match.proximity.values()],
            y          = [*match.proximity]
        ))
    )
