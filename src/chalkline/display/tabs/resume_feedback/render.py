"""
Resume Feedback tab for the Chalkline career report.
"""

import marimo as mo

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import SkillMetrics, Trace


def _build_skill_funnel(skills: SkillMetrics) -> tuple[list[str], list[int]]:
    """
    Build a funnel from total skills down to top strengths, showing how the
    pool narrows at each confidence threshold.
    """
    all_tasks = [
        s for v in skills.skill_groups.values() for s in v
    ]
    total        = len(all_tasks)
    demonstrated = sum(1 for s in all_tasks if s.demonstrated)
    high_conf    = sum(1 for s in all_tasks if s.pct > 70)
    top          = sum(1 for s in all_tasks if s.pct > 85)
    return (
        ["Total Skills", "Demonstrated", "High Confidence (>70%)", "Top Strengths (>85%)"],
        [total, demonstrated, high_conf, top]
    )


def _build_skill_parcats(ctx: TabContext) -> list[dict]:
    """
    Build parallel categories dimensions from scored tasks, flowing Type to
    Outcome to Score Bucket.
    """
    tasks = [
        s for v in ctx.result.tasks_by_type.values() for s in v
    ]
    types    = [ctx.theme.type_label(s.skill_type) for s in tasks]
    outcomes = ["Strength" if s.demonstrated else "Gap" for s in tasks]
    buckets  = [
        "70-100%" if s.pct >= 70
        else "40-70%" if s.pct >= 40
        else "0-40%"
        for s in tasks
    ]
    return [
        {"label": "Skill Type", "values": types},
        {"label": "Outcome", "values": outcomes},
        {"label": "Score Range", "values": buckets}
    ]


def resume_feedback_tab(ctx: TabContext) -> mo.Html:
    """
    Render the Resume Feedback tab comparing the resume against O*NET skill
    definitions.
    """
    tab        = ctx.content.tab("resume_feedback")
    skills     = SkillMetrics.from_result(ctx.result)
    section_kw = {"soc_title": ctx.profile.soc_title}

    type_labels = [ctx.theme.type_label(t) for t in skills.skill_groups]

    return ctx.layout.stack(
        ctx.layout.overview(tab, "how_compare", **section_kw),
        ctx.layout.stats(zip(tab.stat_labels, skills.stat_values)),
        ctx.layout.callout(*tab.hero.render(
            posting_count = len(ctx.profile.postings),
            soc_title     = ctx.profile.soc_title
        )),

        ctx.layout.stack(
            ctx.layout.stack(
                ctx.layout.header(tab, "skill_profile", **section_kw),
                mo.ui.plotly(ctx.charts.bar(
                    height     = 340,
                    horizontal = True,
                    series     = [
                        Trace(
                            color_role = "accent",
                            name       = tab.chart_labels["all_skills_trace"],
                            x          = skills.overall_averages,
                            y          = type_labels
                        ),
                        Trace(
                            color_role = "success",
                            name       = tab.chart_labels["strengths_trace"],
                            x          = skills.strength_averages,
                            y          = type_labels
                        )
                    ],
                    title = ""
                )) if skills.skill_groups
                else mo.md(tab.fallbacks["no_skill_data"])
            ),
            ctx.layout.stack(
                ctx.layout.header(tab, "similarity_dist", **section_kw),
                mo.ui.plotly(ctx.charts.histogram(
                    height    = 340,
                    nbins     = 20,
                    threshold = ctx.result.mean_similarity,
                    x         = ctx.result.all_similarities,
                    x_title   = tab.chart_labels["similarity_x_title"],
                    y_title   = tab.chart_labels["similarity_y_title"]
                )) if ctx.result.all_similarities
                else mo.md(tab.fallbacks["no_similarity"])
            ),
            direction = "h",
            widths    = [1, 1]
        ),

        ctx.layout.stack(
            ctx.layout.stack(
                ctx.layout.header(tab, "demonstration_rate", **section_kw),
                mo.ui.plotly(ctx.charts.bar(
                    color      = "success",
                    height     = max(250, len(skills.demonstration_rates) * 32),
                    horizontal = True,
                    title      = "",
                    x          = [*skills.demonstration_rates.values()],
                    y          = [
                        ctx.theme.type_label(t)
                        for t in skills.demonstration_rates
                    ]
                ))
            ) if skills.demonstration_rates else mo.md(""),
            ctx.layout.stack(
                ctx.layout.header(tab, "skill_funnel", **section_kw),
                mo.ui.plotly((funnel := _build_skill_funnel(skills)) and ctx.charts.funnel(
                    height = max(250, 60 * len(funnel[0])),
                    labels = funnel[0],
                    values = funnel[1]
                ))
            ),
            direction = "h",
            widths    = [1, 1]
        ),

        *ctx.layout.section_if(
            ctx.result.tasks_by_type, tab, "skill_flow",
            mo.ui.plotly(ctx.charts.parcats(
                dimensions = _build_skill_parcats(ctx),
                height     = 400,
                color      = [s.pct for v in ctx.result.tasks_by_type.values() for s in v]
            ))
        ),

        ctx.layout.stack(
            ctx.layout.stack(
                ctx.layout.header(tab, "skill_waterfall", **section_kw),
                mo.ui.plotly(ctx.charts.waterfall(
                    height   = 300,
                    measures = (
                        ["absolute"]
                        + ["relative"] * len(skills.demonstration_rates)
                        + ["total"]
                    ),
                    text     = [
                        str(sum(
                            len(v) for v in skills.skill_groups.values()
                        ))
                    ] + [
                        f"-{len(v) - sum(s.demonstrated for s in v)}"
                        for v in skills.skill_groups.values()
                    ] + [""],
                    x        = ["Total"] + [
                        ctx.theme.type_label(t)
                        for t in skills.skill_groups
                    ] + ["Net"],
                    y        = [
                        sum(len(v) for v in skills.skill_groups.values())
                    ] + [
                        -(len(v) - sum(s.demonstrated for s in v))
                        for v in skills.skill_groups.values()
                    ] + [0]
                ))
            ),
            ctx.layout.stack(
                ctx.layout.header(tab, "score_violin", **section_kw),
                mo.ui.plotly(ctx.charts.violin(
                    groups  = {
                        ctx.theme.type_label(t): [s.pct for s in tasks]
                        for t, tasks in skills.skill_groups.items()
                    },
                    height  = 300,
                    y_title = "Similarity (%)"
                ))
            ),
            direction = "h",
            widths    = [1, 1]
        ),

        ctx.layout.stack(
            ctx.layout.stack(
                ctx.layout.header(tab, "confidence_gauge", **section_kw),
                mo.ui.plotly(ctx.charts.gauge(
                    title = "Match Confidence",
                    value = ctx.result.confidence
                ))
            ),
            ctx.layout.stack(
                ctx.layout.header(tab, "gap_types", **section_kw),
                mo.ui.plotly(ctx.charts.pie(
                    height   = 280,
                    hole     = 0.4,
                    labels   = [
                        ctx.theme.type_label(t)
                        for t in ctx.result.gap_type_counts
                    ],
                    textfont = {"size": 11},
                    textinfo = "label+value",
                    values   = list(ctx.result.gap_type_counts.values())
                ))
            ) if ctx.result.gap_type_counts else mo.md(""),
            direction = "h",
            widths    = [1, 1]
        ),

        ctx.layout.header(tab, "strengths", **section_kw),
        ctx.layout.skill_tree(True, skills.skill_groups, ctx.theme)
        if skills.skill_groups
        else mo.md(tab.fallbacks["no_strengths"]),

        ctx.layout.header(tab, "growth", **section_kw),
        ctx.layout.skill_tree(False, skills.skill_groups, ctx.theme)
        if skills.skill_groups else mo.md(tab.fallbacks["no_gaps"]),

        *ctx.layout.section_if(skills.gap_scatter_points, tab, "gap_priority",
            mo.ui.plotly(ctx.charts.bubble_scatter(
                height  = 350,
                points  = skills.gap_scatter_points,
                x_title = tab.chart_labels["gap_x_title"],
                y_title = tab.chart_labels["gap_y_title"]
            )), soc_title=ctx.profile.soc_title),

        ctx.layout.callout(tab.info)
    )
