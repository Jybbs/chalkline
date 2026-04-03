"""
Resume Feedback tab for the Chalkline career report.
"""

import marimo as mo

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import RadarTrace, SkillMetrics


def resume_feedback_tab(ctx: TabContext) -> mo.Html:
    """
    Render the Resume Feedback tab comparing the resume against O*NET skill
    definitions.
    """
    tab        = ctx.content.tab("resume_feedback")
    skills     = SkillMetrics.from_result(ctx.result)
    section_kw = {"soc_title": ctx.profile.soc_title}

    return ctx.layout.stack(
        ctx.layout.header(tab, "how_compare", **section_kw),
        ctx.layout.stats(zip(tab.stat_labels, skills.stat_values)),
        ctx.layout.callout(*tab.hero.render(
            posting_count = len(ctx.profile.postings),
            soc_title     = ctx.profile.soc_title
        )),

        ctx.layout.stack(
            ctx.layout.stack(
                ctx.layout.header(tab, "skill_profile", **section_kw),
                mo.ui.plotly(ctx.charts.radar(
                    labels = [
                        ctx.theme.type_label(t)
                        for t in skills.skill_groups
                    ],
                    traces = [
                        RadarTrace(
                            color_role = "accent",
                            name       = tab.chart_labels[
                                "all_skills_trace"
                            ],
                            values     = skills.overall_averages
                        ),
                        RadarTrace(
                            color_role = "success",
                            dash       = "dash",
                            name       = tab.chart_labels[
                                "strengths_trace"
                            ],
                            values     = skills.strength_averages
                        )
                    ]
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
                    x_title   = tab.chart_labels[
                        "similarity_x_title"
                    ],
                    y_title   = tab.chart_labels[
                        "similarity_y_title"
                    ]
                )) if ctx.result.all_similarities
                else mo.md(tab.fallbacks["no_similarity"])
            ),
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
