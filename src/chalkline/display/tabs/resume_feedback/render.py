"""
Resume Feedback tab for the Chalkline career report.
"""

import marimo as mo

from functools import partial

from chalkline.display.layout       import callout, header, skill_tree, stat_strip
from chalkline.display.tabs.context import TabContext, load_content

content = load_content(__file__)


def resume_feedback_tab(ctx: TabContext) -> mo.Html:
    """
    Render the Resume Feedback tab comparing the resume against
    O*NET skill definitions.
    """
    radar_fig = ctx.charts.radar(
        labels = [ctx.theme.type_label(t) for t in ctx.data.skills.all_types],
        traces = [
            {
                "alpha"      : 0.15,
                "color_role" : "accent",
                "name"       : "All Skills",
                "values"     : ctx.data.skills.resume_averages
            },
            {
                "alpha"      : 0.15,
                "color_role" : "success",
                "dash"       : "dash",
                "name"       : "Strengths Only",
                "values"     : ctx.data.skills.demo_averages
            }
        ]
    ) if ctx.data.skills.all_types else None

    gap_scatter = ctx.charts.bubble_scatter(
        height = 350,
        points = ctx.data.skills.gap_scatter_points
    ) if ctx.data.skills.gap_scatter_points else None

    section = partial(content.section, soc_title=ctx.data.soc_title)

    return mo.vstack([
        header(*section("how_compare")),
        stat_strip({
            "Strengths"            : str(len(ctx.data.result.demonstrated)),
            "Growth Areas"         : str(len(ctx.data.result.gaps)),
            "Alignment Percentile" : f"{ctx.data.match.percentile}%"
        }),
        callout(*content.hero.render(
            percentile    = ctx.data.match.percentile,
            posting_count = len(ctx.data.profile.postings),
            soc_title     = ctx.data.soc_title
        )),

        header(*section("skill_profile")),
        mo.ui.plotly(radar_fig) if radar_fig
        else mo.md("No skill data available."),

        header(*section("similarity_dist")),
        mo.ui.plotly(ctx.charts.histogram_with_threshold(
            height    = 280,
            scores    = ctx.data.skills.similarity_percentages,
            threshold = ctx.data.skills.threshold_percentage
        )) if ctx.data.skills.all_similarities
        else mo.md("No similarity data."),

        header(*section("strengths")),
        skill_tree(ctx.data.skills.demo_groups, ctx.theme)
        if ctx.data.skills.demo_groups
        else mo.md("No demonstrated skills."),

        header(*section("growth")),
        skill_tree(ctx.data.skills.gap_groups, ctx.theme)
        if ctx.data.skills.gap_groups else mo.md("No gaps identified."),

        *([
            header(*section("gap_priority")),
            mo.ui.plotly(gap_scatter)
        ] if gap_scatter else []),

        callout(content.info)
    ])
