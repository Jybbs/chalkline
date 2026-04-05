"""
Next Steps tab renderer.
"""

import marimo as mo

from collections import Counter

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import HierarchyData


def next_steps_tab(
    ctx       : TabContext,
    target_id : int
) -> mo.Html:
    """
    Render the Next Steps tab.
    """
    tab             = ctx.content.tab("next_steps")
    clusters        = ctx.pipeline.clusters
    profile         = clusters[target_id]
    reach           = ctx.pipeline.graph.reach(target_id)
    reference       = ctx.reference
    apprenticeships = reach.credentials_by_kind.get("apprenticeship", [])
    programs        = reach.credentials_by_kind.get("program", [])

    employers   = reference.match_employers(profile.postings)
    cred_counts = {k.title(): len(v) for k, v in reach.credentials_by_kind.items()}
    wage_ladder = ctx.labor.wage_pairs(e.soc_title for e in reach.advancement)
    boards      = reference.filter_boards(clusters.sector_keywords(profile.sector))
    maine       = boards.get("maine", [])
    national    = boards.get("national", [])

    return ctx.layout.stack(
        ctx.layout.overview(tab, "overview", soc_title=profile.soc_title),

        ctx.layout.stack(
            ctx.layout.stack(
                ctx.layout.header(tab, "credentials"),
                mo.ui.plotly(ctx.charts.pie(
                    height   = 280,
                    hole     = 0.4,
                    labels   = [*cred_counts],
                    textfont = dict(size=11),
                    textinfo = "label+value",
                    values   = [*cred_counts.values()]
                ))
            ) if cred_counts else mo.md(""),
            ctx.layout.stack(
                ctx.layout.header(tab, "apprenticeships"),
                mo.ui.plotly(ctx.charts.bar(
                    color      = (hours := [a.hours for a in apprenticeships]),
                    colorscale = "Teal",
                    height     = max(250, len(apprenticeships) * 32),
                    horizontal = True,
                    title      = tab.chart_labels["min_hours_title"],
                    x          = hours,
                    y          = [a.label for a in apprenticeships]
                ))
            ) if apprenticeships else mo.md(""),
            direction = "h",
            widths    = [1, 2]
        ),

        *ctx.layout.section_if(wage_ladder, tab, "wages",
            mo.ui.plotly(ctx.charts.bar(
                color      = ctx.theme.colors["success"],
                height     = max(200, len(wage_ladder) * 32),
                horizontal = True,
                title      = tab.chart_labels["median_wage_title"],
                x          = [w.wage for w in wage_ladder],
                y          = [w.title for w in wage_ladder]
            ))
        ),

        *ctx.layout.section_if(programs, tab, "programs",
            ctx.layout.grid([
                ctx.layout.program_card(**p.metadata, name=p.label)
                for p in programs
            ]),
            count=len(programs)
        ),

        *ctx.layout.section_if(employers, tab, "employers",
            *([mo.ui.plotly(ctx.charts.treemap(
                data   = HierarchyData(
                    labels = list(emp_types),
                    values = list(emp_types.values())
                ),
                height = 300
            ))]
            if (emp_types := Counter(
                e["member_type"] for e in employers if e["member_type"]
            ))
            else []),
            ctx.layout.grid([
                ctx.layout.employer_card(**e)
                for e in employers
            ]),
            count=len(employers)
        ),

        *ctx.layout.section_if(maine or national, tab, "boards",
            ctx.layout.grid([
                ctx.layout.board_card(**b)
                for b in maine + national
            ]),
            count=len(maine) + len(national)
        ),

        *(
            [ctx.layout.callout(tab.empty_message, kind="warn")]
            if not any((apprenticeships, programs, employers, maine, national))
            else []
        ),

        ctx.layout.callout(tab.info)
    )
