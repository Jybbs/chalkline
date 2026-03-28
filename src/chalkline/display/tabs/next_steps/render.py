"""
Next Steps tab renderer.
"""

import marimo as mo

from collections import Counter

from chalkline.display.layout       import board_card, callout, card_grid
from chalkline.display.layout       import employer_card, header, program_card
from chalkline.display.schemas      import TargetData
from chalkline.display.tabs.context import TabContext, load_content

content = load_content(__file__)


def next_steps_tab(ctx: TabContext, data: TargetData) -> mo.Html:
    """
    Render the Next Steps tab.
    """
    apprenticeships = data.apprenticeship_rows
    programs        = data.program_rows
    employers       = data.employer_rows
    maine, national = ctx.data.board_rows

    return mo.vstack([
        header(*content.section("overview", soc_title=data.soc_title)),

        *([
            header(*content.section("credentials")),
            mo.ui.plotly(ctx.charts.pie(
                280,
                hole     = 0.4,
                labels   = list(data.credential_counts),
                textinfo = "label+value",
                textfont = dict(size=11),
                values   = list(data.credential_counts.values())
            ))
        ] if data.credential_counts else []),

        *([
            header(*content.section("apprenticeships")),
            mo.ui.plotly(ctx.charts.hbar(
                color      = (hours := [a["Min Hours"] for a in apprenticeships]),
                colorscale = "Teal",
                height     = max(250, len(apprenticeships) * 32),
                title      = "Minimum Hours",
                x          = hours,
                y          = [a["Trade"][:35] for a in apprenticeships]
            ))
        ] if apprenticeships else []),

        *([
            header(*content.section("wages")),
            mo.ui.plotly(ctx.charts.hbar(
                color  = ctx.theme.colors["success"],
                height = max(200, len(data.wage_ladder) * 32),
                title  = "Annual Median Wage ($)",
                x      = [w[1] for w in data.wage_ladder],
                y      = [w[0] for w in data.wage_ladder]
            ))
        ] if data.wage_ladder else []),

        *([
            header(*content.section("programs", count=len(programs))),
            card_grid([
                program_card(
                    credential  = p["Credential"],
                    institution = p["Institution"],
                    name        = p["Program"],
                    url         = p["Link"]
                )
                for p in programs
            ])
        ] if programs else []),

        *([
            header(*content.section("employers", count=len(employers))),
            *(
                [mo.ui.plotly(ctx.charts.treemap(
                    height = 300,
                    labels = list(emp_types),
                    values = list(emp_types.values())
                ))]
                if (emp_types := Counter(e["Type"] for e in employers if e["Type"]))
                else []
            ),
            card_grid([
                employer_card(
                    career_url  = e["Career Page"],
                    member_type = e["Type"],
                    name        = e["Company"],
                    posting_url = e["Posting"]
                )
                for e in employers
            ])
        ] if employers else []),

        *([
            header(*content.section("boards", count=len(maine) + len(national))),
            card_grid([
                board_card(
                    best_for = b["Best For"],
                    category = b["Category"],
                    focus    = b["Focus"],
                    name     = b["Name"]
                )
                for b in maine + national
            ])
        ] if maine or national else []),

        *(
            [callout(content.empty_message, kind="warn")]
            if not any([apprenticeships, programs, employers, maine, national])
            else []
        ),

        callout(content.info)
    ])
