"""
Career Paths tab renderer.
"""

import marimo as mo

from chalkline.display.layout       import callout, header
from chalkline.display.schemas      import TargetData
from chalkline.display.tabs.context import TabContext, load_content

content = load_content(__file__)


def career_paths_tab(
    ctx      : TabContext,
    data     : TargetData,
    dropdown : mo.Html
) -> mo.Html:
    """
    Render the Career Paths tab.
    """
    series = [
        {
            "color_role" : color_role,
            "name"       : name,
            "x"          : weights,
            "y"          : labels
        }
        for color_role, name, weights, labels in [
            ("success", "Advancement", data.adv_weights, data.adv_labels),
            ("accent",  "Lateral",     data.lat_weights, data.lat_labels)
        ]
        if labels
    ]

    return mo.vstack([
        header(*content.section("overview")),
        dropdown,
        mo.ui.plotly(ctx.charts.pathways(data.reach, data.target_id)),
        callout(content.info),

        *([
            header(*content.section(
                "movement",
                adv_count = data.advancement_count,
                lat_count = data.lateral_count,
                soc_title = data.soc_title
            )),
            mo.ui.plotly(ctx.charts.pie(
                height   = 260,
                hole     = 0.4,
                labels   = ["Advancement", "Lateral"],
                marker   = {"colors": [
                    ctx.theme.colors["success"],
                    ctx.theme.colors["accent"]
                ]},
                textinfo = "label+value",
                textfont = {"size": 12},
                values   = [data.advancement_count, data.lateral_count]
            ))
        ] if data.advancement_count or data.lateral_count else []),

        *([
            header(*content.section("route_strength")),
            mo.ui.plotly(ctx.charts.grouped_hbar(
                height  = max(280, (len(data.adv_labels) + len(data.lat_labels)) * 26),
                series  = series,
                x_title = "Path Strength (cosine similarity)"
            ))
        ] if series else []),

        *([
            header(*content.section("career_ladder", sector=data.target_profile.sector)),
            mo.ui.plotly(ctx.charts.career_ladder(
                clusters  = data.same_sector,
                target_id = data.target_id
            ))
        ] if data.same_sector else []),

        *([mo.accordion(
            {
                content.sections["credentials"].title.format(
                    count=len(data.credential_rows)
                ):
                mo.ui.table(data.credential_rows)
            },
            multiple = True
        )] if data.credential_rows else [])
    ])
