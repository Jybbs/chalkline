"""
Career Paths tab renderer.
"""

import marimo as mo

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import Trace


def career_paths_tab(
    ctx       : TabContext,
    dropdown  : mo.Html,
    target_id : int
) -> mo.Html:
    """
    Render the Career Paths tab with pathway graph, route strength bars,
    credential table, and career ladder.
    """
    tab     = ctx.content.tab("career_paths")
    profile = ctx.pipeline.clusters[target_id]
    reach   = ctx.pipeline.graph.reach(target_id)
    cols    = tab.columns["credentials"]

    series = [
        Trace(
            color_role = color,
            name       = direction,
            x          = [round(e.weight, 3) for e in edges],
            y          = [e.soc_title for e in edges]
        )
        for edges, color, direction in [
            (reach.advancement, "success", tab.directions["advancement"]),
            (reach.lateral,     "accent",  tab.directions["lateral"])
        ]
        if edges
    ]

    credential_rows: list[dict[str, object]] = [
        {
            cols["credential"] : c.label,
            cols["direction"]  : direction,
            cols["hours"]      : f"{h:,}" if (h := c.hours) else "",
            cols["target"]     : e.soc_title,
            cols["type"]       : c.type_label
        }
        for direction, edges in [
            (tab.directions["advancement"], reach.advancement),
            (tab.directions["lateral"],     reach.lateral)
        ]
        for e in edges
        for c in e.credentials
    ]

    return mo.vstack([
        ctx.layout.header(tab.section("overview")),
        dropdown,
        mo.ui.plotly(ctx.charts.pathways(
            reach     = reach,
            target_id = target_id,
            title     = tab.chart_labels["pathways_title"].format(
                soc_title = profile.soc_title
            )
        )),
        ctx.layout.callout(tab.info),

        *ctx.layout.section_if(
            reach.all_edges, tab, "movement",
            mo.ui.plotly(ctx.charts.pie(
                height   = 260,
                hole     = 0.4,
                labels   = [
                    tab.directions["advancement"],
                    tab.directions["lateral"]
                ],
                marker   = {
                    "colors": [
                        ctx.theme.colors[r]
                        for r in ("success", "accent")
                    ]
                },
                textfont = {"size": 12},
                textinfo = "label+value",
                values   = [
                    len(reach.advancement),
                    len(reach.lateral)
                ]
            )),
            adv_count = len(reach.advancement),
            lat_count = len(reach.lateral),
            soc_title = profile.soc_title
        ),

        *ctx.layout.section_if(
            series, tab, "route_strength",
            mo.ui.plotly(ctx.charts.bar(
                height = max(
                    280,
                    (len(reach.advancement) + len(reach.lateral)) * 26
                ),
                series = series,
                title  = tab.chart_labels["route_strength_title"]
            ))
        ),

        *ctx.layout.section_if(
            sector_peers := ctx.pipeline.clusters.by_sector(profile.sector),
            tab, "career_ladder",
            mo.ui.plotly(ctx.charts.career_ladder(
                clusters    = [p for _, p in sector_peers],
                target_id   = target_id,
                tick_labels = ctx.content.labels.job_zones_abbr.values(),
                x_title     = tab.chart_labels["job_zone_axis"]
            )),
            sector = profile.sector
        ),

        *([mo.accordion(
            {
                tab.sections["credentials"].title.format(
                    count = len(credential_rows)
                ):
                mo.ui.table(credential_rows)
            },
            multiple = True
        )] if credential_rows else [])
    ])
