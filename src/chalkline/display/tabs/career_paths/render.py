"""
Career Paths tab renderer.
"""

import marimo as mo

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import HierarchyData, Trace


def _build_path_sankey(ctx: TabContext, reach, profile) -> dict:
    """
    Build Sankey data from reach edges, with advancement flowing right and
    lateral flowing left from the selected cluster.
    """
    tab     = ctx.content.tab("career_paths")
    labels  = [profile.soc_title]
    colors  = [ctx.theme.colors["primary"]]
    sources, targets, values, link_colors, link_labels = [], [], [], [], []

    for edge in reach.advancement:
        idx = len(labels)
        labels.append(edge.soc_title)
        colors.append(ctx.theme.colors["success"])
        sources.append(0)
        targets.append(idx)
        values.append(max(1, round(edge.weight * 100)))
        link_colors.append("rgba(140, 197, 163, 0.4)")
        link_labels.append(
            f"{len(edge.credentials)} credentials"
            if edge.credentials else ""
        )

    for edge in reach.lateral:
        idx = len(labels)
        labels.append(edge.soc_title)
        colors.append(ctx.theme.colors["accent"])
        sources.append(0)
        targets.append(idx)
        values.append(max(1, round(edge.weight * 100)))
        link_colors.append("rgba(125, 179, 224, 0.4)")
        link_labels.append(
            f"{len(edge.credentials)} credentials"
            if edge.credentials else ""
        )

    return {
        "colors" : colors,
        "labels" : labels,
        "links"  : {
            "color"  : link_colors,
            "label"  : link_labels,
            "source" : sources,
            "target" : targets,
            "value"  : values
        }
    }


def _build_landscape_sunburst(ctx: TabContext) -> HierarchyData:
    """
    Build a sunburst of sector to job zone to SOC title, sized by posting
    count.
    """
    clusters = ctx.pipeline.clusters
    ids, labels, parents, values, colors = [], [], [], [], []

    for sector in clusters.sectors:
        ids.append(sector)
        labels.append(sector)
        parents.append("")
        values.append(0)
        colors.append(ctx.theme.sectors[sector])

    for cluster in clusters.values():
        jz_key = f"{cluster.sector}_JZ{cluster.job_zone}"
        if jz_key not in ids:
            ids.append(jz_key)
            labels.append(f"Job Zone {cluster.job_zone}")
            parents.append(cluster.sector)
            values.append(0)
            colors.append(ctx.theme.sectors[cluster.sector])

        cid = f"c_{cluster.cluster_id}"
        ids.append(cid)
        labels.append(f"{cluster.soc_title} ({cluster.size})")
        parents.append(jz_key)
        values.append(cluster.size)
        colors.append(ctx.theme.sectors[cluster.sector])

    return HierarchyData(
        colors  = colors,
        ids     = ids,
        labels  = labels,
        parents = parents,
        values  = values
    )


def _build_credential_heatmap(reach) -> tuple[list[str], list[str], list[list[int]]]:
    """
    Build a destination x credential-kind matrix showing how many
    credentials bridge each transition.
    """
    kinds = sorted({c.kind for e in reach.all_edges for c in e.credentials})
    if not kinds:
        return [], [], []

    destinations = [e.soc_title for e in reach.all_edges if e.credentials]
    matrix = []
    for edge in reach.all_edges:
        if not edge.credentials:
            continue
        kind_counts = {k: 0 for k in kinds}
        for c in edge.credentials:
            kind_counts[c.kind] += 1
        matrix.append([kind_counts[k] for k in kinds])

    return destinations, kinds, matrix


def career_paths_tab(
    ctx       : TabContext,
    dropdown  : mo.Html,
    target_id : int
) -> mo.Html:
    """
    Render the Career Paths tab with Sankey flow, route strength bars,
    credential heatmap, career ladder, and landscape sunburst.
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

    sankey = _build_path_sankey(ctx, reach, profile)
    sb     = _build_landscape_sunburst(ctx)
    cred_dest, cred_kinds, cred_matrix = _build_credential_heatmap(reach)

    return ctx.layout.stack(
        ctx.layout.overview(tab, "overview"),
        dropdown,

        *ctx.layout.section_if(
            reach.all_edges, tab, "movement",
            mo.ui.plotly(ctx.charts.sankey(
                colors = sankey["colors"],
                height = 400,
                labels = sankey["labels"],
                links  = sankey["links"]
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

        ctx.layout.stack(
            ctx.layout.stack(
                ctx.layout.header(tab, "wage_progression"),
                mo.ui.plotly(ctx.charts.bar(
                    color      = "primary",
                    height     = max(250, (1 + len(reach.all_edges)) * 28),
                    horizontal = True,
                    title      = "",
                    x          = [
                        wp.wage for wp in ctx.labor.wage_pairs(
                            [profile.soc_title]
                            + [e.soc_title for e in reach.all_edges]
                        )
                    ],
                    y          = [
                        wp.title for wp in ctx.labor.wage_pairs(
                            [profile.soc_title]
                            + [e.soc_title for e in reach.all_edges]
                        )
                    ]
                ))
            ) if ctx.labor.wage_pairs(
                [profile.soc_title]
                + [e.soc_title for e in reach.all_edges]
            ) else mo.md(""),
            ctx.layout.stack(
                ctx.layout.header(tab, "credential_hours"),
                mo.ui.plotly(ctx.charts.waterfall(
                    height   = max(250, len(reach.advancement) * 50),
                    measures = ["relative"] * len(reach.advancement) + ["total"],
                    text     = [
                        f"{sum(c.hours or 0 for c in e.credentials):,}h"
                        for e in reach.advancement
                    ] + ["Total"],
                    x        = [
                        e.soc_title for e in reach.advancement
                    ] + ["Total"],
                    y        = [
                        sum(c.hours or 0 for c in e.credentials)
                        for e in reach.advancement
                    ] + [0]
                ))
            ) if reach.advancement else mo.md(""),
            direction = "h",
            widths    = [1, 1]
        ),

        *ctx.layout.section_if(
            cred_matrix, tab, "credential_matrix",
            mo.ui.plotly(ctx.charts.heatmap(
                columns = cred_kinds,
                height  = max(250, len(cred_dest) * 30),
                labels  = cred_dest,
                values  = cred_matrix
            ))
        ),

        ctx.layout.stack(
            ctx.layout.stack(
                ctx.layout.header(
                    tab, "career_ladder",
                    sector=profile.sector
                ),
                mo.ui.plotly(ctx.charts.career_ladder(
                    clusters    = [
                        p for _, p in
                        ctx.pipeline.clusters.by_sector(profile.sector)
                    ],
                    target_id   = target_id,
                    tick_labels = [*ctx.content.labels.job_zones_abbr.values()],
                    x_title     = tab.chart_labels["job_zone_axis"]
                ))
            ),
            ctx.layout.stack(
                ctx.layout.header(tab, "landscape_sunburst"),
                mo.ui.plotly(ctx.charts.sunburst(data=sb, height=400))
            ),
            direction = "h",
            widths    = [1, 1]
        ),

        *([mo.accordion(
            {
                tab.sections["credentials"].title.format(
                    count=len(credential_rows)
                ):
                mo.ui.table(credential_rows)
            },
            multiple = True
        )] if credential_rows else []),

        ctx.layout.callout(tab.info)
    )
