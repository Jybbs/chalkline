"""
ML Internals tab renderer.
"""

import marimo as mo

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import GapScatterPoint, MlMetrics


def ml_internals_tab(ctx: TabContext) -> mo.Html:
    """
    Render the ML Internals diagnostic tab with pipeline statistics,
    variance charts, cluster quality metrics, and SOC heatmaps.
    """
    tab         = ctx.content.tab("ml_internals")
    ml          = MlMetrics.from_pipeline(ctx.pipeline)
    template_kw = ml.model_dump() | {"total_variance": ml.variance.total}
    clusters    = ctx.pipeline.clusters

    centroid_labels = [
        clusters[cid].soc_title for cid in clusters.cluster_ids
    ]

    return ctx.layout.stack(
        ctx.layout.overview(tab, "overview"),
        ctx.layout.stats(zip(tab.stat_labels, ml.stat_values)),

        ctx.layout.header(tab, "pipeline"),
        ctx.layout.process_flow(s.render(**template_kw) for s in tab.process_steps),

        ctx.layout.callout(tab.info),

        ctx.layout.stack(
            ctx.layout.stack(
                ctx.layout.header(tab, "variance", **template_kw),
                mo.ui.plotly(ctx.charts.bar(
                    height = 320,
                    line   = ml.variance.trace,
                    title  = tab.chart_labels["variance_title"],
                    x      = ml.variance.labels,
                    y      = ml.variance.components
                ))
            ),
            ctx.layout.stack(
                ctx.layout.header(tab, "sector_dist"),
                mo.ui.plotly(ctx.charts.bar(
                    color  = [ctx.theme.sectors[s] for s in ml.sector_sizes],
                    height = 320,
                    title  = tab.chart_labels["postings_title"],
                    x      = [*ml.sector_sizes],
                    y      = [*ml.sector_sizes.values()]
                ))
            ),
            direction = "h",
            widths    = [1, 1]
        ),

        ctx.layout.header(tab, "data_funnel"),
        mo.ui.plotly(ctx.charts.funnel(
            height = 280,
            labels = [
                f"Raw Postings ({ml.corpus_size:,})",
                f"Embedding Dims ({ml.component_count * 76})",
                f"SVD Components ({ml.component_count})",
                f"Clusters ({ml.cluster_count})",
                f"Pathway Edges ({ml.edge_count})"
            ],
            values = [
                ml.corpus_size,
                ml.component_count * 76,
                ml.component_count,
                ml.cluster_count,
                ml.edge_count
            ]
        )),

        ctx.layout.stack(
            ctx.layout.stack(
                ctx.layout.header(tab, "cluster_sizes"),
                mo.ui.plotly(ctx.charts.histogram(
                    height  = 280,
                    nbins   = 10,
                    x       = [clusters[cid].size for cid in clusters.cluster_ids],
                    x_title = "Postings per Cluster",
                    y_title = "Count"
                ))
            ),
            ctx.layout.stack(
                ctx.layout.header(tab, "sector_flow"),
                mo.ui.plotly(ctx.charts.parcats(
                    dimensions = [
                        {
                            "label"  : "Sector",
                            "values" : [
                                clusters[cid].sector
                                for cid in clusters.cluster_ids
                            ]
                        },
                        {
                            "label"  : "Job Zone",
                            "values" : [
                                str(clusters[cid].job_zone)
                                for cid in clusters.cluster_ids
                            ]
                        }
                    ],
                    height = 280,
                    color  = ml.silhouette.values
                ))
            ),
            direction = "h",
            widths    = [1, 1]
        ),

        *ctx.layout.section_if(
            ml.brokerage.values, tab, "cluster_parcoords",
            mo.ui.plotly(ctx.charts.parcoords(
                color      = ml.silhouette.values,
                dimensions = [
                    {
                        "label"  : "Size",
                        "values" : [
                            clusters[cid].size
                            for cid in clusters.cluster_ids
                        ]
                    },
                    {
                        "label"  : "Job Zone",
                        "values" : [
                            clusters[cid].job_zone
                            for cid in clusters.cluster_ids
                        ]
                    },
                    {
                        "label"  : "Silhouette",
                        "values" : ml.silhouette.values
                    },
                    {
                        "label"  : "Brokerage",
                        "values" : ml.brokerage.values
                    }
                ],
                height = 350
            ))
        ),

        ctx.layout.header(tab, "treemap"),
        mo.ui.plotly(ctx.charts.treemap(
            branch_values = "total",
            data          = ml.treemap,
            height        = 450
        )),

        ctx.layout.stack(
            ctx.layout.stack(
                ctx.layout.header(tab, "gateways"),
                mo.ui.plotly(ctx.charts.bar(
                    color      = [ctx.theme.sectors[s] for s in ml.brokerage.sectors],
                    height     = max(300, len(ml.brokerage.labels) * 26),
                    horizontal = True,
                    title      = tab.chart_labels["brokerage_title"],
                    x          = ml.brokerage.values,
                    y          = ml.brokerage.labels
                ))
            ),
            ctx.layout.stack(
                ctx.layout.header(tab, "cluster_separation"),
                mo.ui.plotly(ctx.charts.bar(
                    color      = [ctx.theme.sectors[s] for s in ml.silhouette.sectors],
                    height     = max(300, len(ml.silhouette.labels) * 26),
                    horizontal = True,
                    title      = tab.chart_labels["silhouette_title"],
                    x          = ml.silhouette.values,
                    y          = ml.silhouette.labels
                ))
            ),
            direction = "h",
            widths    = [1, 1]
        ),

        *ctx.layout.section_if(
            ml.brokerage.values and ml.silhouette.values,
            tab, "quality_scatter",
            mo.ui.plotly(ctx.charts.bubble_scatter(
                height  = 350,
                points  = [
                    GapScatterPoint(
                        frequency = max(1, round(b * 100)),
                        magnitude = round(s * 100, 1),
                        text      = label,
                    )
                    for b, s, label in zip(
                        ml.brokerage.values,
                        ml.silhouette.values,
                        ml.brokerage.labels
                    )
                ],
                x_title = "Brokerage Centrality",
                y_title = "Silhouette Coefficient"
            ))
        ),

        ctx.layout.stack(
            ctx.layout.stack(
                ctx.layout.header(tab, "distances"),
                mo.ui.plotly(ctx.charts.violin(
                    groups  = ml.pairwise_distances,
                    height  = 350,
                    y_title = tab.chart_labels["pairwise_title"]
                ))
            ),
            ctx.layout.stack(
                ctx.layout.header(tab, "pathway_strength"),
                mo.ui.plotly(ctx.charts.histogram(
                    height  = 350,
                    nbins   = 25,
                    x       = ml.edge_weights,
                    x_title = tab.chart_labels["edge_weight_title"],
                    y_title = tab.chart_labels["count_title"]
                ))
            ),
            direction = "h",
            widths    = [1, 1]
        ),

        ctx.layout.header(tab, "dendrogram"),
        mo.ui.plotly(ctx.charts.dendrogram(
            annotation_text = tab.chart_labels["you"],
            title           = tab.chart_labels["dendrogram"],
            x_title         = tab.chart_labels["career_family"],
            y_title         = tab.chart_labels["ward_distance"]
        )),

        ctx.layout.header(tab, "landscape"),
        mo.ui.plotly(ctx.charts.landscape(
            coordinates     = ctx.result.coordinates,
            legend_families = tab.chart_labels["career_families"],
            legend_resume   = tab.chart_labels["your_resume"],
            title           = tab.chart_labels["landscape"],
            x_title         = tab.chart_labels["svd_component_1"],
            y_title         = tab.chart_labels["svd_component_2"]
        )),

        *ctx.layout.section_if(ml.cluster_heatmap, tab, "cluster_similarity",
            mo.ui.plotly(ctx.charts.heatmap(
                columns = centroid_labels,
                height  = max(400, len(ml.cluster_heatmap) * 28),
                labels  = list(ml.cluster_heatmap),
                values  = list(ml.cluster_heatmap.values())
            ))),

        *ctx.layout.section_if(ml.soc_heatmap, tab, "soc_heatmap",
            mo.ui.plotly(ctx.charts.heatmap(
                columns = [o.title for o in ctx.occupations],
                height  = max(400, len(ml.soc_heatmap) * 28),
                labels  = list(ml.soc_heatmap),
                values  = list(ml.soc_heatmap.values())
            ))),

        ctx.layout.header(tab, "cluster_profiles"),
        mo.accordion({"Cluster Metadata": mo.tree(ml.cluster_profiles)})
    )
