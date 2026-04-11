"""
Methods tab renderer.

Renders the pipeline methodology panel with variance charts, cluster
quality metrics, SOC heatmaps, and career landscape visualizations.
Adapted from the ML Internals tab with the cluster landscape treemap
removed.
"""

from marimo import accordion, Html, tree

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import MlMetrics


def methods_tab(ctx: TabContext) -> Html:
    """
    Render the Methods tab with pipeline statistics, variance
    charts, cluster quality metrics, and SOC heatmaps.
    """
    tab         = ctx.content.tab("methods")
    ml          = MlMetrics.from_pipeline(ctx.pipeline)
    template_kw = ml.template_kwargs
    zone_counts = ml.job_zone.labeled_counts(ctx.content.labels.job_zones)

    return ctx.layout.stack(
        ctx.layout.overview(tab, "overview"),
        ctx.layout.stats(zip(tab.stat_labels, ml.stat_values)),

        ctx.layout.header(tab, "pipeline"),
        ctx.layout.process_flow(s.render(**template_kw) for s in tab.process_steps),

        ctx.layout.callout(tab.info),

        ctx.layout.two_col(
            ctx.layout.panel(tab, "variance", ctx.charts.bar(
                data   = ml.variance.components_dict,
                height = 320,
                line   = ml.variance.cumulative_dict,
                title  = tab.chart_labels["variance_title"]
            ), **template_kw),
            ctx.layout.panel(tab, "sector_dist", ctx.charts.bar(
                color  = ctx.theme.sector_colors(ml.sector_sizes),
                data   = ml.sector_sizes,
                height = 320,
                title  = tab.chart_labels["postings_title"]
            ))
        ),

        ctx.layout.panel(tab, "data_funnel",
            ctx.charts.funnel(height=280, stages=ml.funnel_stages)),

        ctx.layout.panel(tab, "cluster_sizes", ctx.charts.histogram(
            height  = 320,
            nbins   = 10,
            x       = ctx.pipeline.clusters.sizes,
            x_title = "Postings per Cluster",
            y_title = "Count"
        )),

        ctx.layout.two_col(
            ctx.layout.panel(tab, "job_zone_distribution", ctx.charts.bar(
                color      = "primary",
                data       = zone_counts,
                height     = 320,
                horizontal = True,
                title      = tab.chart_labels["clusters_title"]
            )),
            ctx.layout.panel(tab, "sector_job_zone", ctx.charts.heatmap(
                ml.job_zone.matrix,
                columns = list(zone_counts),
                height  = 320
            ))
        ),

        ctx.layout.two_col(
            ctx.layout.panel(tab, "gateways", ctx.charts.ranking_bar(
                ranking = ml.brokerage, 
                title   = tab.chart_labels["brokerage_title"]
            )
            ),
            ctx.layout.panel(tab, "cluster_separation", ctx.charts.ranking_bar(
                ranking = ml.silhouette, 
                title   = tab.chart_labels["silhouette_title"]
            )
            )
        ),

        *ctx.layout.section_if(
            ml.brokerage.values and ml.silhouette.values,
            tab, "quality_scatter",
            ctx.charts.bubble_scatter(
                brokerage  = ml.brokerage,
                height     = 350,
                silhouette = ml.silhouette,
                x_title    = "Brokerage Centrality",
                y_title    = "Silhouette Coefficient"
            )
        ),

        ctx.layout.two_col(
            ctx.layout.panel(tab, "distances", ctx.charts.violin(
                groups  = ml.pairwise_distances,
                height  = 350,
                y_title = tab.chart_labels["pairwise_title"]
            )),
            ctx.layout.panel(tab, "pathway_strength", ctx.charts.histogram(
                height  = 350,
                nbins   = 25,
                x       = ml.edge_weights,
                x_title = tab.chart_labels["edge_weight_title"],
                y_title = tab.chart_labels["count_title"]
            ))
        ),

        ctx.layout.panel(tab, "dendrogram", ctx.charts.dendrogram(
            annotation_text = tab.chart_labels["you"],
            title           = tab.chart_labels["dendrogram"],
            x_title         = tab.chart_labels["career_family"],
            y_title         = tab.chart_labels["ward_distance"]
        )),

        ctx.layout.panel(tab, "landscape", ctx.charts.landscape(
            coordinates     = ctx.result.coordinates,
            legend_families = tab.chart_labels["career_families"],
            legend_resume   = tab.chart_labels["your_resume"],
            title           = tab.chart_labels["landscape"],
            x_title         = tab.chart_labels["svd_component_1"],
            y_title         = tab.chart_labels["svd_component_2"]
        )),

        *ctx.layout.section_if(ml.cluster_heatmap, tab, "cluster_similarity",
            ctx.charts.heatmap(ml.cluster_heatmap)),

        *ctx.layout.section_if(ml.soc_heatmap, tab, "soc_heatmap",
            ctx.charts.heatmap(
                columns = [o.title for o in ctx.occupations],
                data    = ml.soc_heatmap
            )),

        ctx.layout.header(tab, "cluster_profiles"),
        accordion({"Cluster Metadata": tree(ml.cluster_profiles)})
    )
