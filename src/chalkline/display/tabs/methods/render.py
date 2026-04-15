"""
Methods tab renderer.

Renders the pipeline methodology panel with variance charts, cluster quality
metrics, SOC heatmaps, and career landscape visualizations. Adapted from the
ML Internals tab with the cluster landscape treemap removed.
"""

from marimo import accordion, Html, tree

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import MlMetrics


def methods_tab(ctx: TabContext) -> Html:
    """
    Render the Methods tab with pipeline statistics, variance charts,
    cluster quality metrics, and SOC heatmaps.
    """
    tab         = ctx.content.tab("methods")
    ml          = MlMetrics.from_pipeline(ctx.pipeline)
    template_kw = ml.template_kwargs
    zone_counts = ml.job_zone.labeled_counts(ctx.content.labels.job_zones)

    return ctx.layout.stack(
        ctx.layout.overview("overview", tab),
        ctx.layout.stats(zip(tab.stat_labels, ml.stat_values)),

        ctx.layout.header("pipeline", tab),
        ctx.layout.process_flow(s.render(**template_kw) for s in tab.process_steps),

        ctx.layout.callout(tab.info),

        ctx.layout.two_col(
            ctx.layout.panel(ctx.charts.bar(
                data   = ml.variance.components_dict,
                height = 320,
                line   = ml.variance.cumulative_dict,
                title  = tab.chart_labels["variance_title"]
            ), "variance", tab, **template_kw),
            ctx.layout.panel(ctx.charts.bar(
                color  = ctx.theme.sector_colors(ml.sector_sizes),
                data   = ml.sector_sizes,
                height = 320,
                title  = tab.chart_labels["postings_title"]
            ), "sector_dist", tab)
        ),

        ctx.layout.two_col(
            ctx.layout.panel(
                ctx.charts.funnel(height=320, stages=ml.funnel_stages),
                "data_funnel", tab),
            ctx.layout.panel(ctx.charts.histogram(
                height  = 320,
                nbins   = 10,
                x       = ctx.pipeline.clusters.sizes,
                x_title = "Postings per Cluster",
                y_title = "Count"
            ), "cluster_sizes", tab)
        ),

        ctx.layout.two_col(
            ctx.layout.panel(ctx.charts.bar(
                color      = "primary",
                data       = zone_counts,
                height     = 320,
                horizontal = True,
                title      = tab.chart_labels["clusters_title"]
            ), "job_zone_distribution", tab),
            ctx.layout.panel(ctx.charts.heatmap(
                ml.job_zone.matrix,
                columns = list(zone_counts),
                height  = 320
            ), "sector_job_zone", tab)
        ),

        ctx.layout.two_col(
            ctx.layout.panel(ctx.charts.ranking_bar(
                ranking = ml.brokerage,
                title   = tab.chart_labels["brokerage_title"]
            ), "gateways", tab),
            ctx.layout.panel(ctx.charts.ranking_bar(
                ranking = ml.silhouette,
                title   = tab.chart_labels["silhouette_title"]
            ), "cluster_separation", tab)
        ),

        *ctx.layout.section_if(
            ctx.charts.bubble_scatter(
                brokerage  = ml.brokerage,
                height     = 350,
                silhouette = ml.silhouette,
                x_title    = "Brokerage Centrality",
                y_title    = "Silhouette Coefficient"
            ),
            ml.brokerage.values and ml.silhouette.values,
            "quality_scatter", tab
        ),

        ctx.layout.two_col(
            ctx.layout.panel(ctx.charts.violin(
                groups  = ml.pairwise_distances,
                height  = 350,
                y_title = tab.chart_labels["pairwise_title"]
            ), "distances", tab),
            ctx.layout.panel(ctx.charts.histogram(
                height  = 350,
                nbins   = 25,
                x       = ml.edge_weights,
                x_title = tab.chart_labels["edge_weight_title"],
                y_title = tab.chart_labels["count_title"]
            ), "pathway_strength", tab)
        ),

        ctx.layout.panel(ctx.charts.landscape(
            coordinates     = ctx.result.coordinates,
            legend_families = tab.chart_labels["career_families"],
            legend_resume   = tab.chart_labels["your_resume"],
            x_title         = tab.chart_labels["svd_component_1"],
            y_title         = tab.chart_labels["svd_component_2"]
        ), "landscape", tab),

        *ctx.layout.section_if(
            ctx.charts.heatmap(ml.cluster_heatmap),
            ml.cluster_heatmap, "cluster_similarity", tab),

        *ctx.layout.section_if(ctx.charts.heatmap(
                columns = [o.title for o in ctx.occupations],
                data    = ml.soc_heatmap
            ),
            ml.soc_heatmap, "soc_heatmap", tab),

        ctx.layout.header("cluster_profiles", tab),
        accordion({"Cluster Metadata": tree(ml.cluster_profiles)})
    )
