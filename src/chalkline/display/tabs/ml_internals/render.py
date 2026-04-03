"""
ML Internals tab renderer.
"""

import marimo as mo

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import MlMetrics


def ml_internals_tab(ctx: TabContext) -> mo.Html:
    """
    Render the ML Internals diagnostic tab with pipeline statistics,
    variance charts, and cluster quality metrics.
    """
    tab         = ctx.content.tab("ml_internals")
    ml          = MlMetrics.from_pipeline(ctx.pipeline)
    template_kw = ml.model_dump() | {"total_variance": ml.variance.total}

    return ctx.layout.stack(
        ctx.layout.header(tab, "overview"),
        ctx.layout.stats(zip(tab.stat_labels, ml.stat_values)),

        ctx.layout.header(tab, "pipeline"),
        ctx.layout.process_flow(s.render(**template_kw) for s in tab.process_steps),

        ctx.layout.header(tab, "variance", **template_kw),
        mo.ui.plotly(ctx.charts.bar(
            height = 320,
            line   = ml.variance.trace,
            title  = tab.chart_labels["variance_title"],
            x      = ml.variance.labels,
            y      = ml.variance.components
        )),

        ctx.layout.header(tab, "treemap"),
        mo.ui.plotly(ctx.charts.treemap(
            branch_values = "total",
            height        = 450,
            labels        = ml.treemap.labels,
            parents       = ml.treemap.parents,
            sectors       = ml.treemap.sectors,
            values        = ml.treemap.values
        )),

        ctx.layout.header(tab, "gateways"),
        mo.ui.plotly(ctx.charts.bar(
            color      = ctx.charts.sector_colors(ml.brokerage.sectors),
            height     = max(300, len(ml.brokerage.labels) * 26),
            horizontal = True,
            title      = tab.chart_labels["brokerage_title"],
            x          = ml.brokerage.values,
            y          = ml.brokerage.labels
        )),

        ctx.layout.header(tab, "cluster_separation"),
        mo.ui.plotly(ctx.charts.bar(
            color      = ctx.charts.sector_colors(ml.silhouette.sectors),
            height     = max(300, len(ml.silhouette.labels) * 26),
            horizontal = True,
            title      = tab.chart_labels["silhouette_title"],
            x          = ml.silhouette.values,
            y          = ml.silhouette.labels
        )),

        ctx.layout.header(tab, "distances"),
        mo.ui.plotly(ctx.charts.violin(
            groups  = ml.pairwise_distances,
            height  = 350,
            y_title = tab.chart_labels["pairwise_title"]
        )),

        ctx.layout.header(tab, "pathway_strength"),
        mo.ui.plotly(ctx.charts.histogram(
            height  = 300,
            nbins   = 25,
            x       = ml.edge_weights,
            x_title = tab.chart_labels["edge_weight_title"],
            y_title = tab.chart_labels["count_title"]
        )),

        ctx.layout.header(tab, "sector_dist"),
        mo.ui.plotly(ctx.charts.bar(
            color  = ctx.charts.sector_colors(ml.sector_sizes),
            height = 300,
            title  = tab.chart_labels["postings_title"],
            x      = ml.sector_sizes,
            y      = ml.sector_sizes.values()
        )),

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

        ctx.layout.header(tab, "cluster_profiles"),
        mo.tree(ml.cluster_profiles),
        ctx.layout.callout(tab.info)
    )
