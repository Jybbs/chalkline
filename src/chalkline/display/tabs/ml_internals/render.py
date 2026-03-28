"""
ML Internals tab renderer.
"""

import marimo as mo

from chalkline.display.layout       import callout, header, process_flow, stat_strip
from chalkline.display.tabs.context import TabContext, load_content

content = load_content(__file__)


def ml_internals_tab(ctx: TabContext) -> mo.Html:
    """
    Render the ML Internals diagnostic tab with pipeline
    statistics, variance charts, and cluster quality metrics.
    """

    return mo.vstack([
        header(*content.section("overview")),
        stat_strip({
            "Corpus Size"     : f"{ctx.data.ml.corpus_size:,}",
            "Embedding Model" : ctx.data.ml.embedding_model,
            "Clusters (k)"    : str(ctx.data.ml.cluster_count),
            "SVD Components"  : str(ctx.data.ml.component_count),
            "Pathway Edges"   : str(ctx.data.ml.edge_count)
        }),

        header(*content.section("pipeline")),
        process_flow([
            ("1", "Collect", f"{ctx.data.ml.corpus_size:,} postings from AGC Maine"),
            ("2", "Encode",  ctx.data.ml.embedding_model),
            ("3", "Reduce",  (
                f"SVD to {ctx.data.ml.component_count}D "
                f"({ctx.data.ml.total_variance:.0f}% variance)"
            )),
            ("4", "Cluster", f"Ward HAC, k={ctx.data.ml.cluster_count}"),
            ("5", "Graph",   f"{ctx.data.ml.edge_count} pathway edges"),
            ("6", "Match",   "Resume projection + gap analysis")
        ]),

        header(*content.section(
            "variance",
            component_count = ctx.data.ml.component_count,
            total_variance  = ctx.data.ml.total_variance
        )),
        mo.ui.plotly(ctx.charts.bar_line(
            bar_x   = ctx.data.ml.pc_labels,
            bar_y   = ctx.data.ml.explained_variance,
            height  = 320,
            line_x  = ctx.data.ml.pc_labels,
            line_y  = ctx.data.ml.cumulative_variance,
            y_title = "Variance Explained (%)"
        )),

        header(*content.section("treemap")),
        mo.ui.plotly(ctx.charts.sector_treemap(
            height  = 450,
            labels  = ctx.data.ml.treemap_labels,
            parents = ctx.data.ml.treemap_parents,
            sectors = ctx.data.ml.treemap_sectors,
            values  = ctx.data.ml.treemap_values
        )),

        header(*content.section("gateways")),
        mo.ui.plotly(ctx.charts.sector_hbar(
            height  = max(300, len(ctx.data.ml.betweenness_labels) * 26),
            labels  = ctx.data.ml.betweenness_labels,
            sectors = ctx.data.ml.betweenness_sectors,
            title   = "Betweenness Centrality",
            values  = ctx.data.ml.betweenness_values
        )),

        header(*content.section("cluster_separation")),
        mo.ui.plotly(ctx.charts.sector_hbar(
            height  = max(300, len(ctx.data.ml.silhouette_labels) * 26),
            labels  = ctx.data.ml.silhouette_labels,
            sectors = ctx.data.ml.silhouette_sectors,
            title   = "Silhouette Coefficient",
            values  = ctx.data.ml.silhouette_values
        )),

        header(*content.section("distances")),
        mo.ui.plotly(ctx.charts.violin(
            groups  = ctx.data.ml.pairwise_distances,
            height  = 350,
            y_title = "Pairwise Centroid Distance"
        )),

        header(*content.section("pathway_strength")),
        mo.ui.plotly(ctx.charts.histogram(
            height  = 300,
            nbins   = 25,
            x       = ctx.data.ml.edge_weights,
            x_title = "Edge Weight (cosine similarity)",
            y_title = "Count"
        )),

        header(*content.section("sector_dist")),
        mo.ui.plotly(ctx.charts.sector_vbar(
            height = 300,
            labels = ctx.data.ml.sector_labels,
            title  = "Postings",
            values = ctx.data.ml.sector_values
        )),

        header(*content.section("dendrogram")),
        mo.ui.plotly(ctx.charts.dendrogram()),

        header(*content.section("landscape")),
        mo.ui.plotly(ctx.charts.landscape(ctx.data.result.coordinates)),

        header(*content.section("cluster_profiles")),
        mo.tree(ctx.data.ml.cluster_profiles),
        callout(content.info)
    ])
