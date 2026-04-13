"""
Data tab for the Chalkline career report.

Walks through the matched career family's job postings from concrete to
aggregate: sample posting cards first, then hiring companies and
locations, then a TF-IDF distinctive-vocabulary treemap and a t-SNE
projection of the family's posting embeddings, then concrete credential
pathways grouped by kind alongside relevant Maine job boards, then
temporal views via posting timeline and freshness histogram.
"""

from marimo import Html

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import DistinctiveVocabulary, JobPostingMetrics
from chalkline.display.schemas import PostingProjection, RelevantCredentials
from chalkline.display.schemas import RelevantJobBoards


def data_tab(ctx: TabContext) -> Html:
    """
    Render the Data tab from concrete postings up through aggregate
    employer, vocabulary, projection, training, and temporal views.
    """
    tab        = ctx.content.tab("data")
    clusters   = ctx.pipeline.clusters
    postings   = JobPostingMetrics.from_postings(ctx.profile.postings, ctx.reference)
    projection = PostingProjection.from_cluster(ctx.profile)
    boards     = RelevantJobBoards.from_cluster(
        cluster   = ctx.profile,
        clusters  = clusters,
        encoder   = ctx.pipeline.matcher.encoder,
        limit     = 5,
        reference = ctx.reference
    ).boards
    section_kw = {"soc_title": ctx.profile.soc_title}

    return ctx.layout.stack(
        ctx.layout.overview("overview", tab, **section_kw),
        ctx.layout.stats(zip(tab.stat_labels, postings.stat_values)),

        ctx.layout.header("recent", tab, **section_kw),
        ctx.layout.grid(ctx.layout.posting_card(p) for p in postings.recent),

        ctx.layout.two_col(*(
            ctx.layout.panel(ctx.charts.bar(
                color      = color,
                data       = data,
                height     = max(300, len(data) * 28),
                horizontal = True,
                title      = tab.chart_labels["postings_title"]
            ), key, tab, **section_kw)
            for key, color, data in [
                ("whos_hiring", "accent",  postings.companies),
                ("locations",   "success", postings.locations)
            ]
        )),

        ctx.layout.panel(ctx.charts.faceted_treemap(
            descriptions = tab.tier_descriptions,
            facets       = DistinctiveVocabulary.from_cluster(
                cluster           = ctx.profile,
                clusters          = clusters,
                tier_descriptions = tab.tier_descriptions
            ).tiers,
            height       = 480
        ), "distinctive_words", tab, **section_kw),

        *ctx.layout.section_if(
            ctx.charts.category_scatter(
                data    = projection.series,
                height  = 480,
                x_title = tab.chart_labels["tsne_x"],
                y_title = tab.chart_labels["tsne_y"]
            ),
            projection, "posting_projection", tab,
            **section_kw
        ),

        ctx.layout.header("credential_pathways", tab),
        ctx.layout.credential_columns(
            RelevantCredentials.from_cluster(
                cluster  = ctx.profile,
                clusters = clusters,
                graph    = ctx.pipeline.graph
            ).by_kind,
            ctx.theme
        ),

        ctx.layout.header("relevant_boards", tab),
        ctx.layout.grid(
            ctx.layout.board_chip(**b) for b in boards
        ) if boards else ctx.layout.callout(tab.fallbacks["no_boards"]),

        *ctx.layout.stack_if(postings.dated, ctx.layout.two_col(
            ctx.layout.panel(ctx.charts.timeline(
                dates  = postings.dates,
                height = 280,
                hover  = postings.hover
            ), "timeline", tab),
            ctx.layout.panel(ctx.charts.histogram(
                height  = 280,
                nbins   = 15,
                x       = postings.freshness,
                x_title = tab.chart_labels["days_since_posted"],
                y_title = tab.chart_labels["postings"]
            ), "freshness_histogram", tab)
        )),

        ctx.layout.callout(tab.info)
    )
