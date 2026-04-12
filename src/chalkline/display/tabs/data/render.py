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
    tiers      = DistinctiveVocabulary.from_cluster(
        cluster     = ctx.profile,
        clusters    = clusters,
        tier_labels = list(tab.tier_descriptions)
    ).tiers
    series     = PostingProjection.from_cluster(ctx.profile).series
    by_kind    = RelevantCredentials.from_cluster(
        cluster  = ctx.profile,
        clusters = clusters,
        graph    = ctx.pipeline.graph
    ).by_kind
    boards     = RelevantJobBoards.from_cluster(
        cluster   = ctx.profile,
        clusters  = clusters,
        encoder   = ctx.pipeline.matcher.encoder,
        limit     = 5,
        reference = ctx.reference
    ).boards
    section_kw = {"soc_title": ctx.profile.soc_title}

    return ctx.layout.stack(
        ctx.layout.overview(tab, "overview", **section_kw),
        ctx.layout.stats(zip(tab.stat_labels, postings.stat_values)),

        ctx.layout.header(tab, "recent", **section_kw),
        ctx.layout.grid(ctx.layout.posting_card(p) for p in postings.recent),

        ctx.layout.two_col(*(
            ctx.layout.panel(tab, key, ctx.charts.bar(
                color      = color,
                data       = data,
                height     = max(300, len(data) * 28),
                horizontal = True,
                title      = tab.chart_labels["postings_title"]
            ), **section_kw)
            for key, color, data in [
                ("whos_hiring", "accent",  postings.companies),
                ("locations",   "success", postings.locations)
            ]
        )),

        ctx.layout.panel(tab, "distinctive_words",
            ctx.charts.faceted_treemap(
                descriptions = tab.tier_descriptions,
                facets       = tiers,
                height       = 480
            ), **section_kw),

        *ctx.layout.section_if(
            series, tab, "posting_projection",
            ctx.charts.category_scatter(
                data    = series,
                height  = 480,
                x_title = tab.chart_labels["tsne_x"],
                y_title = tab.chart_labels["tsne_y"]
            ),
            **section_kw
        ),

        ctx.layout.header(tab, "credential_pathways"),
        ctx.layout.stack(
            *(
                ctx.layout.grid(
                    ctx.layout.credential_card(c, ctx.theme) for c in cards
                )
                for cards in by_kind.values()
            ),
            direction = "h",
            widths    = [1, 1, 1]
        ),

        ctx.layout.header(tab, "relevant_boards"),
        ctx.layout.grid(
            ctx.layout.board_chip(**b) for b in boards
        ) if boards else ctx.layout.callout(tab.fallbacks["no_boards"]),

        *ctx.layout.stack_if(postings.dated, ctx.layout.two_col(
            ctx.layout.panel(tab, "timeline", ctx.charts.timeline(
                dates  = [d.date  for d in postings.dated],
                height = 280,
                hover  = [d.label for d in postings.dated]
            )),
            ctx.layout.panel(tab, "freshness_histogram", ctx.charts.histogram(
                height  = 280,
                nbins   = 15,
                x       = postings.freshness,
                x_title = tab.chart_labels["days_since_posted"],
                y_title = tab.chart_labels["postings"]
            ))
        )),

        ctx.layout.callout(tab.info)
    )
