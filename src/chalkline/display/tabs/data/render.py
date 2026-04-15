"""
Data tab for the Chalkline career report.

Walks through the matched career family's job postings from aggregate to
concrete: hiring companies and locations, then a TF-IDF
distinctive-vocabulary treemap and t-SNE projection, credential pathways,
temporal views, and a collapsible drawer of individual posting ribbons
ranked by resume similarity.
"""

from htpy       import details, summary
from marimo     import Html
from markupsafe import Markup

from chalkline.display.loaders import TabContext
from chalkline.display.schemas import DistinctiveVocabulary, JobPostingMetrics
from chalkline.display.schemas import PostingProjection, RelevantCredentials


def data_tab(ctx: TabContext) -> Html:
    """
    Render the Data tab from concrete postings up through aggregate
    employer, vocabulary, projection, training, and temporal views.
    """
    tab        = ctx.content.tab("data")
    clusters   = ctx.pipeline.clusters
    postings   = JobPostingMetrics.from_postings(ctx.profile.postings, ctx.reference)
    projection = PostingProjection.from_cluster(ctx.profile)
    ranked     = ctx.pipeline.matcher.score_postings(ctx.profile)
    section_kw = {"soc_title": ctx.profile.soc_title}

    return ctx.layout.stack(
        ctx.layout.overview("overview", tab, **section_kw),
        ctx.layout.stats(zip(tab.stat_labels, postings.stat_values)),

        Html(str(details(".cl-posting-drawer")[
            summary[tab.sections["recent"].title.format(**section_kw)],
            *(
                Markup(ctx.layout.posting_ribbon(
                    color      = ctx.theme.score_color(round(sim * 100)),
                    posting    = p,
                    similarity = sim
                ).text)
                for p, sim in ranked
            )
        ])),

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
