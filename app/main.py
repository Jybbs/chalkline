import marimo

__generated_with = "0.12.0"
app = marimo.App(width="full", css_file="chalkline.css")


# ── Setup ───────────────────────────────────────────────────────────

@app.cell
def _():
    import marimo as mo

    from pathlib import Path

    from chalkline.display.loaders import ContentLoader
    from chalkline.display.theme   import Theme

    content = ContentLoader()
    theme   = Theme()

    return Path, content, mo, theme


# ── Pipeline loading ────────────────────────────────────────────────

@app.cell
def _(Path, mo):
    from chalkline.pipeline.orchestrator import Chalkline
    from chalkline.pipeline.schemas      import PipelineConfig

    with mo.persistent_cache(
        name        = "chalkline",
        pin_modules = True,
        save_path   = ".cache/marimo"
    ):
        pipeline = Chalkline.fit(PipelineConfig(
            lexicon_dir  = Path("data/lexicons"),
            postings_dir = Path("data/postings")
        ))
    return (pipeline,)


# ── Reference data ──────────────────────────────────────────────────

@app.cell
def _(Path, pipeline):
    from chalkline.pathways.loaders import LaborLoader, LexiconLoader
    from chalkline.pathways.loaders import StakeholderReference

    lexicon_dir = pipeline.config.lexicon_dir
    labor       = LaborLoader(lexicon_dir / "labor.json")
    occupations = LexiconLoader(lexicon_dir).occupations
    reference   = StakeholderReference(Path("data/stakeholder/reference"))

    return labor, occupations, reference


# ── Upload widget ───────────────────────────────────────────────────

@app.cell
def _(content, mo):
    upload = mo.ui.file(
        filetypes = [".pdf"],
        kind      = "area",
        label     = content.labels.upload_label
    )
    return (upload,)


# ── Splash page ─────────────────────────────────────────────────────

@app.cell
def _(Path, content, labor, layout, mo, pipeline, upload):
    from chalkline.display.tabs import splash

    mo.stop(bool(upload.value), mo.md(""))
    layout.stack(
        splash(content, labor, layout, Path(__file__).parent / "assets", pipeline),
        upload
    ).style({
        "background"    : "var(--background)",
        "inset"         : "0",
        "place-content" : "center",
        "position"      : "fixed",
        "z-index"       : "100",
    })
    return


# ── Upload gate + matching ──────────────────────────────────────────

@app.cell
def _(content, mo, pipeline, upload):
    mo.stop(not upload.value, mo.md(""))

    with mo.status.spinner(content.labels.spinner_text):
        result = pipeline.match(
            (resume := upload.value[0]).contents,
            label = resume.name
        )

    profile = pipeline.clusters[result.cluster_id]
    return profile, result


# ── Layout ─────────────────────────────────────────────────────────

@app.cell
def _(content, pipeline):
    from chalkline.display.loaders import Layout

    layout = Layout(content, pipeline.substitutions)
    return (layout,)


# ── Charts ──────────────────────────────────────────────────────────

@app.cell
def _(pipeline, result, theme):
    from chalkline.display.charts import Charts

    charts = Charts(
        matched_id = result.cluster_id,
        pathway    = pipeline.graph,
        theme      = theme
    )
    return (charts,)


# ── Routes ─────────────────────────────────────────────────────────

@app.cell
def _(layout, theme):
    from chalkline.display.routes import Routes

    routes = Routes(layout=layout, theme=theme)
    return (routes,)


# ── Forms ──────────────────────────────────────────────────────────

@app.cell
def _(layout, mo):
    from chalkline.display.forms import Forms

    forms = Forms(layout=layout, mo=mo)
    return (forms,)


# ── Wage filter ────────────────────────────────────────────────────

@app.cell
def _(forms, pipeline):
    from chalkline.display.tabs.map.widget import PathwayMap as pathway

    wage_filter = forms.wage_filter(pipeline.clusters)
    wage_slider = wage_filter.slider
    return pathway, wage_filter, wage_slider


# ── Map widget ─────────────────────────────────────────────────────

@app.cell
def _(mo, pathway, pipeline, result, theme):
    widget = mo.ui.anywidget(pathway.from_graph(
        clusters   = pipeline.clusters,
        graph      = pipeline.graph,
        matched_id = result.cluster_id,
        matcher    = pipeline.matcher,
        result     = result,
        theme      = theme,
    ))
    return (widget,)


# ── Map filter sync ────────────────────────────────────────────────

@app.cell
def _(pathway, pipeline, result, theme, wage_slider, widget):
    widget.graph_data = pathway.build_graph_data(
        clusters    = pipeline.clusters,
        graph       = pipeline.graph,
        matched_id  = result.cluster_id,
        matcher     = pipeline.matcher,
        result      = result,
        theme       = theme,
        wage_filter = (wage_slider.value or wage_slider.start, wage_slider.stop),
    )
    return


# ── Route computation ──────────────────────────────────────────────

@app.cell
def _(labor, widget, pipeline, profile, result):
    from chalkline.display.schemas import RouteDetail

    route = RouteDetail.from_selection(
        labor       = labor,
        pipeline    = pipeline,
        profile     = profile,
        result      = result,
        selected_id = widget.value["selected_id"]
    )
    return (route,)


# ── Tab context ─────────────────────────────────────────────────────

@app.cell
def _(
    charts, content, labor, layout, occupations, pipeline,
    profile, reference, result, routes, theme
):
    from chalkline.display.loaders import TabContext

    ctx = TabContext(
        charts      = charts,
        content     = content,
        labor       = labor,
        layout      = layout,
        occupations = occupations,
        pipeline    = pipeline,
        profile     = profile,
        reference   = reference,
        result      = result,
        routes      = routes,
        theme       = theme
    )
    return (ctx,)


# ── Three-tab layout ───────────────────────────────────────────────

@app.cell
def _(ctx, widget, mo, route, wage_filter):
    from chalkline.display.tabs.data.render    import data_tab
    from chalkline.display.tabs.map.render     import map_tab
    from chalkline.display.tabs.methods.render import methods_tab

    tn = ctx.content.labels.tab_names
    mo.ui.tabs(
        {
            tn["map"]     : lambda: map_tab(
                ctx         = ctx,
                route       = route,
                wage_filter = wage_filter.row,
                widget      = widget,
            ),
            tn["data"]    : lambda: data_tab(ctx),
            tn["methods"] : lambda: methods_tab(ctx)
        },
        lazy = True
    )
    return


if __name__ == "__main__":

    app.run()
