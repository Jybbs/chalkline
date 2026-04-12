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
    )
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


# ── Match bar ──────────────────────────────────────────────────────

@app.cell
def _(layout, mo, profile, upload):
    mo.stop(not upload.value, mo.md(""))
    layout.match_bar(profile)
    return


# ── Map widget ─────────────────────────────────────────────────────

@app.cell
def _(content, labor, mo, pipeline, result, theme):
    from chalkline.display.tabs.map.widget import PathwayMap

    map_widget = mo.ui.anywidget(PathwayMap.from_graph(
        clusters   = pipeline.clusters,
        graph      = pipeline.graph,
        labels     = content.labels,
        labor      = labor,
        matched_id = result.cluster_id,
        theme      = theme
    ))
    return (map_widget,)


# ── Sidebar card ───────────────────────────────────────────────────

@app.cell
def _(labor, layout, profile, result, theme):
    sidebar = layout.you_are_here(
        confidence = result.confidence,
        profile    = profile,
        theme      = theme,
        wage       = labor.wage(profile.soc_title)
    )
    return (sidebar,)


# ── Route computation ──────────────────────────────────────────────

@app.cell
def _(labor, map_widget, pipeline, profile, result):
    from chalkline.display.schemas import RouteDetail

    route = RouteDetail.from_selection(
        labor       = labor,
        pipeline    = pipeline,
        profile     = profile,
        result      = result,
        selected_id = map_widget.value["selected_id"]
    )
    return (route,)


# ── Tab context ─────────────────────────────────────────────────────

@app.cell
def _(charts, content, labor, layout, occupations, pipeline, profile, reference, result, theme):
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
        theme       = theme
    )
    return (ctx,)


# ── Three-tab layout ───────────────────────────────────────────────

@app.cell
def _(ctx, map_widget, mo, route, sidebar):
    from chalkline.display.tabs.data.render    import data_tab
    from chalkline.display.tabs.map.render     import map_tab
    from chalkline.display.tabs.methods.render import methods_tab

    tn = ctx.content.labels.tab_names
    mo.ui.tabs(
        {
            tn["map"]     : lambda: map_tab(
                ctx     = ctx,
                route   = route,
                sidebar = sidebar,
                widget  = map_widget
            ),
            tn["data"]    : lambda: data_tab(ctx),
            tn["methods"] : lambda: methods_tab(ctx)
        },
        lazy = True
    )
    return


if __name__ == "__main__":

    app.run()
