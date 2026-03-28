import marimo

__generated_with = "0.12.0"
app = marimo.App(width="full", css_file="chalkline.css")


# ── Setup ───────────────────────────────────────────────────────────

@app.cell
def _():
    import marimo as mo

    from pathlib import Path

    from chalkline.display.theme import Theme

    theme = Theme(dark_fn=lambda: mo.app_meta().theme == "dark")

    return Path, mo, theme


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
            output_dir   = Path(".cache/pipeline"),
            postings_dir = Path("data/postings")
        ))
    return (pipeline,)


# ── Reference data ──────────────────────────────────────────────────

@app.cell
def _(Path):
    from chalkline.pathways.loaders import LaborLoader, LexiconLoader
    from chalkline.pathways.loaders import StakeholderReference

    lexicon_dir = Path("data/lexicons")
    labor       = LaborLoader(lexicon_dir / "labor.json")
    occupations = LexiconLoader(lexicon_dir).occupations
    reference   = StakeholderReference(Path("data/stakeholder/reference"))

    return labor, occupations, reference


# ── Upload widget ───────────────────────────────────────────────────

@app.cell
def _(mo):
    upload = mo.ui.file(
        filetypes = [".pdf"],
        kind      = "area",
        label     = "Drop a resume PDF here"
    )
    return (upload,)


# ── Splash page ─────────────────────────────────────────────────────

@app.cell
def _(Path, labor, mo, pipeline, upload):
    from chalkline.display.schemas import build_splash
    from chalkline.display.tabs    import splash

    mo.stop(bool(upload.value), mo.md(""))
    mo.vstack([
        splash(Path(__file__).parent / "assets", build_splash(pipeline, labor)),
        upload
    ])
    return


# ── Upload gate + matching ──────────────────────────────────────────

@app.cell
def _(labor, mo, occupations, pipeline, reference, upload):
    from chalkline.display.schemas import DisplayData

    mo.stop(not upload.value, mo.md(""))

    with mo.status.spinner("Analyzing your resume..."):
        result = pipeline.match(
            (resume := upload.value[0]).contents,
            label = resume.name
        )

    data = DisplayData(
        labor       = labor,
        occupations = occupations,
        pipeline    = pipeline,
        reference   = reference,
        result      = result
    )
    return (data,)


# ── Match bar ──────────────────────────────────────────────────────

@app.cell
def _(data, mo, theme, upload):
    from chalkline.display.layout import match_bar

    mo.stop(not upload.value, mo.md(""))
    match_bar(
        jz_label  = theme.jz_label(data.profile.job_zone),
        postings  = data.profile.size,
        sector    = data.profile.sector,
        soc_title = data.profile.soc_title
    )
    return


# ── Charts ──────────────────────────────────────────────────────────

@app.cell
def _(data, pipeline, theme):
    from chalkline.display.charts import Charts

    charts = Charts(
        matched_id = data.result.cluster_id,
        pathway    = pipeline.graph,
        theme      = theme
    )
    return (charts,)


# ── Target dropdown ─────────────────────────────────────────────────

@app.cell
def _(data):
    from chalkline.display.layout import target_dropdown

    dropdown = target_dropdown(data.cluster_options, data.profile.display_label)
    return (dropdown,)


# ── Target resolution ───────────────────────────────────────────────

@app.cell
def _(data, dropdown):
    target_data = data.for_target(dropdown.value or data.result.cluster_id)
    return (target_data,)


# ── Tab context ─────────────────────────────────────────────────────

@app.cell
def _(charts, data, theme):
    from chalkline.display.tabs import TabContext

    ctx = TabContext(
        charts = charts,
        data   = data,
        theme  = theme
    )
    return (ctx,)


# ── Tab layout ──────────────────────────────────────────────────────

@app.cell
def _(ctx, dropdown, mo, target_data):
    from chalkline.display import tabs

    mo.ui.tabs(
        {
            "Your Match"      : lambda: tabs.your_match(ctx),
            "Resume Feedback" : lambda: tabs.resume_feedback(ctx),
            "Career Paths"    : lambda: tabs.career_paths(ctx, target_data, dropdown),
            "Job Postings"    : lambda: tabs.job_postings(ctx),
            "Next Steps"      : lambda: tabs.next_steps(ctx, target_data),
            "ML Internals"    : lambda: tabs.ml_internals(ctx)
        },
        lazy = True
    )
    return


if __name__ == "__main__":
    app.run()
