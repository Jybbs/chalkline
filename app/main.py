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
    theme   = Theme(
        dark_fn     = lambda: True,
        jz_labels   = content.labels.job_zones,
        type_labels = content.labels.skill_types
    )

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
def _(content, mo):
    upload = mo.ui.file(
        filetypes = [".pdf"],
        kind      = "area",
        label     = content.labels.upload_label
    )
    return (upload,)


# ── Splash page ─────────────────────────────────────────────────────

@app.cell
def _(Path, content, labor, mo, pipeline, upload):
    from chalkline.display.schemas import SplashMetrics
    from chalkline.display.tabs    import splash

    mo.stop(bool(upload.value), mo.md(""))
    mo.vstack([
        splash(
            content,
            Path(__file__).parent / "assets",
            SplashMetrics.from_pipeline(labor, pipeline)
        ),
        upload
    ])
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


# ── Match bar ──────────────────────────────────────────────────────

@app.cell
def _(layout, mo, profile, theme, upload):
    mo.stop(not upload.value, mo.md(""))
    layout.match_bar(
        jz_label  = theme.jz_label(profile.job_zone),
        postings  = profile.size,
        sector    = profile.sector,
        soc_title = profile.soc_title
    )
    return


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


# ── Target dropdown ─────────────────────────────────────────────────

@app.cell
def _(layout, pipeline, profile):
    dropdown = layout.target_dropdown(
        {p.display_label: cid for cid, p in pipeline.clusters.pairs()},
        profile.display_label
    )
    return (dropdown,)


# ── Target resolution ───────────────────────────────────────────────

@app.cell
def _(dropdown, result):
    target_id = dropdown.value or result.cluster_id
    return (target_id,)


# ── Tab context ─────────────────────────────────────────────────────

@app.cell
def _(charts, content, labor, layout, occupations, pipeline, profile, reference, result, theme):
    from chalkline.display.loaders import TabContext

    ctx = TabContext(
        charts      = charts,
        content     = content,
        layout      = layout,
        labor       = labor,
        occupations = occupations,
        pipeline    = pipeline,
        profile     = profile,
        reference   = reference,
        result      = result,
        theme       = theme
    )
    return (ctx,)


# ── Tab layout ──────────────────────────────────────────────────────

@app.cell
def _(content, ctx, dropdown, mo, target_id):
    from chalkline.display import tabs

    tn = content.labels.tab_names
    mo.ui.tabs(
        {
            tn["your_match"]      : lambda: tabs.your_match(ctx),
            tn["resume_feedback"] : lambda: tabs.resume_feedback(ctx),
            tn["career_paths"]    : lambda: tabs.career_paths(ctx, dropdown, target_id),
            tn["job_postings"]    : lambda: tabs.job_postings(ctx),
            tn["next_steps"]      : lambda: tabs.next_steps(ctx, target_id),
            tn["ml_internals"]    : lambda: tabs.ml_internals(ctx)
        },
        lazy = True
    )
    return


if __name__ == "__main__":
    app.run()
