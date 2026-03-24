import marimo

__generated_with = "0.12.0"
app = marimo.App(width="medium")


# ── Setup ───────────────────────────────────────────────────────────

@app.cell
def _():
    import marimo as mo

    from json    import loads
    from pathlib import Path

    theme = lambda: ["plotly_white", "plotly_dark"][mo.app_meta().theme == "dark"]

    return Path, loads, mo, theme


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
def _(Path, loads):
    reference = {
        name: loads((Path("data/stakeholder/reference") / f"{name}.json").read_text())
        for name in (
            "agc_members",
            "career_urls",
            "job_boards"
        )
    }
    return (reference,)


# ── Splash page ─────────────────────────────────────────────────────

@app.cell
def _(mo, pipeline):
    mo.vstack([
        mo.md(
            '<h1 style="font-family: Georgia, serif; font-weight: 400;">'
            "Chalkline</h1>\n\n"
            "Career mapping for Maine's construction industry. "
            "Upload a resume to see where you sit in the landscape, "
            "what skills separate you from your next role, and how to get there."
        ),

        mo.hstack([
            mo.stat(f"{pipeline.corpus_size:,}",  "Postings"),
            mo.stat(len(pipeline.clusters),       "Career Families"),
            mo.stat(pipeline.sector_count,        "Sectors"),
            mo.stat(pipeline.graph.edge_count,    "Pathway Edges")
        ], gap=1, wrap=True),
        
        (upload := mo.ui.file(
            filetypes = [".pdf"],
            kind      = "area",
            label     = "Drop a resume PDF here to begin"
        ))
    ])
    return (upload,)


# ── Upload gate ─────────────────────────────────────────────────────

@app.cell
def _(mo, pipeline, upload):
    mo.stop(
        not upload.value,
        mo.callout(
            mo.md("Upload a resume above to generate your career report."),
            kind = "neutral"
        )
    )

    with mo.status.spinner("Matching resume to career landscape..."):
        result = pipeline.match(
            (resume := upload.value[0]).contents,
            label = resume.name
        )
    profile = pipeline.clusters[result.cluster_id]
    return profile, result


# ── Match summary ───────────────────────────────────────────────────

@app.cell
def _(profile, mo, result):
    mo.vstack([
        mo.hstack([
            mo.stat(profile.soc_title,        "Career Family"),
            mo.stat(profile.sector,           "Sector"),
            mo.stat(f"JZ {profile.job_zone}", "Job Zone"),
            mo.stat(
                f"{result.match_distance:.3f}",
                "Match Distance",
                direction = "decrease"
            ),
            mo.stat(len(result.gaps),         "Skill Gaps"),
            mo.stat(len(result.demonstrated), "Demonstrated")
        ], gap=1, wrap=True),
        
        mo.callout(
            mo.md(
                f"Your resume most closely matches **{profile.soc_title}** "
                f"in the **{profile.sector}** sector. "
                f"This career family sits at Job Zone {profile.job_zone} "
                f"with {profile.size} postings in the corpus."
            ),
            kind = "success"
        )
    ])
    return


# ── Table builder ──────────────────────────────────────────────────

@app.cell
def _(pipeline, reference, result):
    from chalkline.display.tables import TableBuilder

    tables = TableBuilder(
        pipeline  = pipeline,
        reference = reference,
        result    = result
    )
    return (tables,)


# ── Sidebar ─────────────────────────────────────────────────────────

@app.cell
def _(profile, mo, pipeline, tables):
    target_dropdown = mo.ui.dropdown(
        label      = "Target cluster",
        options    = {p.display_label: cid for cid, p in pipeline.clusters.pairs()},
        searchable = True,
        value      = profile.display_label
    )

    mo.sidebar(
        [
            mo.md(
                '<span style="font-family: Georgia, serif; font-size: 1.4em;">'
                "Chalkline</span>"
            ),
            mo.md(
                f"**{profile.soc_title}**\n\n"
                f"{profile.sector} · JZ {profile.job_zone}"
            ),
            mo.md("---"),
            target_dropdown,
            mo.md("---"),
            mo.download(
                data     = tables.report_text().encode(),
                filename = "chalkline_report.txt",
                label    = "Download Report"
            )
        ],
        
        footer = mo.md(
            '<span style="font-size: 0.85em; color: gray;">'
            "Built for AGC Maine</span>"
        )
    )
    return (target_dropdown,)


# ── Target resolution ───────────────────────────────────────────────

@app.cell
def _(pipeline, result, target_dropdown):
    target_id = v if (v := target_dropdown.value) is not None else result.cluster_id
    target_profile = pipeline.clusters[target_id]
    target_reach   = pipeline.graph.reach(target_id)
    return target_id, target_profile, target_reach


# ── Figure builder ─────────────────────────────────────────────────

@app.cell
def _(pipeline, plotly_theme, result):
    from chalkline.display.figures import FigureBuilder

    figures = FigureBuilder(
        matched_id = result.cluster_id,
        pathway    = pipeline.graph,
        theme      = plotly_theme
    )
    return (figures,)


# ── Career Landscape panel ──────────────────────────────────────────

@app.cell
def _(figures, mo, result):
    def landscape_panel():
        return mo.ui.plotly(figures.landscape(result.coordinates))
    return (landscape_panel,)


# ── Skill Analysis panel ────────────────────────────────────────────

@app.cell
def _(profile, mo, result, tables):
    def skill_analysis_panel():
        gaps  = tables.gap_rows()
        demos = tables.demonstrated_rows()

        return mo.vstack([
            mo.hstack([
                mo.stat(len(result.gaps),         "Gaps"),
                mo.stat(len(result.demonstrated), "Demonstrated")
            ]),

            mo.md(f"Skill profile for **{profile.soc_title}**"),

            mo.accordion(
                {
                    f"Gaps ({len(gaps)})": (
                        mo.ui.table(gaps)
                        if gaps
                        else mo.md("No gaps identified.")
                    ),
                    f"Demonstrated ({len(demos)})": (
                        mo.ui.table(demos)
                        if demos
                        else mo.md("No demonstrated tasks.")
                    )
                },
                multiple = True
            )
        ])
    return (skill_analysis_panel,)


# ── Career Pathways panel ───────────────────────────────────────────

@app.cell
def _(figures, mo, tables, target_id, target_reach):
    def career_pathways_panel():
        fig       = figures.pathways(target_reach, target_id)
        cred_rows = tables.credential_rows(target_reach)

        sections = [mo.ui.plotly(fig)]
        if cred_rows:
            sections.append(
                mo.accordion(
                    {
                        f"Edge Credentials ({len(cred_rows)})":
                        mo.ui.table(cred_rows)
                    },
                    multiple = True
                )
            )
        return mo.vstack(sections)
    return (career_pathways_panel,)


# ── Dendrogram panel ────────────────────────────────────────────────

@app.cell
def _(figures, mo):
    def dendrogram_panel():
        return mo.ui.plotly(figures.dendrogram())
    return (dendrogram_panel,)


# ── Education & Training panel ──────────────────────────────────────

@app.cell
def _(mo, tables, target_profile, target_reach):
    def education_panel():
        data = {
            "Registered Apprenticeships" : tables.apprenticeship_rows(target_reach),
            "Programs"                   : tables.program_rows(target_reach)
        }
        sections = {
            f"{k} ({len(v)})": mo.ui.table(v)
            for k, v in data.items() if v
        }

        return mo.vstack([
            mo.md(
                f"Education and training pathways for "
                f"**{target_profile.soc_title}**"
            ),
            mo.accordion(sections, multiple=True)
            if sections
            else mo.md("No training pathways found for this cluster.")
        ])
    return (education_panel,)


# ── Employer Connections panel ──────────────────────────────────────

@app.cell
def _(mo, tables, target_id, target_profile):
    def employer_panel():
        rows = tables.employer_rows(target_id)

        return mo.vstack([
            mo.stat(len(rows), "AGC Members Matched"),
            mo.md(
                f"Employers in the "
                f"**{target_profile.soc_title}** "
                f"cluster matched against AGC member companies"
            ),
            mo.ui.table(rows)
            if rows
            else mo.md("No AGC member matches found in this cluster.")
        ])
    return (employer_panel,)


# ── Job Boards panel ────────────────────────────────────────────────

@app.cell
def _(profile, mo, tables):
    def job_board_panel():
        data     = dict(zip(["Maine", "National"], tables.board_rows()))
        sections = {
            f"{k} ({len(v)})": mo.ui.table(v)
            for k, v in data.items() if v
        }

        return mo.vstack([
            mo.stat(sum(len(v) for v in data.values()), f"Boards for {profile.sector}"),
            mo.accordion(sections, multiple=True)
            if sections
            else mo.md("No matching job boards found.")
        ])
    return (job_board_panel,)


# ── Pipeline Details panel ──────────────────────────────────────────

@app.cell
def _(mo, pipeline):
    def pipeline_details_panel():
        from chalkline.pipeline import steps
        from inspect            import getmembers, isfunction, signature

        mermaid = "graph LR\n" + "\n".join(
            f"    {param} --> {name}"
            for name, fn in sorted(getmembers(steps, isfunction))
            for param in signature(fn).parameters
            if param not in {"config", "encoder", "lexicons"}
        )

        return mo.vstack([
            mo.hstack([
                mo.stat(f"{pipeline.corpus_size:,}",     "Corpus Size"),
                mo.stat(pipeline.config.embedding_model, "Embedding Model"),
                mo.stat(len(pipeline.clusters),          "Clusters"),
                mo.stat(pipeline.config.component_count, "SVD Components")
            ], gap=1, wrap=True),
            mo.md("#### Pipeline DAG"),
            mo.mermaid(mermaid),
            mo.md("#### Cluster Profiles"),
            mo.tree({
                f"Cluster {cid}: {p.soc_title}": {
                    "Sector"      : p.sector,
                    "Job Zone"    : p.job_zone,
                    "Size"        : p.size,
                    "Modal Title" : p.modal_title
                }
                for cid, p in pipeline.clusters.pairs()
            })
        ])
    return (pipeline_details_panel,)


# ── Layout ──────────────────────────────────────────────────────────

@app.cell
def _(
    career_pathways_panel,
    dendrogram_panel,
    education_panel,
    employer_panel,
    job_board_panel,
    landscape_panel,
    mo,
    pipeline_details_panel,
    skill_analysis_panel
):
    mo.accordion(
        {
            "Career Landscape"     : landscape_panel,
            "Skill Analysis"       : skill_analysis_panel,
            "Career Pathways"      : career_pathways_panel,
            "Dendrogram"           : dendrogram_panel,
            "Education & Training" : education_panel,
            "Employer Connections" : employer_panel,
            "Job Boards"           : job_board_panel,
            "Pipeline Details"     : pipeline_details_panel
        },
        lazy     = True,
        multiple = True
    )
    return


if __name__ == "__main__":
    app.run()
