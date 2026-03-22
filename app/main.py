import marimo

__generated_with = "0.12.0"
app = marimo.App(width="medium")


# ── Setup ───────────────────────────────────────────────────────────

@app.cell
def _():
    import marimo as mo

    from json    import loads
    from pathlib import Path

    def plotly_theme():
        return "plotly_dark" if mo.app_meta().theme == "dark" else "plotly_white"

    return Path, loads, mo, plotly_theme


# ── Pipeline loading ────────────────────────────────────────────────

@app.cell
def _(Path, mo):
    from chalkline.pipeline.orchestrator import Chalkline
    from chalkline.pipeline.schemas      import PipelineConfig

    with mo.persistent_cache(name="chalkline_pipeline", pin_modules=True):
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
    upload = mo.ui.file(
        filetypes = [".pdf"],
        kind      = "area",
        label     = "Drop a resume PDF here to begin"
    )
    mo.vstack([
        mo.md(
            '<h1 style="font-family: Georgia, serif; font-weight: 400;">'
            "Chalkline</h1>\n\n"
            "Career mapping for Maine's construction industry. "
            "Upload a resume to see where you sit in the landscape, "
            "what skills separate you from your next role, and how to get there."
        ),
        mo.hstack([
            mo.stat(label="Postings",        value=f"{pipeline.corpus_size:,}"),
            mo.stat(label="Career Families", value=str(len(pipeline.profiles))),
            mo.stat(label="Sectors",         value=str(pipeline.sector_count)),
            mo.stat(label="Pathway Edges",   value=str(pipeline.graph.edge_count))
        ], gap=1, wrap=True),
        upload
    ])
    return (upload,)


# ── Upload gate ─────────────────────────────────────────────────────

@app.cell
def _(Path, mo, pipeline, upload):
    mo.stop(
        not upload.value,
        mo.callout(
            mo.md("Upload a resume above to generate your career report."),
            kind = "neutral"
        )
    )

    from chalkline.matching.reader import clean_text, extract_pdf
    from tempfile                    import NamedTemporaryFile

    with mo.status.spinner("Matching resume to career landscape..."):
        with NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(upload.value[0].contents)
            tmp.flush()
            result = pipeline.match(clean_text(extract_pdf(Path(tmp.name))))

    matched_profile = pipeline.profiles[result.cluster_id]
    return matched_profile, result


# ── Match summary ───────────────────────────────────────────────────

@app.cell
def _(matched_profile, mo, result):
    mo.vstack([
        mo.hstack([
            mo.stat(label="Career Family", value=matched_profile.soc_title),
            mo.stat(label="Sector",        value=matched_profile.sector),
            mo.stat(label="Job Zone",      value=f"JZ {matched_profile.job_zone}"),
            mo.stat(
                direction = "decrease",
                label     = "Match Distance",
                value     = f"{result.match_distance:.3f}"
            ),
            mo.stat(label="Skill Gaps",   value=str(len(result.gaps))),
            mo.stat(label="Demonstrated", value=str(len(result.demonstrated)))
        ], gap=1, wrap=True),
        mo.callout(
            mo.md(
                f"Your resume most closely matches **{matched_profile.soc_title}** "
                f"in the **{matched_profile.sector}** sector. "
                f"This career family sits at Job Zone {matched_profile.job_zone} "
                f"with {matched_profile.size} postings in the corpus."
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
def _(matched_profile, mo, pipeline, tables):
    target_dropdown = mo.ui.dropdown(
        label      = "Target cluster",
        options    = {
            p.display_label: cid
            for cid, p in sorted(pipeline.profiles.items())
        },
        searchable = True,
        value      = matched_profile.display_label
    )

    mo.sidebar(
        [
            mo.md(
                '<span style="font-family: Georgia, serif; font-size: 1.4em;">'
                "Chalkline</span>"
            ),
            mo.md(
                f"**{matched_profile.soc_title}**\n\n"
                f"{matched_profile.sector} · JZ {matched_profile.job_zone}"
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
    target_neighborhood = pipeline.graph.neighborhood(target_id)
    target_profile      = pipeline.profiles[target_id]
    return target_id, target_neighborhood, target_profile


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
def _(matched_profile, mo, result, tables):
    def skill_analysis_panel():
        gaps  = tables.gap_rows()
        demos = tables.demonstrated_rows()

        return mo.vstack([
            mo.hstack([
                mo.stat(label="Gaps",         value=str(len(result.gaps))),
                mo.stat(label="Demonstrated", value=str(len(result.demonstrated)))
            ]),
            mo.md(f"Skill profile for **{matched_profile.soc_title}**"),
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
def _(figures, mo, tables, target_id, target_neighborhood):
    def career_pathways_panel():
        fig       = figures.pathways(target_neighborhood, target_id)
        cred_rows = tables.credential_rows(target_neighborhood)

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
def _(mo, tables, target_neighborhood, target_profile):
    def education_panel():
        sections = {
            f"{label} ({len(rows)})": mo.ui.table(rows)
            for label, rows in [
                ("Registered Apprenticeships", tables.apprenticeship_rows(target_neighborhood)),
                ("Programs",                   tables.program_rows(target_neighborhood))
            ]
            if rows
        }

        return mo.vstack([
            mo.md(f"Education and training pathways for **{target_profile.soc_title}**"),
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
            mo.stat(label="AGC Members Matched", value=str(len(rows))),
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
def _(matched_profile, mo, tables):
    def job_board_panel():
        maine, national = tables.board_rows()

        sections = {
            f"{label} ({len(rows)})": mo.ui.table(rows)
            for label, rows in [("Maine", maine), ("National", national)]
            if rows
        }

        return mo.vstack([
            mo.stat(
                label = f"Boards for {matched_profile.sector}",
                value = str(len(maine) + len(national))
            ),
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

        mermaid = "\n".join([
            "graph LR",
            *(
                f"    {param} --> {name}"
                for name, fn in sorted(getmembers(steps, isfunction))
                for param in signature(fn).parameters
                if param not in {"config", "model", "lexicons"}
            )
        ])

        return mo.vstack([
            mo.hstack([
                mo.stat(label="Corpus Size",     value=f"{pipeline.corpus_size:,}"),
                mo.stat(label="Embedding Model", value=pipeline.config.embedding_model),
                mo.stat(label="Clusters",        value=str(len(pipeline.profiles))),
                mo.stat(label="SVD Components",  value=str(pipeline.config.component_count))
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
                for cid, p in sorted(pipeline.profiles.items())
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
