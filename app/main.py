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


# ── Sidebar ─────────────────────────────────────────────────────────

@app.cell
def _(matched_profile, mo, pipeline, result):
    target_dropdown = mo.ui.dropdown(
        label      = "Target cluster",
        options    = {
            p.display_label: cid
            for cid, p in sorted(pipeline.profiles.items())
        },
        searchable = True,
        value      = matched_profile.display_label
    )

    from chalkline.display.tables import build_report_text

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
                data     = build_report_text(matched_profile, result).encode(),
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


# ── Career Landscape panel ──────────────────────────────────────────

@app.cell
def _(mo, pipeline, plotly_theme, result):
    def landscape_panel():
        from chalkline.display.figures import landscape_figure

        return mo.ui.plotly(landscape_figure(
            coordinates = result.coordinates,
            matched_id  = result.cluster_id,
            pathway     = pipeline.graph,
            template    = plotly_theme()
        ))
    return (landscape_panel,)


# ── Skill Analysis panel ────────────────────────────────────────────

@app.cell
def _(matched_profile, mo, result):
    def skill_analysis_panel():
        gap_rows = [
            {
                "Similarity" : round(g.similarity, 3),
                "Task"       : g.name
            }
            for g in result.gaps
        ]
        demo_rows = [
            {
                "Similarity" : round(d.similarity, 3),
                "Task"       : d.name
            }
            for d in result.demonstrated
        ]

        return mo.vstack([
            mo.hstack([
                mo.stat(label="Gaps",         value=str(len(result.gaps))),
                mo.stat(label="Demonstrated", value=str(len(result.demonstrated)))
            ]),
            mo.md(f"Skill profile for **{matched_profile.soc_title}**"),
            mo.accordion(
                {
                    f"Gaps ({len(gap_rows)})": (
                        mo.ui.table(gap_rows)
                        if gap_rows
                        else mo.md("No gaps identified.")
                    ),
                    f"Demonstrated ({len(demo_rows)})": (
                        mo.ui.table(demo_rows)
                        if demo_rows
                        else mo.md("No demonstrated tasks.")
                    )
                },
                multiple = True
            )
        ])
    return (skill_analysis_panel,)


# ── Career Pathways panel ───────────────────────────────────────────

@app.cell
def _(mo, pipeline, plotly_theme, result, target_id, target_neighborhood):
    def career_pathways_panel():
        from chalkline.display.figures import pathways_figure
        from chalkline.display.tables  import credential_rows

        fig = pathways_figure(
            matched_id   = result.cluster_id,
            neighborhood = target_neighborhood,
            pathway      = pipeline.graph,
            target_id    = target_id,
            template     = plotly_theme()
        )
        cred_rows = credential_rows(target_neighborhood)

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
def _(mo, pipeline, plotly_theme, result):
    def dendrogram_panel():
        from chalkline.display.figures import dendrogram_figure

        return mo.ui.plotly(dendrogram_figure(
            matched_id = result.cluster_id,
            pathway    = pipeline.graph,
            template   = plotly_theme()
        ))
    return (dendrogram_panel,)


# ── Education & Training panel ──────────────────────────────────────

@app.cell
def _(mo, target_neighborhood, target_profile):
    def education_panel():
        from chalkline.display.tables import apprenticeship_rows, program_rows

        sections = {
            f"{label} ({len(rows)})": mo.ui.table(rows)
            for label, rows in [
                ("Registered Apprenticeships", apprenticeship_rows(target_neighborhood)),
                ("Programs",                   program_rows(target_neighborhood))
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
def _(mo, pipeline, reference, target_id, target_profile):
    def employer_panel():
        from chalkline.display.tables import match_cluster_employers

        rows = match_cluster_employers(
            assignments = pipeline.assignments,
            career_urls = reference["career_urls"],
            cluster_id  = target_id,
            corpus      = pipeline.corpus,
            members     = reference["agc_members"]
        )

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
def _(matched_profile, mo, pipeline, reference):
    def job_board_panel():
        from chalkline.display.tables import filter_boards

        maine_boards, national_boards = filter_boards(
            boards   = reference["job_boards"],
            profiles = pipeline.profiles,
            sector   = matched_profile.sector
        )

        def board_rows(board_list):
            return [
                {
                    "Best For" : b["best_for"],
                    "Category" : b["category"],
                    "Focus"    : b["focus"],
                    "Name"     : b["name"]
                }
                for b in board_list
            ]

        sections = {
            f"{label} ({len(boards)})": mo.ui.table(board_rows(boards))
            for label, boards in [("Maine", maine_boards), ("National", national_boards)]
            if boards
        }

        return mo.vstack([
            mo.stat(
                label = f"Boards for {matched_profile.sector}",
                value = str(len(maine_boards) + len(national_boards))
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
        from chalkline.display.tables import to_mermaid

        return mo.vstack([
            mo.hstack([
                mo.stat(label="Corpus Size",     value=f"{pipeline.corpus_size:,}"),
                mo.stat(label="Embedding Model", value=pipeline.config.embedding_model),
                mo.stat(label="Clusters",        value=str(len(pipeline.profiles))),
                mo.stat(label="SVD Components",  value=str(pipeline.config.component_count))
            ], gap=1, wrap=True),
            mo.md("#### Pipeline DAG"),
            mo.mermaid(to_mermaid()),
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
