import marimo

__generated_with = "0.12.0"
app = marimo.App(width="full", css_file="chalkline.css")


# ── Setup ───────────────────────────────────────────────────────────

@app.cell
def _():
    import marimo               as mo
    import plotly.graph_objects as go

    from json    import loads
    from pathlib import Path

    from chalkline.display.layout import callout, header, stat_strip

    info_dir = Path(__file__).parent / "info"
    info     = lambda name: (info_dir / f"{name}.md").read_text()

    dark   = lambda: mo.app_meta().theme == "dark"
    theme  = lambda: ["plotly_white", "plotly_dark"][dark()]
    MARGIN = dict(b=40, l=10, r=10, t=10)

    LIGHT = {
        "accent"     : "#4a6fa5",
        "error"      : "#c44536",
        "foreground" : "#1a1a1a",
        "muted"      : "#999999",
        "primary"    : "#B8941F",
        "success"    : "#3a7d44"
    }
    DARK = {
        "accent"     : "#6b9fcc",
        "error"      : "#e07a5f",
        "foreground" : "#ebebeb",
        "muted"      : "#777777",
        "primary"    : "#E8C840",
        "success"    : "#81b29a"
    }
    C = lambda: DARK if dark() else LIGHT

    FONT = lambda: dict(
        color  = C()["foreground"],
        family = "Lora, Georgia, serif",
        size   = 13
    )

    def hbar(color, height, title, x, y, **marker_kw):
        fig = go.Figure(go.Bar(
            marker      = dict(color=color, cornerradius=4, **marker_kw),
            orientation = "h",
            x           = x,
            y           = y
        ))
        fig.update_layout(
            font          = FONT(),
            height        = height,
            margin        = MARGIN,
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            template      = theme(),
            xaxis_title   = title,
            yaxis         = dict(autorange="reversed")
        )
        return fig

    def vbar(height, title, x, y, color=None):
        fig = go.Figure(go.Bar(
            marker = dict(
                color        = color,
                cornerradius = 4
            ) if color else dict(cornerradius=4),
            x = x,
            y = y
        ))
        fig.update_layout(
            font          = FONT(),
            height        = height,
            margin        = MARGIN,
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            template      = theme(),
            yaxis_title   = title
        )
        return fig

    return (
        C, FONT, MARGIN, Path, info,
        callout, go, hbar, header,
        loads, mo, stat_strip, theme, vbar
    )


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
    ref_dir   = Path("data/stakeholder/reference")
    reference = {
        name: loads(path.read_text())
        for name in (
            "agc_members",
            "apprenticeships",
            "career_urls",
            "cc_programs",
            "dot_contractors",
            "job_boards",
            "onet_codes",
            "umaine_programs"
        )
        if (path := ref_dir / f"{name}.json").exists()
    }

    labor = {
        r["soc_title"]: r
        for r in loads(
            (Path("data/lexicons") / "labor.json").read_text()
        )
    }
    return labor, reference


# ── Upload widget ───────────────────────────────────────────────────

@app.cell
def _(mo):
    upload = mo.ui.file(
        filetypes = [".pdf"],
        kind      = "area",
        label     = "Drop a resume PDF here"
    )
    return (upload,)


# ── Splash page (always visible) ──────────────────────────────────────

@app.cell
def _(labor, mo, pipeline, stat_strip, upload):
    from base64     import b64encode
    from pathlib    import Path as P
    from statistics import median

    from chalkline.display.layout import stat_html

    logo_b64 = b64encode((P(__file__).parent / "assets/logo.png").read_bytes()).decode()
    logo_src = f"data:image/png;base64,{logo_b64}"

    postings = [
        p for c in pipeline.clusters.values() for p in c.postings
    ]
    companies = len({p.company for p in postings if p.company})
    locations = len({p.location for p in postings if p.location})
    wages = [
        r["wages"]["annual_median"]
        for r in labor.values()
        if (r.get("wages") or {}).get("annual_median")
    ]
    employment = sum(
        r["wages"]["employment"]
        for r in labor.values()
        if (r.get("wages") or {}).get("employment")
    )
    bright = sum(
        1 for r in labor.values()
        if (r.get("outlook") or {}).get("bright_outlook")
    )

    s = stat_html

    splash = mo.Html(
        '<div class="cl-splash">'
        '<div class="cl-brand">'
        f'<span class="cl-logo" style="'
        f"mask-image:url({logo_src});"
        f"-webkit-mask-image:url({logo_src});"
        f'"></span>'
        "<h1>Chalkline</h1>"
        "</div>"
        '<p class="cl-tagline">'
        "Career mapping for Maine's "
        "construction industry</p>"
        '<div class="cl-stats">'
        + s("Job Postings",    f"{pipeline.corpus_size:,}")
        + s("Companies",       str(companies))
        + s("Maine Locations", str(locations))
        + s("Career Families", str(len(pipeline.clusters)))
        + s("Maine Workers",   f"{employment:,}")
        + s("Median Salary",   f"${median(wages):,.0f}")
        + s("Bright Outlook",  str(bright))
        + s("Career Pathways", str(pipeline.graph.edge_count))
        + "</div>"
        "</div>"
    )

    mo.stop(bool(upload.value), mo.md(""))
    mo.vstack([splash, upload])
    return


# ── Upload gate + matching ──────────────────────────────────────────

@app.cell
def _(mo, pipeline, upload):
    mo.stop(not upload.value, mo.md(""))

    with mo.status.spinner("Analyzing your resume..."):
        result = pipeline.match(
            (resume := upload.value[0]).contents,
            label = resume.name
        )
    profile = pipeline.clusters[result.cluster_id]
    return profile, result


# ── Compact header bar (appears after upload) ───────────────────────

@app.cell
def _(mo, profile, upload):
    mo.stop(not upload.value, mo.md(""))

    jz_label = {
        1 : "Entry Level",
        2 : "Some Preparation",
        3 : "Mid-Career",
        4 : "Experienced",
        5 : "Advanced"
    }[profile.job_zone]

    mo.Html(
        '<div class="cl-match-bar">'
        f"<strong>{profile.soc_title}</strong> · "
        f"{profile.sector} · {jz_label} · "
        f"{profile.size} postings"
        "</div>"
    )
    return (jz_label,)


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


# ── Figure builder ─────────────────────────────────────────────────

@app.cell
def _(pipeline, theme, result):
    from chalkline.display.figures import FigureBuilder

    figures = FigureBuilder(
        matched_id = result.cluster_id,
        pathway    = pipeline.graph,
        theme      = theme
    )
    return (figures,)


# ── Target dropdown (outside tabs for lazy rendering) ───────────────

@app.cell
def _(mo, pipeline, profile):
    target_dropdown = mo.ui.dropdown(
        label      = "Explore a career family",
        options    = {p.display_label: cid for cid, p in pipeline.clusters.pairs()},
        searchable = True,
        value      = profile.display_label
    )
    return (target_dropdown,)


# ── Target resolution ───────────────────────────────────────────────

@app.cell
def _(pipeline, result, target_dropdown):
    target_id = (
        v if (v := target_dropdown.value) is not None
        else result.cluster_id
    )
    target_profile = pipeline.clusters[target_id]
    target_reach   = pipeline.graph.reach(target_id)
    return target_id, target_profile, target_reach


# ── Tab: Career Paths ──────────────────────────────────────────────

@app.cell
def _(
    callout, info, figures, header, mo, tables,
    target_dropdown, target_id, target_reach
):
    def career_paths_tab():
        fig       = figures.pathways(target_reach, target_id)
        cred_rows = tables.credential_rows(target_reach)

        sections = [
            header(
                "Career Pathways",
                "Each connection represents a plausible career move "
                "between families at the same experience level (lateral) "
                "or one level up (advancement), weighted by how similar "
                "the roles are."
            ),
            target_dropdown,
            mo.ui.plotly(fig),
            callout(info("career_paths"))
        ]

        if cred_rows:
            sections.append(
                mo.accordion(
                    {f"Credentials bridging these transitions ({len(cred_rows)})":
                     mo.ui.table(cred_rows)},
                    multiple = True
                )
            )
        return mo.vstack(sections)
    return (career_paths_tab,)


# ── Tab: Resume Feedback ───────────────────────────────────────────

@app.cell
def _(
    C, info, callout, hbar, header, mo,
    pipeline, profile, result, stat_strip, tables
):
    def resume_feedback_tab():
        gaps  = tables.gap_rows()
        demos = tables.demonstrated_rows()

        cluster  = pipeline.clusters[result.cluster_id]
        n_closer = sum(
            1 for p in cluster.postings
            if hasattr(p, "distance") and p.distance < result.match_distance
        )
        percentile = round(
            (1 - n_closer / max(len(cluster.postings), 1)) * 100
        )

        def skill_bar(color, rows, title, invert=False):
            names  = [r["Task"][:50] for r in rows[:15]]
            scores = [
                round((1 - r["Similarity"] if invert else r["Similarity"]) * 100, 1)
                for r in rows[:15]
            ]
            return hbar(
                color  = color,
                height = max(300, len(names) * 28),
                title  = title,
                x      = scores,
                y      = names
            ), names

        strength_fig, demo_names = skill_bar(
            color = C()["success"],
            rows  = demos,
            title = "Skill Alignment (%)"
        )
        gap_fig, gap_names = skill_bar(
            color  = C()["error"],
            invert = True,
            rows   = gaps,
            title  = "Gap Magnitude (%)"
        )

        sections = [
            header(
                "How Your Skills Compare",
                "We compared your resume against O*NET task "
                f"definitions for {profile.soc_title} using "
                "language similarity. Scores closer to 100% "
                "mean your resume strongly reflects that skill."
            ),
            stat_strip({
                "Strengths"            : str(len(result.demonstrated)),
                "Growth Areas"         : str(len(result.gaps)),
                "Alignment Percentile" : f"{percentile}%"
            }),
            callout(
                f"Your resume is more aligned with "
                f"**{profile.soc_title}** than "
                f"**{percentile}%** of the "
                f"{len(cluster.postings)} postings "
                f"in this career family.",
                kind = "info"
            ),
            header("Your Strengths",
                   "Tasks from your resume that match "
                   "what employers want."),
            mo.ui.plotly(strength_fig) if demo_names
            else mo.md("No demonstrated tasks."),
            header("Growth Opportunities",
                   "Skills most postings mention that your "
                   "resume doesn't strongly reflect yet."),
            mo.ui.plotly(gap_fig) if gap_names
            else mo.md("No gaps identified."),
            callout(info("skill_alignment")),
            mo.accordion({
                f"Raw Data: Strengths ({len(demos)})":
                    mo.ui.table(demos) if demos
                    else mo.md("None."),
                f"Raw Data: Gaps ({len(gaps)})":
                    mo.ui.table(gaps) if gaps
                    else mo.md("None.")
            }, multiple=True)
        ]

        return mo.vstack(sections)
    return (resume_feedback_tab,)


# ── Tab: Job Postings ──────────────────────────────────────────────

@app.cell
def _(
    C, FONT, MARGIN, go, hbar, header,
    mo, pipeline, profile, result, stat_strip, theme
):
    def job_postings_tab():
        from collections             import Counter
        from chalkline.display.cards import posting_card

        postings = pipeline.clusters[result.cluster_id].postings

        companies = Counter(p.company for p in postings if p.company)
        top_companies = companies.most_common(15)

        hiring_fig = hbar(
            color  = C()["accent"],
            height = max(300, len(top_companies) * 28),
            title  = "Number of Postings",
            x      = [c[1] for c in top_companies],
            y      = [c[0][:30] for c in top_companies]
        )

        locations = Counter(
            p.location for p in postings if p.location
        ).most_common(15)
        location_fig = hbar(
            color  = C()["success"],
            height = max(250, len(locations) * 28),
            title  = "Number of Postings",
            x      = [loc[1] for loc in locations],
            y      = [loc[0][:30] for loc in locations]
        ) if locations else None

        dated = [p for p in postings if p.date_posted]
        timeline_fig = None
        if dated:
            dates  = [p.date_posted for p in dated]
            labels = [p.company or p.title for p in dated]
            timeline_fig = go.Figure(go.Scatter(
                hovertext   = labels,
                marker      = dict(
                    color = C()["accent"],
                    size  = 8
                ),
                mode        = "markers",
                x           = dates,
                y           = [1] * len(dates)
            ))
            timeline_fig.update_layout(
                font          = FONT(),
                height        = 180,
                margin        = MARGIN,
                paper_bgcolor = "rgba(0,0,0,0)",
                plot_bgcolor  = "rgba(0,0,0,0)",
                template      = theme(),
                yaxis         = dict(visible=False)
            )

        posting_cards = [
            posting_card(p)
            for p in sorted(
                postings,
                key     = lambda x: x.date_posted or "",
                reverse = True
            )[:12]
        ]

        sections = [
            header(
                "Real Postings From Your Career Family",
                f"These are actual job postings from Maine "
                f"construction companies that fall into the "
                f"{profile.soc_title} career family. Browse "
                f"to see what employers are looking for."
            ),
            stat_strip({
                "Postings in Family" : str(len(postings)),
                "Companies Hiring"   : str(len(companies)),
                **({"Locations" : str(len(locations))} if locations else {})
            }),
            header(
                "Who's Hiring",
                "Companies with the most postings in this "
                "career family."
            ),
            mo.ui.plotly(hiring_fig)
            if top_companies else mo.md("No company data.")
        ]

        if location_fig:
            sections.extend([
                header(
                    "Where the Jobs Are",
                    "Posting locations across Maine for this "
                    "career family."
                ),
                mo.ui.plotly(location_fig)
            ])

        if timeline_fig:
            sections.extend([
                header(
                    "Posting Timeline",
                    "When these postings were collected, showing "
                    "recent market activity."
                ),
                mo.ui.plotly(timeline_fig)
            ])

        sections.extend([
            header(
                "Recent Postings",
                "The most recent job listings, newest first."
            ),
            *posting_cards
        ])

        return mo.vstack(sections)
    return (job_postings_tab,)


# ── Tab: Next Steps ────────────────────────────────────────────────

@app.cell
def _(
    C, callout, hbar, header, mo, pipeline,
    tables, target_id, target_profile, target_reach, labor
):
    def next_steps_tab():
        apprenticeships = tables.apprenticeship_rows(target_reach)
        programs        = tables.program_rows(target_reach)
        employers       = tables.employer_rows(target_id)
        maine, national = tables.board_rows()

        destinations = [
            (pipeline.clusters[e.cluster_id], e)
            for e in target_reach.advancement
        ]
        wage_ladder = [
            (c.soc_title[:30], w["annual_median"])
            for c, _e in destinations
            if (w := (labor.get(c.soc_title, {}).get("wages") or {}))
            and w.get("annual_median")
        ]

        app_fig = None
        if apprenticeships:
            trades = [a["Trade"][:35] for a in apprenticeships]
            hours  = [
                int(a["Min Hours"].replace(",", ""))
                for a in apprenticeships
            ]
            app_fig = hbar(
                color      = hours,
                colorscale = "Teal",
                height     = max(250, len(trades) * 32),
                title      = "Minimum Hours",
                x          = hours,
                y          = trades
            )

        from chalkline.display.cards import board_card, employer_card, program_card

        program_cards = [
            program_card(
                credential  = p["Credential"],
                institution = p["Institution"],
                name        = p["Program"],
                url         = p["Link"]
            )
            for p in programs
        ]
        employer_cards = [
            employer_card(
                career_url  = e["Career Page"],
                member_type = e["Type"],
                name        = e["Company"],
                posting_url = e["Posting"]
            )
            for e in employers
        ]
        board_cards = [
            board_card(
                best_for = b["Best For"],
                category = b["Category"],
                focus    = b["Focus"],
                name     = b["Name"]
            )
            for b in maine + national
        ]

        sections = [
            header(
                "Training, Credentials, and Employers",
                f"Resources that can help you advance into or within "
                f"the {target_profile.soc_title} career family."
            )
        ]

        if app_fig:
            sections.extend([
                mo.md("#### Apprenticeships"),
                mo.md("Registered earn-while-you-learn programs with "
                       "real hour requirements. These are paid positions."),
                mo.ui.plotly(app_fig)
            ])

        if wage_ladder:
            sorted_wages = sorted(wage_ladder, key=lambda w: w[1])
            sections.extend([
                header(
                    "Where the Money Leads",
                    "Median annual wages in Maine for career "
                    "families you could advance into from here."
                ),
                mo.ui.plotly(hbar(
                    color  = C()["success"],
                    height = max(200, len(sorted_wages) * 32),
                    title  = "Annual Median Wage ($)",
                    x      = [w[1] for w in sorted_wages],
                    y      = [w[0] for w in sorted_wages]
                ))
            ])

        if program_cards:
            sections.extend([
                mo.md(f"#### Educational Programs ({len(program_cards)})"),
                *program_cards
            ])

        if employer_cards:
            sections.extend([
                mo.md(f"#### Employers ({len(employer_cards)})"),
                mo.md("AGC Maine member companies with postings in "
                       "this career family."),
                *employer_cards
            ])

        if board_cards:
            sections.extend([
                mo.md(f"#### Job Boards ({len(board_cards)})"),
                *board_cards
            ])

        if not any([apprenticeships, programs, employers, maine, national]):
            sections.append(callout(
                "No training pathways found for this cluster.",
                kind = "warn"
            ))

        return mo.vstack(sections)
    return (next_steps_tab,)


# ── Tab: Your Match ────────────────────────────────────────────────

@app.cell
def _(
    C, callout, go, hbar, header, info, jz_label, labor,
    mo, pipeline, profile, result, stat_strip, theme, vbar
):
    def your_match_tab():
        rec   = labor.get(profile.soc_title, {})
        w     = rec.get("wages") or {}
        proj  = rec.get("projections") or {}
        outl  = rec.get("outlook") or {}

        salary_text = (
            f" The median salary in Maine is "
            f"**${w['annual_median']:,.0f}** per year."
            if w.get("annual_median") else ""
        )
        outlook_text = (
            f" O*NET designates this occupation **Bright Outlook** "
            f"({', '.join(outl['outlook_reasons'])})."
            if outl.get("bright_outlook") else ""
        )

        hero = callout(
            f"Your resume most closely matches "
            f"**{profile.soc_title}** in "
            f"**{profile.sector}**. "
            f"This is a **{jz_label.lower()}** role "
            f"with {profile.size} postings in the "
            f"corpus.{salary_text}{outlook_text}",
            kind = "success"
        )

        labor_data = {}
        if w.get("annual_median"):
            labor_data["Maine Median Wage"] = f"${w['annual_median']:,.0f}"
        if proj.get("change_percent"):
            labor_data["10-Year Growth"] = f"{proj['change_percent']:+.1f}%"
        if proj.get("openings"):
            labor_data["Projected Openings"] = f"{proj['openings']:,.0f}K/yr"
        if proj.get("education"):
            labor_data["Typical Education"] = proj["education"]

        wage_fig = None
        if w.get("annual_median"):
            percentiles = ["10th", "25th", "Median", "75th", "90th"]
            values = [
                w.get("annual_10", 0),
                w.get("annual_25", 0),
                w["annual_median"],
                w.get("annual_75", 0),
                w.get("annual_90", 0)
            ]
            wage_fig = hbar(
                color  = [
                    C()["muted"],
                    C()["accent"],
                    C()["primary"],
                    C()["accent"],
                    C()["muted"]
                ],
                height = 220,
                title  = "Annual Salary ($)",
                x      = values,
                y      = percentiles
            )

        distances = result.cluster_distances

        prox_fig = hbar(
            color = [
                C()["primary"] if cd.cluster_id == result.cluster_id
                else C()["accent"]
                for cd in distances
            ],
            height = max(400, len(distances) * 24),
            title  = "Distance (lower = closer match)",
            x      = [round(cd.distance, 3) for cd in distances],
            y      = [
                pipeline.clusters[cd.cluster_id].soc_title[:30]
                for cd in distances
            ]
        )

        from collections import defaultdict
        sector_dist = defaultdict(list)
        for cd in distances:
            c = pipeline.clusters[cd.cluster_id]
            sector_dist[c.sector].append(cd.distance)
        sector_avg = {
            s: sum(d) / len(d)
            for s, d in sector_dist.items()
        }

        sector_fig = vbar(
            height = 300,
            title  = "Average Distance",
            x      = list(sector_avg.keys()),
            y      = [round(v, 3) for v in sector_avg.values()]
        )

        sections = [
            hero,
            stat_strip({
                "Career Family"    : profile.soc_title,
                "Sector"           : profile.sector,
                "Experience Level" : jz_label,
                "Postings"         : str(profile.size),
                "Growth Areas"     : str(len(result.gaps)),
                "Strengths"        : str(len(result.demonstrated))
            }),
            callout(info("resume_match"))
        ]

        if labor_data:
            sections.extend([
                header(
                    "Labor Market Snapshot",
                    "Wage and employment data for this occupation "
                    "from the Bureau of Labor Statistics and O*NET."
                ),
                stat_strip(labor_data)
            ])

        if wage_fig:
            sections.extend([
                header(
                    "Maine Wage Distribution",
                    "Annual salary percentiles for this "
                    "occupation in Maine from BLS OEWS 2024."
                ),
                mo.ui.plotly(wage_fig)
            ])

        sections.extend([
            header(
                "Proximity to All Career Families",
                "How close your resume is to each of the 20 career "
                "families. Your best match is highlighted. Lower "
                "distance means a closer fit."
            ),
            mo.ui.plotly(prox_fig),
            header(
                "Sector Affinity",
                "Average distance to career families in each "
                "sector. Lower means your resume aligns more "
                "with that area of construction."
            ),
            mo.ui.plotly(sector_fig)
        ])

        return mo.vstack(sections)
    return (your_match_tab,)


# ── Tab: ML Internals ──────────────────────────────────────────────

@app.cell
def _(
    C, FONT, MARGIN, figures, go, hbar, header,
    mo, pipeline, result, stat_strip, theme, vbar
):
    def ml_internals_tab():
        from networkx import betweenness_centrality

        graph = pipeline.graph.graph
        bc    = betweenness_centrality(graph, weight="weight")

        weights = [
            d["weight"] for _, _, d in graph.edges(data=True)
            if "weight" in d
        ]
        weight_fig = go.Figure(go.Histogram(
            marker = dict(color=C()["accent"], cornerradius=4),
            nbinsx = 25,
            x      = weights
        ))
        weight_fig.update_layout(
            font          = FONT(),
            height        = 300,
            margin        = MARGIN,
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            template      = theme(),
            xaxis_title   = "Edge Weight (cosine similarity)",
            yaxis_title   = "Count"
        )

        sector_color = {
            "Building Construction" : "#4a90d9",
            "Heavy Civil"           : "#d97a4a",
            "Specialty Trade"       : "#6bbf59"
        }
        sc = lambda s: sector_color.get(s, C()["accent"])
        rows = [
            (s, "", 0, sc(s))
            for s in sorted({
                p.sector for _, p in pipeline.clusters.pairs()
            })
        ] + [
            (f"{p.soc_title[:25]} ({p.size})", p.sector, p.size, sc(p.sector))
            for _, p in pipeline.clusters.pairs()
        ]
        labels, parents, values, colors = zip(*rows)

        treemap_fig = go.Figure(go.Treemap(
            branchvalues = "total",
            labels       = labels,
            marker       = dict(colors=colors),
            parents      = parents,
            values       = values
        ))
        treemap_fig.update_layout(
            font          = FONT(),
            height        = 450,
            margin        = MARGIN,
            paper_bgcolor = "rgba(0,0,0,0)",
            template      = theme()
        )

        sector_sizes = {}
        for _, p in pipeline.clusters.pairs():
            sector_sizes[p.sector] = sector_sizes.get(p.sector, 0) + p.size

        return mo.vstack([
            header(
                "Under the Hood",
                "Technical details of how Chalkline analyzes the job "
                "market. This section is for evaluating the methodology."
            ),
            stat_strip({
                "Corpus Size"     : f"{pipeline.corpus_size:,}",
                "Embedding Model" : pipeline.config.embedding_model,
                "Clusters (k)"    : str(len(pipeline.clusters)),
                "SVD Components"  : str(pipeline.config.component_count),
                "Pathway Edges"   : str(pipeline.graph.edge_count)
            }),
            header(
                "Career Landscape Treemap",
                "All career families grouped by sector, sized by posting "
                "count. Larger tiles represent more postings."
            ),
            mo.ui.plotly(treemap_fig),
            header(
                "Gateway Careers (Betweenness Centrality)",
                "Career families that bridge the most pathways. High "
                "centrality means the role is a common stepping stone."
            ),
            mo.ui.plotly(hbar(
                color  = C()["accent"],
                height = max(300, len(bc) * 24),
                title  = "Betweenness Centrality",
                x      = [round(bc[cid], 4) for cid, _ in pipeline.clusters.pairs()],
                y      = [p.soc_title[:30]  for _, p   in pipeline.clusters.pairs()]
            )),
            header(
                "Pathway Strength Distribution",
                "How similar connected career families are. Higher weight "
                "means the transition is more natural."
            ),
            mo.ui.plotly(weight_fig),
            header(
                "Sector Distribution",
                "How the corpus breaks down across construction sectors."
            ),
            mo.ui.plotly(vbar(
                height = 300,
                title  = "Postings",
                x      = list(sector_sizes.keys()),
                y      = list(sector_sizes.values())
            )),
            header(
                "Hierarchical Clustering Dendrogram",
                "Ward-linkage agglomerative clustering builds a tree of "
                "every possible merge, then cuts at k=20."
            ),
            mo.ui.plotly(figures.dendrogram()),
            header(
                "Career Landscape",
                "All 20 career family centroids projected onto the first "
                "two SVD components. Your resume is the gold star."
            ),
            mo.ui.plotly(figures.landscape(result.coordinates)),
            header(
                "Cluster Profiles",
                "Metadata for each career family produced by the pipeline."
            ),
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
    return (ml_internals_tab,)


# ── Tab layout ─────────────────────────────────────────────────────

@app.cell
def _(
    career_paths_tab,
    job_postings_tab,
    ml_internals_tab,
    mo,
    next_steps_tab,
    resume_feedback_tab,
    your_match_tab
):
    mo.ui.tabs(
        {
            "Career Paths"    : career_paths_tab,
            "Resume Feedback" : resume_feedback_tab,
            "Job Postings"    : job_postings_tab,
            "Next Steps"      : next_steps_tab,
            "Your Match"      : your_match_tab,
            "ML Internals"    : ml_internals_tab
        },
        lazy = True
    )
    return


if __name__ == "__main__":
    app.run()
