"""
Pydantic metric models for the Chalkline display layer.

Each model defines a data shape consumed by tab renderers and chart
builders, with a `from_` factory classmethod that constructs the model from
raw pipeline state.
"""

from collections     import Counter
from collections.abc import Iterable, Sequence
from datetime        import date
from itertools       import accumulate
from operator        import attrgetter
from pydantic        import BaseModel, Field, field_validator, model_validator
from statistics      import fmean
from typing          import Literal, NamedTuple, Self

from chalkline.collection.schemas    import Posting
from chalkline.matching.schemas      import MatchResult, ScoredTask
from chalkline.pathways.clusters     import Clusters
from chalkline.pathways.loaders      import LaborLoader, StakeholderReference
from chalkline.pipeline.orchestrator import Chalkline


class GapScatterPoint(BaseModel, extra="forbid"):
    """
    A single point in a bubble scatter plot.

    Shared between gap priority scatter (*resume feedback tab*) where
    magnitude is the non-negative gap size, and quality scatter (*ML
    internals tab*) where magnitude is a signed silhouette coefficient
    scaled to percentage.
    """

    frequency : int   = Field(ge=1)
    magnitude : float = Field(ge=-100, le=100)
    text      : str


class HeroContent(BaseModel, extra="forbid"):
    """
    Callout text and semantic kind for a tab hero banner.
    """

    text: str

    kind: Literal["info", "success", "warn"] = "info"

    def render(self, **kwargs) -> tuple[str, str]:
        """
        Format text and return `(content, kind)` for `callout()`.
        """
        return self.text.format(**kwargs), self.kind


class HierarchyData(BaseModel, extra="forbid"):
    """
    Typed bundle for hierarchical chart data (sunburst, treemap).

    Positionally aligned lists where index `i` across all fields describes
    one tile. `labels` and `values` are required; optional fields enable
    multi-level hierarchies (`parents`, `ids`), custom node colors
    (`colors`), or automatic sector coloring (`sectors`).
    """

    labels : Sequence[str]
    values : Sequence[int | float]

    colors  : Sequence[str] | None = None
    ids     : Sequence[str] | None = None
    parents : Sequence[str] | None = None
    sectors : Sequence[str] | None = None

    @classmethod
    def from_clusters(cls, clusters: Clusters) -> HierarchyData:
        """
        Build a sector → cluster hierarchy from the fitted cluster
        collection.

        Prepends one header row per sector with the total posting count,
        then appends each cluster as a child tile under its sector.
        """
        sector_totals = clusters.sector_sizes
        rows = (
            [(s, "", s, sector_totals.get(s, 0)) for s in clusters.sectors]
            + [(f"{c.soc_title} ({c.size})", c.sector, c.sector, c.size)
               for c in clusters.values()]
        )
        labels, parents, sectors, values = zip(*rows)

        return cls(
            labels  = labels,
            parents = parents,
            sectors = sectors,
            values  = values
        )


class JobPostingMetrics(BaseModel, extra="forbid"):
    """
    Aggregated posting statistics for the matched career family.
    """

    companies    : dict[str, int]
    dated        : dict[str, date]
    descriptions : dict[str, int]
    freshness    : list[int]
    locations    : dict[str, int]
    members      : dict[str, int]
    recent       : list[Posting]
    stat_values  : list[str]
    titles       : dict[str, int]

    @classmethod
    def from_postings(
        cls,
        postings  : list[Posting],
        reference : StakeholderReference
    ) -> JobPostingMetrics:
        """
        Aggregate posting data for one career family.

        Args:
            postings  : Posting objects from the matched cluster.
            reference : Stakeholder reference for AGC member lookup.
        """
        from difflib  import get_close_matches
        from wordfreq import zipf_frequency

        companies = Counter(p.company  for p in postings if p.company)
        locations = Counter(p.location for p in postings if p.location).most_common(15)
        by_date   = attrgetter("date_posted")
        dated     = sorted(filter(by_date, postings), key=by_date, reverse=True)
        today     = date.today()

        agc_names = [m["name"] for m in reference.agc_members]
        agc_types = {m["name"].lower(): m.get("type", "") for m in reference.agc_members}

        agc_matched = {
            c for c in companies
            if get_close_matches(c, agc_names, n=1, cutoff=0.7)
        }

        member_types = Counter(
            agc_types.get(match[0].lower(), "Unknown")
            for c in agc_matched
            if (match := get_close_matches(c, agc_names, n=1, cutoff=0.7))
        )

        freshness = [
            (today - p.date_posted).days
            for p in postings
            if p.date_posted
        ]

        descriptions = dict(
            Counter(
                w for p in postings
                for word in p.description.split()
                if  len(w := word.strip(".,;:!?()[]\"'").lower()) > 3
                and zipf_frequency(w, "en") < 4.5
            ).most_common(20)
        )

        stat_values = [
            str(len(postings)),
            str(len(companies)),
            str(len(locations)),
            str(len(agc_matched)),
            f"{min(freshness)}d" if freshness else "N/A",
            str(len(dated))
        ]

        return cls(
            companies    = {name: n for name, n in companies.most_common(15)},
            dated        = {p.company or p.title: p.date_posted for p in dated},
            descriptions = descriptions,
            freshness    = freshness,
            locations    = dict(locations),
            members      = member_types,
            recent       = dated[:12],
            stat_values  = stat_values,
            titles       = dict(
                Counter(
                    w for p in postings
                    for word in p.title.split()
                    if  len(w := word.strip(".,;:!?()[]").lower()) > 2
                    and zipf_frequency(w, "en") < 5.0
                ).most_common(25)
            )
        )


class LaborMetrics(BaseModel, extra="forbid"):
    """
    BLS and O*NET labor market data for the matched occupation.
    """

    outlook_text : str            = ""
    salary_text  : str            = ""
    stat_strip   : dict[str, str] = Field(default_factory=dict)
    wages        : list[float]    = Field(default_factory=list)

    @classmethod
    def from_record(
        cls,
        labor      : LaborLoader,
        soc_title  : str,
        stat_keys  : dict[str, str],
        templates  : LaborTemplates | None = None
    ) -> LaborMetrics:
        """
        Extract and format BLS labor market data for one occupation.

        Args:
            labor     : BLS labor market data keyed by SOC title.
            soc_title : SOC title to look up.
            stat_keys : Display label per stat, keyed by `"median_wage"`, `"growth"`,
                        `"openings"`, `"education"`.
            templates : Prose templates for outlook and salary text.
        """
        if not (record := labor.get(soc_title)):
            return cls()

        stat_strip = {
            stat_keys[key]: template.format(value)
            for key, value, template in [
                ("growth",      record.change_percent, "{:+.1f}%"),
                ("median_wage", record.annual_median,  "${:,.0f}"),
                ("openings",    record.openings,       "{:,.0f}K/yr")
            ]
            if value
        }

        if templates:
            outlook_text = (
                templates.bright_outlook.format(
                    reasons=", ".join(record.outlook_reasons)
                )
                if record.bright_outlook else ""
            )
            salary_text = (
                templates.salary.format(wage=f"{record.annual_median:,.0f}")
                if record.percentiles else ""
            )
        else:
            outlook_text = salary_text = ""

        return cls(
            outlook_text = outlook_text,
            salary_text  = salary_text,
            stat_strip   = stat_strip,
            wages        = record.percentiles
        )


class LaborTemplates(BaseModel, extra="forbid"):
    """
    Format-string templates for labor market prose callouts.

    Each template uses Python `.format()` placeholders that the consuming
    factory fills at render time.
    """

    bright_outlook : str
    salary         : str


class Labels(BaseModel, extra="forbid"):
    """
    Shared display labels loaded from `tabs/shared/labels.toml`.

    Cross-cutting text used by Theme, layout helpers, and the Marimo
    notebook itself. Validated at load time so that missing keys surface
    immediately rather than at render time.
    """

    card_links        : dict[str, str]
    dropdown_label    : str
    fallback_location : str
    job_zones         : dict[str, str]
    job_zones_abbr    : dict[str, str]
    skill_types       : dict[str, str]
    spinner_text      : str
    tab_names         : dict[str, str]
    upload_label      : str


class MatchMetrics(BaseModel, extra="forbid"):
    """
    Core match quality metrics derived from cluster distances.
    """

    confidence    : int = Field(ge=0, le=100)
    job_zones     : dict[int, int]
    proximity     : dict[str, float]
    runner_up_gap : float
    sectors       : dict[str, str]
    top5          : dict[str, float]

    @field_validator("job_zones", mode="after")
    @classmethod
    def _sort_job_zones(cls, v: dict[int, int]) -> dict[int, int]:
        return dict(sorted(v.items()))

    @classmethod
    def from_result(cls, result: MatchResult) -> MatchMetrics:
        """
        Derive match confidence, top-5 radar, proximity bars, and Job Zone
        distribution.

        Args:
            result: Match result from resume projection.
        """
        cds      = result.cluster_distances
        max_dist = result.max_distance
        return cls(
            confidence    = result.confidence,
            job_zones     = Counter(cd.job_zone for cd in cds[:10]),
            proximity     = {cd.soc_title: round(cd.distance, 3) for cd in cds},
            runner_up_gap = round(cds[1].distance - cds[0].distance, 3) if len(cds) > 1 else 0,
            sectors       = {cd.soc_title: cd.sector for cd in cds},
            top5          = {
                cd.soc_title: round(100 * (1 - cd.distance / max_dist), 1)
                for cd in cds[:5]
            }
        )


class MlMetrics(BaseModel, extra="forbid"):
    """
    Diagnostic metrics for the ML Internals tab.

    Computed lazily on first access because brokerage centrality, silhouette
    analysis, and pairwise centroid distances are only needed when the user
    opens the ML Internals tab.
    """

    brokerage          : SectorRanking
    cluster_count      : int
    cluster_heatmap    : dict[str, list[float]]
    cluster_profiles   : dict[str, dict]
    company_count      : int
    component_count    : int
    corpus_size        : int
    edge_count         : int
    edge_weights       : list[float]
    embedding_model    : str
    pairwise_distances : dict[str, list[float]]
    sector_sizes       : dict[str, int]
    silhouette         : SectorRanking
    soc_heatmap        : dict[str, list[float]]
    treemap            : HierarchyData
    variance           : VarianceBreakdown

    @property
    def median_silhouette(self) -> float:
        """
        Median silhouette coefficient across all clusters.
        """
        return round(sorted(self.silhouette.values)[len(self.silhouette.values) // 2], 3)

    @property
    def stat_values(self) -> list[str]:
        """
        Formatted stat strings in `stat_labels` display order.
        """
        return [
            f"{self.corpus_size:,}",
            str(self.cluster_count),
            str(self.component_count),
            str(self.edge_count),
            f"{self.variance.total:.1f}%",
            str(self.median_silhouette),
            str(self.company_count)
        ]

    @classmethod
    def from_pipeline(cls, pipeline: Chalkline) -> MlMetrics:
        """
        Compute ML diagnostic metrics from the fitted pipeline.

        Imports networkx locally because brokerage centrality is only needed
        when the ML Internals tab opens.

        Args:
            pipeline: Fitted Chalkline pipeline instance.
        """
        clusters  = pipeline.clusters
        ratio     = pipeline.matcher.svd.explained_variance_ratio_
        brokerage = SectorRanking.from_ranking(clusters, pipeline.graph.brokerage)

        soc_heatmap = {
            clusters[cid].display_label: [
                round(float(v), 3) for v in clusters.soc_similarity[i]
            ]
            for i, cid in enumerate(clusters.cluster_ids)
        }

        cluster_labels = [clusters[cid].soc_title for cid in clusters.cluster_ids]
        cluster_heatmap = dict(zip(
            cluster_labels, clusters.cosine_similarity_matrix
        ))

        return cls(
            brokerage          = brokerage,
            cluster_count      = len(clusters),
            cluster_heatmap    = cluster_heatmap,
            cluster_profiles   = clusters.profile_map,
            company_count      = clusters.company_count,
            component_count    = pipeline.config.component_count,
            corpus_size        = pipeline.corpus_size,
            edge_count         = pipeline.graph.edge_count,
            edge_weights       = pipeline.graph.edge_weights,
            embedding_model    = pipeline.config.embedding_model,
            pairwise_distances = clusters.pairwise_distances,
            sector_sizes       = clusters.sector_sizes,
            silhouette         = SectorRanking.from_tuples(clusters.silhouette_scores),
            soc_heatmap        = soc_heatmap,
            treemap            = HierarchyData.from_clusters(clusters),
            variance           = VarianceBreakdown.from_svd(ratio)
        )


class ProcessStep(BaseModel, extra="forbid"):
    """
    One step in the pipeline process flow diagram.
    """

    detail : str
    label  : str
    number : str

    def render(self, **kwargs) -> tuple[str, str, str]:
        """
        Format detail and return `(number, label, detail)` for
        `process_flow()`.
        """
        return self.number, self.label, self.detail.format(**kwargs)


class SectionContent(BaseModel, extra="forbid"):
    """
    Title and description for one chart section.
    """

    title: str

    description: str = ""


class SectorMetrics(BaseModel, extra="forbid"):
    """
    Sector affinity scores from cluster distance averaging.
    """

    scores: dict[str, float]

    @classmethod
    def from_result(cls, result: MatchResult) -> SectorMetrics:
        """
        Average cluster distances by sector into affinity scores.

        Args:
            result: Match result with per-cluster distances.
        """
        return cls(
            scores = {
                s: round((1 - fmean(d) / result.max_distance) * 100, 1)
                for s, d in result.distances_by_sector.items()
            }
        )


class SectorRanking(BaseModel, extra="forbid"):
    """
    Positionally aligned cluster metric ranked by descending score.

    Used for brokerage centrality and silhouette coefficient bar charts
    where each bar needs a label, sector color, and value.
    """

    labels  : list[str]
    sectors : list[str]
    values  : list[float]

    @classmethod
    def from_ranking(
        cls,
        clusters : Clusters,
        ranking  : list[tuple[int, float]]
    ) -> SectorRanking:
        """
        Build from (cluster_id, score) pairs by resolving each cluster's
        title and sector.

        Args:
            clusters : Fitted cluster collection for ID lookup.
            ranking  : Pre-sorted (cluster_id, score) pairs.
        """
        resolved = [(clusters[cid], v) for cid, v in ranking]
        return cls(
            labels  = [c.soc_title for c, _ in resolved],
            sectors = [c.sector    for c, _ in resolved],
            values  = [round(v, 4) for _, v in resolved]
        )

    @classmethod
    def from_tuples(
        cls,
        data: list[tuple[str, str, float]]
    ) -> SectorRanking:
        """
        Unzip a list of (label, sector, value) tuples into parallel lists.

        Args:
            data: Pre-sorted (label, sector, value) triples.
        """
        return cls(
            labels  = [label  for label, _, _  in data],
            sectors = [sector for _, sector, _ in data],
            values  = [value  for _, _, value  in data]
        )


class SkillMetrics(BaseModel, extra="forbid"):
    """
    Grouped skill analysis for the Resume Feedback tab.

    All properties derive from `skill_groups`, which is
    `MatchResult.tasks_by_type` passed straight through. Similarity
    histogram data comes from `MatchResult` directly in the render.
    """

    skill_groups : dict[str, list[ScoredTask]]

    @property
    def gap_scatter_points(self) -> list[GapScatterPoint]:
        """
        Bubble scatter data for gap skills, sized by type frequency.
        """
        gaps_by_type = {
            stype: [s for s in skills if not s.demonstrated]
            for stype, skills in self.skill_groups.items()
        }
        return [
            GapScatterPoint(
                frequency = len(gaps),
                magnitude = max(0.0, round(100 - s.pct, 1)),
                text      = s.name
            )
            for gaps in gaps_by_type.values()
            for s in gaps
        ]

    @property
    def demonstration_rates(self) -> dict[str, float]:
        """
        Percentage of demonstrated skills per O*NET type.
        """
        return {
            stype: round(
                100 * sum(s.demonstrated for s in skills) / len(skills), 1
            )
            for stype, skills in self.skill_groups.items()
            if skills
        }

    @property
    def overall_averages(self) -> list[float]:
        """
        Mean score across all skills per type.
        """
        return [
            round(fmean(s.pct for s in skills), 1)
            for skills in self.skill_groups.values()
        ]

    @property
    def stat_values(self) -> list[str]:
        """
        Formatted stat strings in `stat_labels` display order.
        """
        demo  = sum(s.demonstrated for v in self.skill_groups.values() for s in v)
        total = sum(len(v) for v in self.skill_groups.values())
        return [
            str(demo),
            str(total - demo),
            str(len(self.skill_groups)),
            f"{round(100 * demo / total)}%" if total else "0%",
            str(total)
        ]

    @property
    def top_gaps(self) -> list[ScoredTask]:
        """
        Top 5 gap skills by deficit (lowest similarity).
        """
        return sorted(
            [s for v in self.skill_groups.values() for s in v if not s.demonstrated],
            key=lambda s: s.similarity
        )[:5]

    @property
    def top_strengths(self) -> list[ScoredTask]:
        """
        Top 5 demonstrated skills by similarity score.
        """
        return sorted(
            [s for v in self.skill_groups.values() for s in v if s.demonstrated],
            key=lambda s: s.similarity,
            reverse=True
        )[:5]

    @property
    def strength_averages(self) -> list[float]:
        """
        Mean score across demonstrated skills per type.
        """
        return [
            round(fmean(s.pct for s in skills if s.demonstrated), 1)
            if any(s.demonstrated for s in skills) else 0.0
            for skills in self.skill_groups.values()
        ]

    @classmethod
    def from_result(cls, result: MatchResult) -> SkillMetrics:
        """
        Group scored tasks by O*NET type for display.

        Args:
            result: Match result with scored tasks.
        """
        return cls(skill_groups=result.tasks_by_type)


class SplashMetrics(BaseModel, extra="forbid"):
    """
    Pre-computed statistics for the splash page.
    """

    stat_values : list[str]

    @classmethod
    def from_pipeline(
        cls,
        labor    : LaborLoader,
        pipeline : Chalkline
    ) -> SplashMetrics:
        """
        Compute splash page statistics from the fitted pipeline and typed
        labor records.

        Args:
            labor    : BLS labor market data.
            pipeline : Fitted Chalkline pipeline instance.
        """
        return cls(
            stat_values = [
                f"{pipeline.corpus_size:,}",
                str(pipeline.clusters.company_count),
                str(pipeline.clusters.location_count),
                str(len(pipeline.clusters)),
                f"{labor.total_employment:,}",
                f"${labor.median_annual_wage:,.0f}",
                str(labor.total_bright_outlook),
                str(pipeline.graph.edge_count)
            ]
        )


class TabContent(BaseModel, extra="ignore"):
    """
    Validated content loaded from a tab's `content.toml`.

    Sections are keyed by name and accessed via `content.sections["key"]`.
    Hero is optional because not every tab has one.
    """

    chart_labels    : dict[str, str]            = {}
    chart_lists     : dict[str, list[str]]      = {}
    columns         : dict[str, dict[str, str]] = {}
    directions      : dict[str, str]            = {}
    empty_message   : str                       = ""
    fallbacks       : dict[str, str]            = {}
    hero            : HeroContent               = HeroContent(text="")
    info            : str                       = ""
    labor_stats     : dict[str, str]            = {}
    labor_templates : LaborTemplates | None     = None
    process_steps   : list[ProcessStep]         = []
    sections        : dict[str, SectionContent] = {}
    stat_labels     : list[str]                 = []
    tagline         : str                       = ""
    title           : str                       = ""

    @model_validator(mode="after")
    def _normalize_stat_labels(self) -> Self:
        if not self.stat_labels and "stat_labels" in self.chart_lists:
            self.stat_labels = self.chart_lists.pop("stat_labels")
        return self

    @field_validator("info", mode="before")
    @classmethod
    def _unwrap_info(cls, v) -> str | None:
        return v["text"] if isinstance(v, dict) else v

    def section(self, key: str, **kwargs) -> tuple[str, str]:
        """
        Format a section's description and title for `header()`.

        Returns `(description, title)` to match `header()`'s alphabetized
        parameter order.
        """
        s = self.sections[key]
        return s.description.format(**kwargs), s.title.format(**kwargs)


class Trace(NamedTuple):
    """
    Named x/y series for bar, combo, and grouped charts.
    """

    x          : list
    y          : list

    color_role : str = ""
    name       : str = ""


class VarianceBreakdown(BaseModel, extra="forbid"):
    """
    SVD explained variance packaged for the variance bar chart and
    cumulative overlay.
    """

    components : list[float]
    total      : float

    @property
    def labels(self) -> list[str]:
        """
        Component axis labels (PC1, PC2, ...).
        """
        return [f"PC{i+1}" for i in range(len(self.components))]

    @property
    def trace(self) -> Trace:
        """
        Cumulative variance overlay line for the bar chart.
        """
        cumulative = [round(v, 2) for v in accumulate(self.components)]
        return Trace(self.labels, cumulative)

    @classmethod
    def from_svd(cls, ratios: Iterable[float]) -> VarianceBreakdown:
        """
        Build from raw SVD explained variance ratios (0-1 scale).

        Args:
            ratios: Per-component variance fractions from
                    `TruncatedSVD.explained_variance_ratio_`.
        """
        components = [round(v * 100, 2) for v in ratios]
        return cls(
            components = components,
            total      = round(sum(components), 2)
        )
