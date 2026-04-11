"""
Display-layer data models for the Chalkline career report.

Each model defines a data shape consumed by tab renderers and chart
builders, with a `from_` factory classmethod that constructs the model
from raw pipeline state.
"""

from collections     import Counter, defaultdict
from collections.abc import Iterable
from datetime        import date
from operator        import attrgetter
from pydantic        import BaseModel
from statistics      import median
from typing          import NamedTuple, Self

from chalkline.collection.schemas    import Posting
from chalkline.matching.schemas      import MatchResult, ScoredTask
from chalkline.pathways.clusters     import Cluster, Clusters
from chalkline.pathways.loaders      import LaborLoader, StakeholderReference
from chalkline.pathways.schemas      import Credential
from chalkline.pipeline.orchestrator import Chalkline


class JobPostingMetrics(BaseModel, extra="forbid"):
    """
    Aggregated posting statistics for the matched career family.
    """

    by_month     : dict[str, int]
    companies    : dict[str, int]
    descriptions : dict[str, int]
    locations    : dict[str, int]
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

        top_15    = lambda c: dict(c.most_common(15))
        companies = Counter(p.company  for p in postings if p.company)
        locations = Counter(p.location for p in postings if p.location)
        by_date   = attrgetter("date_posted")
        dated     = sorted(filter(by_date, postings), key=by_date, reverse=True)
        today     = date.today()

        agc_names = [m["name"] for m in reference.agc_members]

        agc_matched = {
            c for c in companies
            if get_close_matches(c, agc_names, n=1, cutoff=0.7)
        }

        by_month = dict(sorted(Counter(
            p.date_posted.strftime("%Y-%m")
            for p in postings
            if p.date_posted
        ).items()))

        descriptions = dict(Counter(
            w for p in postings
            for word in p.description.split()
            if  len(w := word.strip(".,;:!?()[]\"'").lower()) > 3
            and zipf_frequency(w, "en") < 4.5
        ).most_common(20))

        titles = dict(Counter(
            w for p in postings
            for word in p.title.split()
            if  len(w := word.strip(".,;:!?()[]").lower()) > 2
            and zipf_frequency(w, "en") < 5.0
        ).most_common(25))

        stat_values = [
            str(len(postings)),
            str(len(companies)),
            str(len(locations)),
            str(len(agc_matched)),
            f"{(today - dated[0].date_posted).days}d" if dated else "N/A",
            str(len(dated))
        ]

        return cls(
            by_month     = by_month,
            companies    = top_15(companies),
            descriptions = descriptions,
            locations    = top_15(locations),
            recent       = dated[:12],
            stat_values  = stat_values,
            titles       = titles
        )


class JobZoneBreakdown(NamedTuple):
    """
    Cluster counts by Job Zone with cross-tabulated sector breakdown.

    Groups the two related views the methods tab needs together so they
    travel as one field on `MlMetrics` instead of two parallel fields.
    """

    counts : dict[int, int]
    matrix : dict[str, list[int]]

    def labeled_counts(self, names: dict[int, str]) -> dict[str, int]:
        """
        Apply a level-to-name map to produce a label-keyed count dict.
        """
        return {names[level]: count for level, count in self.counts.items()}


class Labels(BaseModel, extra="forbid"):
    """
    Shared display labels loaded from `tabs/shared/labels.toml`.

    Cross-cutting text used by layout helpers and the Marimo notebook
    itself. Validated at load time so that missing keys surface
    immediately rather than at render time.
    """

    fallback_location : str
    job_zones         : dict[int, str]
    map_you_are_here  : str
    spinner_text      : str
    tab_names         : dict[str, str]
    upload_label      : str


class MapGeometry(BaseModel, extra="forbid"):
    """
    Static layout geometry for the pathway map widget.

    Owns every pixel-level constant, opacity tier, and text-fit
    threshold the layout factory consumes, so the JS renderer receives
    a consistent dimensions payload and the Python side has a single
    source of truth instead of scattered literals.
    """

    col_gap              : int               = 90
    default_total_width  : int               = 600
    edge_midpoint_offset : int               = 30
    hop_opacities        : tuple[float, ...] = (1, 1, 0.5, 0.2)
    node_h               : int               = 52
    node_spacing         : int               = 25
    node_w               : int               = 140
    pad                  : int               = 40
    sector_gap           : int               = 60
    title_char_limit     : int               = 22
    top_pad              : int               = 50

    @property
    def col_pitch(self) -> int:
        """
        Horizontal pixels between adjacent Job Zone column centers.
        """
        return self.node_w + self.col_gap

    @property
    def dimensions(self) -> dict[str, int]:
        """
        Card height, card width, and outer padding for the JS payload.
        """
        return self.model_dump(include={"node_h", "node_w", "pad"})

    @property
    def max_hop_index(self) -> int:
        """
        Largest valid index into `hop_opacities` for clamping BFS hops.
        """
        return len(self.hop_opacities) - 1

    @property
    def row_height(self) -> int:
        """
        Vertical pixels per node row including inter-node spacing.
        """
        return self.node_h + self.node_spacing


class MapLayout(NamedTuple):
    """
    Precomputed spatial layout for the career pathway map widget.

    Groups deterministic node positions, Job Zone column metadata, and
    SVG dimensions so they travel together from `from_clusters` through
    serialization into the D3 renderer.
    """

    columns      : list[dict]
    positions    : dict[int, dict[str, int]]
    total_height : int
    total_width  : int

    def labeled_columns(self, names: dict[int, str]) -> list[dict]:
        """
        Pair each column's x position with its label from `names`.

        Resolves Job Zone level integers to display strings so the map
        widget can drop the result straight into its JSON payload
        without re-iterating `columns` at the call site.
        """
        return [
            {
                "label" : names[c["job_zone"]],
                "x"     : c["x"]
            }
            for c in self.columns
        ]

    @classmethod
    def from_clusters(cls, clusters: Clusters, geometry: MapGeometry) -> MapLayout:
        """
        Deterministic grid layout placing clusters in Job Zone columns
        with vertical sector banding.

        Sectors stack top-to-bottom in `clusters.sectors` order, each
        sized by the busiest column it contains so columns inside a band
        share a baseline.

        Args:
            clusters : Fitted cluster container with metadata.
            geometry : Pixel constants for nodes, gaps, and padding.
        """
        def col_center(level: int) -> int:
            return (level - 1) * geometry.col_pitch + geometry.node_w // 2

        groups = defaultdict(list)
        for cluster in clusters.values():
            groups[cluster.job_zone, cluster.sector].append(cluster.cluster_id)

        max_per_sector = defaultdict(int)
        for (_, sector), node_ids in groups.items():
            max_per_sector[sector] = max(max_per_sector[sector], len(node_ids))

        band_y   = {}
        y_cursor = geometry.top_pad
        for sector in clusters.sectors:
            if sector not in max_per_sector:
                continue
            band_y[sector] = y_cursor
            y_cursor += (
                max_per_sector[sector] * geometry.row_height + geometry.sector_gap
            )

        positions = {}
        for (level, sector), node_ids in groups.items():
            base_y = band_y.get(sector, geometry.top_pad)
            for i, node_id in enumerate(node_ids):
                positions[node_id] = {
                    "x" : col_center(level),
                    "y" : base_y + i * geometry.row_height + geometry.node_h // 2
                }

        job_zone_levels = sorted({level for level, _ in groups})
        columns = [
            {
                "job_zone" : level,
                "x"        : col_center(level)
            }
            for level in job_zone_levels
        ]
        total_width = (
            col_center(max(job_zone_levels)) + geometry.node_w // 2
            if job_zone_levels else geometry.default_total_width
        )

        return cls(
            columns      = columns,
            positions    = positions,
            total_height = y_cursor,
            total_width  = total_width
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
    embed_dim          : int
    embedding_model    : str
    job_zone           : JobZoneBreakdown
    pairwise_distances : dict[str, list[float]]
    sector_sizes       : dict[str, int]
    silhouette         : SectorRanking
    soc_heatmap        : dict[str, list[float]]
    variance           : VarianceBreakdown

    @property
    def funnel_stages(self) -> dict[str, int]:
        """
        Pipeline-narrowing stages in display order for the data funnel.
        """
        return {
            "Raw Postings"   : self.corpus_size,
            "Embedding Dims" : self.embed_dim,
            "SVD Components" : self.component_count,
            "Clusters"       : self.cluster_count,
            "Pathway Edges"  : self.edge_count
        }

    @property
    def median_silhouette(self) -> float:
        """
        Median silhouette coefficient across all clusters.
        """
        return round(median(self.silhouette.values), 3)

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

    @property
    def template_kwargs(self) -> dict[str, object]:
        """
        Substitution kwargs for `ProcessStep.detail` and any TOML
        descriptions templated against pipeline state.
        """
        return self.model_dump(include={
            "cluster_count", "component_count", "corpus_size",
            "edge_count", "embedding_model"
        }) | {"total_variance": self.variance.total}

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

        job_zone_counts : Counter = Counter()
        sector_pairs    : Counter = Counter()
        for c in clusters.values():
            job_zone_counts[c.job_zone]        += 1
            sector_pairs[c.sector, c.job_zone] += 1
        job_zone_levels = sorted(job_zone_counts)
        job_zone        = JobZoneBreakdown(
            counts = dict(sorted(job_zone_counts.items())),
            matrix = {
                sector: [sector_pairs[sector, level] for level in job_zone_levels]
                for sector in clusters.sectors
            }
        )

        soc_heatmap     : dict[str, list[float]] = {}
        cluster_heatmap : dict[str, list[float]] = {}
        for i, cluster in enumerate(clusters.values()):
            soc_heatmap[cluster.display_label] = [
                round(float(v), 3) for v in clusters.soc_similarity[i]
            ]
            cluster_heatmap[cluster.soc_title] = clusters.cosine_similarity_matrix[i]

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
            embed_dim          = pipeline.embed_dim,
            embedding_model    = pipeline.config.embedding_model,
            job_zone           = job_zone,
            pairwise_distances = clusters.pairwise_distances,
            sector_sizes       = clusters.sector_sizes,
            silhouette         = SectorRanking.from_tuples(clusters.silhouette_scores),
            soc_heatmap        = soc_heatmap,
            variance           = VarianceBreakdown.from_svd(ratio)
        )


class ProcessStep(BaseModel, extra="forbid"):
    """
    One step in the pipeline process flow diagram.
    """

    detail : str
    label  : str
    number : str

    def render(self, **kwargs) -> Self:
        """
        Return a copy with `detail` formatted against the given kwargs.

        Used by `Layout.process_flow` to substitute corpus-level values
        like `{corpus_size}` and `{cluster_count}` into each step's
        detail line at render time.
        """
        return self.model_copy(update={"detail": self.detail.format(**kwargs)})


class RouteDetail(NamedTuple):
    """
    Joined route data for the Map tab's route card.

    Holds the source and destination `Cluster` objects for dot-notation
    access to SOC titles, sectors, Job Zones, and postings. Constructed
    via `from_selection`, which resolves the matched cluster's adjacent
    reach first and falls back to the widest-path tree query.
    """

    credentials      : list[Credential]
    destination      : Cluster
    destination_wage : float | None
    path             : list[int]
    scored_tasks     : list[ScoredTask]
    source           : Cluster
    source_wage      : float | None
    weight           : float

    @property
    def is_multi_hop(self) -> bool:
        """
        Whether the route traverses intermediate clusters.
        """
        return len(self.path) > 2

    @property
    def step_count(self) -> int:
        """
        Number of hops the route traverses (edges, not nodes).
        """
        return len(self.path) - 1

    @property
    def top_credential(self) -> Credential | None:
        """
        Fastest-path credential (first in the ranked list), or `None`.
        """
        return self.credentials[0] if self.credentials else None

    @property
    def top_gaps(self) -> list[ScoredTask]:
        """
        Five largest skill gaps by deficit (lowest similarity).
        """
        return sorted(
            (t for t in self.scored_tasks if not t.demonstrated),
            key=lambda t: t.similarity
        )[:5]

    @property
    def top_strengths(self) -> list[ScoredTask]:
        """
        Five strongest demonstrated skills. `scored_tasks` is already sorted
        by descending similarity, so taking the first five matches.
        """
        return [t for t in self.scored_tasks if t.demonstrated][:5]

    @property
    def transition_summary(self) -> str:
        """
        Source-to-destination SOC titles joined by a right arrow.
        """
        return f"{self.source.soc_title} \u2192 {self.destination.soc_title}"

    @property
    def wage_delta(self) -> float | None:
        """
        Annual wage difference from source to destination.
        """
        if self.source_wage is None or self.destination_wage is None:
            return None
        return self.destination_wage - self.source_wage

    @classmethod
    def from_selection(
        cls,
        labor       : LaborLoader,
        pipeline    : Chalkline,
        profile     : Cluster,
        result      : MatchResult,
        selected_id : int
    ) -> Self | None:
        """
        Build a route from a clicked map node, or `None` if the click is
        invalid (no selection, the matched cluster itself, or no path
        connects the source and destination).

        Tries the matched cluster's adjacent reach first for a single-hop
        edge; falls back to the widest-path multi-hop tree query.
        """
        if selected_id < 0 or selected_id == result.cluster_id:
            return None

        destination = pipeline.clusters[selected_id]
        adjacent    = next(
            (
                e for e in result.reach.advancement + result.reach.lateral
                if e.cluster_id == selected_id
            ),
            None
        )
        if adjacent is not None:
            edges = [adjacent]
            path  = [profile.cluster_id, selected_id]
        else:
            path  = pipeline.graph.try_widest_path(profile.cluster_id, selected_id)
            edges = pipeline.graph.path_edges(path) if path else []

        if not edges:
            return None

        return cls(
            credentials      = [c for e in edges for c in e.credentials],
            destination      = destination,
            destination_wage = labor.wage(destination.soc_title),
            path             = path,
            scored_tasks     = pipeline.matcher.score_destination(destination),
            source           = profile,
            source_wage      = labor.wage(profile.soc_title),
            weight           = min((e.weight for e in edges), default=0)
        )


class SectionContent(BaseModel, extra="forbid"):
    """
    Title and description for one chart section.
    """

    title: str

    description: str = ""


class SectorRanking(BaseModel, extra="forbid"):
    """
    Positionally aligned cluster metric ranked by descending score.

    Used for brokerage centrality and silhouette coefficient bar charts
    where each bar needs a label, sector color, and value.
    """

    labels  : list[str]
    sectors : list[str]
    values  : list[float]

    @property
    def value_map(self) -> dict[str, float]:
        """
        Label-to-value mapping for chart factories that take `data=`.
        """
        return dict(zip(self.labels, self.values))

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
    def from_tuples(cls, data: list[tuple[str, str, float]]) -> SectorRanking:
        """
        Unzip a list of (label, sector, value) tuples into parallel lists.

        Args:
            data: Pre-sorted (label, sector, value) triples.
        """
        labels, sectors, values = zip(*data) if data else ([], [], [])
        return cls(
            labels  = list(labels),
            sectors = list(sectors),
            values  = list(values)
        )


class SplashMetrics(BaseModel, extra="forbid"):
    """
    Pre-upload corpus statistics rendered on the splash page.

    Mirrors the schema-factory pattern used by `JobPostingMetrics` and
    `MlMetrics` so the splash render stays a thin formatter rather than
    inlining eight ad-hoc casts.
    """

    stat_values: list[str]

    @classmethod
    def from_corpus(cls, labor: LaborLoader, pipeline: Chalkline) -> Self:
        """
        Format the eight splash stats from pipeline and labor state.

        Args:
            labor    : BLS labor market data.
            pipeline : Fitted Chalkline pipeline instance.
        """
        clusters = pipeline.clusters
        return cls(stat_values=[
            f"{pipeline.corpus_size:,}",
            f"{clusters.company_count}",
            f"{clusters.location_count}",
            f"{len(clusters)}",
            f"{labor.total_employment:,}",
            f"${labor.median_annual_wage:,.0f}",
            f"{labor.total_bright_outlook}",
            f"{pipeline.graph.edge_count}"
        ])


class TabContent(BaseModel, extra="ignore"):
    """
    Validated content loaded from a tab's `content.toml`.

    Sections are keyed by name and accessed via `content.sections["key"]`.
    """

    chart_labels  : dict[str, str]            = {}
    fallbacks     : dict[str, str]            = {}
    info          : str                       = ""
    process_steps : list[ProcessStep]         = []
    sections      : dict[str, SectionContent] = {}
    stat_labels   : list[str]                 = []
    tagline       : str                       = ""
    title         : str                       = ""

    def section(self, key: str, **kwargs) -> tuple[str, str]:
        """
        Format a section's description and title for `header()`.

        Returns `(description, title)` to match `header()`'s alphabetized
        parameter order.
        """
        s = self.sections[key]
        return s.description.format(**kwargs), s.title.format(**kwargs)


class VarianceBreakdown(BaseModel, extra="forbid"):
    """
    SVD explained variance packaged for the variance bar chart and
    cumulative overlay.
    """

    components : list[float]
    total      : float

    @property
    def components_dict(self) -> dict[str, float]:
        """
        Per-component variance percentages keyed by axis label.
        """
        return dict(zip(self.labels, self.components))

    @property
    def cumulative(self) -> list[float]:
        """
        Running sum of component variances, rounded to two decimals.
        """
        from itertools import accumulate
        return [round(v, 2) for v in accumulate(self.components)]

    @property
    def cumulative_dict(self) -> dict[str, float]:
        """
        Running cumulative variance percentages keyed by axis label.
        """
        return dict(zip(self.labels, self.cumulative))

    @property
    def labels(self) -> list[str]:
        """
        Component axis labels (PC1, PC2, ...).
        """
        return [f"PC{i+1}" for i in range(len(self.components))]

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
