"""
Display-layer data models for the Chalkline career report.

Each model defines a data shape consumed by tab renderers and chart
builders, with a `from_` factory classmethod that constructs the model
from raw pipeline state.
"""

import numpy as np

from collections     import Counter, defaultdict
from collections.abc import Iterable
from datetime        import date
from difflib         import get_close_matches
from heapq           import nlargest, nsmallest
from itertools       import accumulate, chain, islice
from math            import log
from operator        import attrgetter
from pydantic        import BaseModel
from statistics      import median
from typing          import ClassVar, NamedTuple, Self

from chalkline.collection.schemas    import Posting
from chalkline.matching.schemas      import MatchResult, ScoredTask
from chalkline.pathways.clusters     import Cluster, Clusters
from chalkline.pathways.graph        import CareerPathwayGraph
from chalkline.pathways.loaders      import LaborLoader, StakeholderReference
from chalkline.pathways.schemas      import Credential
from chalkline.pipeline.encoder      import SentenceEncoder
from chalkline.pipeline.orchestrator import Chalkline


class DatedPoint(NamedTuple):
    """
    One posting plotted on the timeline strip scatter.

    `label` is the company name when present, falling back to the
    posting title so every dot has hover text.
    """

    date  : date
    label : str


class DistinctiveVocabulary(BaseModel, extra="forbid"):
    """
    Words distinctive to the matched career family vs the full corpus,
    bucketed into three tiers by document frequency.

    Treats every cluster as a TF-IDF document and combines each
    posting's title and description as one bag of words. The tiers
    split eligible words by how many clusters each one appears in:
    `unique to this family` for words found only here, `rare across
    the corpus` for words found in two to four families, and `notable
    vocabulary` for words found in five or more families that still
    rank high inside the matched family. Each tier is sized
    independently so a sparser tier never gets visually crowded by a
    denser one.

    `tiers` is laid out in the same shape `Charts.faceted_treemap`
    accepts via its `facets=` parameter, so the chart caller can pass
    `vocabulary.tiers` directly. Each tier maps a word to its in-cluster
    count, ordered by descending TF-IDF score; empty tiers are dropped
    in the factory so the chart never receives a blank facet.
    """

    tiers: dict[str, dict[str, int]]

    _corpus_cache: ClassVar[dict[int, tuple[dict[int, Counter], Counter]]] = {}

    @classmethod
    def from_cluster(
        cls,
        cluster     : Cluster,
        clusters    : Clusters,
        tier_labels : list[str],
        min_count   : int = 3,
        per_tier    : int = 12
    ) -> Self:
        """
        Rank vocabulary by TF-IDF over clusters-as-documents and bucket
        into three distinctiveness tiers.

        The corpus-wide `(per_cluster, doc_freq)` pair depends only on
        the cluster collection, not on `cluster`, so it is cached at the
        class level keyed by `id(clusters)`. Per-render work collapses
        to picking the matched cluster's Counter and ranking its
        eligible words by TF-IDF. Pipeline rebuilds invalidate the cache
        naturally because the new `clusters` instance has a different
        identity.

        Args:
            cluster     : Matched career family.
            clusters    : Fitted cluster container with all postings.
            tier_labels : Display labels for the unique, rare, and
                          notable distinctiveness tiers in that order,
                          sourced from the consuming tab's TOML.
            min_count   : Minimum in-cluster occurrences for a word to
                          be eligible, suppressing single-occurrence
                          noise that would otherwise top the ranking.
            per_tier    : Maximum words to keep per distinctiveness tier.
        """
        cache_key = id(clusters)
        if cache_key not in cls._corpus_cache:
            per_cluster = {
                c.cluster_id: Counter(chain.from_iterable(c.distinctive_tokens))
                for c in clusters.values()
            }
            doc_freq = Counter(chain.from_iterable(per_cluster.values()))
            cls._corpus_cache[cache_key] = (per_cluster, doc_freq)

        per_cluster, doc_freq = cls._corpus_cache[cache_key]
        target                = per_cluster[cluster.cluster_id]

        cluster_count = len(per_cluster)
        target_total  = target.total() or 1

        eligible = {
            word: count for word, count in target.items() if count >= min_count
        }
        scores = {
            word: (count / target_total) * log(cluster_count / doc_freq[word])
            for word, count in eligible.items()
        }

        unique, rare, notable = tier_labels
        tier_predicates = [
            (unique,  lambda df: df == 1),
            (rare,    lambda df: 2 <= df <= 4),
            (notable, lambda df: df >= 5)
        ]

        return cls(tiers={
            label: top
            for label, predicate in tier_predicates
            if (top := {
                w: eligible[w] for w in nlargest(
                    per_tier,
                    (w for w in eligible if predicate(doc_freq[w])),
                    key = scores.__getitem__
                )
            })
        })


class JobPostingMetrics(BaseModel, extra="forbid"):
    """
    Aggregated posting statistics for the matched career family.
    """

    companies   : dict[str, int]
    dated       : list[DatedPoint]
    freshness   : list[int]
    locations   : dict[str, int]
    recent      : list[Posting]
    stat_values : list[str]

    @classmethod
    def from_postings(
        cls,
        postings  : list[Posting],
        reference : StakeholderReference
    ) -> Self:
        """
        Aggregate posting data for one career family.

        Args:
            postings  : Posting objects from the matched cluster.
            reference : Stakeholder reference for AGC member lookup.
        """
        today     = date.today()
        by_date   = attrgetter("date_posted")
        dated     = sorted(filter(by_date, postings), key=by_date, reverse=True)
        companies = Counter(p.company for p in postings if p.company)
        locations = Counter(p.location for p in postings if p.location)

        agc_names   = [m["name"] for m in reference.agc_members]
        agc_matched = {
            c for c in companies
            if get_close_matches(c, agc_names, n=1, cutoff=0.7)
        }

        dated_points = [
            DatedPoint(date=p.date_posted, label=p.company or p.title)
            for p in dated
        ]
        freshness = [(today - p.date_posted).days for p in dated]

        return cls(
            companies   = dict(companies.most_common(15)),
            dated       = dated_points,
            freshness   = freshness,
            locations   = dict(locations.most_common(15)),
            recent      = dated[:12],
            stat_values = [
                str(len(postings)),
                str(len(companies)),
                str(len(locations)),
                str(len(agc_matched)),
                f"{freshness[0]}d" if freshness else "N/A",
                str(len(dated))
            ]
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

    @classmethod
    def from_clusters(cls, clusters: Clusters, geometry: MapGeometry) -> Self:
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

        max_per_sector: dict[str, int] = defaultdict(int)
        for (_, sector), node_ids in groups.items():
            max_per_sector[sector] = max(max_per_sector[sector], len(node_ids))

        active_sectors = [s for s in clusters.sectors if s in max_per_sector]
        *band_starts, y_cursor = accumulate(
            (max_per_sector[s] * geometry.row_height + geometry.sector_gap
             for s in active_sectors),
            initial = geometry.top_pad
        )
        band_y = defaultdict(lambda: geometry.top_pad, zip(active_sectors, band_starts))

        positions = {
            node_id: {
                "x" : col_center(level),
                "y" : band_y[sector] + i * geometry.row_height + geometry.node_h // 2
            }
            for (level, sector), node_ids in groups.items()
            for i, node_id in enumerate(node_ids)
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
            "cluster_count",
            "component_count",
            "corpus_size",
            "edge_count",
            "embedding_model"
        }) | {"total_variance": self.variance.total}

    @classmethod
    def from_pipeline(cls, pipeline: Chalkline) -> Self:
        """
        Compute ML diagnostic metrics from the fitted pipeline.

        Args:
            pipeline: Fitted Chalkline pipeline instance.
        """
        clusters  = pipeline.clusters
        ratio     = pipeline.matcher.svd.explained_variance_ratio_
        brokerage = SectorRanking.from_ranking(clusters, pipeline.graph.brokerage)

        job_zone_counts = Counter(c.job_zone for c in clusters.values())
        sector_pairs    = Counter((c.sector, c.job_zone) for c in clusters.values())
        job_zone        = JobZoneBreakdown(
            counts = (sorted_counts := dict(sorted(job_zone_counts.items()))),
            matrix = {
                sector: [sector_pairs[sector, level] for level in sorted_counts]
                for sector in clusters.sectors
            }
        )

        soc_heatmap     = {
            c.display_label: [round(float(v), 3) for v in row]
            for c, row in zip(clusters.values(), clusters.soc_similarity)
        }
        cluster_heatmap = {
            c.soc_title: row
            for c, row in zip(clusters.values(), clusters.cosine_similarity_matrix)
        }

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


class PostingProjection(BaseModel, extra="forbid"):
    """
    2D t-SNE projection of the matched cluster's posting embeddings,
    with k-means sub-clustering on the same vectors for coloring.

    Reads the per-posting embeddings already stored on the matched
    `Cluster` (no encoding work at render time), runs sklearn t-SNE
    with PCA initialization to map them to two dimensions, and runs
    k-means on the same high-dimensional vectors to discover sub-roles
    within the family. Coloring by k-means assignment surfaces actual
    semantic sub-structure rather than employer boilerplate, and pairs
    naturally with t-SNE since both are unsupervised methods.

    `series` is laid out in the same shape `Charts.category_scatter`
    accepts via its `data=` parameter, so the chart caller can pass
    `projection.series` directly without unpacking parallel arrays.
    """

    series: dict[str, dict[str, list]]

    _series_cache: ClassVar[dict[int, dict[str, dict[str, list]]]] = {}

    @classmethod
    def from_cluster(
        cls,
        cluster      : Cluster,
        min_postings : int = 5,
        n_subgroups  : int = 4,
        random_state : int = 42
    ) -> Self:
        """
        Project the cluster's pre-computed embeddings to 2D and color
        by k-means sub-cluster, labeling each sub-cluster with its top
        TF-IDF distinctive words.

        Result is cached at the class level keyed by `id(cluster)`, so
        re-renders that match the same career family (e.g. uploading a
        second resume in the same cluster) skip the t-SNE and k-means
        compute entirely. The Cluster instances live inside the cached
        pipeline, so their identities are stable for the session.

        Args:
            cluster      : Matched career family with stored per-posting
                           embeddings.
            min_postings : Below this size t-SNE is not meaningful;
                           returns an empty projection so the render
                           layer can skip the panel.
            n_subgroups  : Target number of k-means sub-clusters; the
                           actual count is capped at one third of the
                           posting count to keep groups non-trivial.
            random_state : Seed for reproducible projections.
        """
        if (cached := cls._series_cache.get(id(cluster))) is not None:
            return cls(series=cached)

        from sklearn.cluster  import KMeans
        from sklearn.manifold import TSNE

        embeddings = cluster.embeddings
        postings   = cluster.postings
        if len(postings) < min_postings:
            cls._series_cache[id(cluster)] = {}
            return cls(series={})

        perplexity = max(2.0, min(15.0, (len(postings) - 1) / 3))
        coords     = TSNE(
            init         = "pca",
            n_components = 2,
            perplexity   = perplexity,
            random_state = random_state
        ).fit_transform(embeddings)

        k = max(1, min(n_subgroups, len(postings) // 3))
        if k >= 2:
            assignments = KMeans(
                n_clusters   = k,
                n_init       = 10,
                random_state = random_state
            ).fit_predict(embeddings)
        else:
            assignments = np.zeros(len(postings), dtype=int)

        sub_counts: list[Counter] = [Counter() for _ in range(k)]
        for label, bag in zip(assignments, cluster.distinctive_tokens):
            sub_counts[label].update(bag)

        sub_doc_freq = Counter(chain.from_iterable(sub_counts))

        sub_labels: list[str] = []
        for index, counter in enumerate(sub_counts):
            total = counter.total() or 1
            top   = nlargest(
                2,
                ((w, c) for w, c in counter.items() if c >= 2 and sub_doc_freq[w] < k),
                key = lambda wc: (wc[1] / total) * log(k / sub_doc_freq[wc[0]])
            )
            sub_labels.append(
                " · ".join(w.title() for w, _ in top) if top else f"Sub-role {index + 1}"
            )

        series: dict[str, dict[str, list]] = defaultdict(lambda: {
            "hover" : [],
            "x"     : [],
            "y"     : []
        })
        for assignment, (x, y), p in zip(assignments, coords, postings):
            bucket = series[sub_labels[int(assignment)]]
            bucket["hover"].append(f"{p.title} · {p.company}" if p.company else p.title)
            bucket["x"].append(float(x))
            bucket["y"].append(float(y))

        cls._series_cache[id(cluster)] = dict(series)
        return cls(series=cls._series_cache[id(cluster)])


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


class RelevantCredentials(BaseModel, extra="forbid"):
    """
    Top concrete credentials per kind for the matched career family.

    Reuses the cluster-vector and credential-vector cosine machinery
    that powers per-edge credential enrichment in the graph layer.
    Instead of returning per-kind counts, the factory returns the actual
    `Credential` objects ranked by similarity and bucketed by kind, so
    the render layer can surface concrete examples (apprenticeship
    RAPIDS codes and hours, program institutions and links, recognizable
    certification labels) instead of meaningless catalog-side counts.
    Designed for reuse on other tabs (Splash, Your Match) without
    modification.
    """

    by_kind: dict[str, list[Credential]]

    @classmethod
    def from_cluster(
        cls,
        cluster  : Cluster,
        clusters : Clusters,
        graph    : CareerPathwayGraph,
        per_kind : int = 4
    ) -> Self:
        """
        Rank credentials by cosine similarity to the matched cluster
        and keep the top `per_kind` of each kind.

        Slices a single column from `graph.credential_similarity`,
        which the graph already computes once during construction for
        edge enrichment. The factory does no cosine work of its own;
        it just argsorts and bucket-walks the precomputed column.

        Args:
            cluster  : Matched career family.
            clusters : Fitted cluster container with stacked vectors.
            graph    : Fitted career pathway graph, source of both the
                       credential list and the precomputed similarity
                       matrix.
            per_kind : Number of top credentials to keep per kind.
        """
        with_vectors, _ = graph.credential_matrix
        column          = clusters.cluster_index[cluster.cluster_id]
        ranked_creds    = [
            with_vectors[i]
            for i in np.argsort(-graph.credential_similarity[:, column])
        ]
        return cls(by_kind={
            kind: list(islice((c for c in ranked_creds if c.kind == kind), per_kind))
            for kind in ("apprenticeship", "program", "certification")
        })


class RelevantJobBoards(BaseModel, extra="forbid"):
    """
    Job boards aligned with this career family via semantic similarity.

    Encodes each board's `focus + best_for + category` text via the
    pipeline's existing `SentenceEncoder` once at first call, caches
    the flattened board list and the pre-stacked vector matrix together
    on the class so subsequent renders skip both the encoding work and
    the per-render dict-to-matrix rebuild, then ranks boards by cosine
    similarity to the matched cluster's vector. Each returned board
    dict carries a `match_score` percentage so the render layer can
    surface a meaningful relevance signal instead of a near-binary
    keyword count. Designed for reuse on other tabs (Splash, Your
    Match) without modification.
    """

    boards: list[dict[str, object]]

    _board_cache: ClassVar[tuple[list, np.ndarray] | None] = None

    @classmethod
    def from_cluster(
        cls,
        cluster   : Cluster,
        clusters  : Clusters,
        encoder   : SentenceEncoder,
        reference : StakeholderReference,
        limit     : int = 6
    ) -> Self:
        """
        Rank job boards by cosine similarity to the matched cluster.

        Args:
            cluster   : Matched career family.
            clusters  : Fitted cluster container with stacked vectors.
            encoder   : Sentence encoder shared with the pipeline, used
                        once per session to embed every board's focus
                        text into the same space as the cluster vectors.
            reference : Stakeholder reference exposing the job board
                        catalog under `job_boards`.
            limit     : Maximum boards to return, ordered by descending
                        semantic similarity.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        if cls._board_cache is None:
            all_boards = list(chain.from_iterable(reference.job_boards.values()))
            cls._board_cache = (all_boards, encoder.encode([
                f"{b.get('focus', '')} {b.get('best_for', '')} {b.get('category', '')}"
                for b in all_boards
            ]))

        all_boards, matrix = cls._board_cache
        cluster_vector     = clusters.vector_map[cluster.cluster_id]
        similarities       = cosine_similarity(cluster_vector[None, :], matrix)[0]
        ranked             = np.argsort(-similarities)[:limit]
        return cls(boards=[
            {**all_boards[i], "match_score": round(float(similarities[i]) * 100)}
            for i in ranked
        ])


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
        return nsmallest(
            5,
            (t for t in self.scored_tasks if not t.demonstrated),
            key = attrgetter("similarity")
        )

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
        adjacent = next(
            (e for e in chain(result.reach.advancement, result.reach.lateral)
             if e.cluster_id == selected_id),
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
    ) -> Self:
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
    def from_tuples(cls, data: list[tuple[str, str, float]]) -> Self:
        """
        Unzip a list of (label, sector, value) tuples into parallel lists.

        Args:
            data: Pre-sorted (label, sector, value) triples.
        """
        return cls(
            labels  = [label  for label, _,      _     in data],
            sectors = [sector for _,     sector, _     in data],
            values  = [value  for _,     _,      value in data]
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
        return cls(stat_values=[
            f"{pipeline.corpus_size:,}",
            f"{pipeline.clusters.company_count}",
            f"{pipeline.clusters.location_count}",
            f"{len(pipeline.clusters)}",
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

    chart_labels      : dict[str, str]            = {}
    fallbacks         : dict[str, str]            = {}
    info              : str                       = ""
    process_steps     : list[ProcessStep]         = []
    sections          : dict[str, SectionContent] = {}
    stat_labels       : list[str]                 = []
    tagline           : str                       = ""
    tier_descriptions : dict[str, str]            = {}
    title             : str                       = ""

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
    def from_svd(cls, ratios: Iterable[float]) -> Self:
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
