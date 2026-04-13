"""
Display-layer data models for the Chalkline career report.

Each model defines a data shape consumed by tab renderers and chart
builders, with a `from_` factory classmethod that constructs the model
from raw pipeline state.
"""

import numpy as np

from collections     import Counter, defaultdict
from collections.abc import Iterable
from dataclasses     import dataclass
from datetime        import date
from heapq           import nlargest, nsmallest
from itertools       import accumulate, chain, islice
from math            import log
from operator        import attrgetter
from pydantic        import BaseModel
from statistics      import median
from typing          import ClassVar, NamedTuple, Self, TypedDict

from chalkline.collection.schemas    import Posting
from chalkline.matching.schemas      import MatchResult, ScoredTask
from chalkline.pathways.clusters     import Cluster, Clusters
from chalkline.pathways.graph        import CareerPathwayGraph
from chalkline.pathways.loaders      import LaborLoader, StakeholderReference
from chalkline.pathways.schemas      import Credential, Reach
from chalkline.pipeline.encoder      import SentenceEncoder
from chalkline.pipeline.orchestrator import Chalkline


class BoardMatch(TypedDict):
    """
    Job board dict enriched with a cosine-similarity match score.
    """

    best_for    : str
    category    : str
    focus       : str
    match_score : int
    name        : str


@dataclass
class CoverageBuilder:
    """
    Shared context for greedy set-cover across multiple gap-closure
    strategies within a single route.

    Holds the credential-to-gap-index mapping, credential lookup, and
    total gap count so each strategy call only specifies what varies:
    the strategy name, the uncovered set, and an optional seed.
    """

    active         : dict[str, set[int]]
    credential_map : dict[str, Credential]
    n_gaps         : int

    @property
    def best_item(self) -> PathItem | None:
        """
        Single credential covering the most gap tasks, or `None`.
        """
        return max(
            (
                PathItem.from_credential(len(gaps), c, label)
                for label, gaps in self.active.items()
                if (c := self.credential_map.get(label))
            ),
            key     = lambda p: p.coverage,
            default = None
        )

    @property
    def programs(self) -> dict[str, set[int]]:
        """
        Subset of `active` filtered to program-kind credentials.
        """
        return {
            l: s for l, s in self.active.items()
            if (c := self.credential_map.get(l)) and c.kind == "program"
        }

    def path(
        self,
        strategy  : str,
        uncovered : set[int],
        seed      : list[str] | None = None
    ) -> CredentialPath | None:
        """
        Greedy set-cover: pick the credential covering the most
        uncovered gaps each step, up to four items.

        Returns `None` when no credential covers any uncovered gap
        and no seed labels were provided.

        Args:
            strategy  : Display label for this path.
            uncovered : Gap indices not yet covered.
            seed      : Labels to prepend before greedy fill.
        """
        selected  = list(seed or [])
        remaining = set(uncovered)
        for _ in range(4):
            best = max(
                self.active,
                key     = lambda l: len(self.active[l] & remaining),
                default = None
            )
            if not best or not (overlap := self.active.get(best, set()) & remaining):
                break
            selected.append(best)
            remaining -= overlap
        if not selected:
            return None
        return CredentialPath(
            items = [
                PathItem.from_credential(len(self.active[l]), c, l)
                for l in selected
                if (c := self.credential_map.get(l))
            ],
            strategy        = strategy,
            unique_coverage = self.n_gaps - len(remaining)
        )

    def stepping_path(
        self,
        best_item    : PathItem | None,
        graph        : CareerPathwayGraph,
        reach        : Reach,
        route        : RouteDetail
    ) -> CredentialPath | None:
        """
        Stepping-stone strategy: find the best intermediate cluster
        that covers gap tasks, optionally paired with `best_item`.

        Returns `None` when no viable intermediate exists.
        """
        stepping = graph.stepping_stone(
            destination  = route.destination,
            reach        = reach,
            task_vectors = route.gap_vectors
        )
        if not stepping:
            return None
        title, cov = stepping
        step_item = PathItem(
            coverage = cov,
            detail   = "stepping",
            kind     = "career",
            label    = title
        )
        items = [step_item, best_item] if best_item else [step_item]
        return CredentialPath(
            items           = items,
            strategy        = "stepping",
            unique_coverage = min(
                cov + (best_item.coverage if best_item else 0),
                self.n_gaps
            )
        )


class CredentialPath(NamedTuple):
    """
    One gap-closure strategy with its credentials and total coverage.
    """

    items           : list[PathItem]
    strategy        : str
    unique_coverage : int


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

    corpus_cache: ClassVar[dict[int, tuple[dict[int, Counter], Counter]]] = {}

    @classmethod
    def from_cluster(
        cls,
        cluster           : Cluster,
        clusters          : Clusters,
        tier_descriptions : dict[str, str],
        min_count         : int = 3,
        per_tier          : int = 12
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
            cluster           : Matched career family.
            clusters          : Fitted cluster container with all postings.
            tier_descriptions : Tier label-to-description mapping from
                                tab TOML, iterated for tier names.
            min_count         : Minimum in-cluster occurrences for a
                                word to be eligible, suppressing
                                single-occurrence noise that would
                                otherwise top the ranking.
            per_tier          : Maximum words to keep per
                                distinctiveness tier.
        """
        if (cache_key := id(clusters)) not in cls.corpus_cache:
            per_cluster = {
                c.cluster_id: Counter(chain.from_iterable(c.distinctive_tokens))
                for c in clusters.values()
            }
            doc_freq = Counter(chain.from_iterable(per_cluster.values()))
            cls.corpus_cache[cache_key] = (per_cluster, doc_freq)

        per_cluster, doc_freq = cls.corpus_cache[cache_key]
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

        unique, rare, notable = tier_descriptions
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


class GapCoverage(NamedTuple):
    """
    Gap-closure analysis for a career transition.

    Encodes the destination cluster's gap task names, computes
    cosine similarity between each credential's embedding and every
    gap task, thresholds at the per-credential 75th percentile, then
    runs greedy set-cover to propose 2-4 alternative credential
    combinations with different time/coverage tradeoffs.

    The factory mirrors the `RelevantCredentials.from_cluster` and
    `RelevantJobBoards.from_cluster` patterns: it takes individual
    pipeline components rather than `TabContext` so the schema stays
    independent of the display wiring.
    """

    paths: list[CredentialPath]

    @classmethod
    def from_route(
        cls,
        pipeline : Chalkline,
        result   : MatchResult,
        route    : "RouteDetail"
    ) -> Self:
        """
        Assemble gap-closure strategies from the route's pre-computed
        coverage mapping.

        The route carries `coverage` and `gap_vectors` computed during
        `RouteDetail.from_selection`, so this method focuses purely on
        strategy assembly via `CoverageBuilder`.

        Args:
            pipeline : Fitted pipeline for stepping-stone lookup.
            result   : Match result with reach edges for the
                       stepping-stone strategy.
            route    : Route with pre-computed coverage and gap vectors.
        """
        if not route.coverage:
            return cls(paths=[])

        builder  = CoverageBuilder(
            active         = {l: g for l, g in route.coverage.items() if g},
            credential_map = route.credential_map,
            n_gaps         = route.gap_count
        )
        all_gaps   = set(range(builder.n_gaps))
        best_item  = builder.best_item
        paths: list[CredentialPath] = []

        if (fewest := builder.path("fewest", all_gaps)):
            paths.append(fewest)

        if best_item and (not fewest or best_item.label != fewest.items[0].label):
            paths.append(CredentialPath(
                items           = [best_item],
                strategy        = "biggest",
                unique_coverage = best_item.coverage
            ))

        if (programs := builder.programs):
            seed = max(programs, key=lambda l: len(programs[l]))
            education = builder.path(
                "education", all_gaps - programs[seed], [seed]
            )
            if education and (not fewest or education.items != fewest.items):
                paths.append(education)

        if (stepping := builder.stepping_path(
            best_item = best_item,
            graph     = pipeline.graph,
            reach     = result.reach,
            route     = route
        )):
            paths.append(stepping)

        return cls(paths=paths)


class JobPostingMetrics(BaseModel, extra="forbid"):
    """
    Aggregated posting statistics for the matched career family.
    """

    companies   : dict[str, int]
    dated       : list[DatedPoint]
    freshness   : list[int]
    locations   : dict[str, int]
    stat_values : list[str]

    @property
    def dates(self) -> list[date]:
        """
        Posting dates from `dated`, shaped for `Charts.timeline`.
        """
        return [d.date for d in self.dated]

    @property
    def hover(self) -> list[str]:
        """
        Hover labels from `dated`, shaped for `Charts.timeline`.
        """
        return [d.label for d in self.dated]

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

        freshness = [(today - p.date_posted).days for p in dated]

        return cls(
            companies   = dict(companies.most_common(15)),
            dated       = [
                DatedPoint(p.date_posted, p.company or p.title)
                for p in dated
            ],
            freshness   = freshness,
            locations   = dict(locations.most_common(15)),
            stat_values = [
                str(len(postings)),
                str(len(companies)),
                str(len(locations)),
                str(len(reference.match_employers(postings))),
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
    spinner_text      : str
    tab_names         : dict[str, str]
    upload_label      : str


class MapGeometry(BaseModel, extra="forbid"):
    """
    Static layout geometry for the force-directed pathway map widget.

    Owns pixel-level constants for node dimensions, force simulation
    tuning, and text-fit thresholds. The JS renderer receives a
    `dimensions` payload with the values it needs, while the Python
    side uses `title_char_limit` for pre-truncation.
    """

    card_h              : int        = 56
    card_w              : int        = 200
    circle_r            : int        = 24
    default_wage_range  : list[int]  = [30000, 90000]
    height              : int        = 660
    hero_h              : int        = 80
    hero_w              : int        = 280
    pad                 : int        = 50
    title_char_limit    : int        = 28
    width               : int        = 1800

    @property
    def dimensions(self) -> dict[str, int]:
        """
        Layout constants for the JS force simulation and renderer.
        """
        return self.model_dump(exclude={"default_wage_range", "title_char_limit"})


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

        return cls(
            brokerage          = brokerage,
            cluster_count      = len(clusters),
            cluster_heatmap    = clusters.cluster_heatmap,
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
            soc_heatmap        = clusters.soc_heatmap,
            variance           = VarianceBreakdown.from_svd(ratio)
        )


class PathItem(NamedTuple):
    """
    One credential or career node in a gap-closure path.
    """

    coverage : int
    detail   : str
    kind     : str
    label    : str

    @classmethod
    def from_credential(
        cls,
        coverage   : int,
        credential : Credential,
        label      : str
    ) -> Self:
        """
        Build from a `Credential` lookup, pulling `detail_label` and
        `kind` from the credential instance.
        """
        return cls(
            coverage = coverage,
            detail   = credential.detail_label,
            kind     = credential.kind,
            label    = label
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

    series: dict[str, ScatterSeries]

    series_cache: ClassVar[dict[int, dict[str, ScatterSeries]]] = {}

    def __bool__(self) -> bool:
        """
        True when the projection contains at least one sub-role
        series, gating the conditional scatter panel in the render.
        """
        return bool(self.series)

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
        key = id(cluster)
        if (cached := cls.series_cache.get(key)) is not None:
            return cls(series=cached)

        if cluster.size < min_postings:
            cls.series_cache[key] = (result := {})
            return cls(series=result)

        from sklearn.cluster  import KMeans
        from sklearn.manifold import TSNE

        embeddings = cluster.embeddings
        coords = TSNE(
            init         = "pca",
            n_components = 2,
            perplexity   = max(2.0, min(15.0, (cluster.size - 1) / 3)),
            random_state = random_state
        ).fit_transform(embeddings)

        k = max(1, min(n_subgroups, cluster.size // 3))
        assignments = (
            KMeans(
                n_clusters   = k,
                n_init       = 10,
                random_state = random_state
            ).fit_predict(embeddings)
            if k >= 2 else np.zeros(cluster.size, dtype=int)
        )

        sub_labels = cluster.sub_role_labels(assignments, k)
        grouped: dict[str, list[tuple]] = defaultdict(list)
        for assignment, xy, p in zip(assignments, coords, cluster.postings):
            grouped[sub_labels[int(assignment)]].append((
                f"{p.title} · {p.company}" if p.company else p.title,
                float(xy[0]),
                float(xy[1])
            ))

        hover, x, y = range(3)
        cls.series_cache[key] = (result := {
            label: ScatterSeries(
                hover = [p[hover] for p in pts],
                x     = [p[x]     for p in pts],
                y     = [p[y]     for p in pts]
            )
            for label, pts in grouped.items()
        })
        return cls(series=result)


class ProcessStep(BaseModel, extra="forbid"):
    """
    One step in a horizontal process flow diagram.

    Used by `Layout.process_flow` to render numbered cards with optional
    sector-color accents and natural-language hop labels above incoming
    arrows. The Methods tab pipeline diagram uses only the required
    fields; the Map tab career path flow uses the accent and arrow_label
    extensions to convey sector identity and transition difficulty.
    """

    detail : str
    label  : str
    number : str

    accent      : str = ""
    arrow_label : str = ""

    def render(self, **kwargs) -> Self:
        """
        Return a copy with `detail` formatted against the given kwargs.

        Used by `Layout.process_flow` to substitute corpus-level values
        like `{corpus_size}` and `{cluster_count}` into each step's
        detail line at render time. The accent and arrow_label fields
        pass through unchanged because they carry colors and fixed
        phrases that need no substitution.
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

    boards      : list[BoardMatch]
    board_cache : ClassVar[tuple[list, np.ndarray] | None] = None

    @classmethod
    def from_cluster(
        cls,
        cluster   : Cluster,
        clusters  : Clusters,
        encoder   : SentenceEncoder,
        reference : StakeholderReference,
        limit     : int = 5
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

        if cls.board_cache is None:
            all_boards      = list(chain.from_iterable(reference.job_boards.values()))
            cls.board_cache = (
                all_boards,
                encoder.encode([
                    f"{b.get('focus', '')}"
                    f" {b.get('best_for', '')}"
                    f" {b.get('category', '')}"
                    for b in all_boards
                ]))

        all_boards, matrix = cls.board_cache
        cluster_vector     = clusters.vector_map[cluster.cluster_id]
        similarities       = cosine_similarity(cluster_vector[None, :], matrix)[0]
        return cls(boards=[
            BoardMatch(
                **all_boards[i],
                match_score = round(float(similarities[i]) * 100)
            )
            for i in np.argsort(-similarities)[:limit]
        ])


class RouteDetail(NamedTuple):
    """
    Joined route data for the Map tab's recipe card.

    Holds the source and destination `Cluster` objects for dot-notation
    access to SOC titles, sectors, Job Zones, and postings. Constructed
    via `from_selection`, which prefers the direct adjacent edge when
    one exists (the one-hop move the user likely intended by clicking
    a neighbor) and falls back to the widest-path bottleneck tree for
    destinations that aren't direct neighbors.
    """

    coverage         : dict[str, set[int]]
    credentials      : list[Credential]
    destination      : Cluster
    destination_wage : float | None
    display_title    : str
    gap_vectors      : np.ndarray
    scored_tasks     : list[ScoredTask]
    source           : Cluster
    source_wage      : float | None

    bright_outlook   : bool = False

    @property
    def credential_map(self) -> dict[str, Credential]:
        """
        Credential label to `Credential` object for O(1) lookup.
        """
        return {c.label: c for c in self.credentials}

    @property
    def credentials_by_kind(self) -> dict[str, list[Credential]]:
        """
        Route credentials grouped by kind, with empty kinds omitted.

        Reusable for both the recipe section (top picks per kind) and
        the resources drawer (full catalog per kind) so neither has
        to rebuild the grouping.
        """
        by_kind: dict[str, list[Credential]] = {}
        for credential in self.credentials:
            by_kind.setdefault(credential.kind, []).append(credential)
        return by_kind

    @property
    def demonstrated_count(self) -> int:
        """
        Number of destination tasks the resume demonstrates.
        """
        return sum(t.demonstrated for t in self.scored_tasks)

    @property
    def fit_percentage(self) -> int:
        """
        Mean raw cosine similarity as a 0-100 percentage for hero
        display, matching the map donut formula.
        """
        if not self.scored_tasks:
            return 0
        return round(
            100 * sum(t.similarity for t in self.scored_tasks)
            / len(self.scored_tasks)
        )

    @property
    def gap_count(self) -> int:
        """
        Number of destination tasks the resume does not demonstrate.
        """
        return len(self.gap_tasks)

    @property
    def gap_tasks(self) -> list[ScoredTask]:
        """
        Destination tasks the resume does not demonstrate.
        """
        return [t for t in self.scored_tasks if not t.demonstrated]

    @property
    def is_self(self) -> bool:
        """
        True when source and destination are the same cluster,
        meaning the user is viewing their matched career rather
        than a transition route.
        """
        return self.source.cluster_id == self.destination.cluster_id

    @property
    def top_gaps(self) -> list[ScoredTask]:
        """
        Eight largest skill gaps by deficit (lowest similarity).
        """
        return nsmallest(
            8,
            self.gap_tasks,
            key = attrgetter("similarity")
        )

    @property
    def top_strengths(self) -> list[ScoredTask]:
        """
        Eight strongest demonstrated skills, sorted by descending
        similarity.
        """
        return nlargest(
            8,
            (t for t in self.scored_tasks if t.demonstrated),
            key = attrgetter("similarity")
        )

    @property
    def total_tasks(self) -> int:
        """
        Total destination tasks scored against the resume.
        """
        return len(self.scored_tasks)

    @property
    def wage_comparison(self) -> WageComparison:
        """
        Wage bar data for the verdict hero section, with display
        labels and percentages derived from the raw amounts.
        """
        return WageComparison(self.destination_wage, self.source_wage)

    @staticmethod
    def _encode_gaps(
        credentials  : list[Credential],
        pipeline     : Chalkline,
        scored_tasks : list[ScoredTask]
    ) -> tuple[np.ndarray, dict[str, set[int]]]:
        """
        Encode gap task names and compute credential coverage,
        returning empty defaults when there are no gaps or
        credentials to evaluate.
        """
        gaps = [t.name for t in scored_tasks if not t.demonstrated]
        if not gaps or not credentials:
            return np.empty((0, 0)), {}

        vectors = pipeline.matcher.encoder.encode(gaps)
        return (
            vectors,
            pipeline.graph.credential_coverage(
                route_labels = [c.label for c in credentials],
                task_vectors = vectors
            )
        )

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
        Build a route from a clicked map node, or `None` if the click
        is invalid (no selection, the matched cluster itself, or no
        path connects the source and destination).

        Prefers the direct adjacent edge (single-hop) when the
        destination is a direct neighbor of the matched cluster,
        because the user clicked a specific node and the one-hop
        move carries the most relevant credentials. Falls back to the
        widest-path tree for non-adjacent destinations.

        When `selected_id` is negative (no click yet) or equals the
        matched cluster, builds a self-route so the user sees their
        skill profile, credentials, and postings for the matched
        career immediately rather than an empty callout.
        """
        if selected_id < 0 or selected_id == result.cluster_id:
            destination = profile
            edges       = result.reach.edges

        elif (adjacent := next(
            (e for e in result.reach.edges if e.cluster_id == selected_id),
            None
        )):
            destination = pipeline.clusters[selected_id]
            edges       = [adjacent]

        else:
            path  = pipeline.graph.try_widest_path(profile.cluster_id, selected_id)
            edges = pipeline.graph.path_edges(path)
            if not edges:
                return None
            destination = pipeline.clusters[selected_id]

        credentials = [c for e in edges for c in e.credentials]
        scored      = pipeline.matcher.score_destination(destination)
        encoded     = cls._encode_gaps(credentials, pipeline, scored)

        return cls(
            bright_outlook   = labor[destination.soc_title].bright_outlook,
            coverage         = encoded[1],
            credentials      = credentials,
            destination      = destination,
            destination_wage = labor[destination.soc_title].annual_median,
            display_title    = (
                destination.modal_title
                if pipeline.clusters.soc_counts[destination.soc_title] > 1
                else destination.soc_title
            ),
            gap_vectors      = encoded[0],
            scored_tasks     = scored,
            source           = profile,
            source_wage      = labor[profile.soc_title].annual_median
        )


class ScatterSeries(TypedDict):
    """
    Parallel arrays for one trace in a category scatter chart.
    """

    hover : list[str]
    x     : list[float]
    y     : list[float]


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


class WageComparison(NamedTuple):
    """
    Wage bar data for the route card verdict section.

    Accepts nullable wages directly from `RouteDetail` and resolves
    them to zero internally so callers never need `or 0`. Display
    percentages are relative to the higher wage so the longer bar
    is always 100%.
    """

    destination_wage : float | None = None
    source_wage      : float | None = None

    @property
    def delta(self) -> float | None:
        """
        Annual wage difference, or `None` when either wage is
        unavailable.
        """
        if not self.source_wage or not self.destination_wage:
            return None
        return self.destination_wage - self.source_wage

    @property
    def delta_display(self) -> str:
        """
        Signed dollar-per-year string, or empty when unavailable.
        """
        if (d := self.delta) is None:
            return ""
        return f"{'+'  if d >= 0 else ''}${d:,.0f}/yr"

    @property
    def destination_label(self) -> str:
        """
        Formatted destination wage or em dash when unavailable.
        """
        if not self.destination_wage:
            return "\u2014"
        return f"${self.destination_wage / 1000:.0f}k"

    @property
    def destination_percentage(self) -> int:
        """
        Destination wage as a percentage of the higher wage.
        """
        d = self.destination_wage or 0
        s = self.source_wage or 0
        return round(d / max(d, s, 1) * 100)

    @property
    def source_label(self) -> str:
        """
        Formatted source wage or em dash when unavailable.
        """
        if not self.source_wage:
            return "\u2014"
        return f"${self.source_wage / 1000:.0f}k"

    @property
    def source_percentage(self) -> int:
        """
        Source wage as a percentage of the higher wage.
        """
        d = self.destination_wage or 0
        s = self.source_wage or 0
        return round(s / max(d, s, 1) * 100)
