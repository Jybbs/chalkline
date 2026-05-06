"""
Entity and aggregate classes for the career pathway domain.

Behavioral domain objects that wrap schema types with numpy arrays, cached
derived indices, and domain query methods. Separated from `schemas.py` to
keep the schema layer free of computational imports and behavioral logic.
"""

import numpy as np

from collections     import Counter, defaultdict
from collections.abc import Iterator
from dataclasses     import dataclass, field
from functools       import cached_property
from heapq           import nlargest
from itertools       import chain
from math            import log
from operator        import itemgetter
from sklearn.cluster import KMeans
from typing          import NamedTuple

from chalkline.collection.schemas import Posting
from chalkline.pathways.loaders   import LaborLoader
from chalkline.pathways.schemas   import Credential


@dataclass
class Cluster:
    """
    Unified per-cluster representation combining profile metadata, resolved
    postings, per-posting embeddings, and optional O*NET task embeddings for
    gap analysis.

    `embeddings` is the slice of `raw_vectors` that belongs to this cluster,
    retained at fit time so display-layer projections (t-SNE,
    sub-clustering, distance-from-centroid views) operate on the pipeline's
    already-computed encodings rather than re-running the encoder per
    render.
    """

    cluster_id  : int
    embeddings  : np.ndarray
    modal_title : str
    postings    : list[Posting]
    sector      : str
    size        : int
    soc_title   : str

    display_title : str        = field(default="", init=False)
    soc_weights   : np.ndarray = field(default_factory=lambda: np.empty(0), init=False)
    tasks         : list[Task] = field(default_factory=list)
    wage          : float      = field(default=0.0, init=False)
    wage_tier     : int        = field(default=0,   init=False)

    @property
    def display_label(self) -> str:
        """
        Human-readable cluster identifier for dropdown labels, using the
        collision-resolved `display_title` so duplicate SOCs stay
        distinguishable in the picker. The wage tier suffix shows where
        this family sits in the wage-banded career ladder, 1-indexed for
        stakeholder readability while internal storage stays 0-indexed.
        """
        return (
            f"Cluster {self.cluster_id}: {self.display_title} "
            f"(Tier {self.wage_tier + 1})"
        )

    @cached_property
    def distinctive_tokens(self) -> list[list[str]]:
        """
        Per-posting bags of lowercased alphabetic tokens longer than three
        characters whose Zipf frequency falls below the common-word floor,
        lazily computed once per session and cached on the instance.

        Tokenization uses an alphabetic regex rather than whitespace
        splitting so internal punctuation (curly apostrophes in
        possessives like `Maine's`, hyphens in compounds, trademark
        marks) cleanly separates words instead of leaking into token
        strings.

        Display-layer factories that build TF-IDF rankings over postings
        iterate this list to assemble word counters without re-running the
        tokenizer per render. Caching here keeps the matched cluster's
        tokenization shared between cross-corpus distinctiveness ranking and
        intra-cluster sub-role labeling.
        """
        from re       import findall
        from wordfreq import zipf_frequency
        return [
            [
                w for w in findall(r"[a-zA-Z]{4,}", f"{p.title} {p.description}".lower())
                if zipf_frequency(w, "en") < 5.0
            ]
            for p in self.postings
        ]

    @cached_property
    def task_matrix(self) -> np.ndarray:
        """
        Stacked task embedding vectors as a 2D array for cosine similarity
        computation.
        """
        return np.stack([t.vector for t in self.tasks])

    @cached_property
    def task_stems(self) -> list[set[str]]:
        """
        Stemmed content words per task for BM25 scoring, filtering stop
        words and short tokens. One set per task in the same order as
        `self.tasks`.
        """
        from nltk.stem import SnowballStemmer
        from re        import findall
        from wordfreq  import zipf_frequency

        stemmer = SnowballStemmer("english")
        return [
            {
                stemmer.stem(w)
                for w in findall(r"[a-zA-Z]{3,}", t.name.lower())
                if zipf_frequency(w, "en") < 6.0
            }
            for t in self.tasks
        ]

    def sub_role_labels(
        self,
        assignments : np.ndarray,
        k           : int
    ) -> list[str]:
        """
        TF-IDF top-2 word labels for each k-means sub-cluster, derived from
        `distinctive_tokens`.

        Each sub-cluster's word counter is ranked by TF-IDF where documents
        are sub-clusters. Words appearing in all k groups or fewer than
        twice are suppressed. Falls back to a numbered label when no
        distinctive words survive filtering.

        Args:
            assignments : Per-posting k-means cluster assignment array.
            k           : Number of sub-clusters.
        """
        counts: list[Counter] = [Counter() for _ in range(k)]
        for label, bag in zip(assignments, self.distinctive_tokens):
            counts[label].update(bag)

        doc_freq = Counter(w for c in counts for w in c)
        return [
            " · ".join(w.title() for w, _ in top) if top else f"Sub-role {i + 1}"
            for i, counter in enumerate(counts)
            for total in [counter.total() or 1]
            for top in [nlargest(
                2,
                (
                    (w, c) for w, c in counter.items()
                    if c >= 2 and doc_freq[w] < k
                ),
                key = lambda wc: (wc[1] / total) * log(k / doc_freq[wc[0]])
            )]
        ]


@dataclass
class Clusters:
    """
    Indexed collection of clusters with eagerly-derived matrices.

    Constructed once after profiling is complete. Provides the per-cluster
    dict for individual lookups and pre-stacked centroid and embedding
    vector matrices for vectorized operations in graph construction, resume
    matching, and visualization.
    """

    centroids         : np.ndarray
    credentials       : list[Credential]
    items             : dict[int, Cluster]
    labor             : LaborLoader
    occupation_titles : list[str]
    soc_similarity    : np.ndarray
    softmax_tau       : float
    vectors           : np.ndarray
    wage_round        : int
    wage_tier_count   : int
    wage_topk         : int

    cluster_ids: list[int] = field(init=False)

    def __getitem__(self, cluster_id: int) -> Cluster:
        """
        Look up a cluster by ID.
        """
        return self.items[cluster_id]

    def __iter__(self) -> Iterator[int]:
        """
        Iterate sorted cluster IDs.
        """
        return iter(self.cluster_ids)

    def __len__(self) -> int:
        """
        Number of clusters.
        """
        return len(self.items)

    def __post_init__(self):
        """
        Derive sorted cluster IDs for stable iteration order, then fan the
        softmax-derived `soc_weights`, `wage`, K-means-derived `wage_tier`,
        and collision-resolved `display_title` onto each child `Cluster` so
        downstream callers read per-cluster attributes without round-tripping
        through the parent. `wage` must be set before `wage_tier_map` is
        accessed because the partition reads `Cluster.wage` directly.
        """
        self.cluster_ids = sorted(self.items)
        for cid, cluster in self.items.items():
            cluster.soc_weights   = self.soc_weights[cid]
            cluster.wage          = float(self.cluster_wages[cid])
            cluster.display_title = self.display_titles[cid]
        for cid, tier in self.wage_tier_map.items():
            self.items[cid].wage_tier = tier

    @cached_property
    def bm25_average_length(self) -> float:
        """
        Mean stem count across all task descriptions in the corpus, used as
        the average document length for BM25 length normalization.
        """
        return float(np.mean([
            len(ts) for c in self.values() for ts in c.task_stems
        ]))

    @cached_property
    def bm25_idf(self) -> dict[str, float]:
        """
        Inverse document frequency per stem across the union of cluster
        task descriptions and credential descriptions, using the BM25 IDF
        variant with smoothing.

        Counting both sides of the matching pair surfaces what is rare
        across the corpus rather than only what is rare across tasks.
        Boilerplate stems shared by many credentials and many tasks
        (`technician`, `inspector`, `professional`) get suppressed while
        specialty stems on either side stay discriminative, narrowing the
        gap by which broad-text apprenticeships accumulated coverage on
        common stems.
        """
        all_stems  = [ts for c in self.values() for ts in c.task_stems]
        all_stems += [c.stems for c in self.credentials]
        n  = len(all_stems)
        df = Counter(chain.from_iterable(all_stems))

        return {
            stem: log((n - count + 0.5) / (count + 0.5) + 1)
            for stem, count in df.items()
        }

    @cached_property
    def centroid_cosine(self) -> np.ndarray:
        """
        Pairwise cosine similarity matrix between cluster centroids.

        Single source of truth for centroid similarity, consumed by both the
        graph backbone construction (which reads the raw ndarray to build
        stepwise k-NN edges) and `cosine_similarity_matrix` (which rounds
        and converts for heatmap rendering). Caching here means
        `cosine_similarity(self.centroids)` runs once per session instead of
        once for the methods tab and once for the graph.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(self.centroids)

    @cached_property
    def centroid_distance_matrix(self) -> np.ndarray:
        """
        Pairwise Euclidean distance between cluster centroids in the
        reduced SVD space. Single source of truth shared by
        `max_centroid_distance` (resume score normalization denominator)
        and `pairwise_distances` (per-sector distance distributions).
        """
        from sklearn.metrics.pairwise import euclidean_distances
        return euclidean_distances(self.centroids)

    @cached_property
    def cluster_heatmap(self) -> dict[str, list[float]]:
        """
        Centroid cosine similarity keyed by SOC title for the inter-cluster
        heatmap.
        """
        return dict(zip(
            (c.soc_title for c in self.values()),
            self.cosine_similarity_matrix
        ))

    @cached_property
    def cluster_index(self) -> dict[int, int]:
        """
        Cluster ID to its row index in the stacked `vectors` and `centroids`
        matrices, mirroring `vector_map` but exposing the position rather
        than the vector itself. Enables display-layer factories to slice
        precomputed similarity matrices by cluster_id without iterating
        `cluster_ids`.
        """
        return {cid: i for i, cid in enumerate(self.cluster_ids)}

    @cached_property
    def cluster_wages(self) -> np.ndarray:
        """
        Top-K weighted annual median wage per cluster in `cluster_ids`
        order. Zero-masked so missing labor data never drags the expectation
        downward and rounded to `wage_round` to match the granularity of the
        source labor records.
        """
        masked = self.soc_weights * (self.occupation_wages > 0)
        picks  = np.argpartition(masked, -self.wage_topk, axis=1)[:, -self.wage_topk:]

        weights   = np.take_along_axis(self.soc_weights, picks, axis=1)
        weights  /= weights.sum(axis=1, keepdims=True)
        expected  = (weights * self.occupation_wages[picks]).sum(axis=1)
        return np.round(expected / self.wage_round) * self.wage_round

    @property
    def company_count(self) -> int:
        """
        Distinct company names across all cluster postings.
        """
        return len({
            p.company for c in self.values()
            for p in c.postings if p.company
        })

    @cached_property
    def cosine_similarity_matrix(self) -> list[list[float]]:
        """
        Cosine similarity between all cluster centroids as a 2D list of
        rounded floats for heatmap rendering. Derived from `centroid_cosine`
        to share the underlying matrix computation with the graph backbone.
        """
        return np.round(self.centroid_cosine, 3).tolist()

    @cached_property
    def display_titles(self) -> dict[int, str]:
        """
        Cluster ID to display label, with TF-IDF token suffixes that
        disambiguate clusters sharing a SOC title. Single-SOC clusters
        keep the bare title. Cohorts of size `n` retain tokens
        appearing in 2 to `n - 1` siblings (1 to 1 when `n == 2`) that
        also clear a Zipf-frequency floor so brand vocabulary drops
        out before ranking. The suffix is the top-ranked token alone
        when its score exceeds twice the runner-up's, otherwise both
        tokens joined by ` · `. Falls back to `#{cluster_id}` when no
        token survives filtering.
        """
        from wordfreq import zipf_frequency

        cohorts: dict[str, list[int]] = defaultdict(list)
        for cid in self.cluster_ids:
            cohorts[self.items[cid].soc_title].append(cid)

        titles: dict[int, str] = {}
        for soc, siblings in cohorts.items():
            if len(siblings) == 1:
                titles[siblings[0]] = soc
                continue

            tokens = {
                cid: Counter(chain.from_iterable(self.items[cid].distinctive_tokens))
                for cid in siblings
            }
            doc_freq    = Counter(chain.from_iterable(tokens.values()))
            cohort_size = len(siblings)
            min_df      = 2 if cohort_size >= 3 else 1

            for cid, counter in tokens.items():
                total = counter.total() or 1
                top   = nlargest(
                    2,
                    (
                        (w, (c / total) * log(cohort_size / doc_freq[w]))
                        for w, c in counter.items()
                        if c >= 2
                        and min_df <= doc_freq[w] < cohort_size
                        and zipf_frequency(w, "en") >= 2.5
                    ),
                    key=itemgetter(1)
                )
                suffix = f"#{cid}"
                match top:
                    case [(word, _)]:
                        suffix = word.title()
                    case [(winner, hi), (_, lo)] if lo < 0.5 * hi:
                        suffix = winner.title()
                    case [(winner, _), (runner_up, _)]:
                        suffix = f"{winner.title()} · {runner_up.title()}"
                titles[cid] = f"{soc} ({suffix})"

        return titles

    @property
    def location_count(self) -> int:
        """
        Distinct locations across all cluster postings.
        """
        return len({
            p.location for c in self.values()
            for p in c.postings if p.location
        })

    @cached_property
    def max_centroid_distance(self) -> float:
        """
        Maximum Euclidean distance between any two cluster centroids in the
        reduced SVD space. Fixed per-corpus, used as the denominator for
        normalizing resume-to-cluster distances into display match scores.
        """
        return float(self.centroid_distance_matrix.max())

    @cached_property
    def occupation_wages(self) -> np.ndarray:
        """
        Annual median wage per occupation aligned with `occupation_titles`.
        Missing records and null wages resolve to 0 so `cluster_wages` can
        mask and weight the vector directly without per-element None checks.
        """
        return np.array([
            self.labor[t].annual_median or 0
            for t in self.occupation_titles
        ])

    @cached_property
    def pairwise_distances(self) -> dict[str, list[float]]:
        """
        Euclidean distances between all centroid pairs, grouped by sector of
        the first cluster in each pair.
        """
        i, j    = np.triu_indices(len(self.cluster_ids), k=1)
        sectors = self.sector_array[i]
        dists   = self.centroid_distance_matrix[i, j]
        return {
            s: dists[sectors == s].tolist()
            for s in np.unique(sectors)
        }

    @property
    def profile_map(self) -> dict[str, dict]:
        """
        Display label to profile summary for all clusters.
        """
        return {
            c.display_label: {
                "Modal Title" : c.modal_title,
                "Sector"      : c.sector,
                "Size"        : c.size,
                "Wage Tier"   : c.wage_tier + 1
            }
            for c in self.values()
        }

    @cached_property
    def sector_array(self) -> np.ndarray:
        """
        Sector name per cluster as a numpy array for sklearn APIs.
        """
        return np.array([c.sector for c in self.values()])

    @property
    def sector_sizes(self) -> dict[str, int]:
        """
        Total posting count per sector.
        """
        sizes: Counter[str] = Counter()
        for c in self.values():
            sizes[c.sector] += c.size
        return dict(sizes)

    @property
    def sectors(self) -> list[str]:
        """
        Unique sector names in sorted order.
        """
        return sorted({p.sector for p in self.values()})

    @cached_property
    def silhouette_scores(self) -> list[tuple[str, str, float]]:
        """
        Per-cluster silhouette coefficients against sector labels, resolved
        to (title, sector, score) and sorted descending.
        """
        from sklearn.metrics import silhouette_samples
        scores = np.round(np.asarray(
            silhouette_samples(self.centroids, self.sector_array, metric="cosine")
        ), 3)
        return sorted(
            ((c.soc_title, c.sector, float(s)) for c, s in zip(self.values(), scores)),
            key     = itemgetter(2),
            reverse = True
        )

    @cached_property
    def sizes(self) -> list[int]:
        """
        Posting count per cluster in sorted cluster-ID order.
        """
        return [c.size for c in self.values()]

    @cached_property
    def soc_heatmap(self) -> dict[str, list[float]]:
        """
        SOC similarity keyed by display label, rounded to three decimals for
        the SOC assignment heatmap.
        """
        return dict(zip(
            (c.display_label for c in self.values()),
            np.round(self.soc_similarity, 3).tolist()
        ))

    @cached_property
    def soc_weights(self) -> np.ndarray:
        """
        Softmax over `soc_similarity` rows at temperature `softmax_tau`,
        producing a per-cluster distribution over occupations. Feeds the
        weighted wage computation and the runner-up fallback used by
        `display_titles` to differentiate colliding SOC labels.
        """
        from scipy.special import softmax
        return softmax(self.soc_similarity / self.softmax_tau, axis=1)

    @cached_property
    def vector_map(self) -> dict[int, np.ndarray]:
        """
        Cluster ID to its row in the stacked `vectors` matrix, mirroring the
        `wage_tier_map` lookup pattern so display-layer factories can fetch a
        single cluster's vector without indexing into the row order via
        `cluster_ids.index(...)`. Cached because both `RelevantCredentials`
        and `RelevantJobBoards` look it up per render and the underlying
        matrix never changes after fit.
        """
        return dict(zip(self.cluster_ids, self.vectors))

    @cached_property
    def wage_tier_map(self) -> dict[int, int]:
        """
        Cluster ID to wage tier index, ordered so tier 0 is the lowest mean
        log wage and tier `wage_tier_count - 1` is the highest. Forms the
        spine of the career graph in place of O*NET Job Zone, partitioning
        clusters by what they actually pay rather than by O*NET's preparation
        ladder.

        K-means on `log(wage)` separates the wage distribution into bands
        whose boundaries respect the long tail without letting a few
        high-wage clusters dominate Euclidean distance the way they would in
        raw-dollar space. Tier indices come from sorting K-means labels by
        each band's mean log wage, so advancement edges always point toward
        higher pay.
        """
        wages     = np.array([self.items[cid].wage for cid in self.cluster_ids])
        log_wages = np.log(np.where(wages > 0, wages, 1.0)).reshape(-1, 1)
        raw       = KMeans(
            n_clusters   = self.wage_tier_count,
            n_init       = 10,
            random_state = 42
        ).fit_predict(log_wages)
        order = np.argsort([log_wages[raw == j].mean() for j in range(self.wage_tier_count)])
        rank  = {old: new for new, old in enumerate(order)}
        return {cid: rank[int(label)] for cid, label in zip(self.cluster_ids, raw)}

    def values(self) -> Iterator[Cluster]:
        """
        Iterate cluster objects in sorted ID order.
        """
        return (self.items[cid] for cid in self.cluster_ids)


class Task(NamedTuple):
    """
    A single O*NET Task or DWA element with its sentence embedding.

    Produced during SOC task encoding and attached to `Cluster.tasks` for
    per-task cosine gap analysis during resume matching.
    """

    name   : str
    vector : np.ndarray
