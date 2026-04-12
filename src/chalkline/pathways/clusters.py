"""
Entity and aggregate classes for the career pathway domain.

Behavioral domain objects that wrap schema types with numpy arrays, cached
derived indices, and domain query methods. Separated from `schemas.py` to
keep the schema layer free of computational imports and behavioral logic.
"""

import numpy as np

from collections.abc import Iterator
from dataclasses     import dataclass, field
from functools       import cached_property
from typing          import NamedTuple

from chalkline.collection.schemas import Posting


@dataclass
class Cluster:
    """
    Unified per-cluster representation combining profile metadata,
    resolved postings, per-posting embeddings, and optional O*NET task
    embeddings for gap analysis.

    `embeddings` is the slice of `raw_vectors` that belongs to this
    cluster, retained at fit time so display-layer projections (t-SNE,
    sub-clustering, distance-from-centroid views) operate on the
    pipeline's already-computed encodings rather than re-running the
    encoder per render.
    """

    cluster_id  : int
    embeddings  : np.ndarray
    job_zone    : int
    modal_title : str
    postings    : list[Posting]
    sector      : str
    size        : int
    soc_title   : str

    tasks: list[Task] = field(default_factory=list)

    @property
    def display_label(self) -> str:
        """
        Human-readable cluster identifier for dropdown labels.
        """
        return (
            f"Cluster {self.cluster_id}: {self.soc_title} "
            f"(Job Zone {self.job_zone})"
        )

    @cached_property
    def distinctive_tokens(self) -> list[list[str]]:
        """
        Per-posting bags of lowercased tokens longer than three
        characters whose Zipf frequency falls below the common-word
        floor, lazily computed once per session and cached on the
        instance.

        Display-layer factories that build TF-IDF rankings over
        postings iterate this list to assemble word counters without
        re-running the tokenizer per render. Caching here keeps the
        matched cluster's tokenization shared between cross-corpus
        distinctiveness ranking and intra-cluster sub-role labeling.
        """
        from wordfreq import zipf_frequency
        return [
            [
                w for word in f"{p.title} {p.description}".split()
                if  len(w := word.strip(".,;:!?()[]\"'").lower()) > 3
                and zipf_frequency(w, "en") < 5.0
            ]
            for p in self.postings
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

    centroids      : np.ndarray
    items          : dict[int, Cluster]
    soc_similarity : np.ndarray
    vectors        : np.ndarray

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
        Derive sorted cluster IDs for stable iteration order.
        """
        self.cluster_ids = sorted(self.items)

    @cached_property
    def centroid_cosine(self) -> np.ndarray:
        """
        Pairwise cosine similarity matrix between cluster centroids.

        Single source of truth for centroid similarity, consumed by both
        the graph backbone construction (which reads the raw ndarray to
        build stepwise k-NN edges) and `cosine_similarity_matrix` (which
        rounds and converts for heatmap rendering). Caching here means
        `cosine_similarity(self.centroids)` runs once per session
        instead of once for the methods tab and once for the graph.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(self.centroids)

    @cached_property
    def cluster_index(self) -> dict[int, int]:
        """
        Cluster ID to its row index in the stacked `vectors` and
        `centroids` matrices, mirroring `vector_map` but exposing the
        position rather than the vector itself. Enables display-layer
        factories to slice precomputed similarity matrices by
        cluster_id without iterating `cluster_ids`.
        """
        return {cid: i for i, cid in enumerate(self.cluster_ids)}

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
        rounded floats for heatmap rendering. Derived from
        `centroid_cosine` to share the underlying matrix computation
        with the graph backbone.
        """
        return [
            [round(float(v), 3) for v in row]
            for row in self.centroid_cosine
        ]

    @property
    def job_zone_map(self) -> dict[int, int]:
        """
        Cluster ID to Job Zone for stepwise graph constraints.
        """
        return {cid: self.items[cid].job_zone for cid in self.cluster_ids}

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
    def pairwise_distances(self) -> dict[str, list[float]]:
        """
        Euclidean distances between all centroid pairs, grouped by sector of
        the first cluster in each pair.
        """
        from collections import defaultdict
        from itertools   import combinations
        dists = np.linalg.norm(
            self.centroids[:, None] - self.centroids[None, :], axis=2
        )
        result: dict[str, list[float]] = defaultdict(list)
        for i, j in combinations(range(len(self.cluster_ids)), 2):
            result[self.sector_array[i]].append(float(dists[i, j]))
        return dict(result)

    @property
    def profile_map(self) -> dict[str, dict]:
        """
        Display label to profile summary for all clusters.
        """
        return {
            c.display_label: {
                "Job Zone"    : c.job_zone,
                "Modal Title" : c.modal_title,
                "Sector"      : c.sector,
                "Size"        : c.size
            }
            for c in self.values()
        }

    @cached_property
    def sector_array(self) -> np.ndarray:
        """
        Sector name per cluster as a numpy array for sklearn APIs.
        """
        return np.array([self.items[cid].sector for cid in self.cluster_ids])

    @property
    def sector_sizes(self) -> dict[str, int]:
        """
        Total posting count per sector.
        """
        return {
            s: sum(p.size for p in self.values() if p.sector == s)
            for s in self.sectors
        }

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
        scores = np.asarray(
            silhouette_samples(self.centroids, self.sector_array, metric="cosine")
        )
        return sorted(
            (
                (c.soc_title, c.sector, round(float(s), 3))
                for c, s in zip(self.values(), scores)
            ),
            key     = lambda x: x[2],
            reverse = True
        )

    @cached_property
    def sizes(self) -> list[int]:
        """
        Posting count per cluster in sorted cluster-ID order.
        """
        return [c.size for c in self.values()]

    @cached_property
    def vector_map(self) -> dict[int, np.ndarray]:
        """
        Cluster ID to its row in the stacked `vectors` matrix, mirroring
        the `job_zone_map` lookup pattern so display-layer factories can
        fetch a single cluster's vector without indexing into the row
        order via `cluster_ids.index(...)`. Cached because both
        `RelevantCredentials` and `RelevantJobBoards` look it up per
        render and the underlying matrix never changes after fit.
        """
        return dict(zip(self.cluster_ids, self.vectors))

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
