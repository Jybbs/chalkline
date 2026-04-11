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
    resolved postings, and optional O*NET task embeddings for gap
    analysis.
    """

    cluster_id  : int
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
        return f"Cluster {self.cluster_id}: {self.soc_title} (JZ {self.job_zone})"


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
        Cosine similarity between all cluster centroids as a 2D list for
        heatmap rendering.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        return [
            [round(float(v), 3) for v in row]
            for row in cosine_similarity(self.centroids)
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
