"""
Resume matching via sentence embeddings with per-task gap analysis.

Projects an uploaded resume into the fitted SVD space, matches it to the
nearest career family via cluster centroids, identifies demonstrated
competencies and gaps via per-task cosine similarity against O*NET Task+DWA
embeddings, and assembles a neighborhood view with per-edge credential
metadata.
"""

import numpy as np

from dataclasses              import dataclass
from sklearn.decomposition    import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from chalkline.matching.schemas  import ClusterDistance, MatchResult, TaskGap
from chalkline.pipeline.graph    import CareerPathwayGraph
from chalkline.pipeline.schemas  import ClusterProfile, ClusterTasks, Encoder


@dataclass(kw_only=True, slots=True)
class ResumeMatcher:
    """
    Embedding-based resume matching with neighborhood exploration.

    Holds the sentence transformer model, fitted SVD, cluster centroids, and
    per-cluster O*NET task embeddings. The `match()` method encodes resume
    text, projects it into the reduced space, assigns it to the nearest
    cluster, computes per-task gap analysis, and queries the career graph
    for the local neighborhood view.

    Args:
        centroids   : (n_clusters, n_components) in SVD-reduced space.
        cluster_ids : (n_clusters,) aligned with centroid rows.
        graph       : For neighborhood queries post-match.
        model       : For encoding resume text into embedding space.
        profiles    : For sector lookup on the matched cluster.
        soc_tasks   : Per-cluster Task+DWA names and embeddings.
        svd         : For projecting resume embeddings into reduced space.
    """

    centroids   : np.ndarray
    cluster_ids : np.ndarray
    graph       : CareerPathwayGraph
    model       : Encoder
    profiles    : dict[int, ClusterProfile]
    soc_tasks   : dict[int, ClusterTasks]
    svd         : TruncatedSVD

    def _gap_analysis(
        self,
        cluster_id  : int,
        resume_unit : np.ndarray
    ) -> tuple[list[TaskGap], list[TaskGap]]:
        """
        Partition O*NET tasks into demonstrated and gap lists via
        median-split cosine similarity against the resume embedding.

            cos(𝐫, 𝐭ᵢ) ≥ S̃  →  demonstrated
            cos(𝐫, 𝐭ᵢ) < S̃  →  gap

        Demonstrated are sorted descending (strongest first), gaps ascending
        (largest deficits first).

        Args:
            cluster_id  : For Task+DWA occupation lookup.
            resume_unit : L2-normalized resume embedding (1, embedding_dim).

        Returns:
            Tuple of (demonstrated tasks, gap tasks).
        """
        if cluster_id not in self.soc_tasks:
            return [], []

        cluster_tasks = self.soc_tasks[cluster_id]
        similarities  = cosine_similarity(resume_unit, cluster_tasks.vectors)[0]
        threshold     = np.median(similarities)

        tasks = [
            TaskGap(name=name, similarity=float(score))
            for name, score in zip(cluster_tasks.labels, similarities)
        ]
        rank = lambda above: sorted(
            [t for t in tasks if (t.similarity >= threshold) == above],
            key     = lambda t: t.similarity,
            reverse = above
        )
        return rank(True), rank(False)

    def match(self, resume_text: str) -> MatchResult:
        """
        Project resume text into the career landscape and return a full
        match result with gap analysis and neighborhood view.

        Encodes the resume with the sentence transformer, L2-normalizes,
        projects through the fitted SVD, assigns to the nearest cluster
        centroid via Euclidean distance, then computes per-task cosine gaps
        and queries the career graph for neighborhood options with
        credential metadata.

            k* = argmin_k ‖𝐫 − 𝐜ₖ‖₂

        Args:
            resume_text: Raw resume text (post-PDF extraction).

        Returns:
            `MatchResult` with cluster, gaps, and neighborhood.
        """
        resume_unit = self.model.encode([resume_text])
        distances   = np.linalg.norm(
            x    = self.centroids - self.svd.transform(resume_unit)[0],
            axis = 1
        )

        cluster_id         = self.cluster_ids[(ranked := np.argsort(distances))[0]]
        demonstrated, gaps = self._gap_analysis(cluster_id, resume_unit)
        return MatchResult(
            cluster_distances = [
                ClusterDistance(
                    cluster_id = self.cluster_ids[index],
                    distance   = distances[index]
                )
                for index in ranked
            ],
            cluster_id   = cluster_id,
            demonstrated = demonstrated,
            gaps         = gaps,
            neighborhood = self.graph.neighborhood(cluster_id),
            sector       = self.profiles[cluster_id].sector
        )
