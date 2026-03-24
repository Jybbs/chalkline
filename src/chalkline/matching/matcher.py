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

from chalkline.matching.schemas import ClusterDistance, MatchResult, TaskGap
from chalkline.pathways.graph   import CareerPathwayGraph
from chalkline.pathways.schemas import Clusters
from chalkline.pipeline.encoder import SentenceEncoder


@dataclass(kw_only=True)
class ResumeMatcher:
    """
    Embedding-based resume matching with neighborhood exploration.

    Holds the sentence transformer encoder, fitted SVD, and the cluster map
    with pre-stacked centroid matrices. The `match()` method encodes resume
    text, projects it into the reduced space, assigns it to the nearest
    cluster, computes per-task gap analysis, and queries the career graph
    for the local neighborhood view.

    Args:
        clusters : Cluster map with centroids for distance computation.
        encoder  : For encoding resume text into embedding space.
        graph    : For neighborhood queries post-match.
        svd      : For projecting resume embeddings into reduced space.
    """

    clusters : Clusters
    encoder  : SentenceEncoder
    graph    : CareerPathwayGraph
    svd      : TruncatedSVD

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
        tasks = self.clusters[cluster_id].tasks
        if not tasks:
            return [], []

        task_matrix  = np.stack([t.vector for t in tasks])
        similarities = cosine_similarity(resume_unit, task_matrix)[0]
        pairs        = [(t.name, s) for t, s in zip(tasks, similarities)]
        threshold    = np.median(similarities)
        return (
            sorted(
                [TaskGap(name=n, similarity=s) for n, s in pairs if s >= threshold],
                key     = lambda t: t.similarity,
                reverse = True
            ),
            sorted(
                [TaskGap(name=n, similarity=s) for n, s in pairs if s < threshold],
                key = lambda t: t.similarity
            )
        )

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
        resume_unit = self.encoder.encode([resume_text])
        resume_svd  = self.svd.transform(resume_unit)[0]
        distances   = np.linalg.norm(self.clusters.centroids - resume_svd, axis=1)

        ranked             = np.argsort(distances)
        cluster_id         = self.clusters.cluster_ids[ranked[0]]
        demonstrated, gaps = self._gap_analysis(cluster_id, resume_unit)
        return MatchResult(
            cluster_distances = [
                ClusterDistance(
                    cluster_id = self.clusters.cluster_ids[index],
                    distance   = distances[index]
                )
                for index in ranked
            ],
            cluster_id   = cluster_id,
            coordinates  = resume_svd.tolist(),
            demonstrated = demonstrated,
            gaps         = gaps,
            neighborhood = self.graph.neighborhood(cluster_id),
            sector       = self.clusters[cluster_id].sector
        )
