"""
Resume matching via sentence embeddings with per-task gap analysis.

Projects an uploaded resume into the fitted SVD space, matches it to the
nearest career family via cluster centroids, identifies demonstrated
competencies and gaps via per-task cosine similarity against O*NET Task+DWA
embeddings, and assembles a reach view with per-edge credential metadata.
"""

import numpy as np

from dataclasses              import dataclass, field
from heapq                    import nlargest
from sklearn.decomposition    import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from chalkline.collection.schemas import Posting
from chalkline.matching.schemas   import MatchResult, ScoredTask
from chalkline.pathways.clusters  import Cluster, Clusters
from chalkline.pathways.graph     import CareerPathwayGraph
from chalkline.pipeline.encoder   import SentenceEncoder


@dataclass(kw_only=True)
class ResumeMatcher:
    """
    Embedding-based resume matching with reach exploration.

    Holds the sentence transformer encoder, fitted SVD, and the cluster map
    with pre-stacked centroid matrices. The `match()` method encodes resume
    text, projects it into the reduced space, assigns it to the nearest
    cluster, computes per-task gap analysis, and queries the career graph
    for the local reach view.

    Args:
        clusters : Cluster map with centroids for distance computation.
        encoder  : For encoding resume text into embedding space.
        graph    : For reach queries post-match.
        svd      : For projecting resume embeddings into reduced space.
    """

    clusters : Clusters
    encoder  : SentenceEncoder
    graph    : CareerPathwayGraph
    svd      : TruncatedSVD

    global_threshold : float      = field(default=0.0, init=False)
    resume_embedding : np.ndarray = field(init=False)

    def _similarities(self, matrix: np.ndarray) -> np.ndarray:
        """
        Cosine similarity between the stored resume embedding and each
        row of `matrix`, returned as a 1D array.

        Args:
            matrix: (n_items, embedding_dim) against `resume_embedding`.
        """
        return cosine_similarity(self.resume_embedding, matrix)[0]

    def calibrate(self) -> dict[int, float]:
        """
        Compute per-cluster mean task similarity against the stored
        resume embedding and set the global threshold for downstream
        scoring.

        Builds a similarity dict for all task-bearing clusters, stores
        the median across all individual task scores as
        `global_threshold`, and returns the per-cluster means used
        by the map widget for match percentage rendering.

        Returns:
            Cluster ID to mean cosine similarity (0.0 for taskless
            clusters).
        """
        sims = {
            c.cluster_id: self._similarities(c.task_matrix)
            for c in self.clusters.values() if c.tasks
        }
        self.global_threshold = (
            float(np.median(np.concatenate(list(sims.values()))))
            if sims else 0.0
        )
        return {
            cid: float(s.mean()) if (s := sims.get(cid)) is not None else 0.0
            for cid in self.clusters
        }

    def match(self, resume_text: str) -> MatchResult:
        """
        Project resume text into the career landscape and return a full
        match result with gap analysis and reach view.

        Encodes the resume with the sentence transformer, L2-normalizes,
        projects through the fitted SVD, assigns to the nearest cluster
        centroid via Euclidean distance, then computes per-task cosine gaps
        and queries the career graph for reach options with credential
        metadata.

            k* = argmin_k ‖𝐫 − 𝐜ₖ‖₂

        Args:
            resume_text: Raw resume text (post-PDF extraction).

        Returns:
            `MatchResult` with cluster, gaps, and reach.
        """
        self.resume_embedding = self.encoder.encode([resume_text])

        resume_svd = self.svd.transform(self.resume_embedding)[0]
        distances  = np.linalg.norm(self.clusters.centroids - resume_svd, axis=1)
        cluster_id = self.clusters.cluster_ids[np.argmin(distances)]
        return MatchResult(
            cluster_distances = distances.tolist(),
            cluster_id        = cluster_id,
            coordinates       = resume_svd.tolist(),
            reach             = self.graph.reach(cluster_id)
        )

    def score_destination(self, cluster: Cluster) -> list[ScoredTask]:
        """
        Score a destination cluster's O*NET tasks against the user's
        resume embedding stored from the most recent `match` call.

        When `global_threshold` has been set (by `calibrate` batch
        scoring), uses that fixed value so the demonstrated/gap
        split varies meaningfully across clusters. Falls back to
        the per-cluster median when no global threshold is available.

        Args:
            cluster: Destination career family with task embeddings.

        Returns:
            Scored tasks sorted by descending similarity.
        """
        if not cluster.tasks:
            return []

        similarities = self._similarities(cluster.task_matrix)
        threshold    = self.global_threshold or float(np.median(similarities))
        return [
            ScoredTask(
                demonstrated = (s := float(similarities[i])) >= threshold,
                name         = cluster.tasks[i].name,
                similarity   = s
            )
            for i in np.argsort(-similarities)
        ]

    def score_postings(
        self,
        cluster : Cluster,
        limit   : int = 30
    ) -> list[tuple[Posting, float]]:
        """
        Score a cluster's postings by cosine similarity to the resume,
        returned in reverse chronological order.

        Uses the pre-computed per-posting embeddings stored on the
        cluster from pipeline fit, avoiding re-encoding.

        Args:
            cluster : Career family with per-posting embeddings.
            limit   : Maximum postings to return.

        Returns:
            (posting, similarity) pairs sorted most recent first.
        """
        return nlargest(
            limit,
            (
                (p, float(s))
                for p, s in zip(
                    cluster.postings,
                    self._similarities(cluster.embeddings)
                )
                if p.date_posted
            ),
            key = lambda pair: pair[0].date_posted or pair[0].date_collected
        )
