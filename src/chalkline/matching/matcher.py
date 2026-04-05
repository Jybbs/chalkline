"""
Resume matching via sentence embeddings with per-task gap analysis.

Projects an uploaded resume into the fitted SVD space, matches it to the
nearest career family via cluster centroids, identifies demonstrated
competencies and gaps via per-task cosine similarity against O*NET Task+DWA
embeddings, and assembles a reach view with per-edge credential metadata.
"""

import numpy as np

from dataclasses              import dataclass
from sklearn.decomposition    import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from chalkline.matching.schemas  import ClusterDistance, MatchResult, ScoredTask
from chalkline.pathways.clusters import Clusters
from chalkline.pathways.graph    import CareerPathwayGraph
from chalkline.pipeline.encoder  import SentenceEncoder


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

    def _score_tasks(
        self,
        cluster_id  : int,
        resume_unit : np.ndarray
    ) -> list[ScoredTask]:
        """
        Score O*NET tasks by cosine similarity to the resume and flag each
        as demonstrated (>= median) or gap (< median).

        Results are sorted by descending similarity so strengths appear
        first and the largest gaps appear last.

        Args:
            cluster_id  : For Task+DWA occupation lookup.
            resume_unit : L2-normalized resume embedding (1, embedding_dim).
        """
        tasks = self.clusters[cluster_id].tasks
        if not tasks:
            return []

        task_matrix  = np.stack([t.vector for t in tasks])
        similarities = np.clip(
            cosine_similarity(resume_unit, task_matrix)[0],
            -1.0,
            1.0,
        )
        threshold    = float(np.median(similarities))
        return sorted(
            [
                ScoredTask(
                    demonstrated = s >= threshold,
                    name         = t.name,
                    similarity   = s,
                    skill_type   = t.skill_type
                )
                for t, s in zip(tasks, similarities)
            ],
            key     = lambda t: t.similarity,
            reverse = True
        )

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
        resume_unit = self.encoder.encode([resume_text])
        resume_svd  = self.svd.transform(resume_unit)[0]
        distances   = np.linalg.norm(self.clusters.centroids - resume_svd, axis=1)

        ranked     = np.argsort(distances)
        cluster_id = self.clusters.cluster_ids[ranked[0]]
        return MatchResult(
            cluster_distances = [
                ClusterDistance(
                    cluster_id = (cid := self.clusters.cluster_ids[index]),
                    distance   = distances[index],
                    job_zone   = (c := self.clusters[cid]).job_zone,
                    sector     = c.sector,
                    soc_title  = c.soc_title
                )
                for index in ranked
            ],
            cluster_id   = cluster_id,
            coordinates  = resume_svd.tolist(),
            reach        = self.graph.reach(cluster_id),
            scored_tasks = self._score_tasks(cluster_id, resume_unit),
            sector       = self.clusters[cluster_id].sector
        )
