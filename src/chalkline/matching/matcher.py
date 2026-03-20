"""
Resume matching via sentence embeddings with per-task gap analysis.

Projects an uploaded resume into the fitted SVD space, matches it to the
nearest career family via cluster centroids, identifies demonstrated
competencies and gaps via per-task cosine similarity against O*NET Task+DWA
embeddings, and assembles a neighborhood view with per-edge credential
metadata.
"""

import numpy as np

from loguru                   import logger
from sentence_transformers    import SentenceTransformer
from sklearn.decomposition    import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing    import normalize

from chalkline.matching.schemas import ClusterDistance, MatchResult, TaskGap
from chalkline.pipeline.graph   import CareerPathwayGraph
from chalkline.pipeline.schemas import ClusterProfile


class ResumeMatcher:
    """
    Embedding-based resume matching with neighborhood exploration.

    Holds the sentence transformer model, fitted SVD, cluster centroids, and
    per-cluster O*NET task embeddings. The `match()` method encodes resume
    text, projects it into the reduced space, assigns it to the nearest
    cluster, computes per-task gap analysis, and queries the career graph
    for the local neighborhood view.
    """

    def __init__(
        self,
        centroids    : np.ndarray,
        cluster_ids  : list[int],
        graph        : CareerPathwayGraph,
        model        : SentenceTransformer,
        profiles     : dict[int, ClusterProfile],
        svd          : TruncatedSVD,
        task_labels  : dict[int, list[str]],
        task_vectors : dict[int, np.ndarray],
        max_gaps     : int = 10
    ):
        """
        Args:
            centroids    : (n_clusters, n_components) in SVD-reduced space.
            cluster_ids  : Aligned with centroid rows.
            graph        : For neighborhood queries post-match.
            model        : For encoding resume text into embedding space.
            profiles     : For sector lookup on the matched cluster.
            svd          : For projecting resume embeddings into reduced space.
            task_labels  : Per-cluster Task+DWA names aligned with vector rows.
            task_vectors : Per-cluster (n_tasks, embedding_dim) L2-normalized.
            max_gaps     : Maximum number of gaps to return.
        """
        self.centroids    = centroids
        self.cluster_ids  = cluster_ids
        self.graph        = graph
        self.max_gaps     = max_gaps
        self.model        = model
        self.profiles     = profiles
        self.svd          = svd
        self.task_labels  = task_labels
        self.task_vectors = task_vectors

        logger.info(f"Matcher ready for {len(cluster_ids)} clusters")

    def _task_gap_analysis(
        self,
        cluster_id  : int,
        resume_unit : np.ndarray,
        top_k       : int
    ) -> tuple[list[TaskGap], list[TaskGap]]:
        """
        Split O*NET tasks into demonstrated and gap categories via per-task
        cosine similarity against the resume embedding.

            cos(𝐫, 𝐭ᵢ) ≥ S̃  →  demonstrated
            cos(𝐫, 𝐭ᵢ) < S̃  →  gap

        Both lists are sorted by similarity, descending for demonstrated and
        ascending for gaps.

        Args:
            cluster_id  : For Task+DWA occupation lookup.
            resume_unit : L2-normalized resume embedding (1, embedding_dim).
            top_k       : Maximum number of gaps to return.

        Returns:
            Tuple of (demonstrated tasks, gap tasks).
        """
        if cluster_id not in self.task_vectors:
            return [], []

        embeddings   = self.task_vectors[cluster_id]
        names        = self.task_labels[cluster_id]
        similarities = cosine_similarity(resume_unit, embeddings)[0]
        threshold    = np.median(similarities)

        demonstrated = sorted(
            [
                TaskGap(name=name, similarity=score)
                for name, score in zip(names, similarities)
                if score >= threshold
            ],
            key=lambda gap: -gap.similarity
        )

        gaps = sorted(
            [
                TaskGap(name=name, similarity=score)
                for name, score in zip(names, similarities)
                if score < threshold
            ],
            key=lambda gap: gap.similarity
        )[:top_k]

        return demonstrated, gaps

    def match(
        self,
        resume_text : str,
        top_k       : int | None = None
    ) -> MatchResult:
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
            resume_text : Raw resume text (post-PDF extraction).
            top_k       : Override for the default `max_gaps`.

        Returns:
            `MatchResult` with cluster, gaps, and neighborhood.
        """
        resume_embedding   = self.model.encode([resume_text], show_progress_bar=False)
        resume_unit        = normalize(resume_embedding)
        resume_coordinates = self.svd.transform(resume_unit)[0]

        distances       = np.linalg.norm(self.centroids - resume_coordinates, axis=1)
        ranked_indices  = np.argsort(distances)
        nearest_cluster = self.cluster_ids[ranked_indices[0]]

        demonstrated, gaps = self._task_gap_analysis(
            cluster_id  = nearest_cluster,
            resume_unit = resume_unit,
            top_k       = top_k or self.max_gaps
        )

        return MatchResult(
            cluster_distances = [
                ClusterDistance(
                    cluster_id = self.cluster_ids[index],
                    distance   = distances[index]
                )
                for index in ranked_indices
            ],
            cluster_id   = nearest_cluster,
            demonstrated = demonstrated,
            gaps         = gaps,
            neighborhood = self.graph.neighborhood(nearest_cluster),
            sector       = self.profiles[nearest_cluster].sector
        )
