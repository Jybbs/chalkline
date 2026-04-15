"""
Resume matching via BM25-weighted sentence embeddings with gap analysis.

Projects an uploaded resume into the fitted SVD space, matches it to the
nearest career family via cluster centroids, identifies demonstrated
competencies and gaps via BM25-weighted cosine similarity against O*NET
Task+DWA embeddings, and assembles a reach view with per-edge credential
metadata. Task scoring uses sentence-level chunking with max-pooling so that
a specific resume line can drive high similarity for a matching task even
when the document-level mean is diluted by unrelated content. The BM25
weighting suppresses generic verbs that produce artificial similarity
between structurally parallel but semantically unrelated texts.
"""

import numpy as np

from dataclasses              import dataclass, field
from functools                import cached_property
from heapq                    import nlargest
from sklearn.decomposition    import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from chalkline.collection.schemas import Posting
from chalkline.matching.schemas   import BM25Config, MatchResult, ScoredTask
from chalkline.pathways.clusters  import Cluster, Clusters
from chalkline.pipeline.encoder   import SentenceEncoder


@dataclass(kw_only=True)
class ResumeMatcher:
    """
    BM25-weighted resume matching with reach exploration.

    Holds the sentence transformer encoder, fitted SVD, and the cluster map
    with pre-stacked centroid matrices. The `match()` method encodes resume
    text, projects it into the reduced space, assigns it to the nearest
    cluster, computes per-task gap analysis, and queries the career graph
    for the local reach view. Task scoring weights raw cosine similarity by
    a BM25 relevance signal derived from stem overlap between resume chunks
    and task descriptions, suppressing generic verbs that appear across many
    occupations while amplifying domain-specific terms.

    Args:
        clusters : Cluster map with centroids for distance computation.
        encoder  : For encoding resume text into embedding space.
        svd      : For projecting resume embeddings into reduced space.
    """

    clusters : Clusters
    encoder  : SentenceEncoder
    svd      : TruncatedSVD

    bm25                 : BM25Config     = field(default_factory=BM25Config)
    chunk_stems          : list[set[str]] = field(init=False)
    credential_threshold : float          = field(default=0.0, init=False)
    global_threshold     : float          = field(default=0.0, init=False)
    resume_chunks        : np.ndarray     = field(init=False)
    resume_embedding     : np.ndarray     = field(init=False)
    resume_svd           : np.ndarray     = field(init=False)

    @cached_property
    def stemmer(self):
        """
        Snowball stemmer for English, constructed once per session.
        """
        from nltk.stem import SnowballStemmer
        return SnowballStemmer("english")

    def _bm25_score(
        self,
        chunk_stems : set[str],
        task_stems  : set[str]
    ) -> float:
        """
        BM25 relevance score between a single resume chunk and a single task
        description, using the corpus-level IDF table from `self.clusters`.
        Task stems are sets so term frequency is always 1.

        Args:
            chunk_stems : Stemmed content words from one resume chunk.
            task_stems  : Stemmed content words from one task.
        """
        length_ratio = len(task_stems) / self.clusters.bm25_average_length
        scale        = self.bm25.numerator / (
            1 + self.bm25.saturation * (
                self.bm25.base_penalty
                + self.bm25.length_weight * length_ratio
            )
        )
        return scale * sum(
            self.clusters.bm25_idf.get(term, 0)
            for term in chunk_stems & task_stems
        )

    def _content_stems(self, text: str) -> set[str]:
        """
        Stemmed content words from text, filtering high-frequency function
        words via Zipf threshold.
        """
        from re       import findall
        from wordfreq import zipf_frequency

        return {
            self.stemmer.stem(w)
            for w in findall(r"[a-zA-Z]{3,}", text.lower())
            if zipf_frequency(w, "en") < 6.0
        }

    def _posting_similarities(self, matrix: np.ndarray) -> np.ndarray:
        """
        Cosine similarity between the stored resume embedding and each row
        of `matrix`, returned as a 1D array.

        Args:
            matrix: (n_items, embedding_dim) against `resume_embedding`.
        """
        return cosine_similarity(self.resume_embedding, matrix)[0]

    def _stem_gate(
        self,
        credentials : list,
        task_stems  : list[set[str]]
    ) -> np.ndarray:
        """
        Boolean matrix of shape `(n_credentials, n_tasks)` marking pairs
        that share at least one content stem.

        Used to gate per-task cosine similarity in credential coverage so
        zero-lexical-overlap coincidences never clear the threshold, while
        letting semantic strength alone rank the remaining candidates.
        """
        return np.array([
            [bool(c.stems & ts) for ts in task_stems]
            for c in credentials
        ])

    def _task_similarities(self, cluster: Cluster) -> np.ndarray:
        """
        BM25-weighted per-task cosine similarity via max-pooling across
        resume chunks.

        Computes raw cosine max-pool, then weights each task score by a
        normalized BM25 relevance signal derived from stem overlap between
        resume chunks and task descriptions. Generic verbs that appear
        across many occupations get suppressed by IDF weighting, while
        domain-specific terms amplify the score.

        Args:
            cluster: Career family with task embeddings and stems.
        """
        weights = np.array([
            max(self._bm25_score(cs, ts) for cs in self.chunk_stems)
            for ts in cluster.task_stems
        ])
        if weights.max() > 0:
            weights = weights / weights.max()

        return (
            weights *
            cosine_similarity(self.resume_chunks, cluster.task_matrix).max(axis=0)
        )

    def calibrate(self) -> dict[int, float]:
        """
        Set the per-task threshold for demonstrated/gap splits and return
        per-cluster match scores for the map widget.

        The threshold is the median of all BM25-weighted task scores across
        task-bearing clusters, stored on `global_threshold` for
        `score_destination` to classify individual tasks. The returned dict
        reuses `cluster_score` so map donuts and route fit percentages share
        a single SVD-derived formula.

        Returns:
            Cluster ID to match score in [0, 1].
        """
        sims = [
            self._task_similarities(c)
            for c in self.clusters.values() if c.tasks
        ]
        self.global_threshold = (
            float(np.median(np.concatenate(sims))) if sims else 0.0
        )
        return {cid: self.cluster_score(cid) for cid in self.clusters.cluster_ids}

    def calibrate_coverage(self, credentials: list, vectors: np.ndarray):
        """
        Set `credential_threshold` to the median of stem-gated cosine over
        every (credential, cluster-task) matchup in the corpus.

        Cosine carries the semantic signal; the stem gate rejects
        zero-lexical-overlap coincidences where a credential vector happens
        to sit near a task in embedding space without sharing any content
        vocabulary. BM25 magnitude is deliberately excluded because it
        scales with `embedding_text` length and systematically favors long
        program descriptions over short narrow certifications.

        Args:
            credentials : Credentials with cached `.stems`, typically
                          `graph.credential_pool`.
            vectors     : Row-stacked embedding vectors aligned with `credentials`,
                          typically `graph.credential_vectors`.
        """
        if not (tasked := [c for c in self.clusters.values() if c.tasks]):
            self.credential_threshold = 0.0
            return
        scores = np.concatenate([
            cosine_similarity(vectors, cluster.task_matrix)
            * self._stem_gate(credentials, cluster.task_stems)
            for cluster in tasked
        ], axis=None)
        positive                  = scores[scores > 0]
        self.credential_threshold = float(np.median(positive)) if positive.size else 0.0

    def cluster_score(self, cluster_id: int) -> float:
        """
        SVD-derived match score for one cluster, in [0, 1].

        A score of 1 means the resume sits exactly on the cluster centroid
        in reduced SVD space; 0 means the resume is as far from this
        centroid as the corpus-wide maximum centroid separation. The matched
        cluster scores highest by construction because it has the minimum
        distance.

            score_k = 1 − ‖𝐫 − 𝐜ₖ‖₂ / max_{i,j} ‖𝐜ᵢ − 𝐜ⱼ‖₂

        Single source of truth for the map donut and the route fit
        percentage.

        Args:
            cluster_id: ID of the cluster to score.
        """
        index    = self.clusters.cluster_ids.index(cluster_id)
        distance = float(np.linalg.norm(
            self.clusters.centroids[index] - self.resume_svd
        ))
        return 1.0 - distance / self.clusters.max_centroid_distance

    def credential_coverage(
        self,
        credentials : list,
        destination : Cluster
    ) -> dict[str, dict[int, float]]:
        """
        Credential label to the cluster task positions it covers, each mapped
        to the stem-gated cosine affinity on that task. A task is considered
        covered when gated cosine clears `credential_threshold`; keeping the
        score lets the downstream picker rank candidates by per-task semantic
        tightness rather than treating every passing credential as equal.

        Scored against every task in `destination.tasks` so callers can
        distinguish a credential's intrinsic offering from the subset that
        happens to align with a particular resume's gaps.

        Args:
            credentials : Must carry `.stems` and `.vector`. Typically sourced from
                          `graph.credential_pool`.
            destination : Cluster whose tasks are being evaluated.
        """
        if not (scored := [c for c in credentials if c.vector is not None]):
            return {}

        gated = cosine_similarity(
            np.asarray([c.vector for c in scored]),
            destination.task_matrix
        ) * self._stem_gate(scored, destination.task_stems)
        return {
            c.label: {
                int(j): float(gated[i, j])
                for j in np.flatnonzero(gated[i] >= self.credential_threshold)
            }
            for i, c in enumerate(scored)
        }

    def match(self, resume_text: str) -> MatchResult:
        """
        Project resume text into the career landscape and return the
        nearest-cluster assignment with coordinates for downstream
        composition.

        Encodes the resume as both a single document vector for centroid
        matching and as sentence-level chunks for per-task gap analysis.
        Projects the document vector through the fitted SVD and assigns to
        the nearest cluster centroid via Euclidean distance.

            k* = argmin_k ‖𝐫 − 𝐜ₖ‖₂

        The returned result carries an empty `reach`; the orchestrator
        composes reach from `CareerPathwayGraph` at the call site so the
        matcher stays free of graph-layer coupling.

        Args:
            resume_text: Raw resume text (post-PDF extraction).
        """
        from nltk.tokenize import sent_tokenize

        self.resume_embedding = self.encoder.encode([resume_text])
        sentences             = sent_tokenize(resume_text)
        self.resume_chunks    = self.encoder.encode(sentences)
        self.chunk_stems      = [self._content_stems(s) for s in sentences]

        self.resume_svd = self.svd.transform(self.resume_embedding)[0]
        distances       = np.linalg.norm(
            self.clusters.centroids - self.resume_svd, axis=1
        )
        return MatchResult(
            cluster_distances = distances.tolist(),
            cluster_id        = self.clusters.cluster_ids[np.argmin(distances)],
            coordinates       = self.resume_svd.tolist()
        )

    def score_destination(self, cluster: Cluster) -> list[ScoredTask]:
        """
        Score a destination cluster's O*NET tasks against the resume chunks
        stored from the most recent `match` call.

        When `global_threshold` has been set (by `calibrate` batch scoring),
        uses that fixed value so the demonstrated/gap split varies
        meaningfully across clusters. Falls back to the per-cluster median
        when no global threshold is available.

        Args:
            cluster: Destination career family with task embeddings.

        Returns:
            Scored tasks sorted by descending similarity.
        """
        if not cluster.tasks:
            return []

        similarities = self._task_similarities(cluster)
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

        Uses the pre-computed per-posting embeddings stored on the cluster
        from pipeline fit, avoiding re-encoding.

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
                    self._posting_similarities(cluster.embeddings)
                )
                if p.date_posted
            ),
            key = lambda pair: pair[0].date_posted or pair[0].date_collected
        )
