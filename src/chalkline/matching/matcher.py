"""
Resume matching via BM25-weighted sentence embeddings with gap analysis.

Projects an uploaded resume into the fitted SVD space, matches it to the
nearest career family via cluster centroids, identifies demonstrated
competencies and gaps via BM25-weighted cosine similarity against O*NET
Task+DWA embeddings, and assembles a reach view with per-route credential
coverage.

Task scoring uses sentence-level chunking with max-pooling so that a
specific resume line can drive high similarity for a matching task even when
the document-level mean is diluted by unrelated content. Credential coverage
uses the same BM25 weighting with per-credential row normalization so the
downstream DPP picker can distinguish narrow certifications from broad
apprenticeships when computing per-credential gap-coverage strength.
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

    def _bm25_weights(
        self,
        stem_sets   : list[set[str]],
        task_stems  : list[set[str]],
        task_matrix : np.ndarray | None = None,
        vectors     : np.ndarray | None = None
    ) -> np.ndarray:
        """
        BM25 weight matrix of shape `(len(stem_sets), len(task_stems))`.

        When `vectors` and `task_matrix` are provided, row-normalizes and
        multiplies by cosine similarity in one shot, returning the full
        BM25-weighted cosine matrix used by credential coverage. When
        omitted, returns raw weights for callers that need their own
        aggregation (column-max for resume chunks).

        Args:
            stem_sets   : One stem set per row (credential stems or resume chunk stems).
            task_stems  : One stem set per task in the target cluster.
            task_matrix : Cluster task embedding matrix for cosine.
            vectors     : Row-stacked embeddings aligned with `stem_sets`.
        """
        weights = np.array([
            [
                self.bm25.numerator / (
                    self.bm25.base_denominator + self.bm25.length_scale * len(ts)
                    / self.clusters.bm25_average_length
                )
                * sum(self.clusters.bm25_idf.get(t, 0) for t in ss & ts)
                for ts in task_stems
            ]
            for ss in stem_sets
        ])

        if vectors is None or task_matrix is None:
            return weights

        row_max = weights.max(axis=1, keepdims=True)
        np.divide(weights, row_max, out=weights, where=row_max > 0)
        return weights * cosine_similarity(vectors, task_matrix)

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
        weights = self._bm25_weights(self.chunk_stems, cluster.task_stems).max(axis=0)
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
        Set `credential_threshold` to the median of BM25-weighted cosine
        over every (credential, cluster-task) matchup in the corpus.

        BM25 weighting replaces the binary stem gate so that credential
        reach discriminates by term specificity rather than any-overlap
        presence. Per-credential row normalization prevents long
        `embedding_text` from systematically outscoring short narrow
        certifications.

        Args:
            credentials : Credentials with cached `.stems`, typically
                          `graph.credential_pool`.
            vectors     : Row-stacked embedding vectors aligned with `credentials`,
                          typically `graph.credential_vectors`.
        """
        if not (tasked := [c for c in self.clusters.values() if c.tasks]):
            self.credential_threshold = 0.0
            return
        stem_sets = [c.stems for c in credentials]
        scores    = np.concatenate([
            self._bm25_weights(stem_sets, c.task_stems, c.task_matrix, vectors)
            for c in tasked
        ], axis=None)
        positive = scores[scores > 0]
        self.credential_threshold = float(np.percentile(positive, 75)) if positive.size else 0.0

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
        Credential label to the cluster task positions it covers, each
        mapped to the BM25-weighted cosine affinity on that task.

        BM25 weighting produces sharper per-task discrimination than the
        binary stem gate, so the downstream DPP picker can distinguish
        narrow credentials from broad ones when computing each credential's
        gap-coverage quality. Per-credential row normalization follows the
        same pattern as `_task_similarities` to prevent `embedding_text`
        length bias.

        Args:
            credentials : Must carry `.stems` and `.vector`, typically from
                          `graph.credential_pool`.
            destination : Cluster whose tasks are being evaluated.
        """
        if not (scored := [c for c in credentials if c.vector is not None]):
            return {}

        weighted = self._bm25_weights(
            [c.stems for c in scored],
            destination.task_stems,
            destination.task_matrix,
            np.asarray([c.vector for c in scored])
        )
        return {
            c.label: {
                int(j): float(weighted[i, j])
                for j in np.flatnonzero(weighted[i] >= self.credential_threshold)
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
