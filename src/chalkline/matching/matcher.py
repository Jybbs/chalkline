"""
Resume matching with skill gap analysis and PMI-based gap ranking.

Projects an uploaded resume into the fitted PCA space, matches it to the
nearest career family via cluster centroids, identifies skill gaps from
neighboring postings, and ranks gaps by PPMI relevance scoped to the matched
cluster's TF-IDF centroid terms. Cross-references gaps against AGC
apprenticeship programs and educational programs for actionable
recommendations.
"""

import numpy  as np
import pandas as pd

from functools         import cached_property
from loguru            import logger
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline  import Pipeline

from chalkline                         import SkillMap
from chalkline.clustering.hierarchical import HierarchicalClusterer
from chalkline.clustering.schemas      import ClusterLabel
from chalkline.matching.schemas        import ClusterDistance, MatchResult
from chalkline.matching.schemas        import NeighborMatch, RankedGap
from chalkline.pipeline.schemas        import ClusterProfile
from chalkline.pipeline.trades         import TradeIndex

def jaccard(a: set[str], b: set[str]) -> float:
    """
    Jaccard similarity between two skill sets.

        J(A, B) = |A ∩ B| / |A ∪ B|

    Returns 0.0 when both sets are empty to guard against division by zero.

    Args:
        a : First skill set.
        b : Second skill set.

    Returns:
        Jaccard index in [0.0, 1.0].
    """
    return len(a & b) / len(u) if (u := a | b) else 0.0

class ResumeMatcher:
    """
    Resume projection into fitted PCA space with cluster matching and PPMI
    gap ranking.

    Receives fitted pipeline artifacts via the orchestrator's shared state
    objects, computes cluster centroids, and exposes a `match()` method that
    projects a resume's skill list into the shared coordinate space, assigns
    it to the nearest career family, identifies skill gaps from neighboring
    postings, and ranks gaps by centroid-scoped mean PPMI relevance.
    """

    def __init__(
        self,
        cluster_labels    : list[ClusterLabel],
        clusterer         : HierarchicalClusterer,
        extracted_skills  : SkillMap,
        geometry_pipeline : Pipeline,
        ppmi_df           : pd.DataFrame,
        profiles          : dict[int, ClusterProfile],
        trades            : TradeIndex,
        metric            : str = "euclidean",
        top_k_gaps        : int = 10
    ):
        """
        Compute cluster centroids and fit nearest-neighbor models.

        The `clusterer` provides assignments, coordinates, and document
        identifiers from the HAC step. The `trades` index carries
        precomputed prefix lookup dicts for apprenticeship and program
        lookup, while `ppmi_df` provides the PPMI DataFrame for gap ranking.
        The geometry pipeline must be a fitted sklearn `Pipeline` chaining
        `DictVectorizer`, `TfidfTransformer`, `Normalizer`, `TruncatedSVD`,
        and `StandardScaler` so that a single `transform([skill_dict])` call
        projects a resume into the shared PCA space.

        Args:
            cluster_labels    : TF-IDF centroid labels per cluster.
            clusterer         : Fitted hierarchical clusterer with
                                assignments, coordinates, and document
                                identifiers.
            extracted_skills  : Skills per document identifier.
            geometry_pipeline : Fitted vectorization pipeline.
            ppmi_df           : PPMI DataFrame for gap ranking.
            profiles          : Cluster profiles with sector data.
            trades            : Precomputed prefix lookup index.
            metric            : sklearn distance metric string.
            top_k_gaps        : Default number of ranked gaps.
        """
        self.cluster_labels    = cluster_labels
        self.clusterer         = clusterer
        self.extracted_skills  = extracted_skills
        self.geometry_pipeline = geometry_pipeline
        self.ppmi_df           = ppmi_df
        self.profiles          = profiles
        self.trades            = trades
        self.metric            = metric
        self.top_k_gaps        = top_k_gaps
        self.cluster_ids       = clusterer.cluster_ids
        self.centroid_nn       = NearestNeighbors(
            metric      = self.metric,
            n_neighbors = len(self.cluster_ids)
        ).fit(
            np.array([
                clusterer.coordinates[clusterer.assignments == cid].mean(axis = 0)
                for cid in self.cluster_ids
            ])
        )

        logger.info(
            f"Fitted centroid NN for {len(self.cluster_ids)} clusters"
        )

    @cached_property
    def centroid_scope(self) -> dict[int, set[str]]:
        """
        Top TF-IDF terms per cluster for scoping gap relevance.
        """
        return {
            label.cluster_id: set(label.terms)
            for label in self.cluster_labels
        }

    @cached_property
    def family_nn(self) -> dict[int, tuple[list[str], NearestNeighbors]]:
        """
        Pre-fit nearest-neighbor models per cluster with 5+ members,
        avoiding repeated sklearn fitting on each `match()` call.
        """
        return {
            cid: (
                [self.clusterer.document_ids[i] for i in idx],
                NearestNeighbors(
                    metric      = self.metric,
                    n_neighbors = 5
                ).fit(self.clusterer.coordinates[idx])
            )
            for cid in self.cluster_ids
            for idx in [np.nonzero(self.clusterer.assignments == cid)[0]]
            if len(idx) >= 5
        }

    @cached_property
    def ppmi_skills(self) -> frozenset[str]:
        """
        Skill names present in the PPMI matrix, cached because the DataFrame
        is immutable after pipeline fitting.
        """
        return frozenset(self.ppmi_df.index)

    @cached_property
    def skill_sets(self) -> dict[str, set[str]]:
        """
        Skill names as sets per document for Jaccard computation.
        """
        return {
            doc: set(skills)
            for doc, skills in self.extracted_skills.items()
        }

    def _nearest_in_family(
        self,
        cluster_id : int,
        coords     : np.ndarray,
        resume_set : set[str]
    ) -> list[NeighborMatch]:
        """
        Find the top nearest-neighbor postings within a cluster, falling
        back to Jaccard-ranked corpus neighbors when the cluster is too
        small.

        When the assigned cluster has 5+ postings, neighbors are found by
        PCA-space distance within the family. When the cluster is smaller,
        PCA distances are unreliable because the sparse feature space
        concentrates all postings near the origin, so the search falls back
        to Jaccard similarity on discrete skill sets across the full corpus.
        This produces skill-relevant neighbors even when the geometric
        representation lacks discriminative power.

        Args:
            cluster_id : Assigned cluster to search within.
            coords     : Resume PCA coordinates of shape `(1, n_selected)`.
            resume_set : Resume skill names as a set.

        Returns:
            Up to 5 nearest neighbors with distances and Jaccard scores.
        """
        if cluster_id in self.family_nn:
            doc_ids, nn   = self.family_nn[cluster_id]
            dists, nn_idx = nn.kneighbors(coords)

            return [
                NeighborMatch(
                    distance    = dist,
                    document_id = doc_ids[idx],
                    jaccard     = jaccard(resume_set, self.skill_sets[doc_ids[idx]]),
                    skills      = self.extracted_skills[doc_ids[idx]]
                )
                for dist, idx in zip(dists[0], nn_idx[0])
            ]

        logger.debug(
            f"Cluster {cluster_id} has <5 members, "
            f"falling back to corpus-wide Jaccard"
        )
        return [
            NeighborMatch(
                distance    = 1.0 - sim,
                document_id = doc,
                jaccard     = sim,
                skills      = self.extracted_skills[doc]
            )
            for doc, sim in sorted(
                (
                    (doc, jaccard(resume_set, skills))
                    for doc, skills in self.skill_sets.items()
                ),
                key     = lambda p: p[1],
                reverse = True
            )[:5]
            if sim > 0
        ]

    def _rank_gaps(
        self,
        cluster_id : int,
        neighbors  : list[NeighborMatch],
        resume_set : set[str],
        top_k      : int
    ) -> tuple[list[RankedGap], set[str]]:
        """
        Derive skill gaps from neighbors and rank by centroid-scoped mean
        PPMI relevance.

            relevance(g) = (1/|S_r|) * sum PPMI(g,s) for s in S_r

        where S_r is the resume's existing skills restricted to the
        cluster's centroid scope. Gaps are the set difference between all
        neighbor skills and the resume's own skills. Skills outside the
        centroid scope fall back to the full PPMI matrix. Skills absent from
        the PPMI index or with zero mean relevance are returned as
        unrankable.

        Relevance scores are computed via vectorized DataFrame operations
        across two groups (centroid-scoped and unscoped) rather than per-gap
        label lookups. Enrichment with apprenticeship and program
        annotations is deferred until after the top-k selection to avoid
        wasted lookups on gaps that will be discarded.

        Args:
            cluster_id : Assigned cluster for centroid scoping.
            neighbors  : Nearest-neighbor postings whose skills define the
                         gap set.
            resume_set : Resume skill names as a set.
            top_k      : Maximum number of ranked gaps to return.

        Returns:
            Tuple of (ranked gaps with enrichment, unrankable gap set).
        """
        skill_gaps = (
            {skill for n in neighbors for skill in n.skills}
            - resume_set
        )

        scope = self.centroid_scope[cluster_id]
        ref   = resume_set & self.ppmi_skills
        valid = skill_gaps & self.ppmi_skills

        if not ref or not valid:
            return [], skill_gaps

        positive = pd.concat([
            self.ppmi_df.loc[list(g), c].mean(axis=1)
            for g, c in (
                (valid & scope, list(ref & scope or ref)),
                (valid - scope, list(ref))
            )
            if g
        ])[lambda s: s > 0].nlargest(top_k)

        return [
            RankedGap(
                apprenticeships = apps,
                programs        = progs,
                relevance       = rel,
                skill           = gap
            )
            for gap, rel in positive.items()
            for apps, progs in [self.trades.lookup([gap])]
        ], skill_gaps - set(positive.index)

    def match(
        self,
        skills : list[str],
        top_k  : int | None = None
    ) -> MatchResult:
        """
        Project a resume into PCA space and match to career families.

        Converts the resume's canonical skill names to a binary dict,
        projects through the fitted geometry pipeline, assigns to the
        nearest cluster centroid, and finds top-5 neighbors within the
        cluster. Gap derivation and PPMI ranking are handled by
        `_rank_gaps`, with deduplication of programs and apprenticeships
        computed at construction time.

        Args:
            skills : Sorted canonical skill names from
                     `SkillExtractor.extract()`.
            top_k  : Override for the default `top_k_gaps`. Returns at most
                     this many ranked gaps.

        Returns:
            Full `MatchResult` with cluster assignment, neighbors, gaps, and
            recommendations.
        """
        coords = self.geometry_pipeline.transform(
            [dict.fromkeys(skills, 1)]
        )

        distances, indices = self.centroid_nn.kneighbors(coords)
        cluster_id = self.cluster_ids[indices[0, 0]]
        resume_set = set(skills)
        neighbors  = self._nearest_in_family(
            cluster_id = cluster_id,
            coords     = coords,
            resume_set = resume_set
        )

        ranked_gaps, unrankable = self._rank_gaps(
            cluster_id = cluster_id,
            neighbors  = neighbors,
            resume_set = resume_set,
            top_k      = top_k or self.top_k_gaps
        )

        return MatchResult(
            cluster_distances = [
                ClusterDistance(
                    cluster_id = self.cluster_ids[idx],
                    distance   = dist
                )
                for dist, idx in zip(distances[0], indices[0])
            ],
            cluster_id        = cluster_id,
            nearest_neighbors = neighbors,
            pca_coordinates   = coords[0].tolist(),
            programs          = list({
                (p.institution, p.program): p
                for gap in ranked_gaps for p in gap.programs
            }.values()),
            ranked_gaps       = ranked_gaps,
            sector            = self.profiles[cluster_id].sector,
            skills            = sorted(skills),
            skill_gaps        = sorted(
                {g.skill for g in ranked_gaps} | unrankable
            ),
            trade_paths       = list({
                a.rapids_code: a
                for gap in ranked_gaps for a in gap.apprenticeships
            }.values()),
            unrankable_gaps   = sorted(unrankable)
        )
