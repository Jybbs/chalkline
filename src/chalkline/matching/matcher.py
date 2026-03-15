"""
Resume matching with skill gap analysis and PMI-based gap ranking.

Projects an uploaded resume into the fitted PCA space, matches it to
the nearest career family via cluster centroids, identifies skill gaps
from neighboring postings, and ranks gaps by PPMI relevance scoped to
the matched cluster's TF-IDF centroid terms. Cross-references gaps
against AGC apprenticeship programs and educational programs for
actionable recommendations.
"""

import numpy  as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline  import Pipeline

from chalkline.clustering.schemas import ClusterLabel
from chalkline.matching.schemas   import ClusterDistance, MatchResult
from chalkline.matching.schemas   import NeighborMatch, RankedGap
from chalkline.pipeline.schemas   import ApprenticeshipContext
from chalkline.pipeline.schemas   import DistanceMetric, ProgramRecommendation


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _prefix_set(text: str) -> set[str]:
    """
    Extract 4-character word prefixes from text.

    Filters to words of 4+ characters and truncates to 4-char prefixes,
    catching inflectional variants across the construction domain
    (welding/welder, electrical/electrician, scaffolding/scaffold).

    Args:
        text: Raw text to extract prefixes from.

    Returns:
        Set of lowercased 4-character prefixes.
    """
    return {w[:4] for w in text.lower().split() if len(w) >= 4}


def jaccard(a: set[str], b: set[str]) -> float:
    """
    Jaccard similarity between two skill sets.

        J(A, B) = |A ∩ B| / |A ∪ B|

    Returns 0.0 when both sets are empty to guard against division
    by zero.

    Args:
        a : First skill set.
        b : Second skill set.

    Returns:
        Jaccard index in [0.0, 1.0].
    """
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# -------------------------------------------------------------------------
# Matcher
# -------------------------------------------------------------------------

class ResumeMatcher:
    """
    Resume projection into fitted PCA space with cluster matching
    and PPMI gap ranking.

    Receives fitted pipeline artifacts, computes cluster centroids,
    and exposes a `match()` method that projects a resume's skill
    list into the shared coordinate space, assigns it to the nearest
    career family, identifies skill gaps from neighboring postings,
    and ranks gaps by centroid-scoped mean PPMI relevance.
    """

    def __init__(
        self,
        apprenticeships   : list[dict],
        assignments       : np.ndarray,
        cluster_labels    : list[ClusterLabel],
        coordinates       : np.ndarray,
        distance_metric   : DistanceMetric,
        document_ids      : list[str],
        extracted_skills  : dict[str, list[str]],
        geometry_pipeline : Pipeline,
        ppmi_df           : pd.DataFrame,
        programs          : list[ProgramRecommendation],
        top_k_gaps        : int
    ):
        """
        Compute cluster centroids and fit nearest-neighbor models.

        The geometry pipeline must be a fitted sklearn `Pipeline`
        chaining `DictVectorizer`, `TfidfTransformer`, `Normalizer`,
        `TruncatedSVD`, and `StandardScaler` so that a single
        `transform([skill_dict])` call projects a resume into the
        shared PCA space without refitting.

        Args:
            apprenticeships   : Raw dicts from
                                `apprenticeships.json` with
                                `title`, `rapids_code`, and
                                `term_hours` keys.
            assignments       : Cluster ID per posting from
                                `HierarchicalClusterer.assignments`.
            cluster_labels    : TF-IDF centroid labels with terms
                                per cluster for PPMI scoping
                                (top-50 recommended).
            coordinates       : PCA-scaled posting coordinates of
                                shape `(n_postings, n_selected)`.
            distance_metric   : Distance function for neighbor
                                queries.
            document_ids      : Posting identifiers in matrix row
                                order.
            extracted_skills  : Mapping from document identifier to
                                sorted canonical skill names.
            geometry_pipeline : Fitted sklearn `Pipeline` chaining
                                vectorization through scaling.
            ppmi_df           : Symmetric PPMI DataFrame indexed by
                                canonical skill names.
            programs          : Normalized educational program
                                records from `load_programs`.
            top_k_gaps        : Default number of ranked gaps to
                                return.
        """
        self.assignments       = assignments
        self.coordinates       = coordinates
        self.document_ids      = document_ids
        self.extracted_skills  = extracted_skills
        self.geometry_pipeline = geometry_pipeline
        self.ppmi_df           = ppmi_df
        self.programs          = programs
        self.top_k_gaps        = top_k_gaps

        self.skill_sets = {
            doc: set(skills)
            for doc, skills in extracted_skills.items()
        }

        self.metric = (
            "cosine" if distance_metric == DistanceMetric.COSINE
            else "euclidean"
        )

        centroids_df     = pd.DataFrame(coordinates).groupby(assignments).mean()
        self.cluster_ids = centroids_df.index.to_numpy()

        self.centroid_nn = NearestNeighbors(
            metric      = self.metric,
            n_neighbors = len(self.cluster_ids)
        ).fit(centroids_df.values)

        self.centroid_scope = {
            label.cluster_id: set(label.terms)
            for label in cluster_labels
        }

        self.apprenticeship_models = [
            ApprenticeshipContext(
                rapids_code = a["rapids_code"],
                term_hours  = a["term_hours"],
                trade       = a["title"]
            )
            for a in apprenticeships
        ]

        self.trade_prefixes = {
            a.rapids_code: _prefix_set(a.trade)
            for a in self.apprenticeship_models
        }
        self.program_prefixes = {
            (p.institution, p.program): _prefix_set(p.program)
            for p in self.programs
        }

    # -----------------------------------------------------------------
    # Private methods
    # -----------------------------------------------------------------

    def _find_apprenticeships(self, skill: str) -> list[ApprenticeshipContext]:
        """
        Find apprenticeship trades matching a skill via word-root
        overlap.

        Args:
            skill: Canonical skill name to match against.

        Returns:
            Apprenticeships whose trade name shares a 4-char prefix
            with the skill.
        """
        prefixes = _prefix_set(skill)
        return [
            a for a in self.apprenticeship_models
            if prefixes & self.trade_prefixes[a.rapids_code]
        ]

    def _find_programs(self, skill: str) -> list[ProgramRecommendation]:
        """
        Find educational programs matching a skill via word-root
        overlap with the program name.

        Args:
            skill: Canonical skill name to match against.

        Returns:
            Programs whose name shares a 4-char prefix with the
            skill.
        """
        prefixes = _prefix_set(skill)
        return [
            p for p in self.programs
            if prefixes & self.program_prefixes[p.institution, p.program]
        ]

    def _nearest_in_family(
        self,
        cluster_id : int,
        coords     : np.ndarray,
        resume_set : set[str]
    ) -> list[NeighborMatch]:
        """
        Find the top nearest-neighbor postings within a cluster,
        falling back to Jaccard-ranked corpus neighbors when the
        cluster is too small.

        When the assigned cluster has 5+ postings, neighbors are
        found by PCA-space distance within the family. When the
        cluster is smaller, PCA distances are unreliable because
        the sparse feature space concentrates all postings near the
        origin, so the search falls back to Jaccard similarity on
        discrete skill sets across the full corpus. This produces
        skill-relevant neighbors even when the geometric
        representation lacks discriminative power.

        Args:
            cluster_id : Assigned cluster to search within.
            coords     : Resume PCA coordinates of shape
                         `(1, n_selected)`.
            resume_set : Resume skill names as a set.

        Returns:
            Up to 5 nearest neighbors with distances and Jaccard
            scores.
        """
        indices = np.nonzero(self.assignments == cluster_id)[0]

        if len(indices) >= 5:
            family_docs = [self.document_ids[i] for i in indices]
            nn = NearestNeighbors(
                metric      = self.metric,
                n_neighbors = min(5, len(family_docs))
            ).fit(self.coordinates[indices])
            distances, nn_indices = nn.kneighbors(coords)

            return [
                NeighborMatch(
                    distance    = float(dist),
                    document_id = doc,
                    jaccard     = jaccard(
                        resume_set, self.skill_sets[doc]
                    ),
                    skills      = self.extracted_skills[doc]
                )
                for dist, idx in zip(distances[0], nn_indices[0])
                for doc in [family_docs[idx]]
            ]

        scored = sorted(
            (
                (doc, jaccard(resume_set, skill_set))
                for doc, skill_set in self.skill_sets.items()
            ),
            key     = lambda pair: pair[1],
            reverse = True
        )[:5]

        return [
            NeighborMatch(
                distance    = 1.0 - sim,
                document_id = doc,
                jaccard     = sim,
                skills      = self.extracted_skills[doc]
            )
            for doc, sim in scored
            if sim > 0
        ]

    def _rank_gaps(
        self,
        cluster_id : int,
        resume_set : set[str],
        skill_gaps : list[str],
        top_k      : int
    ) -> tuple[list[RankedGap], list[str]]:
        """
        Rank skill gaps by centroid-scoped mean PPMI relevance.

            relevance(g) = (1/|S_r|) * sum PPMI(g,s) for s in S_r

        where S_r is the resume's existing skills restricted to the
        cluster's centroid scope. Skills outside the centroid scope
        fall back to the full PPMI matrix. Skills absent from the
        PPMI index or with zero mean relevance are returned as
        unrankable.

        Relevance scores are computed via vectorized DataFrame
        operations across two groups (centroid-scoped and unscoped)
        rather than per-gap label lookups. Enrichment with
        apprenticeship and program annotations is deferred until
        after the top-k selection to avoid wasted lookups on gaps
        that will be discarded.

        Args:
            cluster_id : Assigned cluster for centroid scoping.
            resume_set : Resume skill names as a set.
            skill_gaps : Sorted list of gap skill names.
            top_k      : Maximum number of ranked gaps to return.

        Returns:
            Tuple of (ranked gaps with enrichment, sorted list of
            unrankable skill names).
        """
        centroid_scope = self.centroid_scope.get(cluster_id, set())
        all_ref        = resume_set & set(self.ppmi_df.columns)

        if not all_ref:
            return [], sorted(skill_gaps)

        valid_set   = set(self.ppmi_df.index)
        valid       = [g for g in skill_gaps if g in valid_set]
        invalid_set = set(skill_gaps) - set(valid)

        if not valid:
            return [], sorted(invalid_set)

        scoped_ref  = all_ref & centroid_scope
        scoped_cols = list(scoped_ref) if scoped_ref else list(all_ref)
        full_cols   = list(all_ref)

        in_scope  = [g for g in valid if g in centroid_scope]
        out_scope = [g for g in valid if g not in centroid_scope]

        parts = []
        if in_scope:
            parts.append(self.ppmi_df.loc[in_scope, scoped_cols].mean(axis = 1))
        if out_scope:
            parts.append(self.ppmi_df.loc[out_scope, full_cols].mean(axis = 1))

        relevances = pd.concat(parts)
        positive   = relevances[relevances > 0].nlargest(top_k)

        unrankable = sorted(invalid_set | (set(valid) - set(positive.index)))

        ranked = [
            RankedGap(
                apprenticeships = self._find_apprenticeships(gap),
                programs        = self._find_programs(gap),
                relevance       = float(rel),
                skill           = gap
            )
            for gap, rel in positive.items()
        ]

        return ranked, unrankable

    # -----------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------

    def match(
        self,
        resume_skills : list[str],
        top_k         : int | None = None
    ) -> MatchResult:
        """
        Project a resume into PCA space and match to career families.

        Converts the resume's canonical skill names to a binary dict,
        projects through the fitted geometry pipeline, assigns to the
        nearest cluster centroid, finds top-5 neighbors within the
        cluster, computes skill gaps, ranks gaps by PPMI relevance,
        and cross-references gaps against apprenticeship and
        educational program reference data.

        Args:
            resume_skills : Sorted canonical skill names from
                            `SkillExtractor.extract()`.
            top_k         : Override for the default `top_k_gaps`.
                            Returns at most this many ranked gaps.

        Returns:
            Full `MatchResult` with cluster assignment, neighbors,
            gaps, and recommendations.
        """
        coords = self.geometry_pipeline.transform([dict.fromkeys(resume_skills, 1)])

        distances, indices = self.centroid_nn.kneighbors(coords)
        cluster_id = int(self.cluster_ids[indices[0, 0]])

        cluster_distances = [
            ClusterDistance(
                cluster_id = int(self.cluster_ids[idx]),
                distance   = float(dist)
            )
            for dist, idx in zip(distances[0], indices[0])
        ]

        resume_set = set(resume_skills)
        neighbors  = self._nearest_in_family(
            cluster_id = cluster_id,
            coords     = coords,
            resume_set = resume_set
        )

        skill_gaps = sorted(
            {skill for n in neighbors for skill in n.skills}
            - resume_set
        )
        ranked_gaps, unrankable = self._rank_gaps(
            cluster_id = cluster_id,
            resume_set = resume_set,
            skill_gaps = skill_gaps,
            top_k      = top_k if top_k is not None else self.top_k_gaps
        )

        seen_trades          = set()
        seen_programs        = set()
        all_apprenticeships  = []
        all_programs         = []

        for gap in ranked_gaps:
            for a in gap.apprenticeships:
                if a.rapids_code not in seen_trades:
                    seen_trades.add(a.rapids_code)
                    all_apprenticeships.append(a)
            for p in gap.programs:
                if (p.institution, p.program) not in seen_programs:
                    seen_programs.add((p.institution, p.program))
                    all_programs.append(p)

        return MatchResult(
            cluster_distances = cluster_distances,
            cluster_id        = cluster_id,
            nearest_neighbors = neighbors,
            programs          = all_programs,
            ranked_gaps       = ranked_gaps,
            resume_skills     = sorted(resume_skills),
            skill_gaps        = skill_gaps,
            trade_paths       = all_apprenticeships,
            unrankable_gaps   = unrankable
        )
