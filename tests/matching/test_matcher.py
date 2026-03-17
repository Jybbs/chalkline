"""
Tests for resume matching, gap analysis, and PPMI gap ranking.
"""

import numpy  as np
import pandas as pd

from pytest           import mark, param
from sklearn.pipeline import Pipeline

from chalkline.matching.matcher  import ResumeMatcher, jaccard
from chalkline.pipeline.enrichment import prefix_set
from chalkline.matching.schemas import MatchResult

class TestResumeMatcher:
    """
    Verify resume projection, cluster assignment, neighbor retrieval,
    skill gap computation, PPMI gap ranking, and enrichment
    cross-referencing.
    """

    def test_prefix_inflection(self):
        """
        4-char prefix catches inflectional variants that the enrichment
        pipeline relies on for apprenticeship and program matching.
        """
        assert prefix_set("welding") & prefix_set("Welder")
        assert prefix_set("electrical wiring") & prefix_set("Electrician")
        assert not prefix_set("scaffolding") & prefix_set("concrete")

    def test_prefix_short_words(self):
        """
        Words shorter than 4 characters are excluded from the prefix
        set to avoid false positives on articles and prepositions.
        """
        assert prefix_set("the NEC code") == {"code"}
        assert prefix_set("on") == set()

    @mark.parametrize("a, b, expected", [
        param({"a", "b"}, {"c", "d"}, 0.0,   id = "disjoint"),
        param(set(),      set(),      0.0,   id = "empty"),
        param({"x", "y", "z"}, {"x", "y", "z"}, 1.0, id = "identical"),
        param({"a", "b", "c"}, {"b", "c", "d"}, 0.5, id = "partial")
    ])
    def test_jaccard(self, a: set[str], b: set[str], expected: float):
        """
        Jaccard similarity returns the correct value for each input
        configuration.
        """
        assert jaccard(a, b) == expected

    def test_assignment_singular(
        self,
        match_result : MatchResult,
        matcher      : ResumeMatcher
    ):
        """
        Career family assignment produces exactly one cluster.
        """
        assert isinstance(match_result.cluster_id, int)
        assert match_result.cluster_id in matcher.cluster_ids

    def test_cluster_distances(self, match_result: MatchResult, matcher: ResumeMatcher):
        """
        Cluster distances cover every cluster in sorted order.
        """
        assert len(match_result.cluster_distances) == len(matcher.cluster_ids)
        distances = [cd.distance for cd in match_result.cluster_distances]
        assert distances == sorted(distances)

    def test_empty_resume(self, matcher: ResumeMatcher):
        """
        An empty skill list produces a valid result without crashing,
        returning no gaps and no ranked gaps.
        """
        result = matcher.match([])
        assert result.skill_gaps == []
        assert result.ranked_gaps == []
        assert isinstance(result.cluster_id, int)

    def test_projection_immutable(
        self,
        geometry_pipeline : Pipeline,
        matcher           : ResumeMatcher,
        resume_skills     : list[str]
    ):
        """
        Projecting a resume does not mutate the fitted transforms.
        """
        probe  = [dict.fromkeys(resume_skills, 1)]
        before = geometry_pipeline.transform(probe).copy()
        matcher.match(resume_skills)
        after = geometry_pipeline.transform(probe)
        np.testing.assert_array_equal(before, after)

    def test_neighbors_bounded(self, match_result: MatchResult):
        """
        Nearest neighbors returns at most 5 results.
        """
        assert 1 <= len(match_result.nearest_neighbors) <= 5

    def test_neighbors_jaccard_fallback(
        self,
        matcher       : ResumeMatcher,
        resume_skills : list[str]
    ):
        """
        Clusters with fewer than 5 postings fall back to corpus-wide
        Jaccard similarity, returning neighbors ranked by skill
        overlap rather than PCA distance. A bug here silently returns
        wrong neighbors for small clusters in production.
        """
        coords = matcher.geometry_pipeline.transform(
            [dict.fromkeys(resume_skills, 1)]
        )
        neighbors = matcher._nearest_in_family(
            cluster_id = 9999,
            coords     = coords,
            resume_set = set(resume_skills)
        )
        assert len(neighbors) > 0
        jaccards = [n.jaccard for n in neighbors]
        assert jaccards == sorted(jaccards, reverse = True)
        assert all(n.distance == 1.0 - n.jaccard for n in neighbors)

    def test_neighbors_sorted(self, match_result: MatchResult):
        """
        Nearest neighbors are sorted by ascending distance.
        """
        distances = [n.distance for n in match_result.nearest_neighbors]
        assert distances == sorted(distances)

    def test_gap_empty_perfect(
        self,
        extracted_skills : dict[str, list[str]],
        matcher          : ResumeMatcher
    ):
        """
        Skill gap is empty when resume contains all target skills.
        """
        all_skills = sorted({
            skill for skills in extracted_skills.values()
            for skill in skills
        })
        assert matcher.match(all_skills).skill_gaps == []

    def test_gap_excludes_resume(
        self,
        match_result  : MatchResult,
        resume_skills : list[str]
    ):
        """
        No resume skill appears in the skill gap set.
        """
        assert set(resume_skills).isdisjoint(set(match_result.skill_gaps))

    def test_gap_nonempty_partial(self, match_result: MatchResult):
        """
        A partial resume produces at least one skill gap.
        """
        assert len(match_result.skill_gaps) > 0

    def test_dedup_unique(self, match_result: MatchResult):
        """
        Aggregate `trade_paths` and `programs` contain no duplicates.

        The deduplication loop in `match()` filters by `rapids_code`
        for apprenticeships and by `(institution, program)` for
        programs. A broken dedup would repeat entries in the career
        report.
        """
        trade_codes = [a.rapids_code for a in match_result.trade_paths]
        prog_keys   = [(p.institution, p.program) for p in match_result.programs]
        assert len(trade_codes) == len(set(trade_codes))
        assert len(prog_keys) == len(set(prog_keys))

    def test_gaps_ranked(self, match_result: MatchResult):
        """
        Ranked gaps are sorted by descending PPMI relevance.
        """
        relevances = [g.relevance for g in match_result.ranked_gaps]
        assert relevances == sorted(relevances, reverse = True)

    def test_rank_all_unrankable(self):
        """
        Gap skills absent from the PPMI vocabulary are returned as
        unrankable rather than silently dropped. A regression would
        lose gap skills from the career report.
        """
        ppmi = pd.DataFrame(
            {"a" : {"a" : 0.0}},
            index = ["a"]
        )
        ranked, unrankable = ResumeMatcher._rank_gaps(
            self       = type("Stub", (), {
                "centroid_scope"        : {0: set()},
                "_find_apprenticeships" : lambda self, s: [],
                "_find_programs"        : lambda self, s: [],
                "ppmi_df"               : ppmi
            })(),
            cluster_id = 0,
            resume_set = {"a"},
            skill_gaps = ["unknown_x", "unknown_y"],
            top_k      = 10
        )
        assert ranked == []
        assert sorted(unrankable) == ["unknown_x", "unknown_y"]

    def test_ranked_relevance_positive(self, match_result: MatchResult):
        """
        Every ranked gap has strictly positive PPMI relevance.
        Zero-relevance gaps in the ranked list would silently populate
        the career report with skills that have no co-occurrence
        relationship to the resume.
        """
        for gap in match_result.ranked_gaps:
            assert gap.relevance > 0

    def test_scoping_affects_rank(self):
        """
        In-scope gaps use centroid-scoped reference columns while
        out-of-scope gaps use the full reference set, producing
        different relevance scores.

        Constructs a minimal PPMI matrix with resume skills
        `{"a", "b"}` and centroid scope `{"a", "x"}`. Gap `"x"`
        is in scope, so its relevance uses only column `"a"`
        yielding `mean([0.5]) = 0.5`. Gap `"y"` is out of scope,
        so it uses both columns yielding `mean([0.8, 0.0]) = 0.4`.
        If the scoping branch were bypassed, both gaps would get
        the same reference set and the ordering would change.
        """
        ppmi = pd.DataFrame(
            {
                "a" : {"x" : 0.5, "y" : 0.8},
                "b" : {"x" : 0.1, "y" : 0.0},
                "x" : {"a" : 0.5, "b" : 0.1},
                "y" : {"a" : 0.8, "b" : 0.0}
            }
        )
        from chalkline.pipeline.enrichment import EnrichmentContext
        ranked, _ = ResumeMatcher._rank_gaps(
            self         = type("Stub", (), {
                "centroid_scope" : {0: {"a", "x"}},
                "enrichment"     : EnrichmentContext([], []),
                "ppmi_df"        : ppmi
            })(),
            cluster_id = 0,
            resume_set = {"a", "b"},
            skill_gaps = ["x", "y"],
            top_k      = 10
        )

        scores = {g.skill: g.relevance for g in ranked}
        assert scores["x"] > scores["y"]

    def test_top_k_limits(self, matcher: ResumeMatcher, resume_skills: list[str]):
        """
        `top_k` parameter caps the number of ranked gaps returned.
        """
        assert len(matcher.match(resume_skills, top_k = 2).ranked_gaps) <= 2

    def test_unrankable_separate(self, match_result: MatchResult):
        """
        Ranked and unrankable gaps together cover all gaps.
        """
        ranked_set = {g.skill for g in match_result.ranked_gaps}
        all_gaps   = set(match_result.skill_gaps)
        covered    = ranked_set | set(match_result.unrankable_gaps)
        assert covered == all_gaps

    def test_result_serializable(self, match_result: MatchResult):
        """
        `MatchResult` is JSON-serializable for Marimo cache
        compatibility.
        """
        data = match_result.model_dump()
        assert isinstance(data, dict)
        assert "cluster_id" in data
        roundtrip = MatchResult(**data)
        assert roundtrip.cluster_id == match_result.cluster_id
