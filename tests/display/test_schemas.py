"""
Tests for display-layer schemas and lazy-loading containers.
"""

import numpy as np

from datetime import date
from pytest   import fixture, mark

from chalkline.display.schemas  import JobPostingMetrics, MapGeometry, ProcessStep
from chalkline.display.schemas  import RouteDetail, SectionContent, SectorRanking
from chalkline.display.schemas  import TabContent, VarianceBreakdown, WageComparison
from chalkline.display.theme    import Theme
from chalkline.matching.schemas import ScoredTask
from chalkline.pathways.loaders import StakeholderReference
from chalkline.pathways.schemas import Credential


class TestCredentialColor:
    """
    Validate credential kind to palette color dispatch.
    """

    @mark.parametrize(("kind", "key"), [
        ("apprenticeship", "cream"),
        ("career",         "highlight"),
        ("certification",  "lavender"),
        ("program",        "accent")
    ])
    def test_known_kinds(self, theme: Theme, kind: str, key: str):
        """
        Each recognized credential kind maps to its designated
        palette color.
        """
        assert theme.credential_color(kind) == theme.colors[key]

    def test_unknown_fallback(self, theme: Theme):
        """
        Unrecognized kinds fall back to the muted palette color.
        """
        assert theme.credential_color("other") == theme.colors["muted"]


class TestJobPostingMetrics:
    """
    Validate posting aggregation and timeline properties.
    """

    def test_dated_newest_first(self, posting, reference):
        """
        `dated` keeps only postings with `date_posted`, sorted
        newest to oldest.
        """
        metrics = JobPostingMetrics.from_postings([
            posting(date_posted=date(2026, 1, 5)),
            posting(date_posted=None),
            posting(date_posted=date(2026, 1, 10))
        ], reference)
        assert len(metrics.dated) == 2
        assert metrics.dated[0].date == date(2026, 1, 10)

    def test_empty_postings(self, reference):
        """
        Empty input produces zero counts, empty collections, and
        "N/A" freshness in stat values.
        """
        metrics = JobPostingMetrics.from_postings([], reference)
        assert metrics.companies == {}
        assert metrics.dated     == []
        assert metrics.dates     == []
        assert metrics.freshness == []
        assert metrics.stat_values[0] == "0"
        assert metrics.stat_values[4] == "N/A"


class TestMapGeometry:
    """
    Validate layout geometry payload and Python-only field exclusion.
    """

    def test_dimensions_excludes_python_only(self):
        """
        The `dimensions` payload sent to JS must not contain fields
        that are only used on the Python side.
        """
        dims = MapGeometry().dimensions
        assert "default_wage_range" not in dims
        assert "title_char_limit" not in dims

    def test_dimensions_includes_layout(self):
        """
        All pixel-level layout constants must be present in the JS
        payload.
        """
        dims = MapGeometry().dimensions
        for key in (
            "card_h", "card_w", "circle_r", "height",
            "hero_h", "hero_w", "pad", "width"
        ):
            assert key in dims

    def test_default_wage_range(self):
        """
        Fallback wage range used when no cluster has wage data.
        """
        assert MapGeometry().default_wage_range == [30000, 90000]


class TestProcessStep:
    """
    Validate template rendering in process flow steps.
    """

    def test_render_substitution(self):
        """
        `render` replaces placeholders in `detail` while preserving
        other fields unchanged.
        """
        step = ProcessStep(
            detail = "{corpus_size} postings across {cluster_count} families",
            label  = "Cluster",
            number = "3"
        )
        rendered = step.render(cluster_count=21, corpus_size=847)
        assert rendered.detail == "847 postings across 21 families"
        assert rendered.label  == "Cluster"
        assert rendered.number == "3"


class TestRouteDetail:
    """
    Validate derived properties on career transition routes.
    """

    @fixture
    def route(self, clusters) -> RouteDetail:
        """
        Minimal route with four scored tasks, two demonstrated.
        """
        src, dst = list(clusters.values())[:2]
        return RouteDetail(
            coverage         = {},
            credentials      = [],
            destination      = dst,
            destination_wage = 65000,
            display_title    = dst.soc_title,
            gap_vectors      = np.empty((0, 0)),
            scored_tasks     = [
                ScoredTask(demonstrated=True,  name="A", similarity=0.9),
                ScoredTask(demonstrated=True,  name="B", similarity=0.7),
                ScoredTask(demonstrated=False, name="C", similarity=0.3),
                ScoredTask(demonstrated=False, name="D", similarity=0.1)
            ],
            source      = src,
            source_wage = 50000
        )

    def test_credentials_by_kind(self, clusters):
        """
        Credentials group by kind with empty kinds omitted.
        """
        src, dst = list(clusters.values())[:2]
        route = RouteDetail(
            coverage    = {},
            credentials = [
                Credential(
                    embedding_text = "x",
                    kind           = "certification",
                    label          = "OSHA 30",
                    metadata       = {"credential": "OSHA 30-Hour"}
                ),
                Credential(
                    embedding_text = "y",
                    kind           = "apprenticeship",
                    label          = "Inside Wireman",
                    metadata       = {"min_hours": 8000, "rapids_code": "0159"}
                ),
                Credential(
                    embedding_text = "z",
                    kind           = "certification",
                    label          = "First Aid",
                    metadata       = {"credential": "First Aid/CPR"}
                )
            ],
            destination      = dst,
            destination_wage = 65000,
            display_title    = dst.soc_title,
            gap_vectors      = np.empty((0, 0)),
            scored_tasks     = [],
            source           = src,
            source_wage      = 50000
        )
        by_kind = route.credentials_by_kind
        assert set(by_kind) == {"apprenticeship", "certification"}
        assert len(by_kind["certification"]) == 2
        assert len(by_kind["apprenticeship"]) == 1

    def test_demonstrated_count(self, route: RouteDetail):
        """
        Demonstrated count is total minus gaps.
        """
        assert route.demonstrated_count == 2

    def test_fit_percentage(self, route: RouteDetail):
        """
        Fit percentage is demonstrated tasks over total (2/4 = 50%).
        """
        assert route.fit_percentage == 50

    def test_gap_tasks(self, route: RouteDetail):
        """
        Gap tasks are those where `demonstrated` is False.
        """
        assert [t.name for t in route.gap_tasks] == ["C", "D"]

    def test_is_not_self(self, route: RouteDetail):
        """
        Route between different clusters is not a self-route.
        """
        assert not route.is_self

    def test_is_self(self, clusters):
        """
        Route from a cluster to itself is detected as a self-route.
        """
        cluster = list(clusters.values())[0]
        route = RouteDetail(
            coverage         = {},
            credentials      = [],
            destination      = cluster,
            destination_wage = 50000,
            display_title    = cluster.soc_title,
            gap_vectors      = np.empty((0, 0)),
            scored_tasks     = [],
            source           = cluster,
            source_wage      = 50000
        )
        assert route.is_self

    def test_top_gaps_by_deficit(self, route: RouteDetail):
        """
        Top gaps are ordered by ascending similarity (largest deficit
        first).
        """
        assert route.top_gaps[0].name == "D"

    def test_top_strengths(self, route: RouteDetail):
        """
        Top strengths are the demonstrated tasks, capped at eight.
        """
        assert [t.name for t in route.top_strengths] == ["A", "B"]

    def test_wage_comparison(self, route: RouteDetail):
        """
        Wage comparison wraps the raw amounts into a WageComparison
        with correct delta.
        """
        assert route.wage_comparison.delta == 15000


class TestScoreColor:
    """
    Validate threshold-based color dispatch.
    """

    def test_score_gradient(self, theme: Theme):
        """
        Scores produce a smooth RGB gradient from red through gold
        to green, with distinct colors at each percentage.
        """
        low  = theme.score_color(10)
        mid  = theme.score_color(50)
        high = theme.score_color(90)
        assert low.startswith("rgb(")
        assert mid.startswith("rgb(")
        assert high.startswith("rgb(")
        assert low != mid != high


class TestSectorRanking:
    """
    Validate parallel-list construction from ranked tuples.
    """

    def test_empty_input(self):
        """
        Empty tuples produce empty parallel lists without raising.
        """
        sr = SectorRanking.from_tuples([])
        assert sr.labels  == []
        assert sr.sectors == []
        assert sr.values  == []

    def test_from_tuples_unzip(self):
        """
        Triples unzip into label, sector, and value lists that
        preserve input order.
        """
        data = [
            ("Electrician", "Specialty", 0.82),
            ("Carpenter", "General", 0.71)
        ]
        sr = SectorRanking.from_tuples(data)
        assert sr.labels  == ["Electrician", "Carpenter"]
        assert sr.sectors == ["Specialty", "General"]
        assert sr.values  == [0.82, 0.71]

    def test_value_map(self):
        """
        `value_map` pairs labels with values for chart factories.
        """
        sr = SectorRanking(
            labels  = ["A", "B"],
            sectors = ["X", "Y"],
            values  = [0.9, 0.5]
        )
        assert sr.value_map == {"A": 0.9, "B": 0.5}


class TestStakeholderReference:
    """
    Validate lazy-loading and missing-file fallback.
    """

    def test_loads_json_on_access(self, tmp_path):
        """
        Attribute access deserializes the corresponding JSON file
        and caches the result.
        """
        (tmp_path / "trades.json").write_text('["electrician"]')
        ref = StakeholderReference(reference_dir=tmp_path)
        assert ref.trades == ["electrician"]
        assert ref.trades is ref.trades

    def test_missing_file_empty(self, tmp_path):
        """
        Accessing a name with no backing JSON file returns an
        empty list rather than raising.
        """
        ref = StakeholderReference(reference_dir=tmp_path)
        assert ref.nonexistent == []


class TestTabContent:
    """
    Validate section formatting and tuple ordering.
    """

    def test_section(self):
        """
        `section()` returns (description, title) to match `header()`'s
        alphabetized parameter order, with `{n}` substitution applied
        to both fields.
        """
        content = TabContent(sections={
            "overview": SectionContent(
                description = "Found {n} clusters",
                title       = "Overview of {n}"
            )
        })
        description, title = content.section("overview", n=21)
        assert description == "Found 21 clusters"
        assert title       == "Overview of 21"


class TestVarianceBreakdown:
    """
    Validate SVD variance percentage conversion.
    """

    def test_from_svd(self):
        """
        Ratios scale to percentages with correct totals, labels,
        cumulative sums, and dict representations.
        """
        vb = VarianceBreakdown.from_svd([0.35, 0.25, 0.15])
        assert vb.components      == [35.0, 25.0, 15.0]
        assert vb.total           == 75.0
        assert vb.labels          == ["PC1", "PC2", "PC3"]
        assert vb.cumulative      == [35.0, 60.0, 75.0]
        assert vb.cumulative_dict == {"PC1": 35.0, "PC2": 60.0, "PC3": 75.0}
        assert vb.components_dict == {"PC1": 35.0, "PC2": 25.0, "PC3": 15.0}

    def test_single_component(self):
        """
        A single SVD component produces one-element lists and a total
        equal to that component.
        """
        vb = VarianceBreakdown.from_svd([0.42])
        assert vb.components == [42.0]
        assert vb.total      == 42.0
        assert vb.labels     == ["PC1"]
        assert vb.cumulative == [42.0]


class TestWageColor:
    """
    Validate signed delta to success/error color dispatch.
    """

    @mark.parametrize(("delta", "key"), [
        (-3000, "error"),
        (0,     "success"),
        (5000,  "success")
    ], ids=["negative", "zero", "positive"])
    def test_wage_color(self, theme: Theme, delta: int, key: str):
        """
        Negative deltas produce the error color; zero and positive
        produce the success color.
        """
        assert theme.wage_color(delta) == theme.colors[key]


class TestWageComparison:
    """
    Validate wage formatting, delta signs, and bar percentages.
    """

    def test_both_none(self):
        """
        Both wages absent produces null delta and 0% bars.
        """
        wc = WageComparison()
        assert wc.delta                  is None
        assert wc.destination_percentage == 0
        assert wc.source_percentage      == 0

    def test_both_zero_safe(self):
        """
        Both wages at zero produces 0% bars rather than dividing by
        zero.
        """
        wc = WageComparison(destination_wage=0, source_wage=0)
        assert wc.source_percentage == 0
        assert wc.destination_percentage == 0

    @mark.parametrize(("dest", "src", "delta", "display"), [
        (60000, 45000, 15000,  "+$15,000/yr"),
        (50000, 50000, 0,      "+$0/yr"),
        (45000, 60000, -15000, "$-15,000/yr")
    ], ids=["positive", "zero", "negative"])
    def test_delta(self, dest: int, src: int, delta: int, display: str):
        """
        Delta is the signed difference between destination and source
        wages, formatted with sign prefix.
        """
        wc = WageComparison(destination_wage=dest, source_wage=src)
        assert wc.delta         == delta
        assert wc.delta_display == display

    def test_delta_missing_wage(self):
        """
        Delta and display are null/empty when either wage is absent.
        """
        wc = WageComparison(destination_wage=60000)
        assert wc.delta is None
        assert wc.delta_display == ""

    def test_labels_present(self):
        """
        Labels format as $Xk when wage is present, em dash when absent.
        """
        wc = WageComparison(destination_wage=75000, source_wage=None)
        assert wc.destination_label == "$75k"
        assert wc.source_label == "\u2014"

    @mark.parametrize(("dest", "src", "dest_pct", "src_pct"), [
        (80000, 40000, 100, 50),
        (40000, 80000, 50,  100)
    ], ids=["higher_dest", "higher_source"])
    def test_percentages(self, dest: int, src: int, dest_pct: int, src_pct: int):
        """
        The higher wage pegs at 100% and the lower scales
        proportionally.
        """
        wc = WageComparison(destination_wage=dest, source_wage=src)
        assert wc.destination_percentage == dest_pct
        assert wc.source_percentage      == src_pct
