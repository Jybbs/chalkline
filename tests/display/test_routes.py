"""
Tests for route card builders.
"""

import numpy as np

from chalkline.display.routes   import Routes
from chalkline.display.schemas  import RouteDetail, TabContent
from chalkline.matching.schemas import ScoredTask


class TestRoutes:
    """
    Validate route-specific HTML card builders.
    """

    def test_skill_card_count(self, routes: Routes):
        """
        One card element per scored task.
        """
        tasks = [
            ScoredTask(
                demonstrated = True,
                name         = "Install conduit",
                similarity   = 0.8
            ),
            ScoredTask(
                demonstrated = False,
                name         = "Read blueprints",
                similarity   = 0.3
            )
        ]
        assert len(routes._skill_cards(tasks)) == 2

    def test_wage_bar_variant_class(self, routes: Routes):
        """
        The variant string appears in the fill element's CSS
        class.
        """
        html = str(routes._wage_bar("$50k", 75, "source"))
        assert "cl-wage-bar-source" in html

    def test_wage_bar_width(self, routes: Routes):
        """
        The bar fill width matches the percentage parameter.
        """
        html = str(routes._wage_bar("$50k", 75, "dest"))
        assert "width:75%" in html

    def test_verdict_self_hides_source(self, clusters, routes: Routes):
        """
        Self-route verdict renders the destination bar but omits the
        source bar.
        """
        cluster = list(clusters.values())[0]
        route = RouteDetail(
            coverage         = {},
            credentials      = [],
            destination      = cluster,
            destination_wage = 50000,
            display_title    = cluster.soc_title,
            gap_vectors      = np.empty((0, 0)),
            match_score      = 1.0,
            scored_tasks     = [
                ScoredTask(demonstrated=True, name="A", similarity=0.8)
            ],
            source      = cluster,
            source_wage = 50000
        )
        tab = TabContent(chart_labels={
            "bright_outlook"  : "Bright Outlook",
            "fit_meter_label" : "match",
            "open_positions"  : "open positions",
            "verdict_match"   : "{demonstrated}/{total} at {fit_pct}% for {soc_title}"
        })
        html = routes.verdict(route, tab).text
        assert "cl-wage-bar-source" not in html
        assert "cl-wage-bar-dest" in html

    def test_verdict_shows_delta(self, clusters, routes: Routes):
        """
        Transition route shows the signed wage delta between source
        and destination wages.
        """
        src, dst = list(clusters.values())[:2]
        route = RouteDetail(
            coverage         = {},
            credentials      = [],
            destination      = dst,
            destination_wage = 65000,
            display_title    = dst.soc_title,
            gap_vectors      = np.empty((0, 0)),
            match_score      = 0.6,
            scored_tasks     = [
                ScoredTask(demonstrated=True, name="A", similarity=0.8)
            ],
            source      = src,
            source_wage = 50000
        )
        tab = TabContent(chart_labels={
            "bright_outlook"  : "Bright Outlook",
            "fit_meter_label" : "match",
            "open_positions"  : "open positions",
            "verdict_match"   : "{demonstrated}/{total} at {fit_pct}% for {soc_title}"
        })
        html = routes.verdict(route, tab).text
        assert "cl-wage-bar-source" in html
        assert "+$15,000/yr" in html
