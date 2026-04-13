"""
Tests for route card builders.
"""

from chalkline.display.routes  import Routes
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
