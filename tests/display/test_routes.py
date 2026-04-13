"""
Tests for route card builders.
"""

from chalkline.display.loaders import Layout
from chalkline.display.routes  import Routes
from chalkline.display.theme   import Theme
from chalkline.matching.schemas import ScoredTask


class TestRoutes:
    """
    Validate route-specific HTML card builders.
    """

    def test_skill_card_count(
        self,
        layout : Layout,
        theme  : Theme
    ):
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
        assert len(Routes(layout, theme)._skill_cards(tasks)) == 2

    def test_wage_bar_variant_class(
        self,
        layout : Layout,
        theme  : Theme
    ):
        """
        The variant string appears in the fill element's CSS
        class.
        """
        html = str(Routes(layout, theme)._wage_bar("$50k", 75, "source"))
        assert "cl-wage-bar-source" in html

    def test_wage_bar_width(
        self,
        layout : Layout,
        theme  : Theme
    ):
        """
        The bar fill width matches the percentage parameter.
        """
        html = str(Routes(layout, theme)._wage_bar("$50k", 75, "dest"))
        assert "width:75%" in html
