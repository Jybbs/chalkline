"""
Tests for layout composition helpers.
"""

from chalkline.display.loaders  import Layout
from chalkline.display.theme    import Theme
from chalkline.matching.schemas import ScoredTask


class TestSkillTree:
    """
    Validate group rendering and empty-group handling in the
    collapsible skill tree.
    """

    def test_all_groups_rendered(self, layout: Layout, theme: Theme):
        """
        Each provided group appears with its type label and individual
        skill names.
        """
        groups = {
            "knowledge" : [ScoredTask(
                demonstrated = True,
                name         = "Mathematics",
                similarity   = 0.70
            )],
            "skill"     : [ScoredTask(
                demonstrated = True,
                name         = "Blueprint Reading",
                similarity   = 0.85
            )]
        }
        html = layout.skill_tree(True, groups, theme).text
        assert "Blueprint Reading" in html
        assert "Mathematics" in html

    def test_missing_group_omitted(self, layout: Layout, theme: Theme):
        """
        Skill types not present in the input dict produce no output.
        """
        groups = {
            "skill" : [ScoredTask(
                demonstrated = True,
                name         = "Blueprint Reading",
                similarity   = 0.85
            )]
        }
        html = layout.skill_tree(True, groups, theme).text
        assert "Blueprint Reading" in html
        assert "Mathematics" not in html
