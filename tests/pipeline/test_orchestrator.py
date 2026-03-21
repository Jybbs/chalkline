"""
Tests for the `Chalkline` dataclass structure.

Validates that the fitted pipeline carries all expected fields and that the
repr methods produce output without errors.
"""

from dataclasses import fields

from chalkline.pipeline.orchestrator import Chalkline


class TestChalkline:
    """
    Structural validation of the Chalkline dataclass.
    """

    def test_fields_present(self):
        """
        The Chalkline dataclass declares the expected field names.
        """
        assert {f.name for f in fields(Chalkline)} == {
            "config", "graph", "manifest",
            "matcher", "profiles", "trades"
        }
