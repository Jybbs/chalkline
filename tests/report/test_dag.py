"""
Tests for Hamilton DAG introspection and Mermaid rendering.
"""

from chalkline.report.dag import to_mermaid


class TestToMermaid:
    """
    Validate that the Mermaid DAG reflects the actual pipeline step
    function signatures.
    """

    def test_starts_with_graph(self):
        """
        Output begins with the Mermaid graph directive.
        """
        assert to_mermaid().startswith("graph LR")

    def test_excludes_inputs(self):
        """
        External inputs (config, model, lexicons) never appear as
        source nodes on the left side of an arrow.
        """
        lines = to_mermaid().splitlines()[1:]
        sources = {
            line.strip().split(" --> ")[0]
            for line in lines
            if " --> " in line
        }
        assert not sources & {"config", "model", "lexicons"}

    def test_contains_known_edge(self):
        """
        At least one well-known dependency edge is present.
        """
        assert "coordinates --> assignments" in to_mermaid()

    def test_no_empty_lines(self):
        """
        Output contains no blank lines that would break Mermaid
        rendering.
        """
        for line in to_mermaid().splitlines():
            assert line.strip()
