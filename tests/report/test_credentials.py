"""
Tests for credential extraction from neighborhood edges.
"""

from chalkline.matching.schemas   import CareerEdge, Neighborhood
from chalkline.pipeline.schemas   import ClusterProfile, Credential
from chalkline.report.credentials import apprenticeship_rows, program_rows


PROFILE = ClusterProfile(
    cluster_id  = 0,
    job_zone    = 3,
    modal_title = "Electrician",
    sector      = "Building",
    size        = 10,
    soc_title   = "Electricians"
)

APPRENTICESHIP = Credential(
    embedding_text = "Electrician",
    kind           = "apprenticeship",
    label          = "Electrician",
    metadata       = {"min_hours": 8000, "rapids_code": "0159"}
)

PROGRAM = Credential(
    embedding_text = "AAS Electrical Technology SMCC",
    kind           = "program",
    label          = "Electrical Technology",
    metadata       = {
        "credential"  : "AAS",
        "institution" : "SMCC",
        "url"         : "https://smcc.edu"
    }
)


def _neighborhood(*edges: CareerEdge) -> Neighborhood:
    """
    Build a neighborhood with all edges as advancement.
    """
    return Neighborhood(advancement=list(edges))


class TestApprenticeshipRows:
    """
    Validate RAPIDS-based deduplication and row formatting.
    """

    def test_deduplicates(self):
        """
        Same RAPIDS code on two edges produces one row.
        """
        edge = CareerEdge(
            credentials = [APPRENTICESHIP],
            profile     = PROFILE,
            weight      = 0.9
        )
        rows = apprenticeship_rows(_neighborhood(edge, edge))
        assert len(rows) == 1
        assert rows[0]["RAPIDS Code"] == "0159"

    def test_empty_edges(self):
        """
        No apprenticeships across edges returns empty list.
        """
        edge = CareerEdge(profile=PROFILE, weight=0.8)
        assert apprenticeship_rows(_neighborhood(edge)) == []


class TestProgramRows:
    """
    Validate institution+program deduplication.
    """

    def test_deduplicates(self):
        """
        Same (institution, program) on two edges produces one row.
        """
        edge = CareerEdge(
            credentials = [PROGRAM],
            profile     = PROFILE,
            weight      = 0.9
        )
        rows = program_rows(_neighborhood(edge, edge))
        assert len(rows) == 1
        assert rows[0]["Institution"] == "SMCC"

    def test_empty_edges(self):
        """
        No programs across edges returns empty list.
        """
        edge = CareerEdge(profile=PROFILE, weight=0.8)
        assert program_rows(_neighborhood(edge)) == []
