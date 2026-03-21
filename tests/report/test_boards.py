"""
Tests for sector-based job board filtering.
"""

from chalkline.pipeline.schemas import ClusterProfile
from chalkline.report.boards    import filter_boards

PROFILES = {
    0: ClusterProfile(
        cluster_id  = 0,
        job_zone    = 3,
        modal_title = "Commercial Electrician",
        sector      = "Building Construction",
        size        = 15,
        soc_title   = "Electricians"
    ),
    1: ClusterProfile(
        cluster_id  = 1,
        job_zone    = 3,
        modal_title = "Framing Carpenter",
        sector      = "Building Construction",
        size        = 12,
        soc_title   = "Carpenters"
    ),
    2: ClusterProfile(
        cluster_id  = 2,
        job_zone    = 4,
        modal_title = "Highway Superintendent",
        sector      = "Heavy Highway Construction",
        size        = 10,
        soc_title   = "Civil Engineers"
    ),
    3: ClusterProfile(
        cluster_id  = 3,
        job_zone    = 3,
        modal_title = "Paving Foreman",
        sector      = "Heavy Highway Construction",
        size        = 8,
        soc_title   = "Paving & Tamping Operators"
    ),
    4: ClusterProfile(
        cluster_id  = 4,
        job_zone    = 5,
        modal_title = "Construction Manager",
        sector      = "Construction Managers",
        size        = 20,
        soc_title   = "Project Management Specialists"
    )
}

BOARDS = {
    "maine": [
        {
            "best_for" : "General Contractors, Trades, Safety Officers",
            "category" : "Industry Association",
            "focus"    : "Commercial & Heavy Civil",
            "name"     : "AGC Maine"
        },
        {
            "best_for" : "Transportation Engineers, Highway Crews",
            "category" : "Government",
            "focus"    : "Public Infrastructure",
            "name"     : "MaineDOT Careers"
        }
    ],
    "national": [
        {
            "best_for" : "Focused on professionalized project management",
            "category" : "Management",
            "focus"    : "Construction Managers (CMs)",
            "name"     : "CMAA Career HQ"
        },
        {
            "best_for" : "Highway and bridge engineering careers",
            "category" : "Transportation",
            "focus"    : "State Highway/DOT Roles",
            "name"     : "AASHTO Jobs"
        }
    ]
}


class TestFilterBoards:
    """
    Validate sector keyword matching derived from cluster profile
    titles.
    """

    def test_building_sector(self):
        """
        Building Construction matches boards mentioning commercial
        or building-related terms derived from profile titles.
        """
        maine, national = filter_boards(BOARDS, PROFILES, "Building Construction")
        names = [b["name"] for b in maine + national]
        assert "AGC Maine" in names

    def test_highway_sector(self):
        """
        Heavy Highway Construction matches boards mentioning
        highway or infrastructure terms from profile titles.
        """
        maine, national = filter_boards(BOARDS, PROFILES, "Heavy Highway Construction")
        names = [b["name"] for b in maine + national]
        assert "MaineDOT Careers" in names
        assert "AASHTO Jobs" in names

    def test_managers_sector(self):
        """
        Construction Managers matches boards mentioning management
        terms derived from profile titles.
        """
        maine, national = filter_boards(BOARDS, PROFILES, "Construction Managers")
        names = [b["name"] for b in national]
        assert "CMAA Career HQ" in names

    def test_unknown_sector(self):
        """
        Unrecognized sector produces empty keyword set from
        profiles, returning no matches.
        """
        maine, national = filter_boards(BOARDS, PROFILES, "Unknown Sector")
        assert maine == []
        assert national == []

    def test_empty_boards(self):
        """
        Empty board dict returns empty lists without error.
        """
        maine, national = filter_boards({}, PROFILES, "Building Construction")
        assert maine == []
        assert national == []
