"""
Tests for display table builders, board filtering, employer matching,
and DAG introspection.
"""

import numpy as np

from chalkline.collection.schemas import Corpus, Posting
from chalkline.display.tables     import apprenticeship_rows, filter_boards
from chalkline.display.tables     import match_cluster_employers, match_member
from chalkline.display.tables     import program_rows, to_mermaid
from chalkline.pathways.schemas   import CareerEdge, ClusterAssignments
from chalkline.pathways.schemas   import ClusterProfile, Credential, Neighborhood


APPRENTICESHIP = Credential(
    embedding_text = "Electrician",
    kind           = "apprenticeship",
    label          = "Electrician",
    metadata       = {"min_hours": 8000, "rapids_code": "0159"}
)

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

MEMBERS = [
    {"name": "Cianbro Corporation", "type": "General Contractors"},
    {"name": "R.J. Grondin and Sons", "type": "General Contractors"},
    {"name": "Reed & Reed", "type": "General Contractors"}
]

PROFILE = ClusterProfile(
    cluster_id  = 0,
    job_zone    = 3,
    modal_title = "Electrician",
    sector      = "Building",
    size        = 10,
    soc_title   = "Electricians"
)

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


def _posting(company: str, title: str = "Worker") -> Posting:
    """
    Build a minimal posting for employer matching tests.
    """
    return Posting(
        company     = company,
        date_posted = None,
        description = "x" * 50,
        source_url  = f"https://example.com/{company.lower().replace(' ', '-')}",
        title       = title
    )


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

    def test_empty_boards(self):
        """
        Empty board dict returns empty lists without error.
        """
        maine, national = filter_boards({}, PROFILES, "Building Construction")
        assert maine == []
        assert national == []

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


class TestMatchClusterEmployers:
    """
    Validate the full employer matching pipeline from cluster
    postings to deduplicated row output.
    """

    def test_deduplicates(self):
        """
        Same company appearing in multiple postings produces
        a single row.
        """
        postings = {
            p.id: p for p in [
                _posting("Cianbro Corporation", "Electrician"),
                _posting("Cianbro Corporation", "Welder")
            ]
        }
        corpus      = Corpus(postings)
        assignments = ClusterAssignments(np.array([0, 0]))

        rows = match_cluster_employers(
            assignments = assignments,
            career_urls = [],
            cluster_id  = 0,
            corpus      = corpus,
            members     = MEMBERS
        )
        assert len(rows) == 1
        assert rows[0]["Company"] == "Cianbro Corporation"

    def test_empty_cluster(self):
        """
        Cluster with no postings returns an empty list.
        """
        postings = {
            p.id: p for p in [_posting("Cianbro Corporation")]
        }
        corpus      = Corpus(postings)
        assignments = ClusterAssignments(np.array([0]))

        rows = match_cluster_employers(
            assignments = assignments,
            career_urls = [],
            cluster_id  = 99,
            corpus      = corpus,
            members     = MEMBERS
        )
        assert rows == []

    def test_joins_career_urls(self):
        """
        Career page URLs from reference data appear in output rows.
        """
        postings = {
            p.id: p for p in [_posting("Reed & Reed")]
        }
        corpus      = Corpus(postings)
        assignments = ClusterAssignments(np.array([0]))
        career_urls = [
            {"company": "Reed & Reed", "url": "https://reedandreed.com/careers"}
        ]

        rows = match_cluster_employers(
            assignments = assignments,
            career_urls = career_urls,
            cluster_id  = 0,
            corpus      = corpus,
            members     = MEMBERS
        )
        assert len(rows) == 1
        assert rows[0]["Career Page"] == "https://reedandreed.com/careers"


class TestMatchMember:
    """
    Validate SequenceMatcher-based company name matching.
    """

    def test_below_threshold(self):
        """
        Unrelated company names return None.
        """
        assert match_member("ACME Corp", MEMBERS) is None

    def test_empty_members(self):
        """
        Empty member list returns None without error.
        """
        assert match_member("Cianbro", []) is None

    def test_exact_match(self):
        """
        Identical names produce a match.
        """
        assert (m := match_member("Cianbro Corporation", MEMBERS)) is not None
        assert m["name"] == "Cianbro Corporation"

    def test_fuzzy_match(self):
        """
        Abbreviation and punctuation differences still match
        above the 0.7 threshold.
        """
        assert (m := match_member("RJ Grondin & Sons", MEMBERS)) is not None
        assert m["name"] == "R.J. Grondin and Sons"


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


class TestToMermaid:
    """
    Validate that the Mermaid DAG reflects the actual pipeline step
    function signatures.
    """

    def test_contains_known_edge(self):
        """
        At least one well-known dependency edge is present.
        """
        assert "coordinates --> assignments" in to_mermaid()

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

    def test_no_empty_lines(self):
        """
        Output contains no blank lines that would break Mermaid
        rendering.
        """
        for line in to_mermaid().splitlines():
            assert line.strip()

    def test_starts_with_graph(self):
        """
        Output begins with the Mermaid graph directive.
        """
        assert to_mermaid().startswith("graph LR")
