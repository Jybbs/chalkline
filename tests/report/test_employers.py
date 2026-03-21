"""
Tests for fuzzy matching of corpus companies against AGC members.
"""

import numpy as np

from chalkline.collection.schemas  import Posting
from chalkline.pipeline.schemas    import ClusterAssignments, Corpus
from chalkline.report.employers    import match_cluster_employers, match_member


MEMBERS = [
    {"name": "Cianbro Corporation", "type": "General Contractors"},
    {"name": "R.J. Grondin and Sons", "type": "General Contractors"},
    {"name": "Reed & Reed", "type": "General Contractors"}
]


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


class TestMatchMember:
    """
    Validate SequenceMatcher-based company name matching.
    """

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
