"""
Tests for corpus collection via job aggregators.

Validates record parsing from JobSpy DataFrame rows into `Posting` records.
"""

from chalkline.collection.collector import Collector


class TestParseRecord:
    """
    Validate `Collector._parse_record` conversion from raw JobSpy rows to
    `Posting` instances.
    """

    def test_missing_field(self):
        """
        A row missing required fields returns `None` instead of raising.
        """
        assert Collector._parse_record({"company": "Cianbro"}) is None

    def test_nan_description(self):
        """
        A NaN description coerces to empty string and fails the
        minimum-length validation, returning `None` rather than allowing
        contentless postings into the corpus.
        """
        assert Collector._parse_record({
            "company"     : "Cianbro",
            "date_posted" : "2026-03-01",
            "description" : float("nan"),
            "job_url"     : "https://example.com",
            "location"    : "Portland, ME",
            "title"       : "Electrician"
        }) is None

    def test_nan_fields(self):
        """
        Pandas `NaN` values coerce to `None` for optional fields.
        """
        result = Collector._parse_record({
            "company"     : "Cianbro",
            "date_posted" : float("nan"),
            "description" : "x" * 50,
            "job_url"     : "https://example.com",
            "location"    : float("nan"),
            "title"       : "Electrician"
        })
        assert result is not None
        assert result.date_posted is None
        assert result.location is None

