"""
Tests for corpus collection via job aggregators.

Validates record parsing from JobSpy DataFrame rows into `Posting` records.
"""

from chalkline.collection.collector import Collector
from chalkline.collection.schemas   import Posting


class TestParseRecord:
    """
    Validate `Collector._parse_record` conversion from raw JobSpy rows to
    `Posting` instances.
    """

    def test_missing_required_field_returns_none(self):
        """
        A row missing required fields returns `None` instead of raising.
        """
        assert Collector._parse_record({"company": "Cianbro"}) is None

    def test_nan_fields_become_none(self):
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
        assert result.date_posted is None
        assert result.location is None

    def test_valid_record(self):
        """
        A complete row converts to a `Posting` with correct field mapping.
        """
        result = Collector._parse_record({
            "company"     : "Cianbro",
            "date_posted" : "2026-03-01",
            "description" : "x" * 50,
            "job_url"     : "https://example.com",
            "location"    : "Portland, ME",
            "title"       : "Electrician"
        })
        assert isinstance(result, Posting)
        assert result.company == "Cianbro"
