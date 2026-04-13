"""
Tests for corpus collection via job aggregators.

Validates record parsing from JobSpy DataFrame rows into `Posting` records.
"""

from pytest import mark

from chalkline.collection.collector import Collector


class TestParseRecord:
    """
    Validate `Collector._parse_record` conversion from raw JobSpy rows to
    `Posting` instances.
    """

    @mark.parametrize("record", [
        {"company": "Cianbro"},
        {
            "company"     : "Cianbro",
            "date_posted" : "2026-03-01",
            "description" : float("nan"),
            "job_url"     : "https://example.com",
            "location"    : "Portland, ME",
            "title"       : "Electrician"
        }
    ], ids=["missing_field", "nan_description"])
    def test_invalid_row_returns_none(self, record: dict):
        """
        Rows with missing required fields or NaN descriptions return
        `None` rather than raising or allowing contentless postings.
        """
        assert Collector._parse_record(record) is None

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

