"""
Shared test fixtures for the Chalkline test suite.
"""

from datetime import date
from pytest   import fixture

from chalkline.collection.schemas import Posting


SAMPLE_DESCRIPTION = (
    "Seeking an experienced electrician for commercial construction "
    "projects. Must have valid journeyman license and OSHA 10 "
    "certification. Responsibilities include conduit bending, "
    "blueprint reading, and NEC code compliance."
)


@fixture
def sample_posting() -> Posting:
    """
    A minimal valid posting for testing.
    """
    return Posting(
        company        = "Cianbro",
        date_collected = date(2026, 3, 5),
        date_posted    = date(2026, 3, 1),
        description    = SAMPLE_DESCRIPTION,
        source_url     = "https://www.cianbro.com/careers-list",
        title          = "Electrician"
    )


@fixture
def second_posting() -> Posting:
    """
    A distinct-company posting for multi-posting tests.
    """
    return Posting(
        company        = "Reed & Reed",
        date_collected = date(2026, 3, 5),
        date_posted    = None,
        description    = SAMPLE_DESCRIPTION,
        source_url     = "https://reed-reed.com/jobs/",
        title          = "Laborer"
    )
