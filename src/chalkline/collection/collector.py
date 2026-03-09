"""
Corpus collection via job aggregators.

Wraps the JobSpy `scrape_jobs` function to query multiple search terms
across configured job boards, convert results to `Posting` records, and
persist deduplicated postings through the storage layer.

    uv run python -m chalkline.collection.collector
"""

import pandas as pd

from jobspy   import scrape_jobs
from logging  import basicConfig, getLogger, INFO
from pathlib  import Path

from chalkline.collection.schemas import Posting
from chalkline.collection.storage import save


logger = getLogger(__name__)


class Collector:
    """
    Corpus collector that queries job aggregators via JobSpy.

    Stores configuration in `__init__` and executes the full
    scrape-convert-save pipeline via `run`.
    """

    def __init__(
        self,
        postings_dir   : Path,
        search_terms   : list[str],
        results_wanted : int       = 10000,
        sites          : list[str] = ["indeed"]
    ):
        self.postings_dir   = postings_dir
        self.results_wanted = results_wanted
        self.search_terms   = search_terms
        self.sites          = sites

    @staticmethod
    def _parse_record(record: dict) -> Posting | None:
        """
        Convert a single JobSpy row into a `Posting`.

        Returns `None` when validation fails because aggregator results
        occasionally include stub listings.
        """
        clean = lambda v: v if pd.notna(v) else None

        try:
            return Posting(
                company     = record["company"],
                date_posted = clean(record["date_posted"]),
                description = clean(record["description"]) or "",
                location    = clean(record["location"]),
                source_url  = record["job_url"],
                title       = record["title"]
            )
        except Exception as error:
            logger.debug(f"Skipped row: {error}")
            return None

    def _scrape(self) -> pd.DataFrame:
        """
        Run `scrape_jobs` once per search term and concatenate all results
        into a single DataFrame.

        Returns an empty DataFrame when every term fails, allowing
        downstream code to iterate without guarding.
        """
        frames = []
        for term in self.search_terms:
            logger.info(f"Searching {term!r}")
            try:
                frames.append(scrape_jobs(
                    location           = "Maine",
                    results_wanted     = self.results_wanted,
                    search_term        = term,
                    site_name          = self.sites,
                    verbose            = 2
                ))
            except Exception as error:
                logger.error(f"{term!r}: {error}")

        return (
            pd.concat(frames, ignore_index=True)
            if frames else pd.DataFrame()
        )

    def run(self):
        """
        Collect postings from all search terms and persist to disk.
        """
        save([
            p for record in self._scrape().to_dict("records")
            if (p := self._parse_record(record))
        ], self.postings_dir)


if __name__ == "__main__":

    basicConfig(
        format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level  = INFO
    )

    Collector(
        postings_dir = Path("data/postings"),
        search_terms = [
            "carpenter",
            "construction",
            "electrician",
            "equipment operator",
            "paving",
            "superintendent"
        ]
    ).run()
