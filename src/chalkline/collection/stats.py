"""
Corpus statistics reporting for collected postings.

Summarizes posting counts, source type distribution, company
coverage, and temporal range. Designed to run on an empty corpus
without error to support pre-collection validation.
"""

from collections import Counter

from chalkline.collection.models import Posting


class CorpusStats:
    """
    Aggregate statistics over a collection of postings.

    Computes company coverage, source type distribution, and date
    range from the raw posting list. All properties handle an empty
    corpus gracefully.
    """

    def __init__(self, postings: list[Posting]):
        """
        Initialize from a list of postings.

        Args:
            postings: The corpus to compute statistics over.
        """
        self._postings = postings

    @property
    def company_counts(self) -> dict[str, int]:
        """
        Number of postings per company, sorted descending by count.

        Returns:
            A dictionary mapping company names to posting counts.
        """
        return dict(
            Counter(p.company for p in self._postings).most_common()
        )

    @property
    def date_range(self) -> tuple[str, str] | None:
        """
        Earliest and latest `date_posted` across all postings.

        Returns `None` when no postings have a `date_posted` value.

        Returns:
            A tuple of ISO date strings, or `None` if no dates
            exist.
        """
        dates = [
            p.date_posted for p in self._postings if p.date_posted
        ]
        if not dates:
            return None
        return (min(dates).isoformat(), max(dates).isoformat())

    @property
    def source_type_counts(self) -> dict[str, int]:
        """
        Number of postings per `SourceType`, sorted descending.

        Returns:
            A dictionary mapping source type values to counts.
        """
        return dict(
            Counter(
                p.source_type.value for p in self._postings
            ).most_common()
        )

    @property
    def total(self) -> int:
        """
        Total number of postings in the corpus.

        Returns:
            The posting count.
        """
        return len(self._postings)

    def report(self) -> str:
        """
        Render a human-readable summary of the corpus.

        Includes total count, company breakdown, date range, and
        source type distribution. The top-10 companies are shown
        when the corpus contains more than 10 distinct employers.

        Returns:
            A multi-line string suitable for logging.
        """
        lines = [
            "Corpus Statistics",
            f"  Total postings: {self.total}",
            f"  Companies: {len(self.company_counts)}",
            f"  Date range: "
            f"{self.date_range or 'no dates available'}"
        ]

        if self.source_type_counts:
            lines.append("  Source types:")
            for source, count in self.source_type_counts.items():
                lines.append(f"    {source}: {count}")

        if self.company_counts:
            lines.append("  Top companies:")
            for company, count in list(
                self.company_counts.items()
            )[:10]:
                lines.append(f"    {company}: {count}")

        return "\n".join(lines)
