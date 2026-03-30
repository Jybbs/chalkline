"""
Tests for corpus ordering.

Validates `Corpus` description alignment with sorted posting keys.
"""

from chalkline.collection.schemas import Corpus, Posting


class TestCorpus:
    """
    Validate corpus key ordering and description alignment.
    """

    def test_descriptions_aligned(self):
        """
        Descriptions follow the same sorted-key order as `posting_ids`.
        """
        postings = {
            f"b_{i}": Posting(
                company     = "Co",
                date_posted = None,
                description = f"{'x' * 50} {i}",
                id          = f"b_{i}",
                source_url  = "https://example.com",
                title       = "Worker"
            )
            for i in range(3)
        }
        corpus = Corpus(postings)
        assert len(corpus.descriptions) == 3
        assert corpus.descriptions[0] == postings[corpus.posting_ids[0]].description
