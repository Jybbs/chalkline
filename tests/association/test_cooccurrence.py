"""
Tests for PMI co-occurrence network and Louvain community detection.

Validates NPMI formula correctness, threshold filtering, graph
construction, and PPMI DataFrame materialization using the synthetic
20-posting fixture chain.
"""

import numpy as np

from chalkline.association.cooccurrence import CooccurrenceNetwork
from chalkline.extraction.vectorize     import SkillVectorizer


class TestCooccurrenceNetwork:
    """
    Validate NPMI, thresholding, and graph construction.
    """

    def test_threshold_filters(self, vectorizer: SkillVectorizer):
        """
        Pairs below the co-occurrence threshold produce no edges.
        """
        strict = CooccurrenceNetwork(
            binary_matrix    = vectorizer.binary_matrix,
            feature_names    = vectorizer.feature_names,
            min_cooccurrence = 0.99
        )
        assert strict.cooccurrence.nnz == 0

    def test_npmi_formula(self, network: CooccurrenceNetwork):
        """
        NPMI for a known pair equals the manual formula, catching
        data-ordering mismatches between PMI and co-occurrence sparse
        arrays that would silently corrupt all downstream graph weights
        and gap ranking.

        Picks the first nonzero entry and computes:

            NPMI = clip(PMI / -log(C_xy / n), 0, 1)
        """
        npmi = network.npmi_matrix
        if npmi.nnz == 0:
            return

        rows, cols = npmi.nonzero()
        r, c       = int(rows[0]), int(cols[0])

        c_xy  = float(network.cooccurrence[r, c])
        n     = float(network.n_docs)
        df_r  = float(network.doc_freq[r])
        df_c  = float(network.doc_freq[c])

        pmi_val   = np.log(n * c_xy / (df_r * df_c))
        denom_val = np.log(n / c_xy)
        expected  = (
            max(0.0, min(1.0, pmi_val / denom_val))
            if denom_val > 0 else 1.0
        )

        assert abs(float(npmi[r, c]) - expected) < 1e-10

    def test_node_names_strings(self, network: CooccurrenceNetwork):
        """
        Graph nodes are canonical skill name strings, not integer
        indices.
        """
        assert all(isinstance(node, str) for node in network.graph().nodes())
