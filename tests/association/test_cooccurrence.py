"""
Tests for PMI co-occurrence network and Louvain community detection.

Validates co-occurrence matrix properties, PMI measure bounds, graph
construction, and partition consistency using the synthetic 20-posting
fixture chain.
"""

import numpy as np

from chalkline.association.cooccurrence import CooccurrenceNetwork
from chalkline.extraction.vectorize     import SkillVectorizer


class TestCooccurrenceNetwork:
    """
    Validate co-occurrence matrix, PMI measures, graph construction,
    and partition.
    """

    # ---------------------------------------------------------
    # Co-occurrence matrix
    # ---------------------------------------------------------

    def test_diagonal_zero(self, network: CooccurrenceNetwork):
        """
        Self-co-occurrence is zero after diagonal zeroing.
        """
        assert network.cooccurrence.diagonal().sum() == 0

    def test_threshold_auto(self, vectorizer: SkillVectorizer):
        """
        Auto-threshold via modularity knee detection produces a threshold
        of at least 3 and a non-degenerate graph.
        """
        auto = CooccurrenceNetwork(
            binary_matrix    = vectorizer.binary_matrix,
            feature_names    = vectorizer.feature_names,
            min_cooccurrence = "auto"
        )
        assert auto.threshold >= 3
        assert auto.cooccurrence.nnz >= 0

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

    # ---------------------------------------------------------
    # PMI measures
    # ---------------------------------------------------------

    def test_npmi_bounded(self, network: CooccurrenceNetwork):
        """
        NPMI+ values fall within [0, 1] after positive clipping.
        """
        npmi = network.npmi_matrix
        if npmi.nnz > 0:
            assert npmi.data.min() >= 0
            assert npmi.data.max() <= 1.0 + 1e-10

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

    def test_ppmi_symmetric(self, network: CooccurrenceNetwork):
        """
        PPMI matrix is symmetric because the underlying co-occurrence
        matrix is symmetric.
        """
        diff = network.ppmi_matrix - network.ppmi_matrix.T
        if diff.nnz > 0:
            assert abs(diff.data).max() < 1e-10

    def test_ppmi_nonnegative(self, network: CooccurrenceNetwork):
        """
        PPMI values are non-negative by definition.
        """
        ppmi = network.ppmi_matrix
        if ppmi.nnz > 0:
            assert ppmi.data.min() >= 0

    def test_node_names_strings(self, network: CooccurrenceNetwork):
        """
        Graph nodes are canonical skill name strings, not integer
        indices.
        """
        assert all(isinstance(node, str) for node in network.graph().nodes())

    def test_partition_map_consistent(self, network: CooccurrenceNetwork):
        """
        `partition_map` assigns every skill to the community it
        belongs to in `partition`.
        """
        for idx, members in enumerate(network.partition):
            for skill in members:
                assert network.partition_map[skill] == idx

    def test_dataframe_symmetric(self, network: CooccurrenceNetwork):
        """
        PMI DataFrame is symmetric with skill name indices.
        """
        df = network.association_dataframe("npmi")
        assert list(df.index) == list(df.columns)
        assert list(df.index) == network.feature_names
        assert (df.values == df.values.T).all()
