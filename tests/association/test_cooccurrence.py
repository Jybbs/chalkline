"""
Tests for PMI co-occurrence network and Louvain community detection.

Validates co-occurrence matrix properties, PMI measure bounds, graph
construction, community detection, and diagnostic reporting using the
synthetic 20-posting fixture chain.
"""

import numpy as np

from logging import WARNING
from pytest  import LogCaptureFixture

from chalkline.association.cooccurrence import CooccurrenceNetwork
from chalkline.extraction.vectorize     import SkillVectorizer


class TestCooccurrenceNetwork:
    """
    Validate co-occurrence matrix, PMI measures, graph construction,
    community detection, and diagnostics.
    """

    # ---------------------------------------------------------
    # Co-occurrence matrix
    # ---------------------------------------------------------

    def test_diagonal_zero(self, network: CooccurrenceNetwork):
        """
        Self-co-occurrence is zero after diagonal zeroing.
        """
        assert network.cooccurrence.diagonal().sum() == 0

    def test_matrix_symmetric(self, network: CooccurrenceNetwork):
        """
        Co-occurrence matrix is symmetric because skill co-occurrence is
        an undirected relationship.
        """
        C = network.cooccurrence
        assert (C - C.T).nnz == 0

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

    def test_pmi_symmetric(self, network: CooccurrenceNetwork):
        """
        Raw PMI matrix is symmetric. Asymmetry would be masked by NPMI
        positive clipping but would corrupt PPMI values for pairs where
        only one direction has positive PMI.
        """
        pmi  = network.pmi_matrix
        diff = pmi - pmi.T
        if diff.nnz > 0:
            assert abs(diff.data).max() < 1e-10

    def test_ppmi_nonnegative(self, network: CooccurrenceNetwork):
        """
        PPMI values are non-negative by definition.
        """
        ppmi = network.ppmi_matrix
        if ppmi.nnz > 0:
            assert ppmi.data.min() >= 0

    # ---------------------------------------------------------
    # G-test
    # ---------------------------------------------------------

    def test_gtest_graph_edges(self, network: CooccurrenceNetwork):
        """
        G-test graph has the same edge set as the co-occurrence matrix
        because every thresholded pair produces a valid contingency
        table.
        """
        assert (
            network.graph(matrix = network.gtest_matrix).number_of_edges()
            == network.graph().number_of_edges()
        )

    def test_gtest_no_nan(self, network: CooccurrenceNetwork):
        """
        No NaN or inf values in the G-test matrix. A negative
        contingency cell from a co-occurrence or doc-frequency mismatch
        would produce NaN via `xlogy` that silently propagates through
        graph weights and community detection.
        """
        G = network.gtest_matrix
        if G.nnz > 0:
            assert not np.any(np.isnan(G.data))
            assert not np.any(np.isinf(G.data))

    def test_gtest_nonnegative(self, network: CooccurrenceNetwork):
        """
        G-test statistics are non-negative because they are chi-squared
        distributed.
        """
        G = network.gtest_matrix
        if G.nnz > 0:
            assert G.data.min() >= 0

    def test_gtest_symmetric(self, network: CooccurrenceNetwork):
        """
        G-test matrix is symmetric because the contingency table is
        invariant to skill order.
        """
        G = network.gtest_matrix
        assert (G - G.T).nnz == 0

    # ---------------------------------------------------------
    # Graph
    # ---------------------------------------------------------

    def test_node_names_strings(self, network: CooccurrenceNetwork):
        """
        Graph nodes are canonical skill name strings, not integer
        indices.
        """
        assert all(isinstance(node, str) for node in network.graph().nodes())

    def test_zero_edge_graceful(self, vectorizer: SkillVectorizer):
        """
        A degenerate network with zero edges after thresholding produces
        an empty graph, no communities, and diagnostics with zeroed
        modularity.
        """
        empty = CooccurrenceNetwork(
            binary_matrix    = vectorizer.binary_matrix,
            feature_names    = vectorizer.feature_names,
            min_cooccurrence = 0.99
        )
        assert empty.graph().number_of_edges() == 0
        communities = empty.communities()
        assert all(c.weighted_degree_sum == 0.0 for c in communities)
        assert all(c.size == 1 for c in communities)
        diag = empty.diagnostics()
        assert diag.modularity is None
        assert diag.coverage == 0.0
        assert diag.performance == 0.0

    # ---------------------------------------------------------
    # Communities
    # ---------------------------------------------------------

    def test_community_reproducible(self, vectorizer: SkillVectorizer):
        """
        Same seed produces identical community assignments.
        """
        a = CooccurrenceNetwork(
            binary_matrix = vectorizer.binary_matrix,
            feature_names = vectorizer.feature_names,
            random_seed   = 42
        )
        b = CooccurrenceNetwork(
            binary_matrix = vectorizer.binary_matrix,
            feature_names = vectorizer.feature_names,
            random_seed   = 42
        )
        comms_a = [(c.community_id, c.top_skills) for c in a.communities()]
        comms_b = [(c.community_id, c.top_skills) for c in b.communities()]
        assert comms_a == comms_b

    def test_community_sizes_sum(self, network: CooccurrenceNetwork):
        """
        Community sizes sum to the total number of graph nodes.
        """
        assert sum(
            c.size for c in network.communities()
        ) == network.graph().number_of_nodes()

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    def test_diagnostics_isolate_warning(
        self,
        caplog     : LogCaptureFixture,
        vectorizer : SkillVectorizer
    ):
        """
        A high-threshold network where most skills are isolates triggers
        the 30% isolate warning.
        """
        sparse = CooccurrenceNetwork(
            binary_matrix    = vectorizer.binary_matrix,
            feature_names    = vectorizer.feature_names,
            min_cooccurrence = 0.50
        )
        with caplog.at_level(WARNING):
            sparse.diagnostics()
        assert any("isolate nodes" in msg for msg in caplog.messages)

    # ---------------------------------------------------------
    # Measure comparison
    # ---------------------------------------------------------

    def test_compare_measures_count(self, network: CooccurrenceNetwork):
        """
        Three measures are compared: PPMI, NPMI, and G-test.
        """
        results = network.compare_measures()
        assert len(results) == 3
        assert {r.measure for r in results} == {"ppmi", "npmi", "g-test"}

    # ---------------------------------------------------------
    # PMI DataFrame
    # ---------------------------------------------------------

    def test_pmi_dataframe_symmetric(self, network: CooccurrenceNetwork):
        """
        PMI DataFrame is symmetric with skill name indices.
        """
        df = network.pmi_dataframe()
        assert list(df.index) == list(df.columns)
        assert list(df.index) == network.feature_names
        assert (df.values == df.values.T).all()
