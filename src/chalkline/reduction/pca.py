"""
Dimensionality reduction via TruncatedSVD with variance-based
component selection.

Fits an initial TruncatedSVD at `max_components` to profile the
explained variance spectrum, selects the effective rank by cumulative
variance threshold, then refits a production `Pipeline` at the
selected rank.

A `StandardScaler(with_mean=False)` is applied so each component has
unit variance, making Euclidean distance equivalent to standardized
Euclidean in downstream matching.
"""

import numpy as np

from scipy.sparse          import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler


class PcaReducer:
    """
    TruncatedSVD reduction with automatic component selection.

    Performs two SVD fits against the input IDF-weighted matrix.
    The first fit at `max_components` produces the full explained
    variance profile for scree analysis. The second fit at the
    selected rank feeds a production `Pipeline` whose output has
    exactly `n_selected` columns with unit variance per component.

    Component selection finds the smallest k where cumulative
    explained variance exceeds the threshold:

        k = argmin{sum_i=1..k lambda_i / sum lambda >= tau}

    where lambda_i are the singular values and tau is
    `variance_threshold`.
    """

    def __init__(
        self,
        max_components     : int,
        random_seed        : int,
        tfidf_matrix       : csr_matrix,
        variance_threshold : float
    ):
        """
        Fit the analysis SVD and production pipeline on the
        sparse TF-IDF matrix from `SkillVectorizer`.

        The maximum component count is capped at one below the
        matrix rank to avoid rank-deficient decomposition, and
        both SVD fits share the same random seed for
        reproducibility.

        Args:
            max_components     : Upper bound on components.
            random_seed        : Reproducibility seed.
            tfidf_matrix       : Sparse TF-IDF matrix.
            variance_threshold : Cumulative variance target.
        """
        variance_profile = TruncatedSVD(
            n_components = (
                effective_max := min(max_components, min(tfidf_matrix.shape) - 1)
            ),
            random_state = random_seed
        ).fit(tfidf_matrix).explained_variance_ratio_

        self.n_selected = min(
            np.searchsorted(np.cumsum(variance_profile), variance_threshold) + 1,
            effective_max
        )

        self.pipeline = Pipeline([
            ("svd", TruncatedSVD(
                n_components = self.n_selected,
                random_state = random_seed
            )),
            ("scaler", StandardScaler(with_mean=False))
        ])
        self.coordinates = self.pipeline.fit_transform(tfidf_matrix)
