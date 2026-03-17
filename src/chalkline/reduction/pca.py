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

from scipy.sparse          import spmatrix
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import StandardScaler

from chalkline.reduction.schemas import ComponentLoading


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
        document_ids       : list[str],
        feature_names      : list[str],
        max_components     : int,
        random_seed        : int,
        tfidf_matrix       : spmatrix,
        variance_threshold : float
    ):
        """
        Fit the analysis SVD and production pipeline on the
        sparse TF-IDF matrix from `SkillVectorizer`.

        Document identifiers and vocabulary terms maintain row
        and column alignment with the input matrix for downstream
        labeling. The maximum component count is capped at one
        below the matrix rank to avoid rank-deficient
        decomposition, and both SVD fits share the same random
        seed for reproducibility.

        Args:
            document_ids       : Posting identifiers in row order.
            feature_names      : Vocabulary terms in column order.
            max_components     : Upper bound on components.
            random_seed        : Reproducibility seed.
            tfidf_matrix       : Sparse TF-IDF matrix.
            variance_threshold : Cumulative variance target.
        """
        self.document_ids  = document_ids
        self.feature_names = feature_names

        effective_max = min(max_components, min(tfidf_matrix.shape) - 1)

        self.explained_variance_ratio = TruncatedSVD(
            n_components = effective_max,
            random_state = random_seed
        ).fit(tfidf_matrix).explained_variance_ratio_

        cumulative               = np.cumsum(self.explained_variance_ratio)
        self.n_selected          = min(
            np.searchsorted(cumulative, variance_threshold) + 1,
            effective_max
        )
        self.cumulative_variance = cumulative[self.n_selected - 1]

        self.pipeline = Pipeline([
            ("svd", TruncatedSVD(
                n_components = self.n_selected,
                random_state = random_seed
            )),
            ("scaler", StandardScaler(with_mean=False))
        ])
        self.coordinates = self.pipeline.fit_transform(tfidf_matrix)

    # -----------------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------------

    def loadings(self, top_n: int = 10) -> list[ComponentLoading]:
        """
        Top loading terms for each selected component.

        Extracts the `top_n` terms with the highest absolute weights
        from each component's loading vector, returning skill names
        rather than column indices.

        Args:
            top_n: Number of top terms per component.

        Returns:
            One `ComponentLoading` per selected component with terms,
            weights, and variance ratio.
        """
        names = np.array(self.feature_names)
        return [
            ComponentLoading(
                index          = i,
                terms          = names[indices].tolist(),
                variance_ratio = self.explained_variance_ratio[i],
                weights        = row[indices].tolist()
            )
            for i, row in enumerate(
                self.pipeline.named_steps["svd"].components_
            )
            for indices in [np.argsort(np.abs(row))[::-1][:top_n]]
        ]
