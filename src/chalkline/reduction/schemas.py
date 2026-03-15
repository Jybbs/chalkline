"""
Data models for PCA dimensionality reduction results.
"""

from pydantic import BaseModel, Field


class ComponentLoading(BaseModel, extra="forbid"):
    """
    Top loading terms for a single PCA component.

    Terms are ordered by descending absolute weight, revealing which
    skills drive each dimension of the reduced space. The
    `variance_ratio` is the fraction of total variance explained by
    this component from the full analysis fit.
    """

    index          : int = Field(ge=0)
    terms          : list[str]
    variance_ratio : float
    weights        : list[float]
