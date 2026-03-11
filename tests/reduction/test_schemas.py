"""
Tests for PCA reduction schemas.
"""

from pytest import mark, param, raises

from chalkline.reduction.schemas import ComponentLoading


class TestComponentLoading:
    """
    Validate ComponentLoading field constraints and extra-field
    rejection.
    """

    # ---------------------------------------------------------
    # Construction
    # ---------------------------------------------------------

    def test_valid(self):
        """
        A well-formed loading with matching terms and weights
        validates successfully.
        """
        loading = ComponentLoading(
            index          = 0,
            terms          = ["welding", "scaffolding"],
            variance_ratio = 0.35,
            weights        = [0.8, 0.6]
        )
        assert loading.index == 0
        assert loading.terms == ["welding", "scaffolding"]

    # ---------------------------------------------------------
    # Rejection
    # ---------------------------------------------------------

    @mark.parametrize("kwargs, match", [
        param(
            {
                "extra_field"    : True,
                "index"          : 0,
                "terms"          : ["welding"],
                "variance_ratio" : 0.3,
                "weights"        : [0.5]
            },
            "Extra inputs",
            id="extra_field"
        ),
        param(
            {
                "index"          : -1,
                "terms"          : ["welding"],
                "variance_ratio" : 0.3,
                "weights"        : [0.5]
            },
            "greater than or equal",
            id="negative_index"
        )
    ])
    def test_invalid(self, kwargs, match):
        """
        Invalid kwargs are rejected by Pydantic validation.
        """
        with raises(ValueError, match=match):
            ComponentLoading(**kwargs)
