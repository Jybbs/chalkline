"""
Tests for PCA reduction schemas.
"""

from pytest import mark, param, raises

from chalkline.reduction.schemas import ComponentLoading


class TestComponentLoading:
    """
    Validate `ComponentLoading` field constraints and extra-field
    rejection.
    """

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
            id = "extra_field"
        ),
        param(
            {
                "index"          : -1,
                "terms"          : ["welding"],
                "variance_ratio" : 0.3,
                "weights"        : [0.5]
            },
            "greater than or equal",
            id = "negative_index"
        )
    ])
    def test_invalid(self, kwargs, match):
        """
        Invalid kwargs are rejected by Pydantic validation.
        """
        with raises(ValueError, match = match):
            ComponentLoading(**kwargs)
