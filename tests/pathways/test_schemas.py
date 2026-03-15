"""
Validate custom field constraints on pathway schema models.

Tests focus on `Field` constraints that could be silently loosened (ge, le
bounds) rather than Pydantic's own type system guarantees (extra-field
rejection, default values, field assignment).
"""

from pytest import mark, raises

from pydantic import ValidationError

from chalkline.pathways.schemas import CareerNode, DagResult, SocMatch


class TestPathwaySchemas:
    """
    Validate custom field constraints on pathway schema models.
    """

    # -----------------------------------------------------------------
    # CareerNode
    # -----------------------------------------------------------------

    @mark.parametrize("job_zone", [0, 6])
    def test_node_job_zone_bounds(self, job_zone: int):
        """
        Job Zone outside [1, 5] is rejected.
        """
        with raises(ValidationError):
            CareerNode(
                cluster_id = 1,
                job_zone   = job_zone,
                sector     = "Building Construction",
                size       = 10,
                skills     = ["welding"],
                terms      = ["welding"]
            )

    def test_node_negative_cluster(self):
        """
        Negative cluster ID is rejected by ge=0.
        """
        with raises(ValidationError):
            CareerNode(
                cluster_id = -1,
                job_zone   = 3,
                sector     = "Building Construction",
                size       = 10,
                skills     = ["welding"],
                terms      = ["welding"]
            )

    # -----------------------------------------------------------------
    # DagResult
    # -----------------------------------------------------------------

    def test_dag_negative_removed(self):
        """
        Negative edges_removed is rejected by ge=0.
        """
        with raises(ValidationError):
            DagResult(
                edges_removed = -1,
                longest_path  = [1, 2],
                path_weight   = 3.0
            )

    # -----------------------------------------------------------------
    # SocMatch
    # -----------------------------------------------------------------

    @mark.parametrize("overlap", [-0.1, 1.5])
    def test_soc_match_overlap_bounds(self, overlap: float):
        """
        Overlap outside [0, 1] is rejected.
        """
        with raises(ValidationError):
            SocMatch(
                job_zone = 3,
                overlap  = overlap,
                soc_code = "47-2111.00"
            )
