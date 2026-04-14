"""
Interactive career pathway map rendered with D3 via AnyWidget.

Produces a force-directed node-link diagram where horizontal position
encodes salary (higher wages drift right), node rendering tiers
distinguish immediate career paths from distant options, and the matched
career renders as an enriched hero card integrated into the SVG. Click
and hover events sync back to Python through traitlets for reactive
panel updates in the Marimo notebook.
"""

from anywidget           import AnyWidget
from json                import dumps
from pathlib             import Path
from traitlets.traitlets import Int, Unicode
from typing              import Self

from chalkline.display.schemas   import MapGeometry
from chalkline.display.theme     import Theme
from chalkline.matching.matcher  import ResumeMatcher
from chalkline.matching.schemas  import MatchResult
from chalkline.pathways.clusters import Clusters
from chalkline.pathways.graph    import CareerPathwayGraph
from chalkline.pathways.loaders  import LaborLoader


class PathwayMap(AnyWidget):
    """
    AnyWidget career pathway map with click-to-select interactivity.

    Traitlets:
        graph_data  : JSON string from `from_graph`.
        matched_id  : Cluster ID of the user's matched career family.
        selected_id : Cluster ID clicked by the user (-1 when none).
    """

    graph_data  = Unicode("{}").tag(sync=True)
    matched_id  = Int(-1).tag(sync=True)
    selected_id = Int(-1).tag(sync=True)

    _esm = Path(__file__).parent / "widget.js"

    @classmethod
    def from_graph(
        cls,
        clusters   : Clusters,
        graph      : CareerPathwayGraph,
        labor      : LaborLoader,
        matched_id : int,
        matcher    : ResumeMatcher,
        result     : MatchResult,
        theme      : Theme
    ) -> Self:
        """
        Build the widget with serialized graph data for the force-directed
        D3 renderer.

        Calls the matcher's calibration to obtain per-cluster match
        percentages, then assembles per-node metadata (wage, sector,
        tier, display title with collision handling), per-edge
        metadata, and enriched hero card data for the matched node.
        The JS renderer runs a force simulation to compute positions
        and handles all visual rendering.

        Args:
            clusters   : Fitted cluster container with metadata.
            graph      : Career pathway graph with edges and credentials.
            labor      : Wage data lookup per SOC title.
            matched_id : Cluster ID of the user's matched career.
            matcher    : Fitted matcher for per-cluster similarity calibration.
            result     : Resume match result with confidence and reach.
            theme      : Palette for sector colors and static UI hues.

        Returns:
            Configured `PathwayMap` ready for Marimo wrapping.
        """
        cluster_mean = matcher.calibrate()
        geometry     = MapGeometry()
        hops         = graph.hops_from(matched_id)

        def select_wage(cluster) -> float | None:
            record = labor[cluster.soc_title]
            if clusters.soc_counts[cluster.soc_title] < 2:
                return record.annual_median
            jz_peers = sorted(
                c.job_zone for c in clusters.values()
                if c.soc_title == cluster.soc_title
            )
            rank = jz_peers.index(cluster.job_zone)
            if rank == 0 and record.annual_25:
                return record.annual_25
            if rank == len(jz_peers) - 1 and record.annual_75:
                return record.annual_75
            return record.annual_median

        nodes = []
        for cluster in clusters.values():
            cid   = cluster.cluster_id
            hop   = hops.get(cid)
            title = (
                cluster.modal_title
                if clusters.soc_counts[cluster.soc_title] > 1
                else cluster.soc_title
            )
            wage  = select_wage(cluster)

            nodes.append({
                "color"     : theme.sector_background(cluster.sector),
                "full_title": title,
                "hop"       : hop,
                "id"        : cid,
                "match_pct" : round(100 * cluster_mean.get(cid, 0.0)),
                "sector"    : cluster.sector,
                "subtitle"  : (
                    f"{cluster.size} postings \u00b7 ${round(wage / 1000)}k"
                    if wage else f"{cluster.size} postings"
                ),
                "tier"      : 1 if hop is not None and hop <= 1 else 2,
                "title"     : title,
                "wage"      : wage
            })

        wages = [n["wage"] for n in nodes if n["wage"]]

        edges = [
            {
                "color"  : theme.sector_background(clusters[source_id].sector),
                "source" : source_id,
                "target" : target_id,
                "weight" : round(float(data["weight"]), 3)
            }
            for source_id, target_id, data in graph.graph.edges(data=True)
        ]

        profile = clusters[matched_id]

        hero = {
            "n_matches"    : len(result.reach.edges),
            "sector_color" : theme.colors["lavender"],
            "size"         : profile.size,
            "title"        : profile.soc_title,
            "wage"         : select_wage(profile)
        }

        return cls(
            graph_data = dumps({
                "dimensions" : geometry.dimensions,
                "edges"      : edges,
                "hero"       : hero,
                "nodes"      : nodes,
                "wage_range" : (
                    [min(wages), max(wages)]
                    if wages else geometry.default_wage_range
                )
            }),
            matched_id = matched_id
        )
