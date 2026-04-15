"""
Interactive career pathway map rendered with D3 via AnyWidget.

Produces a force-directed node-link diagram where horizontal position
encodes salary (higher wages drift right), node rendering tiers distinguish
immediate career paths from distant options, and the matched career renders
as an enriched hero card integrated into the SVG. Click and hover events
sync back to Python through traitlets for reactive panel updates in the
Marimo notebook.
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
    def build_graph_data(
        cls,
        clusters    : Clusters,
        graph       : CareerPathwayGraph,
        labor       : LaborLoader,
        matched_id  : int,
        matcher     : ResumeMatcher,
        result      : MatchResult,
        theme       : Theme,
        wage_filter : tuple[float, float] | None = None
    ) -> str:
        """
        Serialize the force-directed graph payload as JSON.

        Calls the matcher's calibration to obtain per-cluster match
        percentages, then assembles per-node metadata (wage, sector, tier,
        display title with collision handling), per-edge metadata, and
        enriched hero card data for the matched node.

        Tier-2 clusters whose median wage falls outside `wage_filter` are
        dropped before rendering. The matched cluster always renders so the
        user keeps their hero card regardless of where their own wage sits
        in the corpus distribution.

        Separated from `from_graph` so a live widget can recompute its
        payload in response to filter changes and assign the result to its
        `graph_data` traitlet, triggering the JS change listener without a
        remount.

        Args:
            clusters    : Fitted cluster container with metadata.
            graph       : Career pathway graph with edges and credentials.
            labor       : Wage data lookup per SOC title.
            matched_id  : Cluster ID of the user's matched career.
            matcher     : Fitted matcher for per-cluster similarity calibration.
            result      : Resume match result with confidence and reach.
            theme       : Palette for sector colors and static UI hues.
            wage_filter : Inclusive (low, high) wage bounds for tier-2 cards.

        Returns:
            JSON string suitable for the `graph_data` traitlet.
        """
        cluster_mean = matcher.calibrate()
        matcher.calibrate_coverage(graph.credential_pool, graph.credential_vectors)

        geometry = MapGeometry()
        hops     = graph.hops_from(matched_id)

        def in_range(wage: float | None, cid: int) -> bool:
            if cid == matched_id or wage_filter is None:
                return True
            if wage is None:
                return False
            return wage_filter[0] <= wage <= wage_filter[1]

        nodes = []
        for cluster in clusters.values():
            cid  = cluster.cluster_id
            wage = cluster.wage
            if not in_range(wage, cid):
                continue

            hop = hops.get(cid)

            nodes.append({
                "color"      : theme.sector_background(cluster.sector),
                "full_title" : cluster.display_title,
                "hop"        : hop,
                "id"         : cid,
                "match_pct"  : round(100 * cluster_mean.get(cid, 0.0)),
                "sector"     : cluster.sector,
                "subtitle"   : (
                    f"{cluster.size} postings \u00b7 ${round(wage / 1000)}k"
                    if wage else f"{cluster.size} postings"
                ),
                "tier"       : 1 if hop is not None and hop <= 1 else 2,
                "title"      : cluster.display_title,
                "wage"       : wage
            })

        wages    = [n["wage"] for n in nodes if n["wage"]]
        node_ids = {n["id"] for n in nodes}

        edges = [
            {
                "color"  : theme.sector_background(clusters[source_id].sector),
                "source" : source_id,
                "target" : target_id,
                "weight" : round(float(data["weight"]), 3)
            }
            for source_id, target_id, data in graph.graph.edges(data=True)
            if source_id in node_ids and target_id in node_ids
        ]

        profile = clusters[matched_id]

        hero = {
            "match_color"  : theme.colors["lavender"],
            "n_matches"    : len(result.reach.edges),
            "sector_color" : theme.sector_background(profile.sector),
            "size"         : profile.size,
            "title"        : profile.display_title,
            "wage"         : profile.wage
        }

        return dumps({
            "dimensions" : geometry.dimensions,
            "edges"      : edges,
            "hero"       : hero,
            "nodes"      : nodes,
            "wage_range" : (
                [min(wages), max(wages)]
                if wages else geometry.default_wage_range
            )
        })

    @classmethod
    def from_graph(
        cls,
        clusters    : Clusters,
        graph       : CareerPathwayGraph,
        labor       : LaborLoader,
        matched_id  : int,
        matcher     : ResumeMatcher,
        result      : MatchResult,
        theme       : Theme,
        wage_filter : tuple[float, float] | None = None
    ) -> Self:
        """
        Build the widget with its initial serialized graph payload.

        Composes `build_graph_data` with traitlet construction so the
        returned widget is ready for Marimo wrapping. Subsequent filter
        changes should assign a fresh `build_graph_data` result to
        `widget.graph_data` rather than constructing a new widget, avoiding
        an unmount/remount cycle in the frontend.

        Args:
            clusters    : Fitted cluster container with metadata.
            graph       : Career pathway graph with edges and credentials.
            labor       : Wage data lookup per SOC title.
            matched_id  : Cluster ID of the user's matched career.
            matcher     : Fitted matcher for per-cluster similarity calibration.
            result      : Resume match result with confidence and reach.
            theme       : Palette for sector colors and static UI hues.
            wage_filter : Inclusive (low, high) wage bounds for tier-2 cards.

        Returns:
            Configured `PathwayMap` ready for Marimo wrapping.
        """
        return cls(
            graph_data = cls.build_graph_data(
                clusters    = clusters,
                graph       = graph,
                labor       = labor,
                matched_id  = matched_id,
                matcher     = matcher,
                result      = result,
                theme       = theme,
                wage_filter = wage_filter
            ),
            matched_id = matched_id
        )

