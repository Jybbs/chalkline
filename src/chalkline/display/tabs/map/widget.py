"""
Interactive career pathway map rendered with D3 via AnyWidget.

Produces a columnar node-link diagram where columns represent Job Zone
levels, rows group by sector, and edges show career transitions weighted
by cosine similarity. Click events on nodes sync back to Python through
traitlets for reactive panel updates in the Marimo notebook.
"""

from anywidget           import AnyWidget
from json                import dumps
from pathlib             import Path
from traitlets.traitlets import Int, Unicode
from typing              import Self

from chalkline.display.schemas   import Labels, MapGeometry, MapLayout
from chalkline.display.theme     import Theme
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
        labels     : Labels,
        labor      : LaborLoader,
        matched_id : int,
        theme      : Theme
    ) -> Self:
        """
        Build the widget with serialized graph data and theme colors.

        Pre-computes deterministic node positions, BFS hop distances from
        the matched cluster, per-node sector colors and opacity tiers
        from hop distance, truncated display titles, per-edge endpoints,
        midpoints, and colors, and the static theme palette so the JS
        renderer is a thin SVG mapper with no duplicated palette or
        label state.

        Args:
            clusters   : Fitted cluster container with metadata.
            graph      : Career pathway graph with edges and credentials.
            labels     : Shared display labels for Job Zone columns.
            labor      : Wage data lookup per SOC title.
            matched_id : Cluster ID of the user's matched career.
            theme      : Palette for sector colors and static UI hues.

        Returns:
            Configured `PathwayMap` ready for Marimo wrapping.
        """
        geometry   = MapGeometry()
        layout     = MapLayout.from_clusters(clusters, geometry)
        hops       = graph.hops_from(matched_id)
        char_limit = geometry.title_char_limit
        columns    = layout.labeled_columns(labels.job_zones)

        nodes = []
        for cluster in clusters.values():
            soc_title = cluster.soc_title
            wage      = labor[soc_title].annual_median
            position  = layout.positions[cluster.cluster_id]
            hop       = hops.get(cluster.cluster_id)
            nodes.append({
                "color"    : theme.sector_background(cluster.sector),
                "id"       : cluster.cluster_id,
                "opacity"  : (
                    geometry.hop_opacities[-1] if hop is None
                    else geometry.hop_opacities[min(hop, geometry.max_hop_index)]
                ),
                "subtitle" : (
                    f"{cluster.size} postings \u00b7 ${round(wage / 1000)}k"
                    if wage else f"{cluster.size} postings"
                ),
                "title"    : (
                    soc_title if len(soc_title) <= char_limit
                    else soc_title[:char_limit - 1] + "\u2026"
                ),
                "x"        : position["x"],
                "y"        : position["y"]
            })

        edges = []
        for source_id, target_id, data in graph.graph.edges(data=True):
            if matched_id not in (source_id, target_id):
                continue
            source_cluster = clusters[source_id]
            target_cluster = clusters[target_id]
            source_pos     = layout.positions[source_id]
            target_pos     = layout.positions[target_id]
            edges.append({
                "color"            : theme.sector_background(source_cluster.sector),
                "credential_count" : len(data.get("credentials", [])),
                "is_advancement"   : target_cluster.job_zone > source_cluster.job_zone,
                "is_cross_sector"  : source_cluster.sector != target_cluster.sector,
                "mx"               : (source_pos["x"] + target_pos["x"]) / 2,
                "my"               : (
                    (source_pos["y"] + target_pos["y"]) / 2
                    - geometry.edge_midpoint_offset
                ),
                "source"           : source_id,
                "sx"               : source_pos["x"],
                "sy"               : source_pos["y"],
                "target"           : target_id,
                "tx"               : target_pos["x"],
                "ty"               : target_pos["y"],
                "weight"           : round(float(data["weight"]), 3)
            })

        return cls(
            graph_data = dumps({
                "columns"           : columns,
                "dimensions"        : geometry.dimensions,
                "edges"             : edges,
                "matched_x"         : layout.positions[matched_id]["x"],
                "matched_y"         : layout.positions[matched_id]["y"],
                "nodes"             : nodes,
                "total_height"      : layout.total_height,
                "total_width"       : layout.total_width,
                "you_are_here_text" : labels.map_you_are_here
            }),
            matched_id = matched_id
        )
