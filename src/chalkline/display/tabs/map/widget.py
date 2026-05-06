"""
Interactive career pathway map rendered with D3 via AnyWidget.

Produces a force-directed node layout where horizontal position encodes
salary (higher wages drift right), vertical bands group clusters by
sector, render tiers separate strong-match suggestions from the rest of
the corpus, and the matched career anchors the canvas as an enriched
hero card. Click and hover events sync back to Python through traitlets
for reactive panel updates in the Marimo notebook.
"""

from anywidget           import AnyWidget
from json                import dumps
from kneed               import KneeLocator
from pathlib             import Path
from traitlets.traitlets import Int, Unicode
from typing              import Self

from chalkline.display.schemas   import MapGeometry
from chalkline.display.theme     import Theme
from chalkline.matching.matcher  import ResumeMatcher
from chalkline.matching.schemas  import MatchResult
from chalkline.pathways.clusters import Clusters
from chalkline.pathways.graph    import CareerPathwayGraph


def _tier_assignments(
    cluster_mean : dict[int, float],
    matched_id   : int
) -> dict[int, int]:
    """
    Render-tier per cluster id for the map widget.

    Tier 1 covers the matched cluster and a stable handful of
    strong-match neighbors. With more than 10 non-matched candidates,
    the elbow is clamped to `[10, 15]` so the canvas surfaces a
    meaningful set of next-step options without flooding. Smaller
    corpora promote every non-matched cluster, and flat curves fall
    back to the upper bound.

    Args:
        cluster_mean : Per-cluster match score in `[0, 1]`, keyed by id.
        matched_id   : Cluster id rendered as the hero card.

    Returns:
        Cluster id to tier (1 or 2). Tier 1 includes `matched_id`.
    """
    ranked_ids = sorted(
        cluster_mean.keys() - {matched_id},
        key     = cluster_mean.__getitem__,
        reverse = True
    )
    scores = list(map(cluster_mean.__getitem__, ranked_ids))

    if len(scores) <= 10:
        cut = len(scores)
    elif max(scores) == min(scores):
        cut = 15
    else:
        elbow = KneeLocator(
            range(len(scores)),
            scores,
            curve     = "convex",
            direction = "decreasing"
        ).knee
        cut = max(10, min(15, elbow if elbow is not None else 15))

    tier1 = {matched_id, *ranked_ids[:cut]}
    return {cid: 1 if cid in tier1 else 2 for cid in cluster_mean}


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
        matched_id  : int,
        matcher     : ResumeMatcher,
        result      : MatchResult,
        theme       : Theme,
        wage_filter : tuple[float, float] | None = None
    ) -> str:
        """
        Serialize the force-directed graph payload as JSON.

        Calls the matcher's calibration to obtain per-cluster match
        percentages, ranks remaining clusters against the matched one to
        derive tier assignments via `_tier_assignments`, and assembles
        per-node metadata, hero card data, and a sector legend keyed to
        the colors actually present on the canvas.

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

        visible = [
            c for c in clusters.values()
            if c.cluster_id == matched_id
            or wage_filter is None
            or (c.wage is not None and wage_filter[0] <= c.wage <= wage_filter[1])
        ]
        tiers = _tier_assignments(
            {c.cluster_id: cluster_mean[c.cluster_id] for c in visible},
            matched_id
        )
        nodes = [
            {
                "color"      : theme.sector_background(cluster.sector),
                "full_title" : cluster.display_title,
                "id"         : cluster.cluster_id,
                "match_pct"  : round(100 * cluster_mean[cluster.cluster_id]),
                "sector"     : cluster.sector,
                "subtitle"   : (
                    f"{cluster.size} postings \u00b7 ${round(cluster.wage / 1000)}k"
                    if cluster.wage else f"{cluster.size} postings"
                ),
                "suffix"     : (
                    cluster.display_title[len(cluster.soc_title) + 2:-1]
                    if cluster.display_title != cluster.soc_title else ""
                ),
                "tier"       : tiers[cluster.cluster_id],
                "title"      : cluster.soc_title,
                "wage"       : cluster.wage
            }
            for cluster in visible
        ]

        edges = [
            {
                "color"  : n["color"],
                "source" : matched_id,
                "target" : n["id"],
                "weight" : round(cluster_mean[n["id"]], 3)
            }
            for n in nodes
            if n["tier"] == 1 and n["id"] != matched_id
        ]

        wages = [n["wage"] for n in nodes if n["wage"]]

        hero = {
            "match_color"  : theme.colors["lavender"],
            "n_matches"    : len(result.reach.edges),
            "sector_color" : theme.sector_background(clusters[matched_id].sector),
            "size"         : clusters[matched_id].size,
            "title"        : clusters[matched_id].display_title,
            "wage"         : clusters[matched_id].wage
        }

        present_sectors = sorted(
            {n["sector"] for n in nodes} & theme.sectors.keys()
        )
        legend = [{"label": "Your Match", "color": theme.colors["lavender"]}] + [
            {"label": s, "color": theme.sectors[s]} for s in present_sectors
        ]

        return dumps({
            "dimensions" : geometry.dimensions,
            "edges"      : edges,
            "hero"       : hero,
            "legend"     : legend,
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
                matched_id  = matched_id,
                matcher     = matcher,
                result      = result,
                theme       = theme,
                wage_filter = wage_filter
            ),
            matched_id = matched_id
        )

