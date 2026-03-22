"""
Plotly figure builders for the career report panels.

Each function takes pipeline data and returns a configured
`go.Figure`, keeping Plotly construction logic out of the
Marimo notebook cells.
"""

import networkx as nx
import numpy    as np

import plotly.figure_factory as ff
import plotly.graph_objects  as go

from chalkline.pathways.graph   import CareerPathwayGraph
from chalkline.pathways.schemas import Neighborhood


def dendrogram_figure(
    matched_id : int,
    pathway    : CareerPathwayGraph,
    template   : str
) -> go.Figure:
    """
    Ward-linkage dendrogram with the matched cluster annotated.

    Normalizes centroids before linkage so that Ward distances
    reflect direction rather than magnitude. The matched
    cluster's leaf gets a crimson arrow annotation.

    Args:
        matched_id : Cluster ID to annotate.
        pathway    : Fitted career pathway graph.
        template   : Plotly template name for theming.

    Returns:
        Configured dendrogram figure.
    """
    from scipy.cluster.hierarchy import linkage
    from sklearn.preprocessing   import normalize

    normalized = normalize(pathway.centroids)
    Z          = linkage(normalized, method="ward")

    fig = ff.create_dendrogram(
        normalized,
        labels     = [f"C{cid}" for cid in sorted(pathway.profiles)],
        linkagefun = lambda dist: Z
    )

    tick_vals = fig.layout.xaxis.tickvals
    for i, tick in enumerate(fig.layout.xaxis.ticktext or []):
        if tick == f"C{matched_id}":
            fig.update_layout(annotations=[
                *(fig.layout.annotations or []),
                dict(
                    arrowcolor = "crimson",
                    arrowhead  = 2,
                    arrowwidth = 2,
                    font       = dict(color="crimson", size=12),
                    showarrow  = True,
                    text       = "You",
                    x          = tick_vals[i] if tick_vals else i,
                    y          = 0,
                    yshift     = -20
                )
            ])
            break

    fig.update_layout(
        height      = 500,
        template    = template,
        title       = "Hierarchical Clustering Dendrogram",
        xaxis_title = "Career Family",
        yaxis_title = "Ward Distance"
    )
    return fig


def landscape_figure(
    coordinates : list[float],
    matched_id  : int,
    pathway     : CareerPathwayGraph,
    template    : str
) -> go.Figure:
    """
    Scatter plot of cluster centroids in SVD space with the
    resume position overlaid.

    Node sizes scale with betweenness centrality so that gateway
    clusters bridging multiple career families appear larger. The
    matched cluster is highlighted in crimson and the resume
    position rendered as a gold star.

    Args:
        coordinates : Resume SVD position, empty if unavailable.
        matched_id  : Matched cluster ID to highlight.
        pathway     : Fitted career pathway graph.
        template    : Plotly template name for theming.

    Returns:
        Configured landscape scatter figure.
    """
    profiles    = pathway.profiles
    betweenness = nx.betweenness_centrality(pathway.graph, weight="weight")
    cluster_ids = sorted(profiles)
    x = pathway.centroids[:, 0]
    y = (
        pathway.centroids[:, 1]
        if pathway.centroids.shape[1] > 1
        else np.zeros(len(x))
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        hovertext = [
            f"Cluster {cid}<br>{profiles[cid].soc_title}"
            f"<br>JZ {profiles[cid].job_zone} · "
            f"{profiles[cid].size} postings"
            for cid in cluster_ids
        ],
        marker    = dict(
            color = [
                "crimson" if cid == matched_id else "steelblue"
                for cid in cluster_ids
            ],
            line  = dict(color="white", width=1),
            size  = [
                10 + betweenness.get(cid, 0) * 80
                for cid in cluster_ids
            ]
        ),
        mode      = "markers",
        name      = "Career Families",
        x         = x.tolist(),
        y         = y.tolist()
    ))

    if coordinates:
        fig.add_trace(go.Scatter(
            hovertext = ["Your Resume"],
            marker    = dict(
                color  = "gold",
                line   = dict(color="white", width=1.5),
                size   = 14,
                symbol = "star"
            ),
            mode      = "markers",
            name      = "Your Resume",
            x         = [coordinates[0]],
            y         = [coordinates[1]] if len(coordinates) > 1 else [0]
        ))

    fig.update_layout(
        height      = 550,
        hovermode   = "closest",
        showlegend  = True,
        template    = template,
        title       = "Career Landscape (SVD Components 1-2)",
        xaxis_title = "SVD Component 1",
        yaxis_title = "SVD Component 2"
    )
    return fig


def pathways_figure(
    matched_id   : int,
    neighborhood : Neighborhood,
    pathway      : CareerPathwayGraph,
    target_id    : int,
    template     : str
) -> go.Figure:
    """
    Spring-layout network of the target cluster's neighborhood
    with edge annotations showing apprenticeship hours.

    Builds a subgraph from the target and its advancement and
    lateral neighbors, positions nodes via spring layout, and
    annotates edges with up to two apprenticeship programs and
    their minimum hour requirements.

    Args:
        matched_id   : Matched cluster ID for crimson highlighting.
        neighborhood : Advancement and lateral edges from the target.
        pathway      : Fitted career pathway graph.
        target_id    : Center cluster for the neighborhood view.
        template     : Plotly template name for theming.

    Returns:
        Configured pathways network figure.
    """
    profiles = pathway.profiles
    sub = pathway.graph.subgraph({target_id} | {
        edge.profile.cluster_id
        for edge in neighborhood.all_edges
    })
    pos = nx.spring_layout(sub, seed=42, weight="weight")

    edge_x, edge_y = [], []
    edge_annotations = []
    for u, v in sub.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        if hours_parts := [
            f"{c.label}: {c.metadata['min_hours']:,}h"
            for e in neighborhood.all_edges
            if e.profile.cluster_id in (u, v)
            for c in e.credentials
            if c.kind == "apprenticeship"
        ]:
            edge_annotations.append(dict(
                font      = dict(color="gray", size=9),
                showarrow = False,
                text      = "<br>".join(hours_parts[:2]),
                x         = (x0 + x1) / 2,
                y         = (y0 + y1) / 2
            ))

    betweenness = nx.betweenness_centrality(sub, weight="weight")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        hoverinfo = "none",
        line      = dict(color="lightgray", width=1),
        mode      = "lines",
        x         = edge_x,
        y         = edge_y
    ))
    fig.add_trace(go.Scatter(
        hovertext    = [
            f"Cluster {n}<br>{profiles[n].soc_title}<br>"
            f"JZ {profiles[n].job_zone}"
            for n in sub.nodes()
        ],
        marker       = dict(
            color = [
                "crimson" if n == matched_id
                else ("gold" if n == target_id else "steelblue")
                for n in sub.nodes()
            ],
            line  = dict(color="white", width=1),
            size  = [
                15 + betweenness.get(n, 0) * 60
                for n in sub.nodes()
            ]
        ),
        mode         = "markers+text",
        text         = [f"C{n}" for n in sub.nodes()],
        textposition = "top center",
        x            = [pos[n][0] for n in sub.nodes()],
        y            = [pos[n][1] for n in sub.nodes()]
    ))

    hidden_axis = dict(
        showgrid       = False,
        showticklabels = False,
        zeroline       = False
    )
    fig.update_layout(
        annotations = edge_annotations,
        height      = 550,
        hovermode   = "closest",
        showlegend  = False,
        template    = template,
        title       = f"Career Pathways from Cluster {target_id}",
        xaxis       = hidden_axis,
        yaxis       = hidden_axis
    )
    return fig
