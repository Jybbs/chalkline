"""
Credential extraction from career neighborhood edges.

Flattens credentials from neighborhood edges into table row dicts for
the career report, with deduplication for the education panel tables.
"""

from chalkline.matching.schemas import Neighborhood


def apprenticeship_rows(neighborhood: Neighborhood) -> list[dict]:
    """
    Deduplicated apprenticeship rows from neighborhood edges.

    Collects apprenticeships from both advancement and lateral
    edges, deduplicates by RAPIDS code, and sorts by trade
    title for the education panel table.

    Args:
        neighborhood: Advancement and lateral edges to extract from.

    Returns:
        Sorted list of row dicts with `Trade`, `RAPIDS Code`,
        and `Min Hours` keys.
    """
    unique = {
        c.metadata["rapids_code"]: c
        for edge in neighborhood.all_edges
        for c in edge.credentials
        if c.kind == "apprenticeship"
    }
    return [
        {
            "Min Hours"   : f"{c.metadata['min_hours']:,}",
            "RAPIDS Code" : c.metadata["rapids_code"],
            "Trade"       : c.label
        }
        for c in sorted(unique.values(), key=lambda x: x.label)
    ]


def program_rows(neighborhood: Neighborhood) -> list[dict]:
    """
    Deduplicated program rows from neighborhood edges.

    Collects programs from both advancement and lateral edges,
    deduplicates by institution and program name, and sorts
    alphabetically for the education panel table.

    Args:
        neighborhood: Advancement and lateral edges to extract from.

    Returns:
        Sorted list of row dicts with `Credential`,
        `Institution`, `Program`, and `Link` keys.
    """
    unique = {
        (c.metadata["institution"], c.label): c
        for edge in neighborhood.all_edges
        for c in edge.credentials
        if c.kind == "program"
    }
    return [
        {
            "Credential"  : c.metadata["credential"],
            "Institution" : c.metadata["institution"],
            "Link"        : c.metadata["url"],
            "Program"     : c.label
        }
        for c in sorted(unique.values(), key=lambda x: x.label)
    ]


def credential_rows(neighborhood: Neighborhood) -> list[dict]:
    """
    Flatten all credentials on neighborhood edges into table rows.

    Args:
        neighborhood: Advancement and lateral edges to extract from.

    Returns:
        List of row dicts with `Credential`, `Direction`, `Hours`,
        `Target`, and `Type` keys.
    """
    return [
        {
            "Credential" : c.label,
            "Direction"  : direction,
            "Hours"      : (
                f"{c.metadata['min_hours']:,}"
                if "min_hours" in c.metadata
                else ""
            ),
            "Target"     : e.profile.soc_title,
            "Type"       : c.metadata.get("credential", c.kind.title())
        }
        for direction, edges in [
            ("Advancement", neighborhood.advancement),
            ("Lateral", neighborhood.lateral)
        ]
        for e in edges
        for c in e.credentials
    ]
