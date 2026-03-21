"""
Credential extraction from career neighborhood edges.

Flattens apprenticeships, certifications, and programs from
neighborhood edges into table row dicts for the career report.
Both functions centralize the advancement/lateral traversal
that otherwise repeats across notebook cells.
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
        a.rapids_code: a
        for edge in neighborhood.all_edges
        for a in edge.apprenticeships
    }
    return [
        {
            "Min Hours"   : f"{a.min_hours:,}",
            "RAPIDS Code" : a.rapids_code,
            "Trade"       : a.title
        }
        for a in sorted(unique.values(), key=lambda x: x.title)
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
        (p.institution, p.program): p
        for edge in neighborhood.all_edges
        for p in edge.programs
    }
    return [
        {
            "Credential"  : p.credential,
            "Institution" : p.institution,
            "Link"        : p.url,
            "Program"     : p.program
        }
        for p in sorted(unique.values(), key=lambda x: x.program)
    ]


def credential_rows(neighborhood: Neighborhood) -> list[dict]:
    """
    Flatten all credentials on neighborhood edges into table rows.

    Iterates advancement and lateral edges, collecting
    apprenticeships, certifications, and programs into a flat
    list for the career pathways credential table.

    Args:
        neighborhood: Advancement and lateral edges to extract from.

    Returns:
        List of row dicts with `Credential`, `Direction`, `Hours`,
        `Target`, and `Type` keys.
    """
    rows = []
    for direction, edges in [
        ("Advancement", neighborhood.advancement),
        ("Lateral", neighborhood.lateral)
    ]:
        for e in edges:
            for a in e.apprenticeships:
                rows.append({
                    "Credential" : a.title,
                    "Direction"  : direction,
                    "Hours"      : f"{a.min_hours:,}",
                    "Target"     : e.profile.soc_title,
                    "Type"       : "Apprenticeship"
                })
            for c in e.certifications:
                rows.append({
                    "Credential" : c.display_label,
                    "Direction"  : direction,
                    "Hours"      : "",
                    "Target"     : e.profile.soc_title,
                    "Type"       : "Certification"
                })
            for p in e.programs:
                rows.append({
                    "Credential" : p.program,
                    "Direction"  : direction,
                    "Hours"      : "",
                    "Target"     : e.profile.soc_title,
                    "Type"       : p.credential
                })
    return rows
