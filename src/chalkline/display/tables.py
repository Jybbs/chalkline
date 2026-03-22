"""
Row builders, text generation, and data shaping for display panels.

Consolidates all functions that transform pipeline artifacts into row
dicts, filtered board lists, employer matches, and downloadable text
for the Marimo notebook. Also provides Hamilton DAG introspection for
the pipeline details panel.
"""

from difflib  import SequenceMatcher
from inspect  import getmembers, isfunction, signature
from re       import findall

from chalkline.collection.schemas import Corpus
from chalkline.matching.schemas   import MatchResult
from chalkline.pathways.schemas   import ClusterAssignments, ClusterProfile
from chalkline.pathways.schemas   import Credential, Neighborhood


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


def board_rows(board_list: list[dict]) -> list[dict]:
    """
    Format raw board dicts into display-ready table rows.

    Args:
        board_list: From `filter_boards`, each with `best_for`,
                    `category`, `focus`, and `name` keys.

    Returns:
        Row dicts with `Best For`, `Category`, `Focus`, and
        `Name` keys.
    """
    return [
        {
            "Best For" : b["best_for"],
            "Category" : b["category"],
            "Focus"    : b["focus"],
            "Name"     : b["name"]
        }
        for b in board_list
    ]


def build_report_text(
    profile : ClusterProfile,
    result  : MatchResult
) -> str:
    """
    Build a downloadable plain-text career report.

    Summarizes the matched career family, demonstrated skills,
    and skill gaps with per-task similarity scores.

    Args:
        profile : Matched cluster's profile.
        result  : Full match result with gaps and demonstrated tasks.

    Returns:
        Newline-joined report string.
    """
    return "\n".join([
        "Chalkline Career Report",
        "=" * 50,
        "",
        f"Career Family: {profile.soc_title}",
        f"Sector: {profile.sector}",
        f"Job Zone: {profile.job_zone}",
        f"Match Distance: {result.match_distance:.4f}",
        "",
        f"Demonstrated Skills ({len(result.demonstrated)}):",
        *[f"  + {d.name} ({d.similarity:.3f})" for d in result.demonstrated],
        "",
        f"Skill Gaps ({len(result.gaps)}):",
        *[f"  - {g.name} ({g.similarity:.3f})" for g in result.gaps]
    ])


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


def demonstrated_rows(result: MatchResult) -> list[dict]:
    """
    Format demonstrated competencies as table rows.

    Args:
        result: Match result with demonstrated task list.

    Returns:
        Row dicts with `Similarity` and `Task` keys, strongest
        first.
    """
    return [
        {
            "Similarity" : round(d.similarity, 3),
            "Task"       : d.name
        }
        for d in result.demonstrated
    ]


def filter_boards(
    boards   : dict,
    profiles : dict[int, ClusterProfile],
    sector   : str
) -> tuple[list[dict], list[dict]]:
    """
    Filter Maine and national job boards by sector relevance.

    Derives keywords from cluster profiles belonging to the
    matched sector, then checks each board's `focus` and
    `best_for` fields for word overlap. Boards with at least
    one keyword match are included.

    Args:
        boards   : From `job_boards.json` with `maine` and
                   `national` keys, each containing board dicts.
        profiles : Cluster profiles for keyword derivation.
        sector   : Matched cluster's sector name.

    Returns:
        Tuple of (matching Maine boards, matching national boards).
    """
    keywords = _sector_keywords(profiles, sector)

    def is_relevant(board: dict) -> bool:
        text = f"{board.get('focus', '')} {board.get('best_for', '')}".lower()
        return any(kw in text for kw in keywords)

    return tuple(
        [b for b in boards.get(region, []) if is_relevant(b)]
        for region in ("maine", "national")
    )


def gap_rows(result: MatchResult) -> list[dict]:
    """
    Format skill gaps as table rows.

    Args:
        result: Match result with gap task list.

    Returns:
        Row dicts with `Similarity` and `Task` keys, largest
        deficits first.
    """
    return [
        {
            "Similarity" : round(g.similarity, 3),
            "Task"       : g.name
        }
        for g in result.gaps
    ]


def match_cluster_employers(
    assignments : ClusterAssignments,
    career_urls : list[dict],
    cluster_id  : int,
    corpus      : Corpus,
    members     : list[dict]
) -> list[dict]:
    """
    Build the employer panel rows for a cluster.

    Extracts posting companies from the cluster, fuzzy-matches each
    against the AGC member list, joins career page URLs, and
    deduplicates by member name. Each row carries the canonical
    member name, member type, a representative posting URL, and
    the career page URL when available.

    Args:
        assignments : For cluster membership lookup.
        career_urls : From `career_urls.json`, each with `company`
                      and `url` keys.
        cluster_id  : Which cluster to extract companies from.
        corpus      : For posting lookup by index.
        members     : From `agc_members.json`, each with `name`
                      and `type` keys.

    Returns:
        Deduplicated list of row dicts with `Company`, `Type`,
        `Posting`, and `Career Page` keys.
    """
    if cluster_id not in assignments.members:
        return []

    career_url_map = {
        entry["company"].lower(): entry["url"]
        for entry in career_urls
    }

    postings = [
        corpus.postings[corpus.posting_ids[i]]
        for i in assignments.members[cluster_id]
    ]
    member_names = [m["name"].lower() for m in members]
    posting_urls = {p.company: p.source_url for p in reversed(postings)}

    matched = {}
    for company in sorted({p.company for p in postings}):
        m = match_member(company, members, member_names)
        if m is not None and m["name"] not in matched:
            matched[m["name"]] = {
                "Career Page" : career_url_map.get(m["name"].lower(), ""),
                "Company"     : m["name"],
                "Posting"     : posting_urls.get(company, ""),
                "Type"        : m["type"]
            }
    return list(matched.values())


def match_member(
    company      : str,
    members      : list[dict],
    member_names : list[str] | None = None
) -> dict | None:
    """
    Fuzzy-match a corpus company name against the AGC member list.

    Uses `SequenceMatcher.ratio()` with a 0.7 threshold, which
    tolerates abbreviation differences like "RJ Grondin & Sons"
    vs "R.J. Grondin and Sons" while rejecting unrelated names.

    Args:
        company      : Corpus company name to match.
        members      : AGC member records with `name` keys.
        member_names : Pre-lowercased member names to avoid
                       redundant computation across calls.

    Returns:
        Best-matching member dict, or `None` if no match exceeds
        the threshold.
    """
    if not members:
        return None

    if member_names is None:
        member_names = [m["name"].lower() for m in members]

    company_lower = company.lower()
    best_score, best = max(
        (SequenceMatcher(None, company_lower, name).ratio(), m)
        for m, name in zip(members, member_names)
    )
    return best if best_score >= 0.7 else None


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


def _sector_keywords(
    profiles : dict[int, ClusterProfile],
    sector   : str
) -> set[str]:
    """
    Extract keywords from cluster profile titles for a sector.

    Tokenizes SOC titles and modal posting titles into lowercase
    words of 4+ characters, filtering short tokens that would
    produce spurious matches. The sector name itself is included
    so that boards mentioning "highway" or "construction" match
    their namesake sector.

    Args:
        profiles : Cluster ID to profile for title extraction.
        sector   : Sector name to extract vocabulary for.

    Returns:
        Set of lowercase keyword strings.
    """
    text = " ".join(
        f"{p.soc_title} {p.modal_title}"
        for p in profiles.values()
        if p.sector.lower() == sector.lower()
    )
    return {
        w.lower() for w in findall(r"[A-Za-z]+", f"{sector} {text}")
        if len(w) >= 4
    }


def to_mermaid() -> str:
    """
    Build a Mermaid LR flowchart from the pipeline step functions.

    Each function in `chalkline.pipeline.steps` becomes a node, and
    each parameter that is not an external input becomes a directed
    edge from the parameter's source node to the function node.

    Returns:
        Mermaid diagram string starting with `graph LR`.
    """
    from chalkline.pipeline import steps

    return "\n".join([
        "graph LR",
        *(
            f"    {param} --> {name}"
            for name, fn in sorted(getmembers(steps, isfunction))
            for param in signature(fn).parameters
            if param not in {"config", "model", "lexicons"}
        )
    ])
