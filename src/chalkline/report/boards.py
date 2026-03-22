"""
Sector-based job board filtering for the career report.

Derives sector keywords from cluster profile titles rather than
maintaining a hardcoded mapping. Each board's `focus` and
`best_for` fields are checked for word overlap against the
derived vocabulary.
"""

from re import findall

from chalkline.pipeline.schemas import ClusterProfile


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
