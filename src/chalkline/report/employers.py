"""
Fuzzy matching of corpus companies against AGC member companies.

Matches company names from job postings in a given cluster against
the AGC Maine member directory using SequenceMatcher similarity
scoring. Joins career page URLs from a separate reference file and
deduplicates by member name.
"""

from difflib import SequenceMatcher

from chalkline.pipeline.schemas import ClusterAssignments, Corpus


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
