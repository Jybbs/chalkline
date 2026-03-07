"""
Curate the OSHA safety topic lexicon from 29 CFR Parts 1926 and 1910.

Fetches the eCFR Structure API hierarchy for construction safety standards
(Part 1926) and cross-applicable general industry standards (Part 1910),
then extracts section `label_description` values as skill terms with minimal
normalization. Generic section titles like "scope" and "definitions" are
filtered via compiled regex.

Writes a sorted JSON array to `data/lexicons/osha.json` for consumption
by the lexicon loader.
"""

from html    import unescape
from json    import dumps, loads
from pathlib import Path
from re      import compile
from urllib  import request


SKIP = compile(
    r"^(applicability|definitions|equipment|general"
    r"|general provisions|general requirements"
    r"|general requirements for storage|introduction"
    r"|operation|scope|severability|training)$"
    r"|^scope,? |definitions applicable"
)


def find(node: dict, part_id: str) -> dict | None:
    """
    Recursively locate a part node by `identifier`.

    The eCFR nests the requested part several levels deep under title,
    subtitle, and chapter nodes, so a linear scan of `children` would
    miss it.

    Args:
        node    : Current node in the eCFR tree.
        part_id : Part identifier value, namely "1926" or "1910".

    Returns:
        The matching node dict, or `None` if not found.
    """
    if node.get("type") == "part" and node.get("identifier") == part_id:
        return node

    for child in node.get("children", []):
        if found := find(child, part_id):
            return found


def walk(node: dict, allowed_subparts: set[str] | None = None):
    """
    Yield normalized skill terms from non-reserved sections.

    Skips administrative subparts A and B, reserved sections, and
    appendices. Section `label_description` values are lowercased and
    stripped of trailing periods, then filtered against `SKIP` to remove
    generic headings.

    Args:
        node             : Current node in the eCFR tree.
        allowed_subparts : Restricts extraction to these subpart identifiers,
            used for selective pulls from Part 1910.

    Yields:
        Lowercased section titles suitable as safety skill terms.
    """
    if node.get("reserved"):
        return

    match node.get("type"):
        case "appendix":
            return
        case "subpart" if (
            (ident := node.get("identifier")) in {"A", "B"}
            or (allowed_subparts and ident not in allowed_subparts)
        ):
            return
        case "section":
            if (
                (raw := unescape(node.get("label_description", ""))
                 .strip().rstrip(".").lower())
                and not SKIP.search(raw)
            ):
                yield raw
            return

    for child in node.get("children", []):
        yield from walk(child, allowed_subparts)


def main():
    """
    Fetch both CFR parts and write `data/lexicons/osha.json`.

    Collects all section titles from Part 1926 (Construction) and the
    cross-applicable subparts E, I, J, and Z of Part 1910 (General
    Industry), which commonly apply to construction via cross-reference
    or direct applicability. Deduplicates the combined results and writes
    a sorted JSON array of unique skill term strings.
    """
    output = (
        Path(__file__).resolve().parent.parent
        / "data" / "lexicons" / "osha.json"
    )

    skills = set()
    for number, subparts in (("1926", None), ("1910", {"E", "I", "J", "Z"})):
        print(f"Fetching eCFR structure for 29 CFR Part {number}...")

        with request.urlopen(
            f"https://www.ecfr.gov"
            f"/api/versioner/v1/structure/current/title-29.json"
            f"?part={number}"
        ) as resp:
            if not (part := find(loads(resp.read()), number)):
                raise ValueError(f"Part {number} not found in eCFR response")

        skills.update(walk(part, subparts))

    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(dumps(sorted(skills), indent=2) + "\n")
    print(f"Wrote {len(skills)} terms to {output}")


if __name__ == "__main__":

    main()
