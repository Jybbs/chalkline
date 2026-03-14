"""
Shared educational program loader with field normalization.

Loads community college degree programs, statewide workforce
initiatives, and university programs from stakeholder reference data,
normalizing all three into a consistent `ProgramRecommendation`
schema. Consumed by resume matching, career pathway construction,
and report generation.
"""

from json    import loads
from pathlib import Path

from chalkline.pipeline.schemas import ProgramRecommendation


def load_programs(reference_dir: Path) -> list[ProgramRecommendation]:
    """
    Load and normalize educational programs from stakeholder
    reference data.

    Reads three source types from `cc_programs.json` and
    `umaine_programs.json`, normalizing each into the unified
    `ProgramRecommendation` schema:

    - CC degree programs (`college` to `institution`, `credential`
      kept as-is)
    - CC workforce initiatives (`initiative` to `program`,
      `best_for` to `credential`, institution set to `"Statewide"`)
    - UMaine system programs (`campus` to `institution`, `degree`
      to `credential`)

    Initiatives without a URL are included with an empty string
    because they still carry actionable program descriptions in
    the `best_for` field.

    Args:
        reference_dir: Path to `data/stakeholder/reference/`.
    """
    programs: list[ProgramRecommendation] = []

    if (cc_path := reference_dir / "cc_programs.json").exists():
        cc_data = loads(cc_path.read_text())

        for entry in cc_data.get("degrees", []):
            programs.append(ProgramRecommendation(
                credential  = entry["credential"],
                institution = entry["college"],
                program     = entry["program"],
                url         = entry["url"]
            ))

        for entry in cc_data.get("initiatives", []):
            programs.append(ProgramRecommendation(
                credential  = entry["best_for"],
                institution = "Statewide",
                program     = entry["initiative"],
                url         = entry.get("url", "")
            ))

    if (umaine_path := reference_dir / "umaine_programs.json").exists():
        for entry in loads(umaine_path.read_text()):
            programs.append(ProgramRecommendation(
                credential  = entry["degree"],
                institution = entry["campus"],
                program     = entry["program"],
                url         = entry["url"]
            ))

    return programs
