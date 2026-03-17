"""
Occupation-level queries and SOC code matching.

Indexes O*NET occupations by SOC code, resolves bare prefix lookups, and
builds a binary skill matrix for vectorized Jaccard matching via
`scipy.spatial.distance.cdist`.
"""

import numpy as np

from collections            import Counter
from functools              import cached_property
from scipy.spatial.distance import cdist
from statistics             import median

from chalkline.extraction.schemas import OnetOccupation


class OccupationIndex:
    """
    Occupation lookup and SOC matching interface.

    Receives already-loaded O*NET occupations, stores a SOC-keyed
    map, and lazily derives a bare-prefix resolver, skill-to-column
    index, and binary SOC-skill matrix for vectorized Jaccard
    distance computation.
    """

    def __init__(self, occupations: list[OnetOccupation]):
        """
        Store occupation records keyed by SOC code.

        Args:
            occupations: Validated O*NET occupation records.
        """
        self.occupation_map: dict[str, OnetOccupation] = {
            o.soc_code: o for o in occupations
        }

    @cached_property
    def bare_to_full(self) -> dict[str, str]:
        """
        Map bare SOC prefixes to full codes where unambiguous.

        Codes like `"47-2111"` resolve to `"47-2111.00"` when no
        other suffix exists. Ambiguous prefixes where multiple
        suffixes exist (such as `"17-2051"`) are excluded so that
        callers must provide the full code.

        Returns:
            Mapping from bare prefix to full SOC code.
        """
        counts = Counter(c.split(".")[0] for c in self.occupation_map)
        return {
            prefix: code for code in self.occupation_map
            if counts[prefix := code.split(".")[0]] == 1
        }

    @cached_property
    def concrete_profiles(self) -> list[tuple[set[str], int]]:
        """
        Pre-computed concrete skill sets and Job Zones for overlap
        matching.

        Filters each occupation's skills to concrete types (Tasks,
        Technology Skills, Tools, DWAs) and lowercases the names
        for case-insensitive overlap computation against cluster
        skill sets.

        Returns:
            Tuples of (lowercased concrete skill names, Job Zone).
        """
        return [
            (
                {s.name.lower() for s in occ.skills if s.type.is_concrete},
                occ.job_zone
            )
            for occ in self.occupation_map.values()
        ]

    @cached_property
    def skill_to_col(self) -> dict[str, int]:
        """
        Mapping from skill name to column index in the binary
        matrix.

        Enumerates all unique skill names across occupations,
        sorted alphabetically, to produce stable column indices.

        Returns:
            Mapping from skill name to zero-based column index.
        """
        return {
            name: i for i, name in enumerate(sorted({
                s.name
                for o in self.occupation_map.values()
                for s in o.skills
            }))
        }

    @cached_property
    def soc_skill_matrix(self) -> np.ndarray:
        """
        Binary SOC-skill matrix for vectorized Jaccard matching.

        Uses all skill types (including KSAs) because the spec
        keeps abstract labels available for occupation-level
        matching. The matrix has shape `(21, n_unique_skills)`.

        Returns:
            Binary `numpy` array with shape `(n_socs, n_skills)`.
        """
        matrix = np.zeros((len(self.socs), len(self.skill_to_col)), dtype=np.uint8)
        for row, code in enumerate(self.socs):
            matrix[row, [
                self.skill_to_col[s.name]
                for s in self.occupation_map[code].skills
            ]] = 1
        return matrix

    @cached_property
    def socs(self) -> list[str]:
        """
        Sorted SOC codes from the occupation map.

        Returns:
            Alphabetically sorted list of SOC code strings.
        """
        return sorted(self.occupation_map)

    def get(self, soc: str) -> OnetOccupation:
        """
        Resolve a SOC code and return its occupation record.

        Accepts both bare (`"47-2111"`) and suffixed
        (`"47-2111.00"`) formats. Raises `KeyError` for
        unrecognized codes or ambiguous bare prefixes where
        multiple suffixes exist.

        Args:
            soc: SOC code in either format.

        Returns:
            The matching `OnetOccupation` record.
        """
        if soc in self.occupation_map:
            return self.occupation_map[soc]

        if full := self.bare_to_full.get(soc):
            return self.occupation_map[full]

        raise KeyError(f"Unknown or ambiguous SOC code: {soc!r}")

    def job_zone_for_skills(self, skills: set[str]) -> int:
        """
        Assign a Job Zone via overlap coefficient against concrete
        O*NET skill profiles.

        Computes |A & B| / min(|A|, |B|) between the input skill
        set and each occupation's concrete skills, then returns the
        integer median of the top-3 matches. Returns 2 (entry-level
        default) when the input set is empty or no profiles overlap.

        Args:
            skills: Canonical skill names from a cluster.

        Returns:
            Job Zone integer in [1, 5].
        """
        skills = {s.lower() for s in skills}
        if not skills:
            return 2

        matches = sorted(
            [
                (
                    len(skills & profile) / min(len(skills), len(profile)),
                    zone
                )
                for profile, zone in self.concrete_profiles
                if profile
            ],
            reverse=True
        )

        return int(median(
            jz for _, jz in matches[:3]
        )) if matches else 2

    def nearest(self, posting_skills: set[str]) -> str:
        """
        Return the SOC code with maximum Jaccard overlap.

            J(A, B) = |A ∩ B| / |A ∪ B|

        Converts the input skill set into a binary row vector and
        computes Jaccard distance against all SOC codes via
        `cdist`. Because `cdist` returns distance (1 - J) rather
        than similarity, `argmin` gives the most similar
        occupation.

        Args:
            posting_skills: Normalized skill names from a posting.

        Returns:
            The SOC code of the nearest occupation.
        """
        vector = np.zeros((1, len(self.skill_to_col)), dtype=np.uint8)
        vector[0, [
            self.skill_to_col[s]
            for s in posting_skills if s in self.skill_to_col
        ]] = 1

        return self.socs[np.argmin(cdist(vector, self.soc_skill_matrix, "jaccard"))]
