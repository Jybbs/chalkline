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

from chalkline.extraction.schemas import OnetOccupation


class OccupationIndex:
    """
    Occupation lookup and SOC matching interface.

    Receives already-loaded O*NET occupations, stores a SOC-keyed map, and
    lazily derives a bare-prefix resolver, skill-to-column index, and binary
    SOC-skill matrix for vectorized Jaccard distance computation.
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

        Codes like `"47-2111"` resolve to `"47-2111.00"` when no other
        suffix exists. Ambiguous prefixes (where multiple suffixes exist,
        such as `"17-2051"`) are excluded so that callers must provide the
        full code.
        """
        counts = Counter(c.split(".")[0] for c in self.occupation_map)
        return {
            prefix: code for code in self.occupation_map
            if counts[prefix := code.split(".")[0]] == 1
        }

    @cached_property
    def skill_to_col(self) -> dict[str, int]:
        """
        Mapping from skill name to column index in the binary matrix.

        Enumerates all unique skill names across occupations, sorted
        alphabetically, to produce stable column indices.
        """
        return {
            name: i for i, name in enumerate(sorted({
                s.name
                for o in self.occupation_map.values()
                for s in o.skills
            }))
        }

    @cached_property
    def socs(self) -> list[str]:
        """
        Sorted SOC codes from the occupation map.
        """
        return sorted(self.occupation_map)

    @cached_property
    def soc_skill_matrix(self) -> np.ndarray:
        """
        Binary SOC-skill matrix for vectorized Jaccard matching.

        Uses all skill types (including KSAs) because the spec keeps
        abstract labels available for occupation-level matching. The matrix
        has shape `(21, n_unique_skills)`.
        """
        matrix = np.zeros(
            (len(self.socs), len(self.skill_to_col)),
            dtype=np.uint8
        )
        for row, code in enumerate(self.socs):
            matrix[row, [
                self.skill_to_col[s.name]
                for s in self.occupation_map[code].skills
            ]] = 1
        return matrix


    def get(self, soc: str) -> OnetOccupation:
        """
        Resolve a SOC code and return its occupation record.

        Accepts both bare (`"47-2111"`) and suffixed (`"47-2111.00"`)
        formats. Raises `KeyError` for unrecognized codes or ambiguous
        bare prefixes where multiple suffixes exist.

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

    def nearest(self, posting_skills: set[str]) -> str:
        """
        Return the SOC code with maximum Jaccard overlap.

        Converts the input skill set into a binary row vector and computes
        Jaccard distance against all SOC codes via `cdist`. Because `cdist`
        returns distance rather than similarity, `argmin` gives the most
        similar occupation.

        Args:
            posting_skills: Normalized skill names from a posting.

        Returns:
            The SOC code string of the best-matching occupation.
        """
        vector = np.zeros((1, len(self.skill_to_col)), dtype=np.uint8)
        vector[0, [
            self.skill_to_col[s]
            for s in posting_skills if s in self.skill_to_col
        ]] = 1

        distances = cdist(vector, self.soc_skill_matrix, "jaccard")
        return self.socs[np.argmin(distances)]
