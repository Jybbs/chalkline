"""
Row builders, text generation, and data shaping for display panels.

Provides a `TableBuilder` that holds the fitted pipeline, match result, and
reference data, exposing methods that return row dicts for each notebook
panel.
"""

from difflib import SequenceMatcher
from re      import findall
from typing  import TYPE_CHECKING

from chalkline.matching.schemas import MatchResult
from chalkline.pathways.schemas import Reach

if TYPE_CHECKING:
    from chalkline.pipeline.orchestrator import Chalkline


class TableBuilder:
    """
    Stateful row builder for the career report panels.

    Captures the fitted pipeline, match result, and stakeholder reference
    data at construction so that individual table methods receive only
    per-panel arguments like `reach` or `cluster_id`.
    """

    def __init__(self, pipeline: Chalkline, reference: dict, result: MatchResult):
        """
        Args:
            pipeline  : Fitted `Chalkline` instance.
            reference : Stakeholder reference data with `agc_members`, `career_urls`,
                        and `job_boards` keys.
            result    : Match result from resume projection.
        """
        self.pipeline  = pipeline
        self.profile   = pipeline.clusters[result.cluster_id]
        self.reference = reference
        self.result    = result

    @staticmethod
    def _match_member(
        company      : str,
        member_names : list[str],
        members      : list[dict]
    ) -> dict | None:
        """
        Fuzzy-match a corpus company name against the AGC member list.

        Uses `SequenceMatcher.ratio()` with a 0.7 threshold, tolerating
        abbreviation differences like "RJ Grondin & Sons" vs "R.J. Grondin
        and Sons" while rejecting unrelated names.

        Args:
            company      : Corpus company name to match.
            member_names : Pre-lowercased member names.
            members      : AGC member records with `name` keys.

        Returns:
            Best-matching member dict, or `None` if no match exceeds the threshold.
        """
        scored = [
            (SequenceMatcher(None, company, name).ratio(), m)
            for m, name in zip(members, member_names)
        ]
        best_score, best = max(scored, key=lambda pair: pair[0])
        return best if best_score >= 0.7 else None

    def _sector_keywords(self, sector: str) -> set[str]:
        """
        Extract keywords from cluster profile titles for a sector.

        Tokenizes SOC titles and modal posting titles into lowercase words
        of 4+ characters, filtering short tokens that would produce spurious
        matches.

        Args:
            sector: Sector name to extract vocabulary for.

        Returns:
            Set of lowercase keyword strings.
        """
        text = " ".join(
            f"{p.soc_title} {p.modal_title}"
            for p in self.pipeline.clusters.values()
            if p.sector.lower() == sector.lower()
        )
        return {
            w.lower() for w in findall(r"[A-Za-z]+", f"{sector} {text}")
            if len(w) >= 4
        }

    def apprenticeship_rows(self, reach: Reach) -> list[dict]:
        """
        Deduplicated apprenticeship rows from reach edges.

        Collects apprenticeships from both advancement and lateral edges,
        deduplicates by RAPIDS code, and sorts by trade title.

        Args:
            reach: Advancement and lateral edges to extract from.

        Returns:
            Sorted list of row dicts with `Trade`, `RAPIDS Code`, and `Min Hours` keys.
        """
        unique = {
            c.metadata["rapids_code"]: c
            for edge in reach.all_edges
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

    def board_rows(self) -> tuple[list[dict], list[dict]]:
        """
        Filter and format job boards by the matched sector.

        Derives keywords from cluster profiles belonging to the matched
        sector, then checks each board's `focus` and `best_for` fields for
        word overlap. Returns formatted row dicts ready for `mo.ui.table`.

        Returns:
            Tuple of (Maine board rows, national board rows).
        """
        keywords = self._sector_keywords(self.profile.sector)
        relevant = lambda b: any(
            kw in f"{b.get('focus', '')} {b.get('best_for', '')}".lower()
            for kw in keywords
        )
        filtered = lambda region: [
            {
                "Best For" : b["best_for"],
                "Category" : b["category"],
                "Focus"    : b["focus"],
                "Name"     : b["name"]
            }
            for b in self.reference["job_boards"].get(region, [])
            if relevant(b)
        ]
        return filtered("maine"), filtered("national")

    def credential_rows(self, reach: Reach) -> list[dict]:
        """
        Flatten all credentials on reach edges into table rows.

        Args:
            reach: Advancement and lateral edges to extract from.

        Returns:
            List of row dicts with `Credential`, `Direction`, `Hours`, `Target`, and
            `Type` keys.
        """
        return [
            {
                "Credential" : c.label,
                "Direction"  : direction,
                "Hours"      : f"{h:,}" if (h := c.metadata.get("min_hours")) else "",
                "Target"     : self.pipeline.clusters[e.cluster_id].soc_title,
                "Type"       : c.metadata.get("credential", c.kind.title())
            }
            for direction, edges in [
                ("Advancement", reach.advancement),
                ("Lateral",     reach.lateral)
            ]
            for e in edges
            for c in e.credentials
        ]

    def demonstrated_rows(self) -> list[dict]:
        """
        Format demonstrated competencies as table rows.

        Returns:
            Row dicts with `Similarity` and `Task` keys, strongest first.
        """
        return [
            {
                "Similarity" : round(d.similarity, 3),
                "Task"       : d.name
            }
            for d in self.result.demonstrated
        ]

    def employer_rows(self, cluster_id: int) -> list[dict]:
        """
        Build the employer panel rows for a cluster.

        Extracts posting companies from the cluster, fuzzy-matches each
        against the AGC member list, joins career page URLs, and
        deduplicates by member name.

        Args:
            cluster_id: Which cluster to extract companies from.

        Returns:
            Deduplicated list of row dicts with `Company`, `Type`, `Posting`, and
            `Career Page` keys.
        """
        members = self.reference["agc_members"]
        names   = [m["name"].lower() for m in members]

        career_urls = {
            e["company"].lower(): e["url"] 
            for e in self.reference["career_urls"]
        }
        by_company = {
            p.company: p.source_url
            for p in self.pipeline.clusters[cluster_id].postings
        }

        return list({
            m["name"]: {
                "Career Page" : career_urls.get(m["name"].lower(), ""),
                "Company"     : m["name"],
                "Posting"     : by_company[company],
                "Type"        : m["type"]
            }
            for company in sorted(by_company)
            if (m := self._match_member(company.lower(), names, members))
        }.values())

    def gap_rows(self) -> list[dict]:
        """
        Format skill gaps as table rows.

        Returns:
            Row dicts with `Similarity` and `Task` keys, largest deficits first.
        """
        return [
            {
                "Similarity" : round(g.similarity, 3),
                "Task"       : g.name
            }
            for g in self.result.gaps
        ]

    def program_rows(self, reach: Reach) -> list[dict]:
        """
        Deduplicated program rows from reach edges.

        Collects programs from both advancement and lateral edges,
        deduplicates by institution and program name, and sorts
        alphabetically.

        Args:
            reach: Advancement and lateral edges to extract from.

        Returns:
            Sorted list of row dicts with `Credential`, `Institution`, `Program`, and
            `Link` keys.
        """
        unique = {
            (c.metadata["institution"], c.label): c
            for edge in reach.all_edges
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

    def report_text(self) -> str:
        """
        Build a downloadable plain-text career report.

        Summarizes the matched career family, demonstrated skills, and skill
        gaps with per-task similarity scores.

        Returns:
            Newline-joined report string.
        """
        return "\n".join([
            "Chalkline Career Report",
            "=" * 50,
            "",
            f"Career Family: {self.profile.soc_title}",
            f"Sector: {self.profile.sector}",
            f"Job Zone: {self.profile.job_zone}",
            f"Match Distance: {self.result.match_distance:.4f}",
            "",
            f"Demonstrated Skills ({len(self.result.demonstrated)}):",
            *[f"  + {d.name} ({d.similarity:.3f})" for d in self.result.demonstrated],
            "",
            f"Skill Gaps ({len(self.result.gaps)}):",
            *[f"  - {g.name} ({g.similarity:.3f})" for g in self.result.gaps]
        ])
