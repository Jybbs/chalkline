"""
Pydantic model validation for career pathway schemas.
"""

from pydantic import ValidationError
from pytest   import mark, raises

from chalkline.pathways.schemas import Credential, LaborRecord, Occupation, Skill
from chalkline.pathways.schemas import SkillType


class TestCredential:
    """
    Credential metadata validation and display properties.
    """

    def test_apprenticeship_keys(self):
        """
        Apprenticeship kind requires `min_hours` and `rapids_code`
        in metadata.
        """
        with raises(ValidationError, match="Missing keys"):
            Credential(
                embedding_text = "test",
                kind           = "apprenticeship",
                label          = "Test",
                metadata       = {"min_hours": 8000}
            )

    def test_program_keys(self):
        """
        Program kind requires `credential`, `institution`, and `url`
        in metadata.
        """
        with raises(ValidationError, match="Missing keys"):
            Credential(
                embedding_text = "test",
                kind           = "program",
                label          = "Test",
                metadata       = {"institution": "SMCC"}
            )

    def test_card_detail_both(self):
        """
        Card detail joins hours and institution with a centered dot
        when both are present.
        """
        cred = Credential(
            embedding_text = "test",
            kind           = "apprenticeship",
            label          = "Test",
            metadata       = {
                "institution" : "SMCC",
                "min_hours"   : 8000,
                "rapids_code" : "01"
            }
        )
        assert cred.card_detail == "8,000 hours \u00b7 SMCC"

    def test_card_detail_empty(self):
        """
        Card detail returns empty string when neither hours nor
        institution are present.
        """
        cred = Credential(
            embedding_text = "test",
            kind           = "certification",
            label          = "Test"
        )
        assert cred.card_detail == ""

    @mark.parametrize("kind, metadata, expected", [
        (
            "apprenticeship",
            {"min_hours": 4000, "rapids_code": "01"},
            "4,000 hours"
        ),
        (
            "program",
            {"credential": "AAS", "institution": "SMCC", "url": "https://example.com"},
            "SMCC"
        ),
        (
            "certification",
            {},
            "Certification"
        )
    ])
    def test_detail_label(self, kind: str, metadata: dict, expected: str):
        """
        Falls back from hours to institution to titlecased kind.
        """
        cred = Credential(
            embedding_text = "test",
            kind           = kind,
            label          = "Test",
            metadata       = metadata
        )
        assert cred.detail_label == expected

    def test_extra_forbid(self):
        """
        Unexpected fields are rejected.
        """
        with raises(ValidationError):
            Credential(
                embedding_text = "test",
                kind           = "certification",
                label          = "Test",
                unknown        = "bad"
            )

    @mark.parametrize("kind, expected", [
        ("apprenticeship", "Apprenticeship"),
        ("program",        "AAS")
    ])
    def test_type_label(self, kind: str, expected: str):
        """
        Type label returns `credential` metadata when present,
        falls back to titlecased kind otherwise.
        """
        metadata = (
            {"min_hours": 8000, "rapids_code": "01"}
            if kind == "apprenticeship"
            else {
                "credential"  : "AAS",
                "institution" : "SMCC",
                "url"         : "https://example.com"
            }
        )
        cred = Credential(
            embedding_text = "test",
            kind           = kind,
            label          = "Test",
            metadata       = metadata
        )
        assert cred.type_label == expected

    def test_url_absent(self):
        """
        Non-program credentials return empty string for URL.
        """
        cred = Credential(
            embedding_text = "test",
            kind           = "certification",
            label          = "Test"
        )
        assert cred.url == ""

    def test_url_present(self):
        """
        Program credentials expose their URL.
        """
        cred = Credential(
            embedding_text = "test",
            kind           = "program",
            label          = "Test",
            metadata       = {
                "credential"  : "AAS",
                "institution" : "SMCC",
                "url"         : "https://smcc.edu"
            }
        )
        assert cred.url == "https://smcc.edu"


class TestLaborRecord:
    """
    LaborRecord nested-field flattening and validation.
    """

    def test_flatten_nested(self):
        """
        Nested `outlook`, `projections`, and `wages` objects are
        hoisted to top-level fields.
        """
        record = LaborRecord.model_validate({
            "soc_title" : "Electricians",
            "outlook"   : {"bright_outlook": True},
            "wages"     : {"annual_median": 65000.0}
        })
        assert record.bright_outlook is True
        assert record.annual_median == 65000.0

    def test_extra_ignored(self):
        """
        Unknown fields are silently discarded.
        """
        record = LaborRecord.model_validate({
            "soc_title" : "Electricians",
            "unknown"   : "ignored"
        })
        assert record.soc_title == "Electricians"


class TestOccupation:
    """
    Occupation skill filtering and embedding text construction.
    """

    def test_embedding_no_tasks(self):
        """
        Occupation with no TASK or DWA skills produces title-only
        embedding text.
        """
        occ = Occupation(
            job_zone = 3,
            sector   = "Building Construction",
            skills   = [Skill(name="Mathematics", type=SkillType.KNOWLEDGE)],
            soc_code = "47-2111.00",
            title    = "Electricians"
        )
        assert occ.embedding_text == "Electricians: "

    def test_task_elements(self):
        """
        Only TASK and DWA skills are returned by `task_elements`.
        """
        occ = Occupation(
            job_zone = 3,
            sector   = "Building Construction",
            skills   = [
                Skill(name="Install wiring", type=SkillType.TASK),
                Skill(name="Maintain systems", type=SkillType.DWA),
                Skill(name="Mathematics", type=SkillType.KNOWLEDGE)
            ],
            soc_code = "47-2111.00",
            title    = "Electricians"
        )
        assert [s.name for s in occ.task_elements] == [
            "Install wiring",
            "Maintain systems"
        ]

    def test_embedding_text(self):
        """
        Embedding text concatenates title with task element names.
        """
        occ = Occupation(
            job_zone = 3,
            sector   = "Building Construction",
            skills   = [
                Skill(name="Install wiring", type=SkillType.TASK),
                Skill(name="Blueprint Reading", type=SkillType.SKILL)
            ],
            soc_code = "47-2111.00",
            title    = "Electricians"
        )
        assert occ.embedding_text == "Electricians: Install wiring"
