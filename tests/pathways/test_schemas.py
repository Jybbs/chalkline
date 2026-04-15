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

    @mark.parametrize(("kind", "metadata"), [
        ("apprenticeship", {"min_hours": 8000}),
        ("program",        {"institution": "SMCC"})
    ])
    def test_incomplete_metadata_rejected(self, kind: str, metadata: dict):
        """
        Kinds with required metadata keys reject incomplete dicts.
        """
        with raises(ValidationError, match="Missing keys"):
            Credential(
                embedding_text = "test",
                kind           = kind,
                label          = "Test",
                metadata       = metadata
            )

    @mark.parametrize(("kind", "metadata", "expected"), [
        (
            "apprenticeship",
            {"institution": "SMCC", "min_hours": 8000, "rapids_code": "01"},
            "8,000 hours \u00b7 SMCC"
        ),
        ("certification", {}, "")
    ], ids=["both_fields", "empty"])
    def test_card_detail(self, kind: str, metadata: dict, expected: str):
        """
        Card detail joins hours and institution with a centered dot,
        or returns empty when neither is present.
        """
        cred = Credential(
            embedding_text = "test",
            kind           = kind,
            label          = "Test",
            metadata       = metadata
        )
        assert cred.card_detail == expected

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

    @mark.parametrize(("kind", "metadata", "expected"), [
        ("certification", {},                                                    ""),
        ("program",       {"credential": "AAS", "institution": "SMCC",
                           "url": "https://smcc.edu"},                           "https://smcc.edu")
    ], ids=["absent", "present"])
    def test_url(self, kind: str, metadata: dict, expected: str):
        """
        Programs expose their URL; other kinds return empty string.
        """
        cred = Credential(
            embedding_text = "test",
            kind           = kind,
            label          = "Test",
            metadata       = metadata
        )
        assert cred.url == expected


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



class TestOccupation:
    """
    Occupation skill filtering for task-element selection.
    """

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
            title    = "Electricians"
        )
        assert [s.name for s in occ.task_elements] == [
            "Install wiring",
            "Maintain systems"
        ]
