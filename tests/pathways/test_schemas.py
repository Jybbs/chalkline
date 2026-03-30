"""
Tests for pathways schemas.

Validates credential deduplication key dispatch across the three
credential kinds.
"""

from chalkline.pathways.schemas import Credential


class TestCredential:
    """
    Validate `Credential.key` match dispatch per kind.
    """

    def test_key_apprenticeship(self):
        """
        Apprenticeship key is the RAPIDS code.
        """
        assert Credential(
            embedding_text = "x",
            kind           = "apprenticeship",
            label          = "IBEW",
            metadata       = {"min_hours": 8000, "rapids_code": "0123"}
        ).key == "0123"

    def test_key_fallback(self):
        """
        Unknown kinds fall back to the label.
        """
        assert Credential(
            embedding_text = "x",
            kind           = "certification",
            label          = "OSHA 30"
        ).key == "OSHA 30"

    def test_key_program(self):
        """
        Program key is (institution, label) for deduplication.
        """
        assert Credential(
            embedding_text = "x",
            kind           = "program",
            label          = "Electrical Tech",
            metadata       = {
                "credential"  : "AAS",
                "institution" : "SMCC",
                "url"         : "https://example.com"
            }
        ).key == ("SMCC", "Electrical Tech")
