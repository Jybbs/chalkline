"""
Tests for display-layer schemas, lazy-loading containers, and
fuzzy matching utilities.
"""

from chalkline.display.schemas  import _match_member
from chalkline.pathways.loaders import StakeholderReference


class TestStakeholderReference:
    def test_loads_json_on_access(self, tmp_path):
        """
        Attribute access deserializes the corresponding JSON file
        and caches the result.
        """
        (tmp_path / "trades.json").write_text('["electrician"]')
        ref = StakeholderReference(reference_dir=tmp_path)
        assert ref.trades == ["electrician"]
        assert ref.trades is ref.trades

    def test_missing_file_empty(self, tmp_path):
        """
        Accessing a name with no backing JSON file returns an
        empty list rather than raising.
        """
        ref = StakeholderReference(reference_dir=tmp_path)
        assert ref.nonexistent == []


class TestMatchMember:
    def test_above_threshold(self):
        """
        A company name sufficiently similar to a member name
        returns the matched member dict.
        """
        members = [{"name": "Cianbro", "type": "GC"}]
        result = _match_member("cianbro", ["cianbro"], members)
        assert result == members[0]

    def test_below_threshold(self):
        """
        A company name dissimilar to all member names returns
        None.
        """
        members = [{"name": "Cianbro", "type": "GC"}]
        assert _match_member("xyz corp", ["cianbro"], members) is None
