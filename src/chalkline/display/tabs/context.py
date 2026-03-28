"""
Shared dependency container and utilities for tab rendering.
"""

from pathlib  import Path
from pydantic import BaseModel, field_validator
from tomllib  import load
from typing   import Literal, NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from chalkline.display.charts  import Charts
    from chalkline.display.schemas import DisplayData
    from chalkline.display.theme   import Theme


class HeroContent(BaseModel, extra="forbid"):
    """
    Callout text and semantic kind for a tab hero banner.
    """

    text: str

    kind: Literal["info", "success", "warn"] = "info"

    def render(self, **kwargs) -> tuple[str, str]:
        """
        Format text and return `(content, kind)` for `callout()`.
        """
        return self.text.format(**kwargs), self.kind


class SectionContent(BaseModel, extra="forbid"):
    """
    Title and description for one chart section.
    """

    title: str

    description: str = ""


class TabContent(BaseModel, extra="ignore"):
    """
    Validated content loaded from a tab's `content.toml`.

    Sections are keyed by name and accessed via
    `content.sections["key"]`. Hero is optional because not
    every tab has one.
    """

    empty_message : str                       = ""
    hero          : HeroContent               = HeroContent(text="")
    info          : str                       = ""
    sections      : dict[str, SectionContent] = {}
    tagline       : str                       = ""
    title         : str                       = ""

    @field_validator("info", mode="before")
    @classmethod
    def _unwrap_info(cls, v) -> str | None:
        return v["text"] if isinstance(v, dict) else v

    def section(self, key: str, **kwargs) -> tuple[str, str]:
        """
        Format a section's title and description for `header()`.
        """
        s = self.sections[key]
        return s.title.format(**kwargs), s.description.format(**kwargs)


class TabContext(NamedTuple):
    """
    Shared dependencies available to every tab function.

    Constructed once in main.py after upload and passed to all
    tab functions via lambda wrappers. Tab-specific dependencies
    like `target_data` are passed as separate arguments.
    """

    charts : "Charts"
    data   : "DisplayData"
    theme  : "Theme"


def load_content(caller_file: str) -> TabContent:
    """
    Load and validate `content.toml` from the caller's directory.

    Args:
        caller_file: `__file__` of the calling module.
    """
    with (Path(caller_file).parent / "content.toml").open("rb") as f:
        return TabContent.model_validate(load(f))
