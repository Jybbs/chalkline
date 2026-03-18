"""
Career mapping for Maine's construction industry.
"""

from pydantic import Field
from typing   import Annotated


NonEmptyStr  = Annotated[str,   Field(min_length=1)]
SkillMap     = dict[str, list[str]]
UnitInterval = Annotated[float, Field(gt=0, le=1)]
