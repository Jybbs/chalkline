"""
Site-specific scraper implementations for career page extraction.

Importing this package registers all scraper subclasses with
`BaseScraper.__subclasses__()` for automatic dispatch.
"""

from chalkline.collection.scrapers.heuristic import HeuristicScraper
from chalkline.collection.scrapers.workable  import WorkableScraper
from chalkline.collection.scrapers.workday   import WorkdayScraper
