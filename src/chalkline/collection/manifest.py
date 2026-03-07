"""
Crawl manifest generation from stakeholder career URLs.

Reads `career_urls.json`, classifies each URL by scraping approach,
strips Google Ads tracking parameters, and writes the manifest to
`data/postings/manifest.json`. Nine URLs are marked inactive because
they point to application-only forms or PDF downloads rather than
extractable job listing pages.
"""

from json         import dumps, loads
from logging      import getLogger
from pathlib      import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from chalkline.collection.models import ManifestEntry, ScrapeCategory

logger = getLogger(__name__)


GOOGLE_ADS_PARAMS = {
    "gad_campaignid",
    "gad_source",
    "gbraid",
    "gclid",
    "srsltid"
}


_APPLICATION_ONLY_URLS = {
    "cemmaine.com/apply-now",
    "gcmaine.com/contact-us/employment/",
    "gendroncorp.com/application/",
    "prattandsons.net/apply-online/",
    "soderbergconstruction.com/employment-application/",
    "vaughndthibodeau.com/employment-application-2/"
}


_HOSTNAME_CATEGORIES = {
    "engagedtas.com"    : ScrapeCategory.ENGAGEDTAS,
    "myworkdayjobs.com" : ScrapeCategory.WORKDAY,
    "workable.com"      : ScrapeCategory.WORKABLE
}


def _classify(url: str) -> ScrapeCategory:
    """
    Determine the scraping approach for a URL based on its structure.

    Checks against known application-only pages, PDF extensions,
    and ATS domain patterns before falling back to static HTML.

    Args:
        url: The career page URL to classify.

    Returns:
        The `ScrapeCategory` that determines which scraper handles
        this URL.
    """
    hostname = (parsed := urlparse(url)).hostname or ""

    if f"{hostname.removeprefix('www.')}{parsed.path}" in _APPLICATION_ONLY_URLS:
        return ScrapeCategory.APPLICATION_ONLY

    if url.lower().endswith(".pdf"):
        return ScrapeCategory.PDF_ONLY

    return next(
        (cat for domain, cat in _HOSTNAME_CATEGORIES.items()
         if domain in hostname),
        ScrapeCategory.STATIC_HTML
    )


def _strip_tracking_params(url: str) -> str:
    """
    Remove Google Ads tracking parameters from a URL.

    Career URLs in the stakeholder workbook often carry `gclid`,
    `gad_source`, and similar parameters from ad campaigns that
    would cause duplicate manifest entries for the same page.

    Args:
        url: The URL to clean.

    Returns:
        The URL with all Google Ads parameters removed.
    """
    return urlunparse((parsed := urlparse(url))._replace(
        query=urlencode(
            {k: v for k, v in
             parse_qs(parsed.query, keep_blank_values=True).items()
             if k not in GOOGLE_ADS_PARAMS},
            doseq=True
        )
    ))


def generate(output_dir: Path, reference_dir: Path) -> list[ManifestEntry]:
    """
    Build and persist the crawl manifest from stakeholder data.

    Reads `career_urls.json`, classifies each URL, strips tracking
    parameters, and writes the result to `manifest.json`. Returns
    the list of manifest entries for downstream consumption.

    Args:
        output_dir    : Directory to write `manifest.json` into.
        reference_dir : Directory containing `career_urls.json`.

    Returns:
        The complete list of manifest entries.
    """
    entries = [
        ManifestEntry(
            active   = (cat := _classify(url := _strip_tracking_params(
                record["url"]
            ))) not in {ScrapeCategory.APPLICATION_ONLY, ScrapeCategory.PDF_ONLY},
            category = cat,
            company  = record["company"],
            source   = record["source"],
            url      = url
        )
        for record in loads(
            (reference_dir / "career_urls.json").read_text(
                encoding="utf-8"
            )
        )
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        dumps(
            [e.model_dump(mode="json") for e in entries],
            indent=2
        ),
        encoding="utf-8"
    )

    logger.info(
        f"Manifest: {len(entries)} URLs "
        f"({(active := sum(e.active for e in entries))} active, "
        f"{len(entries) - active} inactive)"
    )
    return entries
