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


_PDF_ONLY_EXTENSIONS = {
    ".pdf"
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
    parsed   = urlparse(url)
    path_key = _url_path_key(url)

    if path_key in _APPLICATION_ONLY_URLS:
        return ScrapeCategory.APPLICATION_ONLY

    if any(url.lower().endswith(ext) for ext in _PDF_ONLY_EXTENSIONS):
        return ScrapeCategory.PDF_ONLY

    hostname = parsed.hostname or ""
    if "workable.com" in hostname:
        return ScrapeCategory.WORKABLE
    if "myworkdayjobs.com" in hostname:
        return ScrapeCategory.WORKDAY
    if "engagedtas.com" in hostname:
        return ScrapeCategory.ENGAGEDTAS
    if "cianbro.com" in hostname:
        return ScrapeCategory.CIANBRO

    return ScrapeCategory.STATIC_HTML


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
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    cleaned = {
        k: v for k, v in params.items()
        if k not in GOOGLE_ADS_PARAMS
    }
    return urlunparse(
        parsed._replace(query=urlencode(cleaned, doseq=True))
    )


def _url_path_key(url: str) -> str:
    """
    Extract the domain and path for matching against known patterns.

    Strips the `www.` prefix so that `www.example.com/path` and
    `example.com/path` resolve to the same key.

    Args:
        url: The URL to extract the path key from.

    Returns:
        A string combining hostname and path for pattern matching.
    """
    parsed = urlparse(url)
    host   = (parsed.hostname or "").removeprefix("www.")
    return f"{host}{parsed.path}"


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
    raw = loads(
        (reference_dir / "career_urls.json").read_text(
            encoding="utf-8"
        )
    )

    entries = []
    for record in raw:
        clean_url = _strip_tracking_params(record["url"])
        category  = _classify(clean_url)
        active    = category not in {
            ScrapeCategory.APPLICATION_ONLY,
            ScrapeCategory.PDF_ONLY
        }

        entries.append(ManifestEntry(
            active   = active,
            category = category,
            company  = record["company"],
            source   = record["source"],
            url      = clean_url
        ))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        dumps(
            [e.model_dump(mode="json") for e in entries],
            indent=2
        ),
        encoding="utf-8"
    )

    active_count   = sum(1 for e in entries if e.active)
    inactive_count = len(entries) - active_count
    logger.info(
        f"Manifest: {len(entries)} URLs "
        f"({active_count} active, {inactive_count} inactive)"
    )
    return entries
