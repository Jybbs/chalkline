"""
Splash page renderer.
"""

import marimo as mo

from pathlib import Path

from chalkline.display.layout       import stat_html
from chalkline.display.schemas      import SplashMetrics
from chalkline.display.tabs.context import load_content

content = load_content(__file__)


def splash_page(logo_dir: Path, metrics: SplashMetrics) -> mo.Html:
    """
    Render the pre-upload splash page with branding and corpus
    statistics.

    The logo is read from a pre-encoded base64 file so the splash
    works without a static file server. Stats are pre-computed by
    `build_splash` and formatted here into the branded stat grid.

    Args:
        logo_dir : Directory containing `logo.b64`.
        metrics  : Pre-computed corpus and labor statistics.

    Returns:
        Full-width splash element with logo, tagline, and stats.
    """
    logo_b64 = (logo_dir / "logo.b64").read_text()
    logo_src = f"data:image/png;base64,{logo_b64}"

    stats = "".join([
        stat_html("Job Postings",    f"{metrics.corpus_size:,}"),
        stat_html("Companies",       str(metrics.companies)),
        stat_html("Maine Locations", str(metrics.locations)),
        stat_html("Career Families", str(metrics.num_clusters)),
        stat_html("Maine Workers",   f"{metrics.employment:,}"),
        stat_html("Median Salary",   f"${metrics.median_wage:,.0f}"),
        stat_html("Bright Outlook",  str(metrics.bright_outlook)),
        stat_html("Career Pathways", str(metrics.edge_count))
    ])

    mask = f"mask-image:url({logo_src});-webkit-mask-image:url({logo_src})"

    return mo.Html(
        f'<div class="cl-splash"><div class="cl-brand">'
        f'<span class="cl-logo" style="{mask}"></span>'
        f"<h1>{content.title}</h1></div>"
        f'<p class="cl-tagline">{content.tagline}</p>'
        f'<div class="cl-stats">{stats}</div></div>'
    )
