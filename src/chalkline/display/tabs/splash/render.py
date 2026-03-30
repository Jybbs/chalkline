"""
Splash page renderer.
"""

import marimo as mo

from pathlib import Path

from chalkline.display.loaders import ContentLoader, Layout
from chalkline.display.schemas import SplashMetrics


def splash_page(
    content  : ContentLoader,
    logo_dir : Path,
    metrics  : SplashMetrics
) -> mo.Html:
    """
    Render the pre-upload splash page with branding and corpus statistics.

    The logo is read from a pre-encoded base64 file so the splash works
    without a static file server. Stats are pre-computed by
    `SplashMetrics.from_pipeline` and formatted by `stat_values` into the
    branded stat grid.

    Args:
        content  : Centralized content loader for display-layer TOML.
        logo_dir : Directory containing `logo.b64`.
        metrics  : Pre-computed corpus and labor statistics.

    Returns:
        Full-width splash element with logo, tagline, and stats.
    """
    return Layout(content).splash(
        logo_src = f"data:image/png;base64,{(logo_dir / 'logo.b64').read_text()}",
        metrics  = metrics,
        tab      = content.tab("splash")
    )
