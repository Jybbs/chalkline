"""
Splash page renderer.
"""

from marimo import Html
from pathlib import Path

from chalkline.display.loaders       import ContentLoader, Layout
from chalkline.display.schemas       import SplashMetrics
from chalkline.pathways.loaders      import LaborLoader
from chalkline.pipeline.orchestrator import Chalkline


def splash_page(
    content  : ContentLoader,
    labor    : LaborLoader,
    logo_dir : Path,
    pipeline : Chalkline
) -> Html:
    """
    Render the pre-upload splash page with branding and corpus statistics.

    The logo is read from a pre-encoded base64 file so the splash works
    without a static file server. Stat values come from `SplashMetrics`
    so the formatting lives next to the other tab metrics schemas
    rather than inline in the renderer.

    Args:
        content  : Centralized content loader for display-layer TOML.
        labor    : BLS labor market data.
        logo_dir : Directory containing `logo.b64`.
        pipeline : Fitted Chalkline pipeline instance.

    Returns:
        Full-width splash element with logo, tagline, and stats.
    """
    return Layout(content).splash(
        logo_src    = f"data:image/png;base64,{(logo_dir / 'logo.b64').read_text()}",
        stat_values = SplashMetrics.from_corpus(labor, pipeline).stat_values,
        tab         = content.tab("splash")
    )
