"""
Fit the Chalkline pipeline from the posting corpus.
"""

import typer

from pathlib import Path
from typing  import Annotated


def fit(
    lexicon_dir: Annotated[
        Path,
        typer.Option(help="📚 Path to lexicon JSON files.")
    ] = Path("data/lexicons"),
    output_dir: Annotated[
        Path,
        typer.Option(help="🧱 Cache directory for fitted artifacts.")
    ] = Path(".cache/pipeline"),
    postings_dir: Annotated[
        Path,
        typer.Option(help="🗄️ Directory containing the posting corpus.")
    ] = Path("data/postings"),
):
    """
    🪚 [bold]Fit[/bold] the pipeline and print a summary.

    Runs extraction, vectorization, PCA, clustering, PMI, and graph
    construction. Results are cached so subsequent calls with unchanged code
    and config serve instantly.
    """
    from chalkline.pipeline.orchestrator import Chalkline
    from chalkline.pipeline.schemas      import PipelineConfig

    config = PipelineConfig(
        lexicon_dir  = lexicon_dir,
        output_dir   = output_dir,
        postings_dir = postings_dir,
    )
    typer.echo(Chalkline.fit(config))
