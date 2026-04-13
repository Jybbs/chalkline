"""
Launch the Marimo reactive notebook.

Pre-fits the pipeline in CLI context so that cold-cache encoding and
clustering run with the full Rich progress display rather than Marimo's
minimal bar. If the Hamilton cache is already warm, fit returns in seconds.
"""

import typer

from pathlib    import Path
from subprocess import run
from sys        import executable


def launch(
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help = "🔩 Show diagnostic logs during pre-fit."
    )
):
    """
    🖥️ [bold]Launch[/bold] the Marimo career report notebook.

    Pre-fits the pipeline with the full CLI progress display, then starts
    `marimo run`. If the cache is warm the pre-fit is instant.
    """
    if not (app_path := Path.cwd() / "app" / "main.py").exists():
        typer.echo(
            f"🚧 App not found at {app_path}."
            " Run from the project root.",
            err=True,
        )
        raise typer.Exit(1)

    from chalkline.pipeline.orchestrator import Chalkline
    from chalkline.pipeline.schemas      import PipelineConfig

    Chalkline.fit(
        config = PipelineConfig(
            lexicon_dir  = Path("data/lexicons"),
            postings_dir = Path("data/postings")
        ),
        log_level = "DEBUG" if verbose else "INFO"
    )

    raise typer.Exit(
        run([executable, "-m", "marimo", "run", str(app_path)]).returncode
    )
