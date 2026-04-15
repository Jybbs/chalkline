"""
Fit the Chalkline pipeline from the posting corpus.
"""

from pathlib import Path
from typer   import Option
from typing  import Annotated


def fit(
    lexicon_dir: Annotated[
        Path,
        Option(help="📚  Path to lexicon JSON files.")
    ] = Path("data/lexicons"),
    postings_dir: Annotated[
        Path,
        Option(help="🗄️  Directory containing the posting corpus.")
    ] = Path("data/postings"),
    verbose: Annotated[
        bool,
        Option("--verbose", "-v", help="🔩  Show diagnostic logs.")
    ] = False
):
    """
    🪚  [bold]Fit[/bold] the pipeline and print a summary.

    Encodes postings with a sentence transformer, clusters with Ward HAC,
    builds a stepwise career graph with credential enrichment. Results are
    cached so subsequent calls with unchanged code and config serve
    instantly.
    """
    from rich.console import Console

    from chalkline.pipeline.orchestrator import Chalkline
    from chalkline.pipeline.schemas      import PipelineConfig

    pipeline = Chalkline.fit(
        config    = PipelineConfig(
            lexicon_dir  = lexicon_dir,
            postings_dir = postings_dir
        ),
        log_level = "DEBUG" if verbose else "INFO"
    )
    Console().print(f"\n{pipeline!r}")
