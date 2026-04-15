"""
Command-line interface for the Chalkline pipeline.

Registers subcommands for fitting the pipeline and launching the Marimo
notebook. Run `uv run chalkline --help` for usage.
"""

from typer import Typer

from chalkline.cli.cache  import cache
from chalkline.cli.fit    import fit
from chalkline.cli.launch import launch

app = Typer(
    add_completion   = False,
    no_args_is_help  = True,
    rich_markup_mode = "rich"
)

app.command()(cache)
app.command()(fit)
app.command()(launch)
