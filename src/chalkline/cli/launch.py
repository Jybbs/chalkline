"""
Launch the Marimo reactive notebook.
"""

import typer

from pathlib    import Path
from subprocess import run
from sys        import executable


def launch():
    """
    🖥️ [bold]Launch[/bold] the Marimo career report notebook.

    Starts `marimo run` on the app entry point. Must be run from the project
    root where `app/main.py` exists.
    """
    if not (app_path := Path.cwd() / "app" / "main.py").exists():
        typer.echo(
            f"🚧 App not found at {app_path}."
            " Run from the project root.",
            err=True,
        )
        raise typer.Exit(1)

    raise typer.Exit(
        run([executable, "-m", "marimo", "run", app_path]).returncode
    )
