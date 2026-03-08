"""
Launch the Marimo reactive notebook.

Entry point for the `chalkline` console script defined in
pyproject.toml.
"""

from logging    import getLogger
from pathlib    import Path
from subprocess import run
from sys        import executable, exit

logger = getLogger(__name__)


def main():
    """
    Launch the Marimo app from the project root.
    """
    if not (app_path := Path.cwd() / "app" / "main.py").exists():
        logger.error(f"App not found at {app_path}. Run from the project root.")
        exit(1)

    exit(run([executable, "-m", "marimo", "run", app_path]).returncode)


if __name__ == "__main__":

    main()
