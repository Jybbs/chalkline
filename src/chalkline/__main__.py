"""
Launch the Marimo reactive notebook.

This module serves as the entry point for the ``chalkline`` console
script defined in pyproject.toml.
"""

import subprocess
import sys

from pathlib import Path


def main():
    """
    Launch the Marimo app from the project root.
    """
    app_path = Path.cwd() / "app" / "main.py"

    if not app_path.exists():
        print(f"App not found at {app_path}")
        print("Run this command from the project root.")
        sys.exit(1)

    sys.exit(subprocess.run(
        [sys.executable, "-m", "marimo", "run", str(app_path)]
    ).returncode)


if __name__ == "__main__":

    main()
