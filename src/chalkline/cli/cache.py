"""
Inspect Hamilton's content-addressed cache for the Chalkline pipeline.
"""

from pathlib import Path
from typer   import Exit, Option
from typing  import Annotated


def cache(
    cache_dir: Annotated[
        Path,
        Option(help="📦  Path to the Hamilton cache directory.")
    ] = Path(".cache/hamilton")
):
    """
    🗃️  [bold]Inspect[/bold] the Hamilton cache contents.

    Joins `metadata_store.db` against the files in the cache directory to
    show every cached node with its code version, on-disk file, and size.
    Useful for spotting orphans, confirming which nodes re-ran, and
    estimating footprint.
    """
    from rich.console  import Console
    from rich.filesize import decimal
    from rich.table    import Column, Table
    from sqlite3       import connect

    from chalkline.pipeline.schemas import CacheRow

    console = Console()
    if not (store := cache_dir / "metadata_store.db").exists():
        console.print(f"[red]no cache at {cache_dir}[/red]")
        raise Exit(1)

    rows = [CacheRow(*r) for r in connect(store).execute(
        "SELECT node_name, code_version, data_version, created_at"
        " FROM cache_metadata ORDER BY created_at"
    ).fetchall()]

    table = Table(
        Column("node",    style="cyan"),
        Column("code",    max_width=8,  no_wrap=True, style="dim"),
        Column("file",    max_width=20, style="magenta"),
        Column("created", style="dim"),
        Column("size",    justify="right"),
        title=f"Hamilton cache · {cache_dir}"
    )

    sizes = [
        p.stat().st_size if (p := cache_dir / r.data).exists() else 0
        for r in rows
    ]
    for row, size in zip(rows, sizes):
        table.add_row(*row, decimal(size))

    console.print(table)
    console.print(
        f"[bold]{len(rows)}[/bold] entries · "
        f"[bold]{decimal(sum(sizes))}[/bold] on disk"
    )
