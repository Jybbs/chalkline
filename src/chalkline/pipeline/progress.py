"""
Pipeline progress display for CLI and Marimo contexts.

Provides a Hamilton lifecycle adapter that shows node-level progress during
`Chalkline.fit()`. In CLI context, Rich renders a dynamic progress display
where each executing node gets its own bar, plus transient batch bars for
sentence-transformer encoding. In Marimo context, delegates to
`mo.status.progress_bar`.
"""

from hamilton.lifecycle.api import GraphExecutionHook, NodeExecutionHook
from loguru                 import logger
from sys                    import modules
from time                   import perf_counter


def _in_marimo() -> bool:
    """
    Detect whether code is running inside a Marimo notebook runtime.
    """
    if (mo := modules.get("marimo")):
        return mo.running_in_notebook()
    return False


class MarimoDisplay:
    """
    Progress display backed by `mo.status.progress_bar`.
    """

    def __init__(self, total: int):
        
        import marimo as mo
        self.manager = mo.status.progress_bar(
            completion_title = "Pipeline fitted",
            title            = "Fitting pipeline",
            total            = total
        )
        self.bar = self.manager.__enter__()

    def advance(self, node_name: str):
        """
        Increment the progress bar by one node.
        """
        self.bar.update()

    def start_node(self, node_name: str):
        """
        No-op for Marimo (single bar, no per-node tracking).
        """

    def stop(self):
        """
        Exit the progress bar context manager.
        """
        self.manager.__exit__(None, None, None)


class PipelineProgress(GraphExecutionHook, NodeExecutionHook):
    """
    Hamilton lifecycle adapter for pipeline progress display.

    Dispatches to `RichDisplay` or `MarimoDisplay` once at graph start,
    then all lifecycle methods call the same interface without branching.
    Pass `level="DEBUG"` for verbose output.
    """

    def __init__(self, level: str = "INFO"):
        """
        Args:
            level: Minimum loguru level for the Rich sink.
        """
        self.level = level

    def run_after_graph_execution(self, *, success, **future_kwargs):
        """
        Log completion, then stop the display.
        """
        if success:
            logger.info(f"Pipeline fitted ({self.completed} nodes)")
        else:
            logger.error(
                f"Pipeline failed at node "
                f"{self.completed}/{self.node_count}"
            )
        self.display.stop()

    def run_after_node_execution(self, *, node_name, **future_kwargs):
        """
        Complete the node's task bar and log elapsed time.
        """
        elapsed = perf_counter() - self.timings.pop(
            node_name, perf_counter()
        )
        self.completed += 1
        logger.info(f"· {node_name} ({elapsed:.1f} sec)")
        self.display.advance(node_name)

    def run_before_graph_execution(self, *, execution_path, **future_kwargs):
        """
        Initialize the display backend and timing state.

        `execution_path` sizes the bar to only the nodes that will
        actually execute, excluding cached results.
        """
        self.node_count = len(execution_path)
        self.completed  = 0
        self.timings    = {}
        self.display    = (
            MarimoDisplay(total=self.node_count)
            if _in_marimo()
            else RichDisplay(level=self.level, total=self.node_count)
        )

    def run_before_node_execution(self, *, node_name, **future_kwargs):
        """
        Record start time and create a task bar for the node.
        """
        self.timings[node_name] = perf_counter()
        self.display.start_node(node_name)


class RichDisplay:
    """
    Dynamic Rich progress display with loguru routing.

    `Progress` manages its own `Live` context, rendering the
    pipeline-level task alongside transient per-node and batch bars.
    Encoding batch progress is fed by intercepted tqdm calls from
    sentence-transformers.
    """

    def __init__(self, level: str, total: int):
        from rich.logging  import RichHandler
        from rich.progress import BarColumn, Progress
        from rich.progress import TaskProgressColumn, TextColumn, TimeRemainingColumn
        from rich.table    import Column

        self.progress = Progress(
            TextColumn(
                "{task.description}",
                justify      = "right",
                table_column = Column(min_width=19)
            ),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn()
        )

        logger.remove()
        self.handler_id = logger.add(
            RichHandler(
                console   = self.progress.console,
                show_path = False
            ),
            format = "{message}",
            level  = level
        )

        self.progress.console.rule()
        self.pipeline_task = self.progress.add_task(
            "[bold]pipeline[/bold]",
            total = total
        )
        self.progress.start()

        self.active_nodes  : dict       = {}
        self.batch_context : str | None = None
        self.st_module       = modules["sentence_transformers.SentenceTransformer"]
        self.original_trange = getattr(self.st_module, "trange")
        setattr(self.st_module, "trange", self._rich_trange)

    def _rich_trange(self, *args, desc="", disable=False, **kwargs):
        """
        Replacement for `trange` that renders sentence-transformer
        batches as transient Rich progress bars.
        """
        r = range(*args)
        if disable:
            yield from r
            return
        task = self.progress.add_task(
            f"{self.batch_context or desc} ·",
            total = len(r)
        )
        for value in r:
            yield value
            self.progress.update(task, advance=1)
        self.progress.remove_task(task)

    def advance(self, node_name: str):
        """
        Complete the node's task and advance the pipeline bar.
        """
        if (task := self.active_nodes.pop(node_name, None)) is not None:
            self.progress.remove_task(task)
        self.progress.update(self.pipeline_task, advance=1)

    def start_node(self, node_name: str):
        """
        Add a transient task bar for the executing node. Sets the
        batch context label for sentence-transformer encoding.
        """
        self.active_nodes[node_name] = self.progress.add_task(
            f"{node_name}",
            total = None
        )
        self.batch_context = {
            "credentials" : "credentials",
            "raw_vectors" : "postings",
            "soc_tasks"   : "tasks",
            "soc_vectors" : "occupations"
        }.get(node_name)

    def stop(self):
        """
        Finalize the pipeline bar and restore trange.
        """
        self.progress.update(
            self.pipeline_task,
            description = "[bold]fitted[/bold]"
        )
        self.progress.stop()
        logger.remove(self.handler_id)
        setattr(self.st_module, "trange", self.original_trange)
