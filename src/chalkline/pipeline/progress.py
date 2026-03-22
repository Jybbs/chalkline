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
from typing                 import Any


def _in_marimo() -> bool:
    """
    Detect whether code is running inside a Marimo notebook runtime.
    """
    try:
        import marimo as mo
        return mo.running_in_notebook()
    except Exception:
        return False


class MarimoDisplay:
    """
    Progress display backed by `mo.status.progress_bar`.
    """

    def __init__(self, total: int):
        import marimo as mo
        self.bar: Any = mo.status.progress_bar(
            total = total,
            title = "Fitting pipeline"
        )

    def advance(self, node_name: str):
        """
        Increment the progress bar by one node.
        """
        self.bar.update(increment=1)

    def start_node(self, node_name: str):
        """
        No-op for Marimo (single bar, no per-node tracking).
        """

    def stop(self):
        """
        Close the progress bar.
        """
        self.bar.close()


class RichDisplay:
    """
    Dynamic Rich progress display with loguru routing.

    Uses a single `Progress` instance under a `Live` display. The
    pipeline-level task is persistent, and each executing node gets a
    transient task that appears while running. Encoding batch progress
    is fed by intercepted tqdm calls from sentence-transformers.
    """

    def __init__(self, level: str, total: int):
        st_module: Any = modules["sentence_transformers.SentenceTransformer"]

        from rich.live     import Live
        from rich.logging  import RichHandler
        from rich.progress import BarColumn, Progress, SpinnerColumn
        from rich.progress import TaskProgressColumn, TextColumn
        from rich.progress import TimeRemainingColumn
        from rich.rule     import Rule

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn()
        )
        self.live = Live(self.progress)

        logger.remove()
        self.handler_id = logger.add(
            RichHandler(
                console   = self.live.console,
                show_path = False
            ),
            format = "{message}",
            level  = level
        )

        self.live.console.print(Rule())
        self.pipeline_task = self.progress.add_task(
            f"{'pipeline':<17}",
            total = total
        )
        self.live.start()

        self.active_nodes  : dict = {}
        self.batch_context : list[str | None] = [None]
        self.original_trange = st_module.trange
        st_module.trange     = self._make_trange()

    def _make_trange(self):
        """
        Build a trange replacement that routes sentence-transformer
        batch progress through a transient task on the shared
        Progress instance.
        """
        def rich_trange(*args, desc="", disable=False, **kwargs):
            r = range(*args)
            if disable:
                yield from r
                return
            label = f"· {self.batch_context[0] or desc:<15}"
            task  = self.progress.add_task(label, total=len(r))
            for value in r:
                yield value
                self.progress.update(task, advance=1)
            self.progress.remove_task(task)
        return rich_trange

    def advance(self, node_name: str):
        """
        Complete the node's task and advance the pipeline bar.
        """
        if node_name in self.active_nodes:
            self.progress.remove_task(self.active_nodes.pop(node_name))
        self.progress.update(self.pipeline_task, advance=1)

    def start_node(self, node_name: str):
        """
        Add a transient task bar for the executing node.
        """
        self.active_nodes[node_name] = self.progress.add_task(
            f"› {node_name:<15}",
            total = None
        )

    def stop(self):
        """
        Finalize the pipeline bar and restore trange.
        """
        st_module: Any = modules["sentence_transformers.SentenceTransformer"]
        self.progress.update(
            self.pipeline_task,
            description = f"{'fitted':<17}"
        )
        self.live.stop()
        logger.remove(self.handler_id)
        st_module.trange = self.original_trange


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

    def run_after_graph_execution(
        self,
        *,
        error,
        graph,
        results,
        run_id,
        success,
        **future_kwargs
    ):
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

    def run_after_node_execution(
        self,
        *,
        error,
        node_kwargs,
        node_name,
        node_return_type,
        node_tags,
        result,
        run_id,
        success,
        task_id=None,
        **future_kwargs
    ):
        """
        Complete the node's task bar and log elapsed time.
        """
        elapsed = perf_counter() - self.timings.pop(
            node_name, perf_counter()
        )
        self.completed += 1
        logger.info(f"· {node_name} ({elapsed:.1f} sec)")
        self.display.advance(node_name)

    def run_before_graph_execution(
        self,
        *,
        execution_path,
        final_vars,
        graph,
        inputs,
        overrides,
        run_id,
        **future_kwargs
    ):
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

    def run_before_node_execution(
        self,
        *,
        node_input_types,
        node_kwargs,
        node_name,
        node_return_type,
        node_tags,
        run_id,
        task_id=None,
        **future_kwargs
    ):
        """
        Start a task bar for the node and set the batch context label
        for encoding nodes.
        """
        self.timings[node_name] = perf_counter()
        self.display.start_node(node_name)
        if isinstance(self.display, RichDisplay):
            self.display.batch_context[0] = {
                "credentials" : "credentials",
                "raw_vectors" : "postings",
                "soc_tasks"   : "tasks",
                "soc_vectors" : "occupations"
            }.get(node_name)
