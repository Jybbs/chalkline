"""
Pipeline progress display for CLI and Marimo contexts.

Provides a Hamilton lifecycle adapter that shows node-level progress during
`Chalkline.fit()`. In CLI context, Rich renders a two-level display: a
pipeline bar tracking nodes and a batch bar tracking sentence-transformer
encoding batches. The batch bar intercepts tqdm calls from
sentence-transformers via a monkey-patch on `tqdm.autonotebook.trange`,
routing them through Rich's Group compositing so both bars render in a
single Live display. In Marimo context, delegates to
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
        self.bar: Any = mo.status.progress_bar(total=total, title="Fitting pipeline")

    def advance(self, description: str = ""):
        self.bar.update(increment=1)

    def stop(self):
        self.bar.close()


class RichDisplay:
    """
    Two-level Rich progress display with loguru routing.

    Composes a pipeline-level bar and a batch-level bar in a `Group` under
    one `Live` display. The batch bar is fed by intercepted tqdm calls from
    sentence-transformers. Log messages render above both bars via a
    `RichHandler` bound to the shared console.
    """

    def __init__(self, level: str, total: int):
        st_module: Any = modules["sentence_transformers.SentenceTransformer"]

        from rich.console  import Group
        from rich.live     import Live
        from rich.logging  import RichHandler
        from rich.progress import BarColumn, Progress, SpinnerColumn
        from rich.progress import TaskProgressColumn, TextColumn, TimeRemainingColumn
        from rich.rule     import Rule

        columns = (
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn()
        )
        self.node_progress  = Progress(*columns)
        self.batch_progress = Progress(*columns)
        self.live = Live(Group(self.node_progress, self.batch_progress))

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
        self.node_task = self.node_progress.add_task(
            "Fitting pipeline",
            total = total
        )
        self.live.start()

        self.batch_context: list[str | None] = [None]
        self.original_trange = st_module.trange
        st_module.trange     = self._make_trange()

    def _make_trange(self):
        """
        Build a trange replacement that routes sentence-transformer batch
        progress through the shared Rich batch bar.
        """
        def rich_trange(*args, desc="", disable=False, **kwargs):
            r = range(*args)
            if disable:
                yield from r
                return
            label = self.batch_context[0] or desc
            task = self.batch_progress.add_task(label, total=len(r))
            for value in r:
                yield value
                self.batch_progress.update(task, advance=1)
            self.batch_progress.remove_task(task)
        return rich_trange

    def advance(self, description: str = ""):
        self.node_progress.update(
            self.node_task,
            advance     = 1,
            description = description or None
        )

    def stop(self):
        st_module: Any = modules["sentence_transformers.SentenceTransformer"]
        self.node_progress.update(
            self.node_task, description="Pipeline fitted"
        )
        self.live.stop()
        logger.remove(self.handler_id)
        st_module.trange = self.original_trange


class PipelineProgress(GraphExecutionHook, NodeExecutionHook):
    """
    Hamilton lifecycle adapter for pipeline progress display.

    Dispatches to `RichDisplay` or `MarimoDisplay` once at graph start, then
    all lifecycle methods call the same interface without branching. Pass
    `level="DEBUG"` for verbose output.
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
        Advance the progress bar and log the node name with elapsed time.
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

        `execution_path` sizes the bar to only the nodes that will actually
        execute, excluding cached results.
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
        Record the start time and set the batch progress label for encoding
        nodes.
        """
        self.timings[node_name] = perf_counter()
        if isinstance(self.display, RichDisplay):
            self.display.batch_context[0] = {
                "credentials" : "credentials",
                "raw_vectors" : "postings",
                "soc_tasks"   : "tasks",
                "soc_vectors" : "occupations"
            }.get(node_name)
