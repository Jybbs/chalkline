"""
Pipeline progress display for CLI and Marimo contexts.

Provides a Hamilton lifecycle adapter that shows node-level progress during
`Chalkline.fit()`. In CLI context, Rich renders a dynamic progress display
where each executing node gets its own bar, plus transient batch bars for
encoding driven by a callback on the `SentenceEncoder` instance. In Marimo
context, delegates to `mo.status.progress_bar`.
"""

from collections.abc        import Callable
from hamilton.lifecycle.api import GraphExecutionHook, NodeExecutionHook
from loguru                 import logger
from sys                    import modules
from time                   import perf_counter
from typing                 import TYPE_CHECKING

from chalkline.pipeline.encoder import SentenceEncoder

if TYPE_CHECKING:
    from rich.progress import Progress


class DownloadBar:
    """
    tqdm-compatible progress bar backed by a Rich `Progress` instance.

    Accepts and discards the keyword arguments that `hf_hub_download` passes
    to its `tqdm_class` constructor, forwarding only `total` to the Rich
    task. Use `functools.partial` to bind a `Progress` instance before
    passing as `tqdm_class`.
    """

    def __enter__(self)       : return self
    def __exit__(self, *args) : pass

    def __init__(self, progress: Progress, *args, **kwargs):
        self.progress = progress
        self.task     = progress.add_task("model", total = kwargs.get("total"))

    def close(self): pass

    def update(self, n=1):
        self.progress.update(self.task, advance=n)


class PipelineProgress(GraphExecutionHook, NodeExecutionHook):
    """
    Hamilton lifecycle adapter with display primitives.

    Subclasses override `advance`, `begin_pipeline`, `make_batch_callback`,
    `make_download_tqdm`, `start_node`, and `stop` to render progress in
    their target environment. The lifecycle hooks handle timing, logging,
    and encoder callback wiring automatically.
    """

    encoder: SentenceEncoder

    def advance(self, node_name: str):
        """
        Mark a node complete in the display.
        """

    def begin_pipeline(self, total: int):
        """
        Initialize the display once the node count is known.
        """

    def make_batch_callback(self, label: str) -> Callable[[int, int], None]:
        """
        Return a callback for batch-level encoding progress.
        """
        return lambda *_: None

    def make_download_tqdm(self) -> type[DownloadBar] | None:
        """
        Return a tqdm-compatible callable for model download progress.
        """
        return None

    def run_after_graph_execution(self, *, success, **future_kwargs):
        """
        Log completion, then stop the display.
        """
        if success:
            logger.info(f"Pipeline fitted ({self.completed} nodes)")
        else:
            logger.error(f"Pipeline failed at node {self.completed}/{self.node_count}")
        self.stop()

    def run_after_node_execution(self, *, node_name, **future_kwargs):
        """
        Complete the node's task bar, clear any batch callback, and log
        elapsed time.
        """
        self.encoder.on_batch = None
        elapsed = perf_counter() - self.timings.pop(node_name, perf_counter())
        self.completed += 1
        logger.info(f"· {node_name} ({elapsed:.1f} sec)")
        self.advance(node_name)

    def run_before_graph_execution(self, *, execution_path, **future_kwargs):
        """
        Initialize timing state and begin the display.

        `execution_path` sizes the bar to only the nodes that will actually
        execute, excluding cached results.
        """
        self.node_count = len(execution_path)
        self.completed  = 0
        self.timings    = {}
        self.begin_pipeline(self.node_count)

    def run_before_node_execution(self, *, node_name, node_tags, **future_kwargs):
        """
        Record start time, create a task bar for the node, and wire the
        batch callback for encoding nodes tagged with `batch_label`.
        """
        self.timings[node_name] = perf_counter()
        self.start_node(node_name)

        if (label := node_tags.get("batch_label")):
            self.encoder.on_batch = self.make_batch_callback(label)

    def start_node(self, node_name: str):
        """
        Show a node as in-progress in the display.
        """

    def stop(self):
        """
        Tear down the display.
        """


class MarimoDisplay(PipelineProgress):
    """
    Progress display backed by `mo.status.progress_bar`.
    """

    def advance(self, node_name: str):
        """
        Increment the progress bar by one node.
        """
        self.bar.update()

    def begin_pipeline(self, total: int):
        """
        Create the Marimo progress bar once the node count is known.
        """
        import marimo as mo
        self.manager = mo.status.progress_bar(
            completion_title = "Pipeline fitted",
            title            = "Fitting pipeline",
            total            = total
        )
        self.bar = self.manager.__enter__()

    @classmethod
    def detect(cls) -> MarimoDisplay | None:
        """
        Return an instance if running inside Marimo, else `None`.
        """
        if (mo := modules.get("marimo")) and mo.running_in_notebook():
            return cls()
        return None

    def stop(self):
        """
        Exit the progress bar context manager.
        """
        self.manager.__exit__(None, None, None)


class RichDisplay(PipelineProgress):
    """
    Dynamic Rich progress display with loguru routing.

    `Progress` manages its own `Live` context, rendering the pipeline-level
    task alongside transient per-node and batch bars. Encoding batch
    progress is driven by callbacks from the `SentenceEncoder` rather than
    monkey-patching tqdm.
    """

    def __init__(self, level: str = "INFO"):
        from rich.progress import BarColumn, Progress
        from rich.progress import TaskProgressColumn, TextColumn, TimeRemainingColumn
        from rich.table    import Column

        self.active_nodes: dict = {}
        self.level    = level
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

    def advance(self, node_name: str):
        """
        Complete the node's task and advance the pipeline bar.
        """
        if (task := self.active_nodes.pop(node_name, None)) is not None:
            self.progress.remove_task(task)
        self.progress.update(self.pipeline_task, advance = 1)

    def begin_pipeline(self, total: int):
        """
        Start the live display, wire loguru through Rich, and add the
        pipeline-level progress bar.
        """
        from rich.logging import RichHandler

        logger.remove()
        self.handler_id = logger.add(
            RichHandler(
                console   = self.progress.console,
                show_path = False
            ),
            format = "{message}",
            level  = self.level
        )

        self.progress.console.rule()
        self.progress.start()
        self.pipeline_task = self.progress.add_task(
            "[bold]pipeline[/bold]",
            total = total
        )

    def make_batch_callback(self, label: str) -> Callable[[int, int], None]:
        """
        Return a closure that drives a transient Rich task bar for encoding
        batches.
        """
        progress = self.progress

        def on_batch(current: int, total: int):
            if current == 0:
                on_batch.task_id = progress.add_task(
                    f"{label} ·", total = total
                )
            progress.update(on_batch.task_id, completed = current + 1)
            if current + 1 == total:
                progress.remove_task(on_batch.task_id)

        return on_batch

    def make_download_tqdm(self) -> type[DownloadBar]:
        """
        Return a tqdm-compatible callable that renders the HuggingFace model
        download as a transient Rich task bar. Passed to
        `hf_hub_download(tqdm_class=...)`.
        """
        progress = self.progress
        class BoundDownloadBar(DownloadBar):
            def __init__(self, *args, **kwargs):
                super().__init__(progress, *args, **kwargs)
        return BoundDownloadBar

    def start_node(self, node_name: str):
        """
        Add a transient task bar for the executing node.
        """
        self.active_nodes[node_name] = self.progress.add_task(
            node_name,
            total = None
        )

    def stop(self):
        """
        Finalize the pipeline bar.
        """
        self.progress.update(
            self.pipeline_task,
            description = "[bold]fitted[/bold]"
        )
        self.progress.stop()
        logger.remove(self.handler_id)
