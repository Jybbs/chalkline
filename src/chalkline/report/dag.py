"""
Hamilton DAG introspection for Mermaid rendering.

Inspects the function signatures in `chalkline.pipeline.steps` to
reconstruct the DAG dependency graph as a Mermaid flowchart string.
Driver-level parameters (`config`, `model`, `lexicons`) are excluded
because they are injected inputs rather than computed node outputs.
"""

from inspect import getmembers, isfunction, signature


def to_mermaid() -> str:
    """
    Build a Mermaid LR flowchart from the pipeline step functions.

    Each function in `chalkline.pipeline.steps` becomes a node, and
    each parameter that is not an external input becomes a directed
    edge from the parameter's source node to the function node.

    Returns:
        Mermaid diagram string starting with `graph LR`.
    """
    from chalkline.pipeline import steps

    return "\n".join([
        "graph LR",
        *(
            f"    {param} --> {name}"
            for name, fn in sorted(getmembers(steps, isfunction))
            for param in signature(fn).parameters
            if param not in {"config", "model", "lexicons"}
        )
    ])
