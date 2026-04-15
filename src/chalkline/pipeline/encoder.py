"""
ONNX sentence encoder with CLS pooling.

Downloads the ONNX model and tokenizer from HuggingFace on first use (cached
locally by huggingface_hub thereafter), then runs inference via onnxruntime
with batch-level progress callbacks.
"""

import numpy as np

from collections.abc import Callable
from loguru          import logger
from pathlib         import Path


class SentenceEncoder:
    """
    ONNX sentence encoder with CLS pooling for the Hamilton pipeline.

    Downloads the ONNX model and tokenizer from HuggingFace on first use
    (cached locally by huggingface_hub thereafter), then runs inference via
    onnxruntime. The `name` field is the full HuggingFace repo ID.

    The optional `on_batch` callback enables progress reporting without
    coupling the encoder to any display framework. Set it before calling
    `encode()` and clear it after.
    """

    def __init__(
        self,
        name       : str,
        batch_size : int         = 32,
        tqdm_class : type | None = None
    ):
        from onnxruntime import InferenceSession
        from tokenizers  import Tokenizer

        self.batch_size = batch_size
        self.name       = name

        logger.info(f"Initializing ONNX encoder ({name!r})")

        self.session   = InferenceSession(self._path("onnx/model.onnx", tqdm_class))
        self.tokenizer = Tokenizer.from_file(self._path("tokenizer.json"))
        self.tokenizer.enable_padding()

        self.dimension = self.session.get_outputs()[0].shape[-1]
        self.on_batch: Callable[[int, int], None] | None = None

    def __reduce__(self) -> tuple:
        return (SentenceEncoder, (self.name,))

    def __repr__(self) -> str:
        return f"SentenceEncoder(name={self.name!r})"

    def _infer_batch(
        self,
        batch       : list[str],
        batch_index : int,
        total       : int
    ) -> np.ndarray:
        """
        Tokenize, run ONNX inference, and extract CLS embeddings for one
        pre-sliced batch. Calls `on_batch` after inference completes.
        """
        encoded  = self.tokenizer.encode_batch(batch)
        to_int64 = lambda key: np.array(
            [getattr(e, key) for e in encoded], dtype=np.int64
        )

        hidden = np.asarray(self.session.run(None, {
            "input_ids"      : to_int64("ids"),
            "attention_mask" : to_int64("attention_mask")
        })[0])

        if self.on_batch is not None:
            self.on_batch(batch_index, total)

        return hidden[:, 0, :]

    def _path(
        self,
        filename   : str,
        tqdm_class : type | None = None
    ) -> str:
        """
        Resolve a file from the HuggingFace repo for this encoder,
        preferring the local cache to skip the slow hub metadata check.
        Downloads on first access only. Bypasses the ~10-second round-trip
        that `Tokenizer.from_pretrained` performs on every instantiation.
        """
        from huggingface_hub import hf_hub_download, try_to_load_from_cache

        cache_dir = Path(".cache/models")
        cached    = try_to_load_from_cache(self.name, filename, cache_dir)
        if isinstance(cached, str):
            return cached
        return hf_hub_download(
            self.name,
            cache_dir  = cache_dir,
            filename   = filename,
            tqdm_class = tqdm_class,
        )

    def encode(self, texts: list[str], unit: bool = True) -> np.ndarray:
        """
        Encode texts with CLS pooling over ONNX transformer output.

        Processes in fixed-size batches with optional progress reporting via
        the `on_batch` callback.

        Args:
            texts : Strings to encode.
            unit  : L2-normalize output (default True).
        """
        starts = range(0, len(texts), self.batch_size)
        total  = len(starts)
        result = np.vstack([
            self._infer_batch(texts[s:s + self.batch_size], i, total)
            for i, s in enumerate(starts)
        ])

        if unit:
            result = result / np.linalg.norm(result, axis=1, keepdims=True)

        return result
