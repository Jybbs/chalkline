"""
ONNX sentence encoder with CLS pooling.

Downloads the ONNX model and tokenizer from HuggingFace on first use (cached
locally by huggingface_hub thereafter), then runs inference via onnxruntime
with batch-level progress callbacks.
"""

import numpy as np

from collections.abc import Callable
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
        from huggingface_hub import hf_hub_download
        from onnxruntime     import InferenceSession
        from tokenizers      import Tokenizer

        self.batch_size = batch_size
        self.name       = name
        self.session    = InferenceSession(hf_hub_download(
            name,
            cache_dir  = Path(".cache/models"),
            filename   = "onnx/model.onnx",
            tqdm_class = tqdm_class
        ))
        self.tokenizer  = Tokenizer.from_pretrained(name)
        self.tokenizer.enable_padding()

        self.on_batch: Callable[[int, int], None] | None = None

    def __reduce__(self) -> tuple:
        return (SentenceEncoder, (self.name,))

    def __repr__(self) -> str:
        return f"SentenceEncoder(name={self.name!r})"

    def _infer_batch(
        self,
        batch_index : int,
        texts       : list[str],
        total       : int
    ) -> np.ndarray:
        """
        Tokenize, run ONNX inference, and extract CLS embeddings for one
        batch. Calls `on_batch` after inference completes.
        """
        encoded = self.tokenizer.encode_batch(texts[
            (start := batch_index * self.batch_size)
            : start + self.batch_size
        ])

        ids, masks = zip(*((e.ids, e.attention_mask) for e in encoded))
        hidden     = np.asarray(self.session.run(None, {
            "input_ids"      : np.array(ids, dtype=np.int64),
            "attention_mask" : np.array(masks, dtype=np.int64)
        })[0])

        if self.on_batch is not None:
            self.on_batch(batch_index, total)
            
        return hidden[:, 0, :]

    def encode(self, texts: list[str], unit: bool = True) -> np.ndarray:
        """
        Encode texts with CLS pooling over ONNX transformer output.

        Processes in fixed-size batches with optional progress reporting via
        the `on_batch` callback.

        Args:
            texts : Strings to encode.
            unit  : L2-normalize output (default True).
        """
        total  = -(-len(texts) // self.batch_size)
        result = np.vstack([
            self._infer_batch(batch_index=i, texts=texts, total=total)
            for i in range(total)
        ])

        if unit:
            result = result / np.linalg.norm(result, axis=1, keepdims=True)

        return result
