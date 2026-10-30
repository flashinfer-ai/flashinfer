# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Backend of the experimental Qwen 3.8 Flash-Next gated attention block API (``flashinfer.qwen_3_8_block``).

The whole Qwen 3.8 Flash-Next sparse-attention sub-layer -- the DECODE step, and the EXTEND step behind the ``extend`` /
``prefill`` forms (a uniform chunk of every sequence behind a cached prefix; it writes the serving caches a decode step reads) --:
the QKV + gate + indexer projection, QK-RMSNorm with partial RoPE, the paged-KV write-through, the in-block indexer (the raw-key
ring, the paged compressed-key cache, the paged scorer, the per-row top-k), block-sparse GQA attention at head dim 256 over the
selected 4-token blocks, the sigmoid gate and the out projection -- in ONE call over the serving stack's own caches, on SM107 (Rubin, cc 10.7) only, bf16, as CuTe DSL kernels
vendored from cuDNN Frontend.  The vendored tree is the Qwen 3.5 package's (``flashinfer.experimental.qwen_3_5_block._cudnn_frost``:
one drop, one manifest, one re-vendor command; this package adds nothing under it).  Nothing here imports the DSL, the kernels or
torch at package-import time: the lightweight ``support`` module answers the dispatch question, ``api`` builds and runs plans on
first use.  The three forms are one vendored declaration form and are served together (``support.FORMS == ("prefill", "extend",
"decode")`` on a served device); the extend forms take a chunk of any length up to the block table's reach, at any batch, up to 65,535
rows per run at this pin (the band slab's projection rides the JIT route at every row count since the cuDNN Frontend 6b21be0e0 pin; the
vendored indexer's query compress launches with the run's row count as a CUDA grid dimension, so a run past that count is a typed decline
-- ``support.EXTEND_ROWS_GRID_MAX`` -- until the compressor's grid is flattened upstream and re-vendored).
"""

__all__ = ["api", "support", "weights", "rope"]


def __getattr__(name):
    # A deferred support-check import must not load the execution implementation (the cudnn_frost precedent).
    if name in __all__:
        import importlib

        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
