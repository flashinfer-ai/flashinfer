# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Backend of the experimental Qwen 3.8 Flash-Next gated attention block API (``flashinfer.qwen_3_8_block``).

The whole Qwen 3.8 Flash-Next sparse-attention sub-layer at DECODE -- the QKV + gate + indexer projection, QK-RMSNorm with
partial RoPE, the paged-KV write-through, the in-block indexer (the raw-key ring, the paged compressed-key cache, the paged
scorer, the per-row top-k), block-sparse GQA attention at head dim 256 over the selected 4-token blocks, the sigmoid gate and the
out projection -- in ONE call over the serving stack's own caches, on SM107 (Rubin, cc 10.7) only, bf16, as CuTe DSL kernels
vendored from cuDNN Frontend.  The vendored tree is the Qwen 3.5 package's (``flashinfer.experimental.qwen_3_5_block._cudnn_frost``:
one drop, one manifest, one re-vendor command; this package adds nothing under it).  Nothing here imports the DSL, the kernels or
torch at package-import time: the lightweight ``support`` module answers the dispatch question, ``api`` builds and runs plans on
first use.  DECODE only: the prefill / extend forms of the indexer are a later cuDNN Frontend PR's and decline by name here.
"""

__all__ = ["api", "support", "weights", "rope"]


def __getattr__(name):
    # A deferred support-check import must not load the execution implementation (the cudnn_frost precedent).
    if name in __all__:
        import importlib

        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
