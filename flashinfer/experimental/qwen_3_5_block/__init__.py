# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Backend of the experimental Qwen 3.5 gated attention block API (``flashinfer.qwen_3_5_block``).

The whole Qwen3.5-style full-attention sub-layer -- QKV + gate projection, QK-RMSNorm with partial RoPE, paged-KV
write-through, GQA attention at head dim 256, sigmoid gate, out projection -- in three serving forms (dense prefill,
EXTEND over a cached prefix, DECODE), on SM107 (Rubin, cc 10.7) only, bf16, as CuTe DSL kernels vendored from cuDNN
Frontend (``_cudnn_frost/``, see ``_cudnn_frost/VENDOR.md`` and ``README.md``).  Nothing here imports the DSL, the
kernels or torch at package-import time: the lightweight ``support`` module answers the dispatch question, ``api``
builds and runs plans on first use.
"""

__all__ = ["api", "support", "weights", "rope"]


def __getattr__(name):
    # A deferred support-check import must not load the execution implementation (the cudnn_frost precedent).
    if name in __all__:
        import importlib

        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
