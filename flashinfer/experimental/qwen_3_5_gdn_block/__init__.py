# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Backend of the experimental Qwen 3.5 GDN block API (``flashinfer.qwen_3_5_gdn_block``).

The Qwen3.5-style gated-delta-net sub-layer between the QKVZ projection and the out projection -- BA projection,
causal conv + SiLU, gated delta rule over the SSM state pool, gated RMSNorm, static FP8 output quantization -- in two
forms (``decode``: one token per request; ``verify``: an MTP verify of T tokens per request), behind a backend registry
(``api.register_backend``). The first backend, ``"frost"``, is CuTe DSL for SM100 (``backends/frost/``; see
``README.md``). Nothing here imports the DSL or the kernels at package-import time: ``support`` answers the dispatch
question, ``api`` builds and runs plans, and a backend's kernels load when its first plan is built.
"""

__all__ = ["api", "support"]


def __getattr__(name):
    if name in __all__:
        import importlib

        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
