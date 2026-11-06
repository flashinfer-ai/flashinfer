"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Cake backend, non-deterministic: the GLM-5.2 dense projection GEMMs for training
# and the FP32 router-gate GEMM on SM100 / SM103 / SM107
# (flashinfer-ai/flashinfer#5677), with programs that add split-K partials in
# the order they finish.  The public module is
# flashinfer.cake_dense_gemm_nondet_train; it selects the mode, so no entry
# point takes a mode argument.  The rows that run fixed-order programs reuse the
# programs of the deterministic sibling package cake_dense_gemm_det_train.
# Importing this package imports nothing: ``cake_backend``, ``cake_jit`` and
# ``cake_build`` load on first attribute access.

__all__ = ["cake_backend", "cake_build", "cake_jit"]


def __getattr__(name: str):
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
