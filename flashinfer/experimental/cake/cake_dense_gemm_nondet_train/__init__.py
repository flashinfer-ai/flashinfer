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

Cake backend, non-deterministic: the GLM-5.2 dense projection GEMMs for training
and the FP32 router-gate GEMM on SM100 / SM103 / SM107
(flashinfer-ai/flashinfer#5677), with programs that add split-K partials in the
order they finish.  The public module is ``flashinfer.cake_dense_gemm_nondet_train``.
The mode is the package name: no entry point takes a mode argument, and no
environment variable selects it.  Importing this package imports nothing:
``cake_backend``, ``cake_jit`` and ``cake_build`` load on first attribute access.

Standalone build (flashinfer-ai/flashinfer#6553): this package needs the
deterministic backend directory ``cake_dense_gemm_det_train`` shipped next to it
(``build.json``: ``common.requires.packages``).
The rows that run fixed-order programs reuse that package's programs; this
package imports it lazily, on first use, as its sibling (the top-level package
``cake_dense_gemm_det_train`` when this package is vendored as a top-level
package).  It imports nothing else outside its own directory, and it imports
``flashinfer`` only in ``cake_build``'s guarded fallback to FlashInfer's JIT when
a library is not prebuilt.
"""

__all__ = ["cake_backend", "cake_build", "cake_jit"]


def __getattr__(name: str):
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
