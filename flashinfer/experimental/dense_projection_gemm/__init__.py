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

# Experimental Cake backend: the GLM-5.2 dense projection GEMMs for training
# and the FP32 router-gate GEMM on SM100 / SM103 / SM107 with an explicit
# numerical mode (flashinfer-ai/flashinfer#5677).  Every entry point of
# ``cake_backend`` takes the keyword-only ``deterministic: bool = True``: the
# default calls the deterministic package
# (``flashinfer.experimental.dense_projection_det_gemm``: the same programs and
# the same bits as before this package existed); ``deterministic=False`` selects
# the programs of this package, which add split-K partials in the order they
# finish.  Importing this package imports nothing: ``cake_backend``,
# ``cake_jit`` and ``cake_build`` load on first attribute access.

__all__ = ["cake_backend", "cake_build", "cake_jit"]


def __getattr__(name: str):
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
