"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Quantization Kernels (EXPERIMENTAL)
===================================

.. warning::
    This subpackage is **experimental** and not part of the stable FlashInfer API.
    The interfaces, implementations, and behaviors may change or be removed
    in future versions without notice. Use at your own risk for production workloads.

This subpackage hosts the backend kernel implementations behind the public
quantization APIs:

- CuTe-DSL kernels for MXFP4, MXFP8, and NVFP4 formats. These require SM100+
  (Blackwell) GPUs and the ``nvidia-cutlass-dsl`` package.
- A cuTile kernel for grouped MXFP8 quantization in the ``cutile`` subpackage
  (``cutile/mxfp8_grouped_quantize_cutile.py``). This requires SM100+ and the
  ``cuda.tile`` package and does not depend on ``nvidia-cutlass-dsl``.

The CuTe-DSL symbols are re-exported lazily (PEP 562 ``__getattr__`` below), so
importing this package does not pull in ``cutlass`` until one of those symbols
is accessed. Kernels that depend only on ``cuda.tile`` therefore avoid the
``nvidia-cutlass-dsl`` requirement entirely.
"""

import importlib
from typing import Any

# Map each lazily exported symbol to the submodule that defines it. Every
# submodule below imports ``cutlass`` at module load, so importing them eagerly
# here would force any consumer that merely touches this package -- including
# sibling kernels that have no CuTe-DSL dependency -- to require
# ``nvidia-cutlass-dsl``. Defer the import to first attribute access via
# PEP 562 so the cutlass cost is only paid when a CuTe-DSL symbol is used.
_LAZY_EXPORTS = {
    "MXFP4QuantizeLinearKernel": ".mxfp4_quantize",
    "MXFP4QuantizeSwizzledKernel": ".mxfp4_quantize",
    "mxfp4_quantize_cute_dsl": ".mxfp4_quantize",
    "MXFP8QuantizeLinearKernel": ".mxfp8_quantize",
    "MXFP8QuantizeSwizzledKernel": ".mxfp8_quantize",
    "mxfp8_quantize_cute_dsl": ".mxfp8_quantize",
    "NVFP4QuantizePerTokenKernel": ".nvfp4_quantize",
    "NVFP4QuantizeSwizzledKernel": ".nvfp4_quantize",
    "nvfp4_quantize_cute_dsl": ".nvfp4_quantize",
    "nvfp4_quantize_per_token_cute_dsl": ".nvfp4_quantize",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, name)
    # Cache on the package so later lookups skip this hook entirely.
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
