# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Quantization ops for flashinfer.experimental.sm12x.

- ``mxfp8``: BF16/FP16 rows -> dense-GEMM MXFP8 layout (one-shot CuTe kernel).
- ``nvfp4``: BF16 -> packed NVFP4 + e4m3 MMA-layout scales (TMA tile kernel;
  revived by the CUTLASS DSL 4.6 migration).
"""

from __future__ import annotations

import importlib
from typing import Any

_OP_MODULES = ("mxfp8", "nvfp4")


def __getattr__(name: str) -> Any:
    if name in _OP_MODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_OP_MODULES)


__all__ = list(_OP_MODULES)
