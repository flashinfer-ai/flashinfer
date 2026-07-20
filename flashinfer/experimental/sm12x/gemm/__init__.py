# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""GEMM ops for flashinfer.experimental.sm12x.

- ``blockscaled``: one-shot dense block-scaled GEMM (NVFP4 / MXFP4 / MXFP8).
- ``block_fp8_linear``: DeepSeek-style serialized block-FP8 linear via MXFP8.
- ``mxfp8_linear``: ModelOpt MXFP8 linear (one-shot).
- ``wo_projection``: fused MLA WO-A/WO-B projections (+ inverse-RoPE variant).
"""

from __future__ import annotations

import importlib
from typing import Any

_OP_MODULES = ("blockscaled", "block_fp8_linear", "mxfp8_linear", "wo_projection")


def __getattr__(name: str) -> Any:
    if name in _OP_MODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_OP_MODULES)


__all__ = list(_OP_MODULES)
