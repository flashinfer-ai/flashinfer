# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""MoE ops for flashinfer.experimental.sm12x.

- ``fused_moe``: fused tensor-parallel routed-expert FFN (route -> FC1 ->
  activation -> FC2 -> scatter); recipes nvfp4/mxfp4/w4a8_mx/w4a8_nvfp4/w4a16.
- ``ep_moe``: expert-parallel MoE (replicated input -> local partial;
  cross-rank reduction is the caller's job, typically ``comm.pcie``).
"""

from __future__ import annotations

import importlib
from typing import Any

_OP_MODULES = ("fused_moe", "ep_moe")


def __getattr__(name: str) -> Any:
    if name in _OP_MODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_OP_MODULES)


__all__ = list(_OP_MODULES)
