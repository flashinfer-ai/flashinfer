"""GEMM ops for b12x.

- ``mm``: block-scaled dense matrix multiplication over ``[M,K,L]`` stacks.
- ``bmm``: dtype-dispatched batched matrix multiplication.
- ``blockscaled``: prepared dense block-scaled GEMM (NVFP4 / MXFP4 / MXFP8).
- ``block_fp8_linear``: DeepSeek-style serialized block-FP8 linear via MXFP8.
- ``bf16_vocab_projection``: prepared decode-time BF16 vocabulary projection.
- ``mxfp8_linear`` / ``tensor_fp8_linear``: compatibility aliases for
  ``blockscaled`` packed-weight calls.
- ``mla_query_projection``: fused MXFP8 MLA query projection and assembly.
- ``trellis_linear``: native EXL3 Trellis W4A16/direct-W4A8 dense linear.
- ``wo_projection``: fused MLA WO-A/WO-B projections (+ inverse-RoPE variant).
"""

from __future__ import annotations

import importlib
from typing import Any

_OP_MODULES = (
    "bf16_vocab_projection",
    "blockscaled",
    "block_fp8_linear",
    "mxfp8_linear",
    "tensor_fp8_linear",
    "mla_query_projection",
    "trellis_linear",
    "wo_projection",
)
_FUNCTIONS = {
    "mm": ("._preparation", "mm"),
    "plan": ("._preparation", "plan"),
    "DenseGemmQuery": ("._tuning", "DenseGemmQuery"),
    "DenseGemmConfig": ("._tuning", "DenseGemmConfig"),
    "bmm": ("._bmm.api", "bmm"),
    "plan_bmm": ("._bmm.api", "plan"),
    "can_implement_bmm": ("._bmm.api", "can_implement_bmm"),
    "is_bmm_supported": ("._bmm.api", "is_bmm_supported"),
}


def __getattr__(name: str) -> Any:
    if name in _OP_MODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _FUNCTIONS:
        module_name, attribute = _FUNCTIONS[name]
        value = getattr(importlib.import_module(module_name, __name__), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*_OP_MODULES, *_FUNCTIONS])


__all__ = [*_FUNCTIONS, *_OP_MODULES]
