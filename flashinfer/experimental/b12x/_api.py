"""b12x — consumer-Blackwell (SM120/SM121) kernels.

CuTe-DSL and Triton kernels for NVFP4/MXFP4/MXFP8 GEMM, fused MoE, attention
(paged, dense/sparse/compressed MLA, DSA indexing, and QSA decode),
quantization, multi-stream residual mixing, recurrent/sequence features, and
PCIe collectives. One grammar everywhere:

- ops live at ``b12x.<group>.<op>`` and declare themselves via ``META``;
- operations declare ``Plan`` values without allocating or compiling;
- ``PreparationSession`` selects, compiles and primes each ``Plan``, which then
  carries its own prepared state and a stable integer ``handle``;
- binding and custom ops consume the prepared ``Plan``; run and graph replay
  never resolve kernels.

Importing this module is cheap and side-effect free; kernels, cutlass, and
torch custom ops load on first op use.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from ._lib.meta import OpMeta
from ._lib.runtime_control import (
    KernelResolutionFrozenError,
    kernel_resolution_frozen,
)

# Static logical-op registry, kept in lockstep with public op directories and
# the explicit private-module overrides below by tests/test_registry.py.
_OPS: tuple[str, ...] = (
    "attention.paged",
    "attention.dense_mla",
    "attention.sparse_mla",
    "attention.compressed_sparse_mla",
    "attention.mla_compress",
    "attention.dsa_indexer",
    "attention.mla_compress",
    "attention.qsa",
    "attention.varlen",
    "comm.pcie",
    "comm.roce",
    "gemm.bf16_gemv",
    "gemm.bf16_vocab_projection",
    "gemm.blockscaled",
    "gemm.block_fp8_linear",
    "gemm.bmm",
    "gemm.mxfp8_linear",
    "gemm.tensor_fp8_linear",
    "gemm.mla_query_projection",
    "gemm.trellis_linear",
    "gemm.wo_projection",
    "moe.fused_moe",
    "moe.ep_moe",
    "norm.hyperconnection",
    "norm.mhc",
    "quantization.mxfp8",
    "quantization.nvfp4",
    "sequence.ple_hash",
    "sequence.ple_embedding",
    "sequence.ple",
    "sequence.engram",
    "sequence.embedding",
    "sequence.gdn_decode",
    "sequence.kda_prefill",
    "sequence.gdn_prefill",
    "sequence.mtp_feedback",
    "sequence.engram",
)

# A group-level function cannot share its name with an imported child module.
# These registry entries keep their public qualname while their metadata and
# implementation live under a private package.
_OP_MODULE_OVERRIDES: dict[str, str] = {
    "gemm.bmm": "gemm._bmm",
}
_CACHE_CLEAR_OVERRIDES: dict[str, str] = {
    "gemm.bmm": "clear_bmm_caches",
}

_GROUPS = (
    "attention",
    "comm",
    "gemm",
    "moe",
    "norm",
    "quantization",
    "sequence",
)
_LAZY_ROOT_ATTRS: dict[str, tuple[str, str]] = {
    # public name -> (module, attribute)
    "ScratchBufferSpec": ("._lib.scratch", "ScratchBufferSpec"),
    "Plan": (".preparation", "Plan"),
    "PreparationRequest": (".preparation", "PreparationRequest"),
    "PreparationSession": (".preparation", "PreparationSession"),
}


def _op_module_path(qualname: str) -> str:
    return _OP_MODULE_OVERRIDES.get(qualname, qualname)


def list_ops() -> tuple[OpMeta, ...]:
    """Import every op's (cheap) ``__init__`` and return their ``META``s."""
    return tuple(
        importlib.import_module(f".{_op_module_path(op_path)}", __name__).META
        for op_path in _OPS
    )


def find_op(qualname: str) -> OpMeta:
    """Look up one op's ``META`` by ``"<group>.<op>"`` qualname."""
    if qualname not in _OPS:
        raise KeyError(
            f"unknown experimental b12x op {qualname!r}; known ops: {sorted(_OPS)}"
        )
    return importlib.import_module(f".{_op_module_path(qualname)}", __name__).META


def clear_all_caches() -> None:
    """Clear caches of every op already imported; never forces imports."""
    for op_path in _OPS:
        module_path = _op_module_path(op_path)
        api = sys.modules.get(f"{__name__}.{module_path}.api")
        clear_name = _CACHE_CLEAR_OVERRIDES.get(op_path, "clear_caches")
        clear = getattr(api, clear_name, None) if api is not None else None
        if clear is not None:
            clear()
    compiler = sys.modules.get(f"{__name__}._lib.compiler")
    if compiler is not None:
        compiler.clear_compile_cache()


def __getattr__(name: str) -> Any:
    if name in _GROUPS:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _LAZY_ROOT_ATTRS:
        module_name, attr = _LAZY_ROOT_ATTRS[name]
        value = getattr(importlib.import_module(module_name, __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*__all__, *_GROUPS])


__all__ = [
    "KernelResolutionFrozenError",
    "OpMeta",
    "ScratchBufferSpec",  # noqa: F822 (resolved by __getattr__).
    "Plan",  # noqa: F822 (resolved by __getattr__).
    "PreparationRequest",  # noqa: F822 (resolved by __getattr__).
    "PreparationSession",  # noqa: F822 (resolved by __getattr__).
    "clear_all_caches",
    "find_op",
    "kernel_resolution_frozen",
    "list_ops",
]
