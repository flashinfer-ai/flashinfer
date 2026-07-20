# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""flashinfer.experimental.sm12x — consumer-Blackwell (SM120/SM121) kernels.

CuTe-DSL kernels for NVFP4/MXFP4/MXFP8 GEMM, fused MoE, attention (paged,
sparse/compressed MLA, NSA indexing), quantization, mHC residual, and PCIe
collectives, ported from the b12x project.  One grammar everywhere:

- ops live at ``sm12x.<group>.<op>`` and declare themselves via ``META``;
- planned ops share the lifecycle ``Caps -> plan() -> bind() ->
  run*()`` (``plan`` may allocate; ``bind`` builds views only and never
  allocates; ``run*`` is CUDA-graph-capture safe);
- one-shot ops are plain functions; comm collectives are classes.

Serving controls (`freeze_kernel_resolution` & friends) live here at the
arch root because they guard the shared compiler: warm every kernel shape,
then freeze so a cache miss raises instead of compiling inside a live
request or graph capture.

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
    compilation_frozen,
    freeze_compilation,
    freeze_kernel_resolution,
    kernel_resolution_frozen,
    unfreeze_compilation,
    unfreeze_kernel_resolution,
)

# Static op registry: one entry per op directory, kept in lockstep by
# tests/experimental/test_registry.py.  Grows as port phases land.
_OPS: tuple[str, ...] = ()

_GROUPS = ("attention", "comm", "gemm", "moe", "norm", "quantization")
_LAZY_ROOT_ATTRS: dict[str, tuple[str, str]] = {
    # public name -> (module, attribute)
    "ScratchBufferSpec": ("._lib.scratch", "ScratchBufferSpec"),
}


def list_ops() -> tuple[OpMeta, ...]:
    """Import every op's (cheap) ``__init__`` and return their ``META``s."""
    return tuple(
        importlib.import_module(f".{op_path}", __name__).META for op_path in _OPS
    )


def find_op(qualname: str) -> OpMeta:
    """Look up one op's ``META`` by ``"<group>.<op>"`` qualname."""
    if qualname not in _OPS:
        raise KeyError(
            f"unknown experimental sm12x op {qualname!r}; known ops: {sorted(_OPS)}"
        )
    return importlib.import_module(f".{qualname}", __name__).META


def clear_all_caches() -> None:
    """Clear caches of every op already imported; never forces imports."""
    for op_path in _OPS:
        api = sys.modules.get(f"{__name__}.{op_path}.api")
        clear = getattr(api, "clear_caches", None) if api is not None else None
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
    "ScratchBufferSpec",
    "clear_all_caches",
    "compilation_frozen",
    "find_op",
    "freeze_compilation",
    "freeze_kernel_resolution",
    "kernel_resolution_frozen",
    "list_ops",
    "unfreeze_compilation",
    "unfreeze_kernel_resolution",
]
