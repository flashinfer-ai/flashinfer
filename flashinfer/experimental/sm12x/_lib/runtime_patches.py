# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/cute/runtime_patches.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
from __future__ import annotations

import inspect
import linecache
import os
from functools import wraps
from typing import Any

_COMPILE_ONLY_CACHE_WARNING = "Cache is disabled as user wants to compile only."
_WARNING_PATCHED = False
_MEMORY_DEBUG_PATCHED = False
_DIRECT_FRAMEINFO_PATCHED = False
_MEMORY_DEBUG_SNAPSHOT = {
    "free": None,
    "total": None,
    "used": None,
    "torch_allocated": None,
    "torch_reserved": None,
    "external": None,
    "device": None,
}


def apply_cutlass_runtime_patches() -> None:
    if _env_flag("FLASHINFER_EXP_SM12X_DISABLE_CUTLASS_RUNTIME_PATCHES", default=False):
        return
    _apply_compile_only_warning_patch()
    _apply_memory_debug_patch()
    _apply_direct_frameinfo_patch()


def cutlass_runtime_patch_status() -> tuple[tuple[str, bool], ...]:
    """Return effective patch state for compile-manifest provenance."""
    return (
        ("compile_only_warning", _WARNING_PATCHED),
        ("memory_debug", _MEMORY_DEBUG_PATCHED),
        ("direct_frameinfo", _DIRECT_FRAMEINFO_PATCHED),
    )


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _apply_compile_only_warning_patch() -> None:
    global _WARNING_PATCHED
    if _WARNING_PATCHED:
        return

    try:
        from cutlass.base_dsl.dsl import BaseDSL
    except Exception:
        return

    original_print_warning = BaseDSL.print_warning
    original_print_warning_once = BaseDSL.print_warning_once

    @wraps(original_print_warning)
    def patched_print_warning(self, message):
        if message == _COMPILE_ONLY_CACHE_WARNING:
            return None
        return original_print_warning(self, message)

    @wraps(original_print_warning_once)
    def patched_print_warning_once(self, message):
        if message == _COMPILE_ONLY_CACHE_WARNING:
            return None
        return original_print_warning_once(self, message)

    BaseDSL.print_warning = patched_print_warning
    BaseDSL.print_warning_once = patched_print_warning_once
    _WARNING_PATCHED = True


def _apply_memory_debug_patch() -> None:
    global _MEMORY_DEBUG_PATCHED
    if _MEMORY_DEBUG_PATCHED:
        return
    if _env_flag("CUTLASS_DSL_CUDA_MEMORY_DEBUG", default=False):
        return

    try:
        from cutlass.base_dsl.runtime import cuda as cuda_helpers
    except Exception:
        return
    if not hasattr(cuda_helpers, "_memory_debug_snapshot") or not hasattr(
        cuda_helpers, "_memory_debug_log"
    ):
        _MEMORY_DEBUG_PATCHED = True
        return
    if getattr(cuda_helpers, "_sm12x_memory_debug_patched", False):
        _MEMORY_DEBUG_PATCHED = True
        return

    if not hasattr(cuda_helpers, "_sm12x_original_memory_debug_snapshot"):
        cuda_helpers._sm12x_original_memory_debug_snapshot = (
            cuda_helpers._memory_debug_snapshot
        )
    if not hasattr(cuda_helpers, "_sm12x_original_memory_debug_log"):
        cuda_helpers._sm12x_original_memory_debug_log = cuda_helpers._memory_debug_log

    def _empty_memory_debug_snapshot() -> dict[str, int | None]:
        return dict(_MEMORY_DEBUG_SNAPSHOT)

    def _empty_memory_debug_log(
        label: str, before: dict[str, int | None] | None = None
    ) -> None:
        return None

    cuda_helpers._memory_debug_snapshot = _empty_memory_debug_snapshot
    cuda_helpers._memory_debug_log = _empty_memory_debug_log
    cuda_helpers._sm12x_memory_debug_patched = True
    _MEMORY_DEBUG_PATCHED = True


class _DirectFrameInfoInspectProxy:
    """Preserve CUTLASS source locations without ``inspect.findsource``.

    CUTLASS asks for frame information once per emitted DSL operation.  The
    standard ``inspect.getframeinfo(..., context=1)`` implementation first
    scans backwards from the current line to rediscover the enclosing Python
    function.  That is quadratic for large monolithic kernels: dynamic W4A8
    emits roughly 41k operations from a function spanning about 4k lines.

    The code position already identifies the exact source line.  Fetching its
    context directly from ``linecache`` preserves CUTLASS's filename, line,
    column, function, and source-text location while avoiding the backwards
    regular-expression scan.
    """

    _sm12x_direct_frameinfo = True

    def __init__(self, inspect_module: Any) -> None:
        self._inspect = inspect_module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inspect, name)

    def getframeinfo(self, frame: Any, context: int = 1) -> Any:
        frame_info = self._inspect.getframeinfo(frame, context=0)
        if context <= 0:
            return frame_info

        lines = linecache.getlines(frame_info.filename)
        if not lines:
            code_context = None
            index = None
        else:
            start = frame_info.lineno - 1 - context // 2
            start = max(0, min(start, len(lines) - context))
            code_context = lines[start : start + context]
            index = frame_info.lineno - 1 - start

        trace_args = (
            frame_info.filename,
            frame_info.lineno,
            frame_info.function,
            code_context,
            index,
        )
        if hasattr(frame_info, "positions"):
            return inspect.Traceback(
                *trace_args,
                positions=frame_info.positions,
            )
        # Python 3.10 predates PEP 657's ``positions`` field; CUTLASS already
        # handles that legacy Traceback shape by using lineno and column zero.
        return inspect.Traceback(*trace_args)


def _apply_direct_frameinfo_patch() -> None:
    global _DIRECT_FRAMEINFO_PATCHED
    if _DIRECT_FRAMEINFO_PATCHED:
        return

    try:
        # CUTLASS DSL 4.6 promotes the MLIR helpers to the top-level package.
        from cutlass._mlir_helpers import op as op_helpers
    except Exception:
        return

    current_inspect = op_helpers.inspect
    if getattr(current_inspect, "_sm12x_direct_frameinfo", False):
        _DIRECT_FRAMEINFO_PATCHED = True
        return

    op_helpers.inspect = _DirectFrameInfoInspectProxy(current_inspect)
    _DIRECT_FRAMEINFO_PATCHED = True
