# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional IKET (In-Kernel Event Tracing) markers for the MXFP8 x MXFP4 kernel.

The markers compile to nothing unless the CuTe DSL compiler option ``iket`` is
set (``CUTE_DSL_COMPILER_OPT=iket``); capture traces with
``run-iket -o <dir> profile -- <command>``. Every call site is guarded by
``cutlass.const_expr(self.iket)`` (set from ``iket_enabled()``), so normal
builds are unchanged.

Per-K-tile wait ranges (MMA operand wait, TMA-A and TMA-B free-stage waits)
emit hundreds of thousands of events per launch, so they are recorded only on
the first ``FLASHINFER_MXFP8_MXFP4_IKET_FINE_CTAS`` CTAs (default 0: off).
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32

IKET_FINE_CTAS_ENV = "FLASHINFER_MXFP8_MXFP4_IKET_FINE_CTAS"


def iket_enabled() -> bool:
    """True when the DSL compiles IKET instrumentation (``iket`` option)."""

    tokens = os.environ.get("CUTE_DSL_COMPILER_OPT", "").split()
    return any(token.split("{", 1)[0] == "iket" for token in tokens)


def iket_fine_ctas() -> int:
    """CTAs that record per-K-tile wait ranges (IKET builds only)."""

    if not iket_enabled():
        return 0
    return max(0, int(os.environ.get(IKET_FINE_CTAS_ENV, "0") or 0))


def is_fine_cta(limit: int):
    """Device predicate: this CTA's linear index is below ``limit``."""

    bx, by, bz = cute.arch.block_idx()
    gx, gy, _ = cute.arch.grid_dim()
    return bx + gx * (by + gy * bz) < Int32(limit)


class _NoIket:
    """No-op markers for CuTe DSL builds without the IKET dialect."""

    @staticmethod
    def range_push(_name):
        return None

    @staticmethod
    def range_pop():
        return None


def load_iket() -> Any:
    """The DSL's IKET module, or no-op markers when it is unavailable."""

    for module_name in ("cutlass.cute.experimental", "cutlass.cute"):
        try:
            return importlib.import_module(module_name).iket
        except Exception:  # availability depends on the DSL build
            pass
    return _NoIket()


__all__ = ["iket_enabled", "iket_fine_ctas", "is_fine_cta", "load_iket"]
