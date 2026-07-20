# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""BF16/FP16 row quantization into the dense-GEMM MXFP8 layout (one-shot).

Writes packed e4m3 values plus both scale layouts (row-major e8m0 and the
MMA-swizzled physical layout) into caller-provided output tensors — no
allocation, CUDA-graph safe.

Example:
    from flashinfer.experimental.sm12x.quantization import mxfp8

    mxfp8.quantize_rows(x, values, scale_rows, scale_mma)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="mxfp8",
    group="quantization",
    api_style="oneshot",
    entry_points=("quantize_rows", "is_supported"),
    dtypes=("bf16", "fp16"),
    recipes=("mxfp8",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/gemm/mxfp8_quant_cute.py",),
    ),
    test_path="tests/experimental/quantization/test_mxfp8.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import is_supported, quantize_rows  # noqa: F401

install_lazy_api(globals(), META)
