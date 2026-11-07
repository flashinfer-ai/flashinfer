"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Validated problem matrix of the experimental Cake per-token NVFP4 backend
(``flashinfer.experimental.cake_nvfp4_per_token``).

:func:`validated_problems` lists the ``(kind, K, N, M, x_is_bf16, out_f16,
fold)`` rows the quantizer and the GEMM were validated on (the matrix the
package README describes), and :func:`required_kernel_keys` runs the
backend's own planner (``cake_backend.quant_plan`` / ``cake_backend.gemm_plan``)
over those rows to enumerate the kernels the matrix launches on one
architecture at its validated SM count (:data:`ARCH_SM_COUNT`); a plan that runs
the nearest registered program stands for the unregistered key it replaces
(``unregistered_key``), so the matrix never relies on that fallback.  The tests in
``tests/gemm/test_cake_mm_fp4.py`` and
``tests/quantization/test_cake_nvfp4_quantize.py`` assert that every such key
is registered in ``cake_jit.KERNELS``; nothing in the package reads these
names.
"""

from __future__ import annotations

from typing import Optional

from flashinfer.experimental.cake_nvfp4_per_token.cake_backend import (
    gemm_plan,
    quant_plan,
)

# Streaming multiprocessors of the parts the dispatch was validated on
# (sm_100a: B200; sm_103a: GB300).  The GEMM tactic depends on the SM count;
# :func:`required_kernel_keys` enumerates the registered kernels for these
# counts.
ARCH_SM_COUNT = {"sm_100a": 148, "sm_103a": 152}

# Validated problem matrix (kernel keys enumerated by :func:`required_kernel_keys`).
# GEMM families are ``(K, N)`` of the per-token quantize + GEMM chains the backend was
# measured on; rows are the token counts of the validated matrix (tails 17 /
# 130 / 257 included) and the ragged correctness rows.
VALIDATED_GEMM_FAMILIES: tuple[tuple[int, int], ...] = (
    (7168, 2112),
    (7168, 1536),
    (16384, 7168),
    (7168, 18432),
    (18432, 7168),
    (8192, 8192),
    (8192, 28672),
    (28672, 8192),
)
VALIDATED_ROWS: tuple[int, ...] = (1, 8, 32, 128, 512, 2048, 8192, 17, 130, 257)
VALIDATED_RAGGED_ROWS: tuple[int, ...] = (3, 1000, 4097)
VALIDATED_QUANTIZE_K: tuple[int, ...] = (7168, 8192, 16384, 18432, 28672)
# Correctness-only GEMM rows beyond the bf16 no-fold matrix: ragged M on two
# families, fp16 activations and a folded output scale on the first family.
VALIDATED_RAGGED_FAMILIES: tuple[tuple[int, int], ...] = ((7168, 2112), (8192, 8192))
VALIDATED_F16_INPUT_ROWS: tuple[int, ...] = (8, 257, 2048)
VALIDATED_FOLD_ROWS: tuple[int, ...] = (1, 130, 8192)
# Coverage rows: the small token counts between the swapped-orientation tiles
# (M in 9..16 on five families), a (K, N) pair outside the measured families
# and quantizer K values outside the measured families (K % 16 == 0 only).
VALIDATED_SMALL_M_ROWS: tuple[int, ...] = (9, 12, 16)
VALIDATED_SMALL_M_FAMILIES: tuple[tuple[int, int], ...] = (
    (7168, 2112),
    (8192, 8192),
    (7168, 18432),
    (8192, 28672),
    (16384, 7168),
)
VALIDATED_OFF_MATRIX_FAMILY: tuple[int, int] = (4096, 3200)
VALIDATED_OFF_MATRIX_ROWS: tuple[int, ...] = (16, 300)
VALIDATED_OFF_MATRIX_QUANTIZE_K: tuple[int, ...] = (2688, 4096)
VALIDATED_OFF_MATRIX_QUANTIZE_ROWS: tuple[int, ...] = (1, 16, 130, 2048, 8192)


def validated_problems() -> tuple[tuple[str, int, int, int, bool, bool, bool], ...]:
    """``(kind, K, N, M, x_is_bf16, out_f16, fold)`` rows of the validated matrix
    (``kind`` ``"gemm"`` = quantizer + GEMM chain, ``"quantize"`` = quantizer only)."""
    rows: list[tuple[str, int, int, int, bool, bool, bool]] = []
    for k, n in VALIDATED_GEMM_FAMILIES:
        for m in VALIDATED_ROWS:
            for out_f16 in (False, True):
                rows.append(("gemm", k, n, m, True, out_f16, False))
    for k, n in VALIDATED_RAGGED_FAMILIES:
        for m in VALIDATED_RAGGED_ROWS:
            rows.append(("gemm", k, n, m, True, False, False))
    k, n = VALIDATED_GEMM_FAMILIES[0]
    for m in VALIDATED_F16_INPUT_ROWS:
        rows.append(("gemm", k, n, m, False, False, False))
    for m in VALIDATED_FOLD_ROWS:
        rows.append(("gemm", k, n, m, True, False, True))
    for k in VALIDATED_QUANTIZE_K:
        for m in VALIDATED_ROWS:
            for fold in (False, True):
                rows.append(("quantize", k, 0, m, True, False, fold))
    for k, n in VALIDATED_SMALL_M_FAMILIES:
        for m in VALIDATED_SMALL_M_ROWS:
            rows.append(("gemm", k, n, m, True, False, False))
    k, n = VALIDATED_OFF_MATRIX_FAMILY
    for m in VALIDATED_OFF_MATRIX_ROWS:
        rows.append(("gemm", k, n, m, True, False, False))
    for k in VALIDATED_OFF_MATRIX_QUANTIZE_K:
        for m in VALIDATED_OFF_MATRIX_QUANTIZE_ROWS:
            for fold in (False, True):
                rows.append(("quantize", k, 0, m, True, False, fold))
    return tuple(rows)


def required_kernel_keys(arch: str, sm_count: Optional[int] = None) -> tuple[str, ...]:
    """Every logical kernel the validated matrix launches on ``arch`` (``sm_count``
    defaults to :data:`ARCH_SM_COUNT`), quantizer keys first, in first-use order."""
    if sm_count is None:
        sm_count = ARCH_SM_COUNT[arch]
    keys: list[str] = []
    for kind, k, n, m, x_bf16, out_f16, fold in validated_problems():
        plans = [quant_plan(m, k, x_bf16, fold, arch, sm_count)]
        if kind == "gemm":
            plans.append(gemm_plan(m, n, k, out_f16, arch, sm_count))
        plan_keys = [plan.unregistered_key or plan.kernel_key for plan in plans]
        for key in plan_keys:
            if key not in keys:
                keys.append(key)
    return tuple(keys)
