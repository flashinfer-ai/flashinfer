# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""SM8x per-tensor FP8 ``mm_fp8`` / ``bmm_fp8`` on the W4A16 MoE records GEMM's dense path.

SM8x has no FP8 MMA. ``prep_f8`` restores ``A`` (E4M3 or E5M2) to BF16 exactly, and ``gemm_f8``
reads the FP8 weight as given (E4M3 or E5M2: the unprocessed column-major ``B``, or the trtllm
backend's prepared ``(K / 128, N, 128)`` weight in its row shuffle) and writes
``A_scale * B_scale * (A @ B)`` in BF16 or FP16, the scales' product formed first, as cuBLAS
and cuDNN apply them. Every backend name keeps its SM100+ requirements and runs here; on SM89
the native cuBLAS and cuDNN backends keep their own runners.
"""

from typing import List, Sequence

import torch

from ..autotuner import OptimizationProfile, TunableRunner
from ..fused_moe.core import _w4a16_moe_sm80_module
from .sm8x_fp4_dense import _out_f16_module, _pad

_E4M3, _E5M2 = torch.float8_e4m3fn, torch.float8_e5m2
_FMT = {_E4M3: 1, _E5M2: 2}
_I32_MAX = 2**31 - 1


def _weight(b: torch.Tensor, k: int, n: int, shuffled: bool):
    """``(flat uint8 weight, row, k-step, batch strides)`` of ``b``: the trtllm ``(K / 128, N,
    128)`` weight, a column-major ``(batch, K, N)`` weight in place, or a copy in ``(batch, N,
    K)`` order."""
    if shuffled:
        if not b.is_contiguous() or b.data_ptr() % 16:
            b = b.clone(memory_format=torch.contiguous_format)
        return b.view(torch.uint8).reshape(-1), 128, n * 128, 0
    batches = int(b.shape[0])
    s0, s1, s2 = b.stride()
    s0 = 0 if batches == 1 else s0
    if s1 == 1 and s2 % 16 == 0 and s0 % 16 == 0 and b.data_ptr() % 16 == 0:
        span = (batches - 1) * s0 + (n - 1) * s2 + k
        return b.view(torch.uint8).as_strided((span,), (1,)), s2, 128, s0
    w = b.transpose(1, 2).clone(memory_format=torch.contiguous_format)
    return w.view(torch.uint8).reshape(-1), k, 128, n * k


def _run_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: torch.Tensor,
    r: int,
    shuffled: bool,
) -> None:
    """``out = scale_a * scale_b * (a @ b)``: ``a`` ``(batch, M, K)``, ``b`` ``(batch, K, N)``
    (``shuffled``: the trtllm ``(K / 128, N, 128)`` weight, batch 1), ``out`` ``(batch, M, N)``."""
    batches, m, k = (int(x) for x in a.shape)
    n = int(out.shape[2])
    if m == 0 or batches == 0 or n == 0:
        return
    kp = _pad(k, 32)
    if batches * m * kp > _I32_MAX:  # the GEMM's activation offsets are 32-bit
        rows = max(1, _I32_MAX // kp)
        for i in range(batches):
            bi = b if shuffled else b[i : i + 1]
            for r0 in range(0, m, rows):
                _run_fp8(
                    a[i : i + 1, r0 : r0 + rows], bi, scale_a, scale_b,
                    out[i : i + 1, r0 : r0 + rows], r, shuffled,
                )  # fmt: skip
        return
    x = a.view(torch.uint8)
    if not x.is_contiguous() or x.data_ptr() % 4:
        x = x.clone(memory_format=torch.contiguous_format)
    w, rs, ks, bs = _weight(b, k, n, shuffled)
    rows, max_blocks = batches * m, batches * -(-m // r)
    ints = (1 + 2 * max_blocks, max_blocks * r)
    meta, rowmap = torch.empty(sum(ints), dtype=torch.int32, device=a.device).split(
        ints
    )
    act = torch.empty((rows, kp), dtype=torch.bfloat16, device=a.device)
    module = _w4a16_moe_sm80_module(None, False, None, False)
    module.prep_f8(
        x.reshape(-1), act, meta, rowmap, batches, k, r, max_blocks, _FMT[a.dtype]
    )
    direct = n % 8 == 0 and out.is_contiguous() and out.data_ptr() % 16 == 0
    target = (
        out.view(rows, n)
        if direct
        else torch.empty((rows, _pad(n, 8)), dtype=out.dtype, device=a.device)
    )
    gemm = _out_f16_module() if out.dtype == torch.float16 else module
    gemm.gemm_f8(
        w, scale_a, scale_b, act, meta, rowmap, target, r, n, k, max_blocks, batches, rs, ks, bs,
        _FMT[b.dtype], shuffled,
    )  # fmt: skip
    if not direct:
        out.copy_(target[:, :n].view(batches, m, n))


def _check(
    names: Sequence[str], a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, *scales
) -> bool:
    """The SM100+ runners' dtype and layout contract for ``names``, with ``a`` ``(batch, M, K)``,
    ``b`` ``(batch, K, N)`` (or the trtllm weight) and ``out`` ``(batch, M, N)`` on one device;
    True: ``b`` is the trtllm backend's prepared weight."""
    if any(t.device != a.device for t in (b, out, *scales)):
        raise ValueError("FP8 GEMM operands, scales and output must be on one device")
    if a.dtype not in _FMT or b.dtype not in _FMT:
        raise ValueError(f"FP8 GEMM requires FP8 operands, got {a.dtype} and {b.dtype}")
    k = int(a.shape[2])
    if k % 16:
        raise ValueError(f"FP8 GEMM requires K % 16 == 0, got K={k}")
    shuffled = "trtllm_low_latency" in names
    if shuffled:
        if a.dtype != _E4M3 or b.dtype != _E4M3:
            raise ValueError("trtllm_low_latency mm_fp8 requires e4m3 operands")
        if b.dim() != 3 or int(b.shape[0]) * 128 != k or int(b.shape[2]) != 128:
            raise ValueError(
                "trtllm_low_latency mm_fp8 requires the weight prepared by "
                "prepare_low_latency_gemm_weights, shape (k // 128, n, 128)"
            )
        if int(b.shape[1]) % 32:
            raise ValueError("trtllm_low_latency mm_fp8 requires N % 32 == 0")
    # cuBLAS rejects e5m2 x e5m2, cuDNN and cutedsl_low_latency take it; an empty list is an
    # explicit cutlass / cute-dsl call whose requirement already admitted the dtypes
    elif (
        a.dtype == _E5M2
        and b.dtype == _E5M2
        and names
        and not {"cudnn", "cutedsl_low_latency"} & set(names)
    ):
        raise ValueError("Unsupported combination: both A and B are e5m2")
    batches, m = int(a.shape[0]), int(a.shape[1])
    if (
        b.dim() != 3
        or (not shuffled and tuple(b.shape[:2]) != (batches, k))
        or tuple(out.shape) != (batches, m, int(b.shape[1 if shuffled else 2]))
    ):
        raise ValueError(
            f"FP8 GEMM shapes: a {tuple(a.shape)}, b {tuple(b.shape)}, out {tuple(out.shape)}"
        )
    return shuffled


class Sm8xFp8Runner(TunableRunner):
    """``_run_fp8`` behind ``fp8_gemm_sm100`` (``mm_fp8``, ``bmm_fp8``) as an autotuner
    candidate on its inputs ``[a, b, scale_a, scale_b, out, workspace]``; tactic: the row tile."""

    def __init__(self, names: Sequence[str]):
        self.names = tuple(names)

    def get_cache_key_extras(self, inputs) -> tuple:
        return (inputs[0].dtype, inputs[1].dtype, inputs[4].dtype, self.names)

    def get_valid_tactics(self, inputs, profile: OptimizationProfile) -> List:
        return [8, 16, 32, 64]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        a, b, scale_a, scale_b, out = inputs[:5]
        shuffled = _check(self.names, a, b, out, scale_a, scale_b)
        _run_fp8(
            a,
            b,
            scale_a,
            scale_b,
            out,
            64 if tactic in (None, -1) else tactic,
            shuffled,
        )
        return out
