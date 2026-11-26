# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""SM8x FP8 ``mm_fp8`` / ``bmm_fp8`` (per-tensor) and ``mm_mxfp8`` / ``bmm_mxfp8`` on the W4A16
MoE records GEMM's dense path.

SM8x has no FP8 MMA. ``prep_f8`` restores ``A`` (E4M3 or E5M2) to BF16 exactly, and ``gemm_f8``
reads the FP8 weight as given (E4M3 or E5M2: the unprocessed column-major ``B``, or the trtllm
backend's prepared weight in its row shuffle) and writes ``A_scale * B_scale * (A @ B)`` in BF16
or FP16, the scales' product formed first, as cuBLAS and cuDNN apply them; MXFP8 applies the
E8M0 block scales of ``A`` and ``B`` (128x4-swizzled, ``A``'s 8x4 with ``use_8x4_sf_layout``)
per 32-element k block; groupwise (``gemm_fp8_nt_groupwise``, ``gemm_fp8_nt_blockscaled``) the
FP32 scales of ``A``'s 1- or 128-row groups and ``B``'s 128-row groups per 128-element k step;
``group_deepgemm_fp8_nt_groupwise`` and ``batch_deepgemm_fp8_nt_groupwise`` the UE8M0 exponents
the SM100 DeepGEMM takes of their scales, as MXFP8 scales, the rows routed by ``m_indices`` or
``masked_m``; ``group_gemm_mxfp8_mxfp4_nt_groupwise`` the MXFP4 weights on the dense
``mm_fp4`` GEMM (``gemm_fp4``) with ``A`` restored to BF16 in its routed row order, as
``masked_scaled_bmm`` and ``ragged_scaled_bmm`` their FP4 weights (MXFP8: ``gemm_f8``).
``mm_bf16_fp8`` and ``mm_fp16_fp8`` run ``gemm_f8`` on the 16-bit activation with the E4M3
weight restored exactly in its type, the bias added to the FP32 sum before the one rounding.
Every backend name keeps its SM100+ requirements and runs here; on SM89 the native per-tensor
cuBLAS and cuDNN backends keep their own runners.
"""

import functools
from typing import List, Optional, Sequence, Tuple

import torch

from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..fused_moe.core import _w4a16_moe_sm80_module
from .sm8x_fp4_dense import _ones, _out_module, _pad

_E4M3, _E5M2 = torch.float8_e4m3fn, torch.float8_e5m2
_FMT = {_E4M3: 1, _E5M2: 2}
_OUT = (torch.bfloat16, torch.float16, torch.float32)  # the GEMM2 output modules
_FP4_FMT = {"nvfp4": 0, "mxfp4": 5, "mixed": None}  # the prep's A format (mixed: FP8)
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


def _restack(sf: torch.Tensor, batches: int, rows: int, k: int, g: int) -> torch.Tensor:
    """The ``g``x4-swizzled E8M0 scales ``sf`` of ``batches`` stacked ``rows``-row operands
    (``mxfp8_quantize`` of the 3-D tensor) with each batch starting at a ``g``-row atom."""
    nka, p = -(-k // 128), _pad(rows, g)
    if sf.numel() < _pad(batches * rows, g) * nka * 4:
        raise ValueError(
            "MXFP8 scales are shorter than their operand's swizzled layout"
        )
    t = torch.arange(batches * rows, device=sf.device)
    j = torch.arange(nka * 4, device=sf.device)
    jo = j // 4 * g * 4 + j % 4

    def off(x: torch.Tensor) -> torch.Tensor:
        within = x % 32 * 16 + x % 128 // 32 * 4 if g == 128 else x % 8 * 4
        return x // g * nka * g * 4 + within

    out = torch.zeros(batches * p * nka * 4, dtype=torch.uint8, device=sf.device)
    out[off(t // rows * p + t % rows)[:, None] + jo] = sf.reshape(-1)[
        off(t)[:, None] + jo
    ]
    return out


def _contiguous(*named: Tuple[str, Optional[torch.Tensor]]) -> None:
    for name, t in named:
        if t is not None and not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def _aligned(t: torch.Tensor, align: int) -> torch.Tensor:
    if not t.is_contiguous() or t.data_ptr() % align:
        t = t.clone(memory_format=torch.contiguous_format)
    return t.reshape(-1)


def _group_rows(
    starts: torch.Tensor,
    ends: torch.Tensor,
    rows: int,
    r: int,
    max_blocks: int,
    meta: torch.Tensor,
    rowmap: torch.Tensor,
    perm: Optional[torch.Tensor] = None,
) -> None:
    """``meta`` and ``rowmap`` of ``gemm_f8`` routing rows ``starts[i] .. ends[i]`` (disjoint,
    clamped to ``0 .. rows``; ``perm``: rows ``perm[starts[i]] .. perm[ends[i] - 1]``) to batch
    ``i`` in blocks of ``r`` rows, on the device (no host sync)."""
    groups = int(starts.shape[0])
    s = starts.clamp(0, rows)
    sizes = (ends.clamp(0, rows) - s).clamp_(min=0)
    nbg = (sizes + r - 1) // r
    end = nbg.cumsum(0)
    blk = torch.arange(max_blocks, device=s.device)
    e = torch.searchsorted(end, blk, right=True).clamp_(max=groups - 1)
    local = blk - end[e] + nbg[e]
    cnt = (sizes[e] - local * r).clamp_(0, r)
    j = torch.arange(r, device=s.device)
    meta[:1].copy_(end[-1:])
    meta[1 : 1 + max_blocks].copy_(e)
    meta[1 + max_blocks :].copy_(cnt)
    row = s[e, None] + local[:, None] * r + j
    if perm is not None:
        row = perm[row.clamp_(max=rows - 1)]
    rowmap.view(max_blocks, r).copy_(torch.where(j < cnt[:, None], row, -1))


def _run_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: torch.Tensor,
    r: int,
    shuffled: bool,
    mx: bool = False,
    a8x4: bool = False,
    gm: int = 0,
    m_indptr: Optional[torch.Tensor] = None,
    m_keys: Optional[torch.Tensor] = None,
    masked_m: Optional[torch.Tensor] = None,
) -> None:
    """``out = scale_a * scale_b * (a @ b)``: ``a`` ``(batch, M, K)``, ``b`` ``(batch, K, N)``
    (``shuffled``: the trtllm weight, batch 1: per-tensor ``(K / 128, N, 128)``, MXFP8 its
    rows in the shuffle), ``out`` ``(batch, M, N)``; ``mx``: the MXFP8 block scales of the
    stacked rows (``a8x4``: ``A``'s in the 8x4 layout); ``gm`` (batch 1): groupwise, the FP32
    scales ``scale_a`` ``[ceil(M / gm)][ceil(K / 128)]`` and ``scale_b`` ``[batch][ceil(N /
    128)][ceil(K / 128)]`` (any strides). ``a`` and ``out`` batch 1 routed to ``b[i]``:
    ``m_indptr`` (groupwise) rows ``m_indptr[i] .. m_indptr[i + 1]``, ``m_keys`` (MXFP8,
    ``scale_b`` each group's at a 128-row atom) the rows of key ``i``; ``masked_m``: the first
    ``masked_m[i]`` rows of batch ``i``. Rows routed nowhere are left unwritten."""
    batches, m, k = (int(x) for x in a.shape)
    n = int(out.shape[2])
    routed = m_indptr is not None or m_keys is not None
    groups = int(b.shape[0]) if routed else batches
    if m == 0 or batches == 0 or n == 0 or groups == 0:
        return
    kp, nka, sf_bstride = _pad(k, 32), -(-k // 128), 0
    if mx:
        scale_a, scale_b = _aligned(scale_a, 4), _aligned(scale_b, 16)
        if groups > 1:
            sf_bstride = _pad(n, 128) * nka * 4
            if batches > 1 and n % 128:  # each batch's weight scales at a 128-row atom
                scale_b = _restack(scale_b, batches, n, k, 128)
    g = (8 if a8x4 else 128) if mx else max(gm, 1)
    # the GEMM's activation offsets are 32-bit: one batch (and g-row atom chunk) per call
    if batches * m * kp > _I32_MAX and (batches > 1 or m > g):
        if mx and batches > 1 and m % g:
            scale_a = _restack(scale_a, batches, m, k, g)
        mp, rows = _pad(m, g), max(g, _I32_MAX // kp // g * g)
        for i in range(batches):
            bi = b if (shuffled and not mx) or routed else b[i : i + 1]
            sb = scale_b[i * sf_bstride :] if mx and not routed else scale_b
            for r0 in range(0, m, rows):
                sa = (
                    scale_a[(i * mp + r0) * nka * 4 :]
                    if mx
                    else scale_a[r0 // g :]
                    if gm
                    else scale_a
                )
                _run_fp8(
                    a[i : i + 1, r0 : r0 + rows], bi, sa, sb,
                    out[i : i + 1, r0 : r0 + rows], r, shuffled, mx, a8x4, gm,
                    None if m_indptr is None else m_indptr - r0,
                    None if m_keys is None else m_keys[r0 : r0 + rows],
                    None if masked_m is None else masked_m[i : i + 1] - r0,
                )  # fmt: skip
        return
    x = a.view(torch.uint8)
    if not x.is_contiguous() or x.data_ptr() % 4:
        x = x.clone(memory_format=torch.contiguous_format)
    w, rs, ks, bs = _weight(b, k, n, shuffled and not mx)
    rows, max_blocks = batches * m, batches * -(-m // r)
    if routed:  # each group's last block partial: at most M / r + groups
        max_blocks = m // r + groups
    ints = (1 + 2 * max_blocks, max_blocks * r)
    meta, rowmap = torch.empty(sum(ints), dtype=torch.int32, device=a.device).split(
        ints
    )
    act = torch.empty((rows, kp), dtype=torch.bfloat16, device=a.device)
    module = _w4a16_moe_sm80_module(None, False, None, False)
    module.prep_f8(
        x.reshape(-1), act, meta, rowmap, batches, k, r, max_blocks, _FMT[a.dtype]
    )
    if m_indptr is not None:  # the offsets made nondecreasing: disjoint groups
        ip = torch.cummax(m_indptr.clamp(0, m).long(), 0).values
        _group_rows(ip[:-1], ip[1:], m, r, max_blocks, meta, rowmap)
    elif m_keys is not None:  # rows ordered by key, keys outside 0 .. groups - 1 last
        keys = torch.where((m_keys >= 0) & (m_keys < groups), m_keys, groups).long()
        sk, perm = torch.sort(keys, stable=True)
        ip = torch.searchsorted(sk, torch.arange(groups + 1, device=a.device))
        _group_rows(ip[:-1], ip[1:], m, r, max_blocks, meta, rowmap, perm)
    elif masked_m is not None:
        st = torch.arange(batches, device=a.device) * m
        ends = st + masked_m.long().clamp(0, m)
        _group_rows(st, ends, rows, r, max_blocks, meta, rowmap)
    direct = n % 8 == 0 and out.is_contiguous() and out.data_ptr() % 16 == 0
    target = (
        out.view(rows, n)
        if direct
        else torch.empty((rows, _pad(n, 8)), dtype=out.dtype, device=a.device)
    )
    if not direct and (routed or masked_m is not None):  # rows routed nowhere kept
        target[:, :n].view(batches, m, n).copy_(out)
    gemm = module if out.dtype == torch.bfloat16 else _out_module(out.dtype)
    alpha, beta, sf, asf, gsa, gsb = scale_a, scale_b, None, None, None, None
    if mx:
        alpha, beta, sf, asf = _ones(a.device), None, scale_b, scale_a
    elif gm:
        alpha, beta, gsa, gsb = _ones(a.device), None, scale_a, scale_b
    gemm.gemm_f8(
        w, alpha, beta, act, meta, rowmap, target, r, n, k, max_blocks, groups, rs, ks, bs,
        _FMT[b.dtype], shuffled, sf, asf, a8x4, sf_bstride, gsa, gsb, max(gm, 1), None,
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


class Sm8xMxfp8Runner(TunableRunner):
    """``_run_fp8`` on MXFP8 operands as an autotuner candidate behind ``mm_mxfp8`` (inputs
    ``[a, b, a_descale, b_descale, out_dtype, out, workspace]``, 2-D) and ``mxfp8_gemm_sm100``
    (``bmm_mxfp8``: ``[a, b, a_scale, b_scale, out, workspace]``, 3-D); tactic: the row tile.
    ``shuffled``: ``b`` and its scales in the trtllm backend's row shuffle; ``a8x4``: ``A``'s
    scales in the 8x4 layout."""

    def __init__(self, shuffled: bool = False, a8x4: bool = False):
        self.shuffled = shuffled
        self.a8x4 = a8x4

    def get_cache_key_extras(self, inputs) -> tuple:
        out = inputs[5 if len(inputs) == 7 else 4]
        return (inputs[0].dtype, inputs[1].dtype, out.dtype, self.shuffled, self.a8x4)

    def get_valid_tactics(self, inputs, profile: OptimizationProfile) -> List:
        return [8, 16, 32, 64]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        a, b, scale_a, scale_b = inputs[:4]
        res = out = inputs[5 if len(inputs) == 7 else 4]
        if any(t.device != a.device for t in (b, scale_a, scale_b, out)):
            raise ValueError(
                "MXFP8 GEMM operands, scales and output must be on one device"
            )
        if a.dim() == 2:
            a, b, out = a[None], b[None], out[None]
        _run_fp8(
            a, b, scale_a, scale_b, out, 64 if tactic in (None, -1) else tactic,
            self.shuffled, True, self.a8x4,
        )  # fmt: skip
        return res


def sm8x_mxfp8_runner(
    device: torch.device, shuffled: bool = False, a8x4: bool = False
) -> Sm8xMxfp8Runner:
    """The MXFP8 runner of every backend on SM8x (after the backend's requirement check)."""
    _ones(device)  # the constant alpha exists before any CUDA graph capture
    return Sm8xMxfp8Runner(shuffled, a8x4)


class Sm8xFp8GroupwiseRunner(TunableRunner):
    """``_run_fp8`` with groupwise scales behind ``gemm_fp8_nt_groupwise`` as an autotuner
    candidate on ``[a, b, a_scale, b_scale, out]`` (2-D, the scales as ``_run_fp8`` reads them,
    ``A``'s in ``gm``-row groups); tactic: the row tile."""

    def __init__(self, gm: int):
        self.gm = gm

    def get_cache_key_extras(self, inputs) -> tuple:
        return (inputs[0].dtype, inputs[4].dtype, self.gm)

    def get_valid_tactics(self, inputs, profile: OptimizationProfile) -> List:
        return [8, 16, 32, 64]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        a, b, scale_a, scale_b, out = inputs
        _run_fp8(
            a[None], b.t()[None], scale_a, scale_b[None], out[None],
            64 if tactic in (None, -1) else tactic, False, gm=self.gm,
        )  # fmt: skip
        return out


class Sm8xGroupFp8GroupwiseRunner(Sm8xFp8GroupwiseRunner):
    """``_run_fp8`` over ``m_indptr``'s row groups behind ``group_gemm_fp8_nt_groupwise`` on
    ``[a, b, a_scale, b_scale, m_indptr, out]`` (``b`` ``(batch, N, K)``, the scales as
    ``_run_fp8`` reads them); tactic: the row tile."""

    def get_cache_key_extras(self, inputs) -> tuple:
        return (inputs[0].dtype, inputs[5].dtype, self.gm)

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        a, b, scale_a, scale_b, m_indptr, out = inputs
        _run_fp8(
            a[None], b.transpose(1, 2), scale_a, scale_b, out[None],
            64 if tactic in (None, -1) else tactic, False, gm=self.gm, m_indptr=m_indptr,
        )  # fmt: skip
        return out


@functools.cache
def _groupwise_tuning_config(gm: int, out_index: int = 4) -> TuningConfig:
    from ..fused_moe.utils import (
        get_hybrid_num_tokens_buckets,
        map_to_hybrid_bucket_uncapped,
    )

    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                (0,), (0,), get_hybrid_num_tokens_buckets, map_to_hybrid_bucket_uncapped
            ),
        ),
        constraint_specs=(
            *(
                (ConstraintSpec(2, 0, lambda shapes: -(-shapes[0][0] // gm)),)
                if gm  # gm 0: per-tensor scales
                else ()
            ),
            ConstraintSpec(out_index, 0, lambda shapes: shapes[0][0]),
        ),
    )


def sm8x_gemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    scale_major_mode: Optional[str],
    mma_sm: int,
    scale_granularity_mnk: Tuple[int, int, int],
    backend: str,
) -> None:
    """``gemm_fp8_nt_groupwise`` (``gemm_fp8_nt_blockscaled``) on SM8x after the backend's
    requirement check, with its SM100+ operand contract: ``cutlass`` reads the scales in
    ``scale_major_mode`` (K: ``a_scale`` ``(M / gm, K / 128)``, ``b_scale`` ``(N / 128, K /
    128)``; MN: their transposes), ``trtllm`` the scales' memory as ``(K / 128, M)`` and ``(N /
    128, K / 128)`` row-major, ``cutile`` K-major scales of exactly those shapes."""
    m, k = (int(x) for x in a.shape)
    n, nka, gm = int(b.shape[0]), -(-k // 128), int(scale_granularity_mnk[0])
    if any(t.device != a.device for t in (b, a_scale, b_scale, out)):
        raise ValueError("FP8 GEMM operands, scales and output must be on one device")
    # cutlass reads b in a's dtype, trtllm and cutile take e4m3
    if (
        a.dtype not in _FMT
        or b.dtype != a.dtype
        or (backend != "cutlass" and a.dtype != _E4M3)
    ):
        raise ValueError(
            f"{backend} FP8 groupwise GEMM requires FP8 operands of one dtype"
            f"{'' if backend == 'cutlass' else ' (e4m3)'}, got {a.dtype} and {b.dtype}"
        )
    if k % 16:
        raise ValueError(f"FP8 GEMM requires K % 16 == 0, got K={k}")
    if backend == "cutile":
        if not (a.is_contiguous() and b.is_contiguous()):
            raise ValueError("a and b must be contiguous")
        if out.shape != (m, n):
            raise ValueError(f"out must be ({m},{n}); got {tuple(out.shape)}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if not (a_scale.dim() == 2 and b_scale.dim() == 2):
            raise ValueError("scales must be 2D")
        nb = -(-n // scale_granularity_mnk[1])
        if a_scale.shape != (m, nka):
            raise ValueError(
                f"a_scale must be ({m}, {nka}); got {tuple(a_scale.shape)}"
            )
        if b_scale.shape != (nb, nka):
            raise ValueError(
                f"b_scale must be ({nb}, {nka}); got {tuple(b_scale.shape)}"
            )
    elif tuple(out.shape) != (m, n):
        raise ValueError(f"out shape {tuple(out.shape)} != expected {(m, n)}")
    if backend == "trtllm":  # mma_sm and scale_major_mode are ignored
        if not (a.is_contiguous() and b.is_contiguous() and out.is_contiguous()):
            raise ValueError("trtllm FP8 GEMM requires contiguous a, b and out")
        if out.dtype != torch.bfloat16:
            raise ValueError("trtllm FP8 GEMM writes a bfloat16 output")
        a_scale = a_scale.as_strided((m, nka), (1, m))
        b_scale = b_scale.as_strided((-(-n // 128), nka), (nka, 1))
    elif backend != "cutile":
        if tuple(scale_granularity_mnk) not in ((1, 128, 128), (128, 128, 128)):
            raise ValueError(
                f"Unsupported scale granularity: {tuple(scale_granularity_mnk)}"
            )
        if mma_sm not in (1, 2):
            raise ValueError(f"Unsupported MMA SM: {mma_sm}")
        if scale_major_mode not in ("K", "MN"):
            raise ValueError(f"Unsupported Scale Major Mode: {scale_major_mode}")
        if scale_major_mode == "MN":
            a_scale, b_scale = a_scale.t(), b_scale.t()
    if m == 0 or n == 0:
        return
    runner = Sm8xFp8GroupwiseRunner(gm)
    inputs = [a, b, a_scale, b_scale, out]
    chosen, tactic = AutoTuner.get().choose_one(
        "gemm_fp8_nt_groupwise_sm8x", [runner], _groupwise_tuning_config(gm), inputs
    )
    chosen(inputs=inputs, tactic=tactic)


def sm8x_group_gemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    out: torch.Tensor,
    scale_granularity_mnk: Tuple[int, int, int],
    scale_major_mode: str,
    backend: str,
) -> None:
    """``group_gemm_fp8_nt_groupwise`` on SM8x after its problem-size check, with its SM100+
    operand contract: ``trtllm`` reads the scales in ``scale_major_mode`` (K: ``a_scale``
    ``(cum_m / gm, K / 128)``, ``b_scale`` ``(batch, N / 128, K / 128)``; MN: their
    transposes), ``cutile`` K-major e4m3 scales of its granularity's exact shapes."""
    cum_m, k = int(a.shape[0]), int(a.shape[-1])
    groups, n = int(b.shape[0]), int(b.shape[1])
    nka = -(-k // 128)
    if any(t.device != a.device for t in (b, a_scale, b_scale, m_indptr, out)):
        raise ValueError("FP8 GEMM operands, scales and output must be on one device")
    gm, n_g, k_g = (int(x) for x in scale_granularity_mnk)
    if backend == "cutile":
        if scale_major_mode != "K":
            raise NotImplementedError(
                f"cuTile group_gemm_fp8_nt_groupwise only supports scale_major_mode='K'; "
                f"got {scale_major_mode!r}."
            )
        if gm != 1:
            raise NotImplementedError(
                f"cuTile group_gemm_fp8_nt_groupwise requires scale_granularity_mnk[0] == 1; "
                f"got {gm}."
            )
        if not (a.is_contiguous() and b.is_contiguous()):
            raise ValueError("a and b must be contiguous")
        if a.dim() != 2:
            raise ValueError("a must be 2D (cum_m, k)")
        if b.dim() != 3:
            raise ValueError("b must be 3D (batch_size, n, k)")
        if m_indptr.dim() != 1 or m_indptr.shape[0] != groups + 1:
            raise ValueError(f"m_indptr must be 1D with {groups + 1} elements")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        ka, nb = -(-k // k_g), -(-n // n_g)
        if a_scale.dim() != 2 or a_scale.shape != (cum_m, ka):
            raise ValueError(
                f"a_scale must be ({cum_m}, {ka}); got {tuple(a_scale.shape)}"
            )
        if b_scale.dim() != 3 or b_scale.shape != (groups, nb, ka):
            raise ValueError(
                f"b_scale must be ({groups}, {nb}, {ka}); got {tuple(b_scale.shape)}"
            )
        if not (a_scale.is_contiguous() and b_scale.is_contiguous()):
            raise ValueError("a_scale and b_scale must be contiguous")
        if a.dtype != _E4M3 or b.dtype != _E4M3:
            raise ValueError(
                f"cutile FP8 group GEMM requires e4m3 operands, got {a.dtype} and {b.dtype}"
            )
        if n_g % 128 or k_g % 128:
            raise NotImplementedError(
                "cuTile group_gemm_fp8_nt_groupwise on SM8x requires N and K scale "
                f"granularities in multiples of 128; got {(gm, n_g, k_g)}."
            )
        if (n_g, k_g) != (128, 128):  # each 128-element step's scale repeated
            a_scale = a_scale.repeat_interleave(k_g // 128, 1)[:, :nka]
            b_scale = b_scale.repeat_interleave(n_g // 128, 1)
            b_scale = b_scale.repeat_interleave(k_g // 128, 2)[:, : -(-n // 128), :nka]
    else:  # mma_sm is ignored; trtllm reads b in a's dtype
        if b.dtype != a.dtype:
            raise ValueError(
                f"trtllm FP8 group GEMM requires FP8 operands of one dtype, got {a.dtype} "
                f"and {b.dtype}"
            )
        if (gm, n_g, k_g) not in ((1, 128, 128), (128, 128, 128)):
            raise ValueError(f"Unsupported scale granularity: {(gm, n_g, k_g)}")
        if scale_major_mode == "MN":
            a_scale, b_scale = a_scale.t(), b_scale.transpose(1, 2)
    if cum_m == 0 or groups == 0:
        return
    _run_group_groupwise(a, b, a_scale, b_scale, m_indptr, out, gm)


def _run_group_groupwise(a, b, a_scale, b_scale, m_indptr, out, gm: int) -> None:
    runner = Sm8xGroupFp8GroupwiseRunner(gm)
    inputs = [a, b, a_scale, b_scale, m_indptr, out]
    chosen, tactic = AutoTuner.get().choose_one(
        "group_gemm_fp8_nt_groupwise_sm8x",
        [runner],
        _groupwise_tuning_config(gm, 5),
        inputs,
    )
    chosen(inputs=inputs, tactic=tactic)


def sm8x_grouped_mm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    """``grouped_mm_fp8`` on SM80 / SM86 / SM87: ``alpha * (a @ b[e]^T)`` over cuDNN's MOE
    segments (the starts ``m_indptr[:-1]``, the last segment to ``a.shape[0]``)."""
    cum_m, k = (int(x) for x in a.shape)
    if k % 16:
        raise ValueError(f"FP8 GEMM requires K % 16 == 0, got K={k}")
    if out is None:
        out = torch.empty((cum_m, b.shape[1]), dtype=out_dtype, device=a.device)
    if cum_m and b.shape[0]:
        ip = torch.cat((m_indptr[:-1], m_indptr.new_full((1,), cum_m)))
        one = _ones(a.device)
        _run_group_groupwise(a, b, one if alpha is None else alpha, one, ip, out, 0)
    return out


def sm8x_ragged_block_scaled_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: Optional[torch.Tensor],
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    max_m_device: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
    segment_alignment: int,
) -> torch.Tensor:
    """``ragged_block_scaled_bmm`` on SM8x after its requirement check, with the cuTile
    backend's operand contract: ``a`` ``(total_m, K)``, ``b`` ``(Q, N, K)``, the FP32 block
    scales ``a_scale`` ``(total_m, K / VEC)`` (or None) and ``b_scale`` ``(Q, N / VEC, K /
    VEC)``; VEC a multiple of 128 dividing N and K here."""
    _contiguous(
        ("A matrix", a), ("B matrix", b), ("A scale matrix", a_scale),
        ("B scale matrix", b_scale), ("m_indptr", m_indptr),
    )  # fmt: skip
    total_m, k = (int(x) for x in a.shape)
    q, n, k_b = (int(x) for x in b.shape)
    if k != k_b:
        raise ValueError(f"K dimensions must match: {k} != {k_b}")
    if m_indptr.shape[0] != q + 1:
        raise ValueError(
            f"m_indptr must have Q+1 ({q + 1}) elements; got {m_indptr.shape[0]}"
        )
    q_sb, rnb, rkb = (int(x) for x in b_scale.shape)
    vec = k // rkb
    if a_scale is not None and a_scale.shape[0] != total_m:
        raise ValueError(
            f"a_scale total_m dimension mismatch: {a_scale.shape[0]} != {total_m}"
        )
    if q_sb != q:
        raise ValueError(f"b_scale Q dimension mismatch: {q_sb} != {q}")
    out_dtype = out_dtype or torch.bfloat16
    if max_m_device is not None and max_m_device.device != a.device:
        raise ValueError(
            f"max_m_device must be on {a.device}; got {max_m_device.device}"
        )
    if segment_alignment % 128:
        raise ValueError(
            f"BLOCK_M (128) must divide segment_alignment ({segment_alignment})"
        )
    if n // rnb != vec:
        raise ValueError(
            f"b_scale N granularity ({n // rnb}) must equal BLOCK_K ({vec})"
        )
    if a_scale is not None and k // int(a_scale.shape[1]) != vec:
        raise ValueError(
            f"a_scale K granularity ({k // int(a_scale.shape[1])}) must equal BLOCK_K "
            f"({vec})"
        )
    scales = (b_scale, m_indptr) + (() if a_scale is None else (a_scale,))
    if any(t.device != a.device for t in (b, *scales)):
        raise ValueError("FP8 GEMM operands, scales and output must be on one device")
    if vec % 128 or n % vec or k % vec or out_dtype not in _OUT:
        raise NotImplementedError(
            "ragged_block_scaled_bmm on SM8x requires block scales of a multiple of 128 "
            f"dividing N and K, a bfloat16, float16 or float32 output; got VEC={vec}, "
            f"N={n}, K={k}, out_dtype={out_dtype}."
        )
    out = torch.empty((total_m, n), dtype=out_dtype, device=a.device)
    if total_m == 0 or q == 0 or n == 0:
        return out
    j = vec // 128  # each 128-element step's scale repeated
    if a_scale is None:
        a_scale = _ones(a.device).expand(total_m, k // 128)
    elif j > 1:
        a_scale = a_scale.repeat_interleave(j, 1)
    if j > 1:
        b_scale = b_scale.repeat_interleave(j, 1).repeat_interleave(j, 2)
    _run_group_groupwise(a, b, a_scale, b_scale, m_indptr, out, 1)
    return out


def _ue8m0(
    sf: torch.Tensor,
    mn: int,
    k: int,
    recipe: Tuple[int, int, int],
    num_groups: Optional[int],
    is_sfa: bool,
) -> torch.Tensor:
    """The UE8M0 exponents ``[(group,) mn, ceil(K / 128)]`` of the DeepGEMM scales ``sf`` that
    the SM100 ``transform_sf_into_required_layout`` accepts: FP32 of granularity ``(1, 128)``
    or ``(128, 128)`` (its exponent, the sign dropped), or the packed int32 UE8M0 scales."""
    from ..deep_gemm import check_sf_layout

    gran = (recipe[0 if is_sfa else 1], recipe[2])
    known = gran in ((1, 128), (128, 128))
    if sf.dtype != torch.int or not known:
        check_sf_layout(sf, mn=mn, k=k, gran=gran, num_groups=num_groups)
    if not known:
        raise AssertionError(f"Unknown cases: {sf.dtype=}, {gran=}")
    if sf.dtype == torch.float:
        if gran[0] == 128:
            sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // 128)
        return (sf.view(torch.int) >> 23).to(torch.uint8)
    sf = check_sf_layout(
        sf, mn=mn, k=k, gran=(1, 128), num_groups=num_groups, tma_stride_check=True,
        type_check=torch.int,
    )  # fmt: skip
    e = sf[..., None] >> torch.arange(0, 32, 8, device=sf.device, dtype=torch.int)
    return e.flatten(-2)[..., : -(-k // 128)].to(torch.uint8)


def _mx_rows(e: torch.Tensor) -> torch.Tensor:
    """The 128x4-swizzled E8M0 scales (each 128-element step's exponent for its four 32-element
    blocks) of the exponents ``e`` ``[(group,) rows, ceil(K / 128)]``, each group's rows padded
    to a 128-row atom."""
    rows, nka = int(e.shape[-2]), int(e.shape[-1])
    x = e.new_zeros(*e.shape[:-2], _pad(rows, 128), nka)
    x[..., :rows, :] = e
    x = x.view(-1, 4, 32, nka).permute(0, 3, 2, 1)
    return x[..., None].expand(-1, -1, -1, -1, 4).reshape(-1)


class Sm8xDeepGemmFp8Runner(TunableRunner):
    """``_run_fp8`` on the DeepGEMM scales as MXFP8 behind ``group_deepgemm_fp8_nt_groupwise``
    (``[a, b, a_sf, b_sf, m_indices, out]``, 2-D ``a``) and ``batch_deepgemm_fp8_nt_groupwise``
    (``[a, b, a_sf, b_sf, masked_m, out]``, 3-D ``a``), and on the MXFP8 scales of
    ``ragged_scaled_bmm`` and ``masked_scaled_bmm``; tactic: the row tile."""

    def get_valid_tactics(self, inputs, profile: OptimizationProfile) -> List:
        return [8, 16, 32, 64]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        a, b, a_sf, b_sf, rows, out = inputs
        r = 64 if tactic in (None, -1) else tactic
        if a.dim() == 2:
            _run_fp8(
                a[None], b.transpose(1, 2), a_sf, b_sf, out[None], r, False, True,
                m_keys=rows,
            )  # fmt: skip
        else:
            _run_fp8(
                a, b.transpose(1, 2), a_sf, b_sf, out, r, False, True, masked_m=rows
            )
        return out


@functools.cache
def _deepgemm_tuning_config(masked: bool) -> TuningConfig:
    if masked:
        return TuningConfig()
    from ..fused_moe.utils import (
        get_hybrid_num_tokens_buckets,
        map_to_hybrid_bucket_uncapped,
    )

    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                (0,), (0,), get_hybrid_num_tokens_buckets, map_to_hybrid_bucket_uncapped
            ),
        ),
        constraint_specs=(
            ConstraintSpec(
                2,
                0,
                lambda shapes: _pad(shapes[0][0], 128) * -(-shapes[0][1] // 128) * 4,
            ),
            ConstraintSpec(4, 0, lambda shapes: shapes[0][0]),
            ConstraintSpec(5, 0, lambda shapes: shapes[0][0]),
        ),
    )


def sm8x_deepgemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    rows: torch.Tensor,
    out: torch.Tensor,
    recipe: Optional[Tuple[int, int, int]],
) -> None:
    """``group_deepgemm_fp8_nt_groupwise`` (``a`` ``(M, K)``, ``rows`` the ``m_indices``) and
    ``batch_deepgemm_fp8_nt_groupwise`` (``a`` ``(G, M, K)``, ``rows`` the ``masked_m``) on SM8x
    after their problem-size checks, the scales as the SM100 DeepGEMM accepts and applies them
    (UE8M0, per 128-element k step); the rows of keys outside ``0 .. G - 1`` and past
    ``masked_m`` stay unwritten."""
    masked = a.dim() == 3
    m, k = int(a.shape[-2]), int(a.shape[-1])
    groups, n = int(b.shape[0]), int(b.shape[1])
    if m == 0:
        return
    if any(t.device != a.device for t in (b, a_scale, b_scale, rows, out)):
        raise ValueError("FP8 GEMM operands, scales and output must be on one device")
    if recipe is None:  # the SM100 default recipe
        assert a_scale.dtype in (torch.float, torch.int)
        recipe = (1, 128, 128) if b_scale.dtype == torch.float else (1, 1, 128)
    ea = _ue8m0(a_scale, m, k, recipe, groups if masked else None, True)
    eb = _ue8m0(b_scale, n, k, recipe, groups, False)
    if masked and groups > 1:  # the SM100 masked kernel's 128-row block
        assert m % 128 == 0
    nka = int(ea.shape[-1])
    # A's scales of the stacked rows; B's of each group at a 128-row atom (masked: stacked,
    # _run_fp8 restacks the batches)
    a_sf = _mx_rows(ea.reshape(-1, nka))
    b_sf = _mx_rows(eb.reshape(-1, nka) if masked else eb)
    inputs = [a, b, a_sf, b_sf, rows.reshape(-1), out]
    chosen, tactic = AutoTuner.get().choose_one(
        f"{'batch' if masked else 'group'}_deepgemm_fp8_nt_groupwise_sm8x",
        [Sm8xDeepGemmFp8Runner()],
        _deepgemm_tuning_config(masked),
        inputs,
    )
    chosen(inputs=inputs, tactic=tactic)


def _run_rows_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    alpha: torch.Tensor,
    asf_off: torch.Tensor,
    out: torch.Tensor,
    r: int,
    fmt: Optional[int] = None,
) -> None:
    """``out`` rows ``starts[i] .. ends[i]`` (clamped, disjoint) ``= alpha[i] (a @ b[i]^T)`` on
    ``gemm_fp4``: ``b`` ``(groups, N, K / 2)`` FP4, its 128x4-swizzled scales (flat) group
    ``i``'s at ``i pad(N, 128)`` rows; ``a`` restored to BF16 exactly in its blocks' row order,
    packed FP4 by the prep (``fmt`` 0: NVFP4 with its scales, 5: MXFP4 without them, times
    2^119), FP8 (``fmt`` None) times 2^119 like ``mm_fp4``'s MXFP4 activation, E5M2, past BF16
    there, times 2^112, its 2^7 added back to each k32 group's scale exponent (``act_shift``);
    MX: row ``t``'s E8M0 scales those of row ``t + asf_off[i]``."""
    rows, k = int(a.shape[0]), 2 * int(b.shape[2])
    groups, n = int(b.shape[0]), int(b.shape[1])
    if rows == 0 or groups == 0 or n == 0:
        return
    max_blocks = rows // r + groups  # each group's last block partial
    ints = (1 + 2 * max_blocks, max_blocks * r, -(-rows // 2048) + 1)
    meta, rowmap, workspace = torch.empty(
        sum(ints), dtype=torch.int32, device=a.device
    ).split(ints)
    module = _w4a16_moe_sm80_module(None, False, None, False)
    asf = _aligned(a_scale.view(torch.uint8), 4)
    if fmt is not None:  # in token order, its blocks replaced by the groups' below
        tok = torch.empty((rows, k), dtype=torch.bfloat16, device=a.device)
        module.prep(
            torch.zeros((rows, 1), dtype=torch.int32, device=a.device), meta, rowmap,
            workspace, 1, r, max_blocks, _aligned(a.view(torch.uint8), 16).view(rows, -1),
            asf, tok, True, None, None, fmt, None, None, None, None, None,
        )  # fmt: skip
    _group_rows(starts, ends, rows, r, max_blocks, meta, rowmap)
    idx, e5 = rowmap.clamp(min=0), a.dtype == torch.float8_e5m2
    if fmt is None:
        act = a.view(torch.uint8).index_select(0, idx).view(a.dtype)
        act = act.to(torch.bfloat16).mul_(2.0 ** (112 if e5 else 119))
    else:
        act = tok.index_select(0, idx)
    direct = out.is_contiguous() and out.data_ptr() % 16 == 0  # N % 8 == 0
    target = out if direct else out.clone(memory_format=torch.contiguous_format)
    gemm = module if out.dtype == torch.bfloat16 else _out_module(out.dtype)
    gemm.gemm_fp4(
        _aligned(b.view(torch.uint8), 16), _aligned(b_scale.view(torch.uint8), 16), alpha,
        act, meta, rowmap, target, r, n, k, max_blocks, False, None if fmt == 0 else asf,
        False, asf_off, 7 if e5 else 0,
    )  # fmt: skip
    if not direct:
        out.copy_(target)


class Sm8xRowsFp4Runner(TunableRunner):
    """``_run_rows_fp4`` as an autotuner candidate on ``[a, b, a_scale, b_scale, starts, ends,
    alpha, asf_off, out]`` (the scales flat); tactic: the row tile."""

    def __init__(self, fmt: Optional[int]):
        self.fmt = fmt

    def get_cache_key_extras(self, inputs) -> tuple:
        return (inputs[0].dtype, inputs[8].dtype, self.fmt)

    def get_valid_tactics(self, inputs, profile: OptimizationProfile) -> List:
        return [8, 16, 32, 64]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        _run_rows_fp4(*inputs, 64 if tactic in (None, -1) else tactic, self.fmt)
        return inputs[8]


@functools.cache
def _rows_tuning_config(sf_div: int) -> TuningConfig:
    """``a``'s rows bucketed, its scales (a row's bytes / ``sf_div``) sized for SM100's per-group
    padding."""
    from ..fused_moe.utils import (
        get_hybrid_num_tokens_buckets,
        map_to_hybrid_bucket_uncapped,
    )

    return TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                (0,), (0,), get_hybrid_num_tokens_buckets, map_to_hybrid_bucket_uncapped
            ),
        ),
        constraint_specs=(
            ConstraintSpec(
                2, 0, lambda s: _pad(s[0][0] + 127 * s[1][0], 128) * (s[0][1] // sf_div)
            ),
            ConstraintSpec(8, 0, lambda s: s[0][0]),
        ),
    )


def _tuned_rows_fp4(name: str, fmt: Optional[int], inputs: list) -> None:
    div = 32 if fmt is None else 16 if fmt else 8  # FP8, MXFP4, NVFP4
    chosen, tactic = AutoTuner.get().choose_one(
        name, [Sm8xRowsFp4Runner(fmt)], _rows_tuning_config(div), inputs
    )
    chosen(inputs=inputs, tactic=tactic)


def _tuned_mx8(name: str, masked: bool, inputs: list) -> None:
    chosen, tactic = AutoTuner.get().choose_one(
        name, [Sm8xDeepGemmFp8Runner()], _deepgemm_tuning_config(masked), inputs
    )
    chosen(inputs=inputs, tactic=tactic)


def sm8x_group_gemm_mxfp8_mxfp4_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """``group_gemm_mxfp8_mxfp4_nt_groupwise`` on SM8x after its problem-size check (the tile
    arguments select no SM8x kernel): ``_run_rows_fp4`` with its row tile tuned, the E8M0 scales
    where the SM100 kernel reads them: ``b_scale`` group ``i``'s 128x4 at ``i pad(N, 128)``
    rows, ``a_scale`` group ``i``'s rows from ``(m_indptr[i] + 127 i) // 128 * 128``."""
    cum_m, groups = int(a.shape[0]), int(b.shape[0])
    if any(t.device != a.device for t in (b, a_scale, b_scale, m_indptr, out)):
        raise ValueError(
            "MXFP4 group GEMM operands, scales and output must be on one device"
        )
    if cum_m == 0 or groups == 0:
        return
    need = (cum_m + 127 * groups) // 128 * 128 * -(-int(a.shape[1]) // 128) * 4
    if a_scale.numel() < need:
        raise ValueError(f"a_scale must hold {need} scales, got {a_scale.numel()}")
    ip = torch.cummax(m_indptr.clamp(0, cum_m).long(), 0).values
    sf_off = (ip[:-1] + 127 * torch.arange(groups, device=a.device)) // 128 * 128
    _tuned_rows_fp4(
        "group_gemm_mxfp8_mxfp4_nt_groupwise_sm8x", None,
        [a, b, a_scale.reshape(-1), b_scale.reshape(-1), ip[:-1], ip[1:],
         torch.ones(groups, device=a.device), (sf_off - ip[:-1]).int(), out],
    )  # fmt: skip


def _scaled_bmm_k(block_scale_type: str, k_a: int, k_b: int) -> int:
    """K after the cuTile ``*_scaled_bmm`` check of the operands' packed K."""
    epb_a = 2 if "fp4" in block_scale_type else 1
    epb_b = 1 if block_scale_type == "mxfp8" else 2
    if k_a * epb_a != k_b * epb_b:
        raise ValueError(
            f"incompatible dimensions: K_A*ELEM_PER_BYTE_A ({k_a * epb_a}) "
            f"must match K_B*ELEM_PER_BYTE_B ({k_b * epb_b})"
        )
    return k_a * epb_a


def _nt_only(transpose_a: bool, transpose_b: bool) -> None:
    if transpose_a or not transpose_b:
        raise ValueError(
            "Only NT layout is supported (transpose_a=False, transpose_b=True)"
        )


def sm8x_masked_scaled_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    block_scale_type: str,
    max_m_device: Optional[torch.Tensor],
    transpose_a: bool,
    transpose_b: bool,
    out_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    """``masked_scaled_bmm`` on SM8x after the cuTile backend's checks (its module needs
    ``cuda.tile``): ``c[q, :masked_m[q]] = a[q] @ b[q]^T``, ``c[q]``'s other rows unwritten."""
    q, m, k_a = (
        a.shape[0],
        a.shape[2 if transpose_a else 1],
        a.shape[1 if transpose_a else 2],
    )
    q_b, n, k_b = (
        b.shape[0],
        b.shape[1 if transpose_b else 2],
        b.shape[2 if transpose_b else 1],
    )
    k = _scaled_bmm_k(block_scale_type, k_a, k_b)
    if q != q_b:
        raise ValueError(f"incompatible dimensions: Q_A ({q}) must match Q_B ({q_b})")
    _nt_only(transpose_a, transpose_b)
    _contiguous(
        ("A matrix", a), ("B matrix", b), ("A scale matrix", a_scale),
        ("B scale matrix", b_scale), ("Masked matrix", masked_m),
    )  # fmt: skip
    if masked_m.numel() != q:
        raise ValueError(f"masked_m must have the same number of elements as Q ({q})")
    if any(t.device != a.device for t in (b, a_scale, b_scale, masked_m)):
        raise ValueError(
            "Scaled BMM operands, scales and masked_m must be on one device"
        )
    masked_m = masked_m.reshape(-1)
    if block_scale_type not in ("nvfp4", "mxfp4", "mxfp8", "mixed"):
        raise ValueError(f"Invalid block scale type: {block_scale_type}")
    vec = 16 if block_scale_type == "nvfp4" else 32
    q_sa, rm, rka, m2, k256a = a_scale.shape
    q_sb, rn, rkb, n2, k256b = b_scale.shape
    if m2 != 2 or n2 != 2:
        raise ValueError("incompatible dimensions: scale dim -2 must be 2")
    if k256a != 256 or k256b != 256:
        raise ValueError("incompatible dimensions: scale dim -1 must be 256")
    if q_sa != q_sb or q_sa != q:
        raise ValueError("incompatible dimensions: scale batch must match Q")
    if rm * 128 != m or rn * 128 != n:
        raise ValueError(
            "incompatible dimensions: M/N must be a multiple of 128 matching scale shape"
        )
    if rka * 4 * vec != k or rkb * 4 * vec != k:
        raise ValueError("incompatible dimensions: K must match scale K shape")
    out_dtype = out_dtype or torch.bfloat16
    if max_m_device is not None:
        if not isinstance(max_m_device, torch.Tensor) or not max_m_device.is_cuda:
            raise ValueError(
                "max_m_device must be a CUDA tensor (a device-side copy of max_m); "
                f"got {type(max_m_device).__name__}. Pass a CUDA int32 tensor or None."
            )
        if max_m_device.device != a.device:
            raise ValueError(
                f"max_m_device must be on {a.device}; got {max_m_device.device}."
            )
    c = torch.empty((q, m, n), device=a.device, dtype=out_dtype)
    if c.numel() == 0:
        return c
    t = c if out_dtype in _OUT else torch.empty_like(c, dtype=torch.float32)
    sf = (a_scale.reshape(-1).view(torch.uint8), b_scale.reshape(-1).view(torch.uint8))
    if block_scale_type == "mxfp8":
        _tuned_mx8("masked_scaled_bmm_sm8x", True, [a, b, *sf, masked_m, t])
    else:
        st = torch.arange(q, device=a.device) * m
        _tuned_rows_fp4(
            "masked_scaled_bmm_sm8x", _FP4_FMT[block_scale_type],
            [a.view(q * m, k_a), b, *sf, st, st + masked_m.long().clamp(0, m),
             torch.ones(q, device=a.device), torch.zeros_like(masked_m, dtype=torch.int32),
             t.view(q * m, n)],
        )  # fmt: skip
    if t is not c:
        c.copy_(t)
    return c


def sm8x_ragged_scaled_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    block_scale_type: str,
    transpose_a: bool,
    transpose_b: bool,
    swizzled_layout_a: bool,
    a_global_scale: Optional[torch.Tensor],
    b_global_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    """``ragged_scaled_bmm`` on SM8x after the cuTile backend's checks (its module needs
    ``cuda.tile``): FP32 ``c[m_indptr[q] : m_indptr[q + 1]] = (a @ b[q]^T) / (a_global_scale
    b_global_scale[q])``, the other rows unwritten."""
    total_m, k_a = a.shape
    q, n, k_b = b.shape
    if block_scale_type not in ("nvfp4", "mxfp4", "mxfp8", "mixed"):
        raise ValueError(f"Invalid block scale type: {block_scale_type}")
    k = _scaled_bmm_k(block_scale_type, k_a, k_b)
    _nt_only(transpose_a, transpose_b)
    if not swizzled_layout_a:
        raise ValueError(
            "Only swizzled_layout_a=True is supported by the cuTile backend"
        )
    _contiguous(
        ("A matrix", a), ("B matrix", b), ("A scale matrix", a_scale),
        ("B scale matrix", b_scale), ("m_indptr", m_indptr),
    )  # fmt: skip
    if m_indptr.numel() != q + 1:
        raise ValueError(f"m_indptr must have Q+1 ({q + 1}) elements")
    gss = [t for t in (a_global_scale, b_global_scale) if t is not None]
    if any(t.device != a.device for t in (b, a_scale, b_scale, m_indptr, *gss)):
        raise ValueError(
            "Scaled BMM operands, scales and m_indptr must be on one device"
        )
    m_indptr = m_indptr.reshape(-1)
    vec = 16 if block_scale_type == "nvfp4" else 32
    q_sb, rnb, rkb, n2, k256b = b_scale.shape
    if q_sb != q:
        raise ValueError("incompatible b_scale batch dim")
    if n2 != 2 or k256b != 256:
        raise ValueError("incompatible b_scale inner dims")
    if rnb * 128 != n:
        raise ValueError("incompatible b_scale N dim")
    if rkb * 4 * vec != k:
        raise ValueError("incompatible b_scale K dim")
    total_ma, rka, m2, k256a = a_scale.shape
    if total_ma * 128 != total_m:
        raise ValueError("incompatible a_scale total_m dim")
    if m2 != 2 or k256a != 256:
        raise ValueError("incompatible a_scale inner dims")
    if rka * 4 * vec != k:
        raise ValueError("incompatible a_scale K dim")
    c = torch.empty((total_m, n), device=a.device, dtype=torch.float32)
    if q == 0 or c.numel() == 0:
        return c
    ip = torch.cummax(m_indptr.clamp(0, total_m).long(), 0).values
    gs = torch.ones(q, device=a.device)  # the cuTile kernel's 1 / (a_gs b_gs)
    if a_global_scale is not None or b_global_scale is not None:
        ag = (
            _ones(a.device)
            if a_global_scale is None
            else a_global_scale.reshape(-1)[:1]
        )
        bg = _ones(a.device) if b_global_scale is None else b_global_scale.reshape(-1)
        gs = (1.0 / (ag.float() * bg.float())).expand(q).contiguous()
    sf = (a_scale.reshape(-1).view(torch.uint8), b_scale.reshape(-1).view(torch.uint8))
    if block_scale_type == "mxfp8":
        seg = torch.searchsorted(ip, torch.arange(total_m, device=a.device), right=True)
        _tuned_mx8("ragged_scaled_bmm_sm8x", False, [a, b, *sf, seg - 1, c])
        if a_global_scale is not None or b_global_scale is not None:
            c.mul_(gs[(seg - 1).clamp_(0, q - 1)][:, None])
    else:
        _tuned_rows_fp4(
            "ragged_scaled_bmm_sm8x", _FP4_FMT[block_scale_type],
            [a, b, *sf, ip[:-1], ip[1:], gs, torch.zeros(q, dtype=torch.int32,
             device=a.device), c],
        )  # fmt: skip
    return c


def _run_16(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    r: int,
) -> None:
    """``out = alpha * (a @ b.T) + bias``: ``a`` ``(M, K)`` BF16 or FP16, ``b`` ``(N, K)`` E4M3,
    ``bias`` ``(N,)`` of ``a``'s type or None, ``out`` ``(M, N)``, contiguous."""
    m, k = (int(x) for x in a.shape)
    n, kp = int(b.shape[0]), _pad(k, 32)
    rows = _I32_MAX // kp  # the GEMM's activation offsets are 32-bit
    if m > rows:
        for r0 in range(0, m, rows):
            _run_16(a[r0 : r0 + rows], b, alpha, bias, out[r0 : r0 + rows], r)
        return
    max_blocks = -(-m // r)
    ints = (1 + 2 * max_blocks, max_blocks * r)
    meta, rowmap = torch.empty(sum(ints), dtype=torch.int32, device=a.device).split(
        ints
    )
    act = torch.empty((m, kp), dtype=a.dtype, device=a.device)
    module = _w4a16_moe_sm80_module(None, False, None, a.dtype == torch.float16)
    module.prep_f8(a.reshape(-1), act, meta, rowmap, 1, k, r, max_blocks, 0)
    module.gemm_f8(
        b.view(torch.uint8).reshape(-1), alpha.reshape(1), None, act, meta, rowmap, out, r, n,
        k, max_blocks, 1, k, 128, 0, 3, False, None, None, False, 0, None, None, 1, bias,
    )  # fmt: skip


class Sm8x16BitFp8Runner(TunableRunner):
    """``_run_16`` as an autotuner candidate on ``[a, b, alpha, out]`` (plus ``bias`` when
    given); tactic: the row tile."""

    def get_cache_key_extras(self, inputs) -> tuple:
        return (inputs[0].dtype, len(inputs))

    def get_valid_tactics(self, inputs, profile: OptimizationProfile) -> List:
        return [8, 16, 32, 64]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        a, b, alpha, out = inputs[:4]
        bias = inputs[4] if len(inputs) > 4 else None
        _run_16(a, b, alpha, bias, out, 64 if tactic in (None, -1) else tactic)
        return out


def sm8x_mm_16bit_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
) -> None:
    """``mm_bf16_fp8`` / ``mm_fp16_fp8`` on SM8x: ``launch_w8a16_fp8_gemm``'s validated 2-D
    operands (``b`` ``(N, K)`` E4M3, ``alpha`` one FP32)."""
    from ..fused_moe.utils import (
        get_hybrid_num_tokens_buckets,
        map_to_hybrid_bucket_uncapped,
    )

    tuning = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                (0,), (0,), get_hybrid_num_tokens_buckets, map_to_hybrid_bucket_uncapped
            ),
        ),
        constraint_specs=(ConstraintSpec(3, 0, lambda shapes: shapes[0][0]),),
    )
    inputs = [a, b, alpha, out] + ([] if bias is None else [bias])
    chosen, tactic = AutoTuner.get().choose_one(
        "mm_16bit_fp8_sm8x", [Sm8x16BitFp8Runner()], tuning, inputs
    )
    chosen(inputs=inputs, tactic=tactic)
