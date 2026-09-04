# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""cuTeDSL-backed reference ops for bit-exact hardware semantics.

The runner's reference computation uses a handful of tiny kernels whose only
job is to mirror device-side instruction semantics bit-exactly
(``rcp.approx.ftz.f32``, ``ex2.approx.f32``, ``cvt.rn.satfinite.e2m1x2.f32``,
sequential ``add.rn.bf16``).  They are expressed as cuTeDSL kernels so every
supported GPU follows one deterministic path, without depending on Triton's
bundled ptxas knowing the device architecture.  The instruction-level helpers
use the same PTX instructions via ``llvm.inline_asm``.

bf16 note: sequential bf16 adds are emitted as ``(f32(a) + f32(b)).to(bf16)``.
The f32 sum of two bf16 values is exact (8-bit mantissas into a 24-bit
mantissa), and the f32->bf16 convert rounds to nearest even, so each step is
exactly ``add.rn.bf16`` -- the same rounding the device's
``red.global.add.noftz.v2.bf16x2`` applies per add.
"""

import functools
from itertools import permutations

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack

_BLOCK = 256


def _f32_asm1(asm: str, x):
    """One-operand f32 PTX op (e.g. ``rcp.approx.ftz.f32 $0, $1;``)."""
    return cutlass.Float32(llvm.inline_asm(
        cutlass.Float32.mlir_type,
        [x.ir_value()],
        asm, "=f,f",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


# ---------------------------------------------------------------------------
# rcp.approx.ftz.f32
# ---------------------------------------------------------------------------

@cute.kernel
def _rcp_approx_kernel(x: cute.Tensor, y: cute.Tensor, n: Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * _BLOCK + tidx
    if i < n:
        y[i] = _f32_asm1("rcp.approx.ftz.f32 $0, $1;", x[i])


@cute.jit
def _rcp_approx_host(x: cute.Tensor, y: cute.Tensor):
    n = cute.size(x)
    _rcp_approx_kernel(x, y, Int32(n)).launch(
        grid=((n + _BLOCK - 1) // _BLOCK, 1, 1), block=(_BLOCK, 1, 1),
    )


# ---------------------------------------------------------------------------
# SwiGLU pair matching the kernel-side PTX op sequence
# (mul, ex2.approx of -gate*log2e, +1, rcp.approx, mul)
# ---------------------------------------------------------------------------

@cute.kernel
def _swiglu_pair_kernel(g: cute.Tensor, u: cute.Tensor, o: cute.Tensor, n: Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * _BLOCK + tidx
    if i < n:
        gate = g[i]
        up = u[i]
        ug = up * gate
        neg_g_l2e = gate * cutlass.Float32(-1.4426950408889634)
        exp_neg = _f32_asm1("ex2.approx.f32 $0, $1;", neg_g_l2e)
        one_plus = exp_neg + cutlass.Float32(1.0)
        sigmoid = _f32_asm1("rcp.approx.ftz.f32 $0, $1;", one_plus)
        o[i] = ug * sigmoid


@cute.jit
def _swiglu_pair_host(g: cute.Tensor, u: cute.Tensor, o: cute.Tensor):
    n = cute.size(g)
    _swiglu_pair_kernel(g, u, o, Int32(n)).launch(
        grid=((n + _BLOCK - 1) // _BLOCK, 1, 1), block=(_BLOCK, 1, 1),
    )


# ---------------------------------------------------------------------------
# FP4 E2M1 pack: cvt.rn.satfinite.e2m1x2.f32 (odd -> high nibble, even -> low)
# ---------------------------------------------------------------------------

@cute.kernel
def _pack_fp4_kernel(x: cute.Tensor, out: cute.Tensor, n_pairs: Int64):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    # Int64 offsets: the combine round-trip packs the whole (world*tok, topk,
    # hidden) term tensor, so 2*i overflows int32 at large token counts.
    i = Int64(bidx) * Int64(_BLOCK) + Int64(tidx)
    if i < n_pairs:
        even = x[i * 2]
        odd = x[i * 2 + 1]
        packed = Int32(llvm.inline_asm(
            Int32.mlir_type,
            [odd.ir_value(), even.ir_value()],
            "{\n"
            "  .reg .b8 r;\n"
            "  cvt.rn.satfinite.e2m1x2.f32 r, $1, $2;\n"
            "  mov.b32 $0, {r, r, r, r};\n"
            "}",
            "=r,f,f",
            has_side_effects=True, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        ))
        out[i] = cutlass.Uint8(packed & Int32(0xFF))


@cute.jit
def _pack_fp4_host(x: cute.Tensor, out: cute.Tensor):
    n_pairs = cute.size(out)
    _pack_fp4_kernel(x, out, Int64(n_pairs)).launch(
        grid=((n_pairs + _BLOCK - 1) // _BLOCK, 1, 1), block=(_BLOCK, 1, 1),
    )


# ---------------------------------------------------------------------------
# bf16 sequential sums over K (plain + all-K!-permutations variant)
# ---------------------------------------------------------------------------

def _bf16_add_rn(acc, v):
    # Exact f32 sum of two bf16 + RN convert back == add.rn.bf16 per step.
    return (acc.to(cutlass.Float32) + v.to(cutlass.Float32)).to(cutlass.BFloat16)


@functools.lru_cache(None)
def _make_seq_sum_k(K: int):
    @cute.kernel
    def kern(src: cute.Tensor, dst: cute.Tensor, T: Int32, H: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        h = bidy * _BLOCK + tidx
        t = bidx
        if h < H:
            acc = cutlass.BFloat16(0.0)
            for k in cutlass.range_constexpr(K):
                acc = _bf16_add_rn(acc, src[Int64(t) * K * H + k * H + h])
            dst[Int64(t) * H + h] = acc

    @cute.jit
    def host(src: cute.Tensor, dst: cute.Tensor, T: Int32, H: Int32):
        kern(src, dst, T, H).launch(
            grid=(T, (H + _BLOCK - 1) // _BLOCK, 1), block=(_BLOCK, 1, 1),
        )

    return host


@functools.lru_cache(None)
def _make_all_perms(K: int):
    perms = tuple(permutations(range(K)))
    n_perms = len(perms)

    @cute.kernel
    def kern(src: cute.Tensor, dst: cute.Tensor, T: Int32, H: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        h = bidy * _BLOCK + tidx
        t = bidx
        if h < H:
            for p in cutlass.range_constexpr(n_perms):
                acc = cutlass.BFloat16(0.0)
                for k_pos in cutlass.range_constexpr(K):
                    k = perms[p][k_pos]  # python int, folds at trace time
                    acc = _bf16_add_rn(acc, src[Int64(t) * K * H + k * H + h])
                dst[(Int64(t) * H + h) * n_perms + p] = acc

    @cute.jit
    def host(src: cute.Tensor, dst: cute.Tensor, T: Int32, H: Int32):
        kern(src, dst, T, H).launch(
            grid=(T, (H + _BLOCK - 1) // _BLOCK, 1), block=(_BLOCK, 1, 1),
        )

    return host, n_perms


# ---------------------------------------------------------------------------
# Host-facing wrappers (compile-once caches keyed by op / K)
# ---------------------------------------------------------------------------

_compiled = {}


def _flat_dyn(t: torch.Tensor):
    return from_dlpack(t.view(-1)).mark_layout_dynamic()


def rcp_approx_ftz_f32(x: torch.Tensor) -> torch.Tensor:
    """Bit-match ``rcp.approx.ftz.f32`` (== cute.arch.rcp_approx) via cuTeDSL."""
    x_c = x.contiguous()
    y = torch.empty_like(x_c)
    if x_c.numel() == 0:
        return y.view_as(x)
    key = "rcp"
    if key not in _compiled:
        _compiled[key] = cute.compile(_rcp_approx_host, _flat_dyn(x_c), _flat_dyn(y))
    _compiled[key](_flat_dyn(x_c), _flat_dyn(y))
    torch.cuda.synchronize()
    return y.view_as(x)


def swiglu_pair_hw_match(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU with the kernel-side ex2/rcp approx sequence via cuTeDSL."""
    g_c, u_c = gate.contiguous(), up.contiguous()
    o = torch.empty_like(g_c)
    if g_c.numel() == 0:
        return o
    key = "swiglu"
    if key not in _compiled:
        _compiled[key] = cute.compile(
            _swiglu_pair_host, _flat_dyn(g_c), _flat_dyn(u_c), _flat_dyn(o)
        )
    _compiled[key](_flat_dyn(g_c), _flat_dyn(u_c), _flat_dyn(o))
    torch.cuda.synchronize()
    return o


def pack_f32_to_fp4_u8(flat_f32: torch.Tensor) -> torch.Tensor:
    """cvt.rn.satfinite.e2m1x2.f32 nibble pack -> uint8 pairs via cuTeDSL."""
    n_pairs = flat_f32.numel() // 2
    out = torch.empty(n_pairs, dtype=torch.uint8, device=flat_f32.device)
    if n_pairs == 0:
        return out
    key = "pack_fp4"
    if key not in _compiled:
        _compiled[key] = cute.compile(_pack_fp4_host, _flat_dyn(flat_f32), _flat_dyn(out))
    _compiled[key](_flat_dyn(flat_f32), _flat_dyn(out))
    torch.cuda.synchronize()
    return out


def bf16_seq_sum_k(src: torch.Tensor) -> torch.Tensor:
    """Sequential bf16 sum over K of (T, K, H) bf16, add.rn.bf16 per step."""
    T, K, H = src.shape
    src_c = src.contiguous()
    dst = torch.empty((T, H), dtype=torch.bfloat16, device=src.device)
    key = ("seq_sum", K)
    if key not in _compiled:
        _compiled[key] = cute.compile(
            _make_seq_sum_k(K), _flat_dyn(src_c), _flat_dyn(dst), Int32(T), Int32(H)
        )
    _compiled[key](_flat_dyn(src_c), _flat_dyn(dst), Int32(T), Int32(H))
    torch.cuda.synchronize()
    return dst


def bf16_all_perm_sums(src: torch.Tensor) -> torch.Tensor:
    """(T, K, H) bf16 -> (T, H, K!) bf16: every K-permutation's sequential sum."""
    T, K, H = src.shape
    host, n_perms = _make_all_perms(K)
    src_c = src.contiguous()
    dst = torch.empty((T, H, n_perms), dtype=torch.bfloat16, device=src.device)
    key = ("all_perms", K)
    if key not in _compiled:
        _compiled[key] = cute.compile(
            host, _flat_dyn(src_c), _flat_dyn(dst), Int32(T), Int32(H)
        )
    _compiled[key](_flat_dyn(src_c), _flat_dyn(dst), Int32(T), Int32(H))
    torch.cuda.synchronize()
    return dst
