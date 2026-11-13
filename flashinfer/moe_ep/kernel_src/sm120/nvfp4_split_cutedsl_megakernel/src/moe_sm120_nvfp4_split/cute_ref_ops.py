# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Small cuTeDSL reference ops for exact SM120 instruction semantics."""

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
