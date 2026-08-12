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

CuTe-DSL backend for serving-native packed Kimi K3 T=1 recurrent decode.

INTERNAL IMPLEMENTATION -- not public API. The supported way to reach these
kernels is the public ``flashinfer.recurrent_kda`` operation, whose T=1 fast
path dispatches eligible decode calls here (toggle with
``FLASHINFER_KDA_T1_FAST_PATH=0``). ``run_packed_kda_decode_cute`` is kept
for tests and benchmarks of the packed-input entry point.

Same numerical contract and tensor layouts as the exported Cake backend in
``flashinfer.kda_kernels.packed_kda_decode``: packed bf16 QKV rows, raw gate
and beta logits, ``scale = 1/sqrt(128)``, L2 epsilon ``1e-6``,
``lower_bound = -5``, bf16 state pool updated in place, and ``-1`` state
indices producing zero output without touching the pool.

Kernel design (ported from the GDN wide-vec T=1 kernel in
``flashinfer.gdn_kernels.gdn_decode_bf16_state`` and the register-tile
recurrent KDA kernel in ``flashinfer.kda_kernels.recurrent_kda``):

- CTAs are ``num_groups`` groups of 16 lanes; each lane owns a contiguous
  eight-element K vector (LDG.128 / STG.128 on the bf16 state). Grid is
  ``num_v_tiles x H x B`` linearised; each CTA owns one ``tile_v`` V-row
  tile of one ``(batch, head)`` pair.
- Barrier-free staging: each warp independently loads q/k/gate (four
  elements per lane across 32 lanes), reduces the L2 norms with a full-warp
  butterfly, computes the per-channel decay, and shuffle-transposes into the
  16-lane-by-8 register layout.
- State stays packed bf16 in registers; unpack is a shift/mask bit trick and
  repack a single ``cvt.rn.bf16x2.f32`` (full-rate ALU instead of the
  conversion pipe). FMAs use packed F32x2 pairs on SM100+.
- Two main-loop variants, selected per batch size by ``_select_config``:
  a register-prefetch kernel (double-buffered LDG.128, CTA width tunable
  down to 32 threads for small-batch latency), and a cp.async kernel that
  streams state rows through a shared-memory ring (``n_stages`` deep,
  ``cp.async.cg`` L1 bypass) so the in-flight read volume is not bounded by
  the register file — the DRAM-bound large-batch regime runs there.
"""

import functools
import os
from typing import Optional

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm as mlir_llvm
from cutlass.cute.runtime import make_fake_stream
import tvm_ffi  # noqa: F401 -- TVM FFI required for zero-overhead dispatch

from ..jit.cpp_ext import is_cuda_version_at_least
from ..utils import get_compute_capability

_HEADS = 12
_HEAD_DIM = 128
_MIXED_WIDTH = 3 * _HEADS * _HEAD_DIM
_SCALE = float(_HEAD_DIM) ** -0.5
_EPS = 1.0e-6
_LOWER_BOUND = -5.0
_LOG2_E = 1.4426950408889634

_LANES_PER_ROW = 16  # 16 lanes cover K=128 with 8 bf16 each
_ELEMS_PER_LANE = 8
_NUM_THREADS = 128
_NUM_GROUPS = _NUM_THREADS // _LANES_PER_ROW  # 8
_VALS_PER_LANE = _HEAD_DIM // 32  # 4: warp-staged q/k/gate elements per lane


@cute.jit
def _bf16x2_to_f32x2(u):
    """Unpack one register holding two bf16 into two f32 via bit ops.

    bf16 widens to f32 by appending 16 zero mantissa bits, so a shift and a
    mask on the packed word replace two CVT instructions (full-rate ALU
    instead of the conversion pipe — the same trick the Cake-generated
    kernel uses via inline PTX). Element 0 sits in the low half.
    """
    lo = cutlass.Float32(
        mlir_arith.bitcast(cutlass.Float32.mlir_type, (u << 16).ir_value())
    )
    hi = cutlass.Float32(
        mlir_arith.bitcast(cutlass.Float32.mlir_type, (u & -65536).ir_value())
    )
    return lo, hi


@cute.jit
def _f32x2_to_bf16x2(h0, h1):
    """Pack two f32 into one register of two bf16 with a single CVT.

    ``cvt.rn.bf16x2.f32`` rounds both halves to nearest-even — bitwise
    identical to two scalar BFloat16() conversions (and to CUDA's
    ``__float22bfloat162_rn``) — replacing two conversion-pipe CVTs plus a
    register merge. ``h0`` lands in the low half (element 0).
    """
    packed = mlir_llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [
            cutlass.Float32(h1).ir_value(),
            cutlass.Float32(h0).ir_value(),
        ],
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(packed)


def _cp_async_bf16x8_cg(base_addr_i64, bf16_elem_offset, smem_addr_i32, l2_hint=0):
    """cp.async.cg, 16 B (8 bf16), L1 bypass (state is a stream-once input).

    Same inline-PTX shape as the GDN wy kernel helpers: the u32 element
    offset is widened and scaled into the u64 global address in-asm.
    ``l2_hint`` of 128/256 adds an ``.L2::<n>B`` allocation hint — each 8 KiB
    chunk densely covers its sectors, so full-sector allocation can cut DRAM
    request overhead (constexpr; compiled in).
    """
    hint = f".L2::{int(l2_hint)}B" if l2_hint else ""
    r = mlir_llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [
            smem_addr_i32.ir_value(),
            base_addr_i64.ir_value(),
            bf16_elem_offset.ir_value(),
        ],
        "{ .reg .u64 _a; mad.wide.u32 _a, $3, 2, $2;"
        f" cp.async.cg.shared.global{hint} [$1], [_a], 16;"
        " mov.u32 $0, 0; }",
        "=r,r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(r)


def _l2_policy_evict_last():
    """64-bit L2 cache policy: evict_last (created once per thread)."""
    r = mlir_llvm.inline_asm(
        cutlass.Int64.mlir_type,
        [],
        "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;",
        "=l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int64(r)


def _l2_policy_evict_first():
    r = mlir_llvm.inline_asm(
        cutlass.Int64.mlir_type,
        [],
        "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
        "=l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int64(r)


def _stg_hint_v4_b32(gmem_addr_i64, u0, u1, u2, u3, policy_i64):
    """st.global.L2::cache_hint.v4.b32 — state write with an L2 policy.

    evict_last keeps the freshly written state resident in L2 so its
    write-back drains after kernel-end (overlapping the next kernel in a
    serving pipeline) instead of competing with the in-kernel read stream.
    """
    r = mlir_llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [
            gmem_addr_i64.ir_value(),
            u0.ir_value(),
            u1.ir_value(),
            u2.ir_value(),
            u3.ir_value(),
            policy_i64.ir_value(),
        ],
        "{ st.global.L2::cache_hint.v4.b32 [$1], {$2, $3, $4, $5}, $6;"
        " mov.u32 $0, 0; }",
        "=r,l,r,r,r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(r)


def _cp_async_bf16x8_cg_hint(
    base_addr_i64, bf16_elem_offset, smem_addr_i32, policy_i64
):
    """cp.async.cg with an L2 cache policy (evict_first for the read stream)."""
    r = mlir_llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [
            smem_addr_i32.ir_value(),
            base_addr_i64.ir_value(),
            bf16_elem_offset.ir_value(),
            policy_i64.ir_value(),
        ],
        "{ .reg .u64 _a; mad.wide.u32 _a, $3, 2, $2;"
        " cp.async.cg.shared.global.L2::cache_hint [$1], [_a], 16, $4;"
        " mov.u32 $0, 0; }",
        "=r,r,l,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(r)


def _stg_v4_b32(gmem_addr_i64, u0, u1, u2, u3):
    """st.global.v4.b32 at a raw byte address (default cache policy)."""
    r = mlir_llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [
            gmem_addr_i64.ir_value(),
            u0.ir_value(),
            u1.ir_value(),
            u2.ir_value(),
            u3.ir_value(),
        ],
        "{ st.global.v4.b32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0; }",
        "=r,l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(r)


def _stg_cg_v4_b32(gmem_addr_i64, u0, u1, u2, u3):
    """st.global.cg.v4.b32 — stream the state write past L1 (write-once)."""
    r = mlir_llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [
            gmem_addr_i64.ir_value(),
            u0.ir_value(),
            u1.ir_value(),
            u2.ir_value(),
            u3.ir_value(),
        ],
        "{ st.global.cg.v4.b32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0; }",
        "=r,l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(r)


def _cp_async_commit_group():
    r = mlir_llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [],
        "{ cp.async.commit_group; mov.u32 $0, 0; }",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(r)


def _cp_async_wait_group_n(n_const):
    """Wait until at most ``n_const`` cp.async groups remain in flight.

    The ``~{memory}`` clobber matters in the barrier-free private ring:
    without it the compiler may hoist the subsequent smem loads above the
    wait (the barriered path is protected by bar.sync instead).
    """
    r = mlir_llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [],
        f"{{ cp.async.wait_group {int(n_const)}; mov.u32 $0, 0; }}",
        "=r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(r)


def _cp_async_bulk_store(gmem_addr_i64, smem_addr_i32, size_bytes):
    """One shared->global bulk store (async proxy, bulk_group completion).

    The issuing warp's lane 0 pushes its warp's contiguous smem region out
    through the TMA path — state writes leave the SM without occupying
    LSU/L1TEX wavefronts.
    """
    r = mlir_llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [gmem_addr_i64.ir_value(), smem_addr_i32.ir_value()],
        "{ cp.async.bulk.global.shared::cta.bulk_group [$1], [$2], "
        f"{int(size_bytes)};"
        " mov.u32 $0, 0; }",
        "=r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(r)


def _cp_async_bulk_read(smem_addr_i32, gmem_addr_i64, mbar_addr_i32, size_bytes):
    """One global->shared bulk read completing on an mbarrier (TMA-1D)."""
    r = mlir_llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [
            smem_addr_i32.ir_value(),
            gmem_addr_i64.ir_value(),
            mbar_addr_i32.ir_value(),
        ],
        "{ cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
        f" [$1], [$2], {int(size_bytes)}, [$3];"
        " mov.u32 $0, 0; }",
        "=r,r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(r)


def _cp_async_bulk_commit():
    r = mlir_llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [],
        "{ cp.async.bulk.commit_group; mov.u32 $0, 0; }",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(r)


def _cp_async_bulk_wait_read(n_const):
    """Wait until at most n bulk groups have unread smem sources."""
    r = mlir_llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [],
        f"{{ cp.async.bulk.wait_group.read {int(n_const)}; mov.u32 $0, 0; }}",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(r)


def _cp_async_bulk_wait(n_const):
    r = mlir_llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [],
        f"{{ cp.async.bulk.wait_group {int(n_const)}; mov.u32 $0, 0; }}",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(r)


def _fence_proxy_async():
    """Order generic-proxy smem writes before async-proxy bulk reads."""
    r = mlir_llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [],
        "{ fence.proxy.async.shared::cta; mov.u32 $0, 0; }",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(r)


def _sync_warp():
    r = mlir_llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [],
        "{ bar.warp.sync 0xffffffff; mov.u32 $0, 0; }",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(r)


@cute.kernel
def _kda_packed_t1_kernel(
    state: cute.Tensor,  # [pool, H, V, K] bf16, padded slot stride allowed
    q: cute.Tensor,  # [B, H, K] bf16 strided view of mixed_qkv
    k: cute.Tensor,  # [B, H, K] bf16 strided view of mixed_qkv
    v: cute.Tensor,  # [B, H, V] bf16 strided view of mixed_qkv
    g: cute.Tensor,  # [B, H, K] bf16 strided view of raw_gate
    beta: cute.Tensor,  # [B, H] bf16 (row stride dynamic)
    A_log: cute.Tensor,  # [H] f32
    dt_bias: cute.Tensor,  # [H, K] f32
    o: cute.Tensor,  # [B, H, V] bf16 contiguous
    state_indices: cute.Tensor,  # [B] i32
    scale: cutlass.Constexpr[float],
    eps: cutlass.Constexpr[float],
    lb_log2e: cutlass.Constexpr[float],
    precomputed: cutlass.Constexpr[bool],  # lower_bound * log2(e)
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    ilp_rows: cutlass.Constexpr[int],
    num_groups: cutlass.Constexpr[int],
    use_packed_fma: cutlass.Constexpr[bool],
    evict_first_state: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Grid: (num_v_tiles x H x B) linearised, i_v fastest so consecutive CTAs
    # touch adjacent V tiles of the same state slot (L2 locality).
    i_v = bidx % num_v_tiles
    tmp = bidx // num_v_tiles
    i_h = tmp % H
    i_n = tmp // H

    lane = tidx % 32
    k_lane = tidx % _LANES_PER_ROW
    group_idx = tidx // _LANES_PER_ROW

    ROWS_PER_GROUP: cutlass.Constexpr[int] = tile_v // num_groups
    ITERS: cutlass.Constexpr[int] = ROWS_PER_GROUP // ilp_rows

    vec: cutlass.Constexpr[int] = _ELEMS_PER_LANE
    vals: cutlass.Constexpr[int] = _VALS_PER_LANE

    # Plain-Python unroll indices: the DSL stages `for x in range(...)` loops,
    # whose loop variable cannot index Python lists of tiles/accumulators.
    # Iterating tuples keeps the loop in Python (fully unrolled at trace).
    ROWS = tuple(range(ilp_rows))
    VECI = tuple(range(vec))
    VECP = tuple(range(0, vec, 2))
    VALSI = tuple(range(vals))
    ITERI = tuple(range(ITERS))

    raw_slot = state_indices[i_n]

    # All register tensors are allocated up front, before any dynamic branch
    # (allocas must not live inside scf.if regions; same structure as the GDN
    # wide-vec kernel and the register-tile recurrent KDA kernel).
    q_src = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.Float32
    )
    k_src = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.Float32
    )
    d_src = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.Float32
    )
    q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.BFloat16
    )
    k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.BFloat16
    )
    g_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.BFloat16
    )
    dtb_f32 = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(cute.make_layout((vec,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec,), stride=(1,)), cutlass.Float32)
    r_d = cute.make_rmem_tensor(cute.make_layout((vec,), stride=(1,)), cutlass.Float32)
    # Double-buffered bf16 staging: while iteration `it` computes, iteration
    # `it+1`'s state rows and v values are already in flight (the kernel is
    # otherwise latency-bound — nothing saturates at base clocks).
    r_hb = [
        [
            cute.make_rmem_tensor(
                cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
            )
            for _ in ROWS
        ]
        for _ in range(2)
    ]
    r_hb32 = [[cute.recast_tensor(t, cutlass.Int32) for t in bufs] for bufs in r_hb]
    r_v_bf16 = [
        cute.make_rmem_tensor(
            cute.make_layout((ilp_rows,), stride=(1,)), cutlass.BFloat16
        )
        for _ in range(2)
    ]
    r_o_bf16 = cute.make_rmem_tensor(
        cute.make_layout((ilp_rows,), stride=(1,)), cutlass.BFloat16
    )

    slot_dead = (raw_slot < 0) | (cutlass.Int64(raw_slot) >= state.shape[0])
    if slot_dead:
        # Inactive CUDA-graph padding row: zero this CTA's output rows and do
        # not touch the state pool. Uniform per CTA (no divergence hazards).
        for r in ROWS:
            r_o_bf16[r] = cutlass.BFloat16(0.0)
        if k_lane == 0:
            for it in ITERI:
                vb = i_v * tile_v + group_idx * ROWS_PER_GROUP + it * ilp_rows
                ot = cute.local_tile(o, (1, 1, ilp_rows), (i_n, i_h, vb // ilp_rows))
                cute.autovec_copy(r_o_bf16, ot)
    else:
        # Issue iteration 0's state loads FIRST: they depend only on the slot
        # index, and the whole q/k/gate staging chain below (LDG + norm
        # butterflies + shuffles) then executes while they are in flight.
        # Mirrors the GDN wide-vec tile_v=32 hoist.
        h_slot = state[(cutlass.Int64(raw_slot), i_h, None, None)]
        v_base = i_v * tile_v + group_idx * ROWS_PER_GROUP

        ht_of = [
            [
                cute.local_tile(h_slot, (1, vec), (v_base + it * ilp_rows + r, k_lane))
                for r in ROWS
            ]
            for it in ITERI
        ]
        vt_of = [
            cute.local_tile(
                v, (1, 1, ilp_rows), (i_n, i_h, (v_base + it * ilp_rows) // ilp_rows)
            )
            for it in ITERI
        ]

        for r in ROWS:
            if cutlass.const_expr(evict_first_state):
                cute.autovec_copy(
                    ht_of[0][r],
                    r_hb[0][r],
                    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_FIRST,
                )
            else:
                cute.autovec_copy(ht_of[0][r], r_hb[0][r])
        cute.autovec_copy(vt_of[0], r_v_bf16[0])

        # ------------------------------------------------------------------
        # Warp-local staging: 32 lanes load q/k/gate (4 elems each), butterfly
        # the L2 norms, compute per-channel decay, then shuffle-transpose into
        # the (k_lane, 8) register layout every group needs.
        # ------------------------------------------------------------------
        q_tile = cute.local_tile(q, (1, 1, vals), (i_n, i_h, lane))
        k_tile = cute.local_tile(k, (1, 1, vals), (i_n, i_h, lane))
        g_tile = cute.local_tile(g, (1, 1, vals), (i_n, i_h, lane))
        dtb_tile = cute.local_tile(dt_bias, (1, vals), (i_h, lane))
        cute.autovec_copy(q_tile, q_bf16)
        cute.autovec_copy(k_tile, k_bf16)
        cute.autovec_copy(g_tile, g_bf16)
        cute.autovec_copy(dtb_tile, dtb_f32)

        if cutlass.const_expr(precomputed):
            # Pre-computed convention: beta arrives already sigmoided.
            a_exp = cutlass.Float32(0.0)
            r_beta = cutlass.Float32(beta[(i_n, i_h)])
        else:
            a_exp = cute.exp(cutlass.Float32(A_log[i_h]), fastmath=True)
            b_logit = cutlass.Float32(beta[(i_n, i_h)])
            r_beta = cute.rcp(
                cute.exp(-b_logit, fastmath=True) + 1.0, approx=True, ftz=True
            )

        sum_q = cutlass.Float32(0.0)
        sum_k = cutlass.Float32(0.0)
        for i in VALSI:
            q_val = cutlass.Float32(q_bf16[i])
            k_val = cutlass.Float32(k_bf16[i])
            q_src[i] = q_val
            k_src[i] = k_val
            sum_q += q_val * q_val
            sum_k += k_val * k_val
            if cutlass.const_expr(precomputed):
                # g is the log-space decay: d = exp(g).
                d_src[i] = cute.exp2(
                    cutlass.Float32(g_bf16[i]) * _LOG2_E, fastmath=True
                )
            else:
                gate_x = cutlass.Float32(g_bf16[i]) + dtb_f32[i]
                sig = cute.rcp(
                    cute.exp(-(a_exp * gate_x), fastmath=True) + 1.0,
                    approx=True,
                    ftz=True,
                )
                d_src[i] = cute.exp2(lb_log2e * sig, fastmath=True)

        for offset in [16, 8, 4, 2, 1]:
            sum_q += cute.arch.shuffle_sync_bfly(sum_q, offset=offset, mask=0xFFFFFFFF)
            sum_k += cute.arch.shuffle_sync_bfly(sum_k, offset=offset, mask=0xFFFFFFFF)
        inv_q = cute.rsqrt(sum_q + eps, fastmath=True) * scale
        inv_k = cute.rsqrt(sum_k + eps, fastmath=True)

        for i in VECI:
            source_lane = 2 * k_lane + i // vals
            source_value = i % vals
            r_q[i] = (
                cute.arch.shuffle_sync(
                    q_src[source_value], offset=source_lane, mask=0xFFFFFFFF
                )
                * inv_q
            )
            r_k[i] = (
                cute.arch.shuffle_sync(
                    k_src[source_value], offset=source_lane, mask=0xFFFFFFFF
                )
                * inv_k
            )
            r_d[i] = cute.arch.shuffle_sync(
                d_src[source_value], offset=source_lane, mask=0xFFFFFFFF
            )

        # ------------------------------------------------------------------
        # Main loop: ilp_rows V rows in registers per iteration, with the
        # next iteration's loads issued before the current compute chain.
        # ------------------------------------------------------------------
        for it in ITERI:
            vb = v_base + it * ilp_rows
            cur = it % 2
            if cutlass.const_expr(it + 1 < ITERS):
                for r in ROWS:
                    if cutlass.const_expr(evict_first_state):
                        cute.autovec_copy(
                            ht_of[it + 1][r],
                            r_hb[1 - cur][r],
                            l1c_evict_priority=(
                                cute.nvgpu.CacheEvictionPriority.EVICT_FIRST
                            ),
                        )
                    else:
                        cute.autovec_copy(ht_of[it + 1][r], r_hb[1 - cur][r])
                cute.autovec_copy(vt_of[it + 1], r_v_bf16[1 - cur])
            ht = ht_of[it]

            # Pass 1: s = (h * decay) . k, reduced over K. The fp32 decayed
            # state is transient — state stays packed bf16 in registers (the
            # Cake register-economy trick: ~32 fewer live registers, so more
            # CTAs fit per SM; the decay product is recomputed identically in
            # pass 2).
            s_e = [cutlass.Float32(0.0) for _ in ROWS]
            s_o = [cutlass.Float32(0.0) for _ in ROWS]
            for i in VECP:
                for r in ROWS:
                    h0, h1 = _bf16x2_to_f32x2(r_hb32[cur][r][i // 2])
                    if cutlass.const_expr(use_packed_fma):
                        h0, h1 = cute.arch.fma_packed_f32x2(
                            src_a=(h0, h1),
                            src_b=(r_d[i], r_d[i + 1]),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        s_e[r], s_o[r] = cute.arch.fma_packed_f32x2(
                            src_a=(h0, h1),
                            src_b=(r_k[i], r_k[i + 1]),
                            src_c=(s_e[r], s_o[r]),
                        )
                    else:
                        h0 = h0 * r_d[i]
                        h1 = h1 * r_d[i + 1]
                        s_e[r] = s_e[r] + h0 * r_k[i]
                        s_o[r] = s_o[r] + h1 * r_k[i + 1]
            s = [s_e[r] + s_o[r] for r in ROWS]
            for offset in [8, 4, 2, 1]:
                for r in ROWS:
                    s[r] += cute.arch.shuffle_sync_bfly(
                        s[r], offset=offset, mask=0xFFFFFFFF
                    )

            # Delta rule: vn = (v - s) * beta.
            vn = [(cutlass.Float32(r_v_bf16[cur][r]) - s[r]) * r_beta for r in ROWS]

            # Pass 2: recompute h*decay, apply the rank-1 update h += k * vn,
            # accumulate o = (h_new * q), and pack h_new straight back into
            # the bf16 staging registers (which then feed the STG.128s).
            o_e = [cutlass.Float32(0.0) for _ in ROWS]
            o_o = [cutlass.Float32(0.0) for _ in ROWS]
            for i in VECP:
                for r in ROWS:
                    h0, h1 = _bf16x2_to_f32x2(r_hb32[cur][r][i // 2])
                    if cutlass.const_expr(use_packed_fma):
                        h0, h1 = cute.arch.fma_packed_f32x2(
                            src_a=(h0, h1),
                            src_b=(r_d[i], r_d[i + 1]),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        h0, h1 = cute.arch.fma_packed_f32x2(
                            src_a=(r_k[i], r_k[i + 1]),
                            src_b=(vn[r], vn[r]),
                            src_c=(h0, h1),
                        )
                        o_e[r], o_o[r] = cute.arch.fma_packed_f32x2(
                            src_a=(h0, h1),
                            src_b=(r_q[i], r_q[i + 1]),
                            src_c=(o_e[r], o_o[r]),
                        )
                    else:
                        h0 = h0 * r_d[i]
                        h1 = h1 * r_d[i + 1]
                        h0 = r_k[i] * vn[r] + h0
                        h1 = r_k[i + 1] * vn[r] + h1
                        o_e[r] = o_e[r] + h0 * r_q[i]
                        o_o[r] = o_o[r] + h1 * r_q[i + 1]
                    r_hb32[cur][r][i // 2] = _f32x2_to_bf16x2(h0, h1)

            # State write-back: h_new is already packed bf16; the STG.128s
            # issue before the output butterfly so they drain during the
            # shuffle reduction.
            for r in ROWS:
                cute.autovec_copy(r_hb[cur][r], ht[r])

            o_val = [o_e[r] + o_o[r] for r in ROWS]
            for offset in [8, 4, 2, 1]:
                for r in ROWS:
                    o_val[r] += cute.arch.shuffle_sync_bfly(
                        o_val[r], offset=offset, mask=0xFFFFFFFF
                    )

            if k_lane == 0:
                for r in ROWS:
                    r_o_bf16[r] = cutlass.BFloat16(o_val[r])
                ot = cute.local_tile(o, (1, 1, ilp_rows), (i_n, i_h, vb // ilp_rows))
                cute.autovec_copy(r_o_bf16, ot)


@cute.kernel
def _kda_packed_t1_smem_kernel(
    state: cute.Tensor,  # [pool, H, V, K] bf16, padded slot stride allowed
    q: cute.Tensor,  # [B, H, K] bf16 strided view of mixed_qkv
    k: cute.Tensor,  # [B, H, K] bf16 strided view of mixed_qkv
    v: cute.Tensor,  # [B, H, V] bf16 strided view of mixed_qkv
    g: cute.Tensor,  # [B, H, K] bf16 strided view of raw_gate
    beta: cute.Tensor,  # [B, H] bf16 (row stride dynamic)
    A_log: cute.Tensor,  # [H] f32
    dt_bias: cute.Tensor,  # [H, K] f32
    o: cute.Tensor,  # [B, H, V] bf16 contiguous
    state_indices: cute.Tensor,  # [B] i32
    scale: cutlass.Constexpr[float],
    eps: cutlass.Constexpr[float],
    lb_log2e: cutlass.Constexpr[float],
    precomputed: cutlass.Constexpr[bool],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    ilp_rows: cutlass.Constexpr[int],
    n_stages: cutlass.Constexpr[int],
    chunk_rows: cutlass.Constexpr[int],
    cp_l2_hint: cutlass.Constexpr[int],
    heads_per_cta: cutlass.Constexpr[int],
    bulk_store: cutlass.Constexpr[bool],
    tma_read: cutlass.Constexpr[bool],
    l2_policy_mode: cutlass.Constexpr[int],
    private_ring: cutlass.Constexpr[bool],
    use_packed_fma: cutlass.Constexpr[bool],
):
    """cp.async-pipelined variant for the DRAM-bound large-batch regime.

    State rows stream through a shared-memory ring buffer via cp.async.cg
    (L1 bypass): unlike register prefetching, in-flight bytes are not bounded
    by the register file, so each CTA keeps ``(n_stages - 1) * 8 KiB`` of
    reads outstanding while computing — the standard GEMM mainloop shape.
    Requires a 16-byte-aligned state pool (cp.async constraint); the
    register-prefetch kernel remains the fallback for odd pools.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # Multiple heads per CTA (heads are adjacent compact 32 KiB blocks in
    # the slot, so the ring streams one contiguous span and only the q/k/gate
    # staging re-runs at head boundaries). Requires tile_v == V for HPC > 1.
    HPC: cutlass.Constexpr[int] = heads_per_cta
    HB: cutlass.Constexpr[int] = H // HPC
    i_v = bidx % num_v_tiles
    tmp = bidx // num_v_tiles
    i_h0 = (tmp % HB) * HPC
    i_n = tmp // HB

    lane = tidx % 32
    k_lane = tidx % _LANES_PER_ROW
    group_idx = tidx // _LANES_PER_ROW

    # One pipeline chunk covers SUBI compute sub-iterations of
    # ilp_rows rows per group (bigger chunks halve the wait+barrier count
    # without touching the register profile).
    CHUNK_ROWS: cutlass.Constexpr[int] = chunk_rows
    ITERS_H: cutlass.Constexpr[int] = tile_v // CHUNK_ROWS
    ITERS: cutlass.Constexpr[int] = HPC * ITERS_H
    SUB: cutlass.Constexpr[int] = CHUNK_ROWS // (_NUM_GROUPS * ilp_rows)

    vec: cutlass.Constexpr[int] = _ELEMS_PER_LANE
    vals: cutlass.Constexpr[int] = _VALS_PER_LANE

    ROWS = tuple(range(ilp_rows))
    VECI = tuple(range(vec))
    VECP = tuple(range(0, vec, 2))
    VALSI = tuple(range(vals))
    ITERI = tuple(range(ITERS))
    ITERI_H = tuple(range(ITERS_H))
    SUBI = tuple(range(SUB))

    # cp.async issue geometry: 128 threads cover one chunk with
    # CHUNK_ROWS * 2 / 128 sixteen-byte copies each.
    THREADS_PER_ROW: cutlass.Constexpr[int] = _NUM_THREADS // CHUNK_ROWS
    COL_SPAN: cutlass.Constexpr[int] = K // THREADS_PER_ROW
    crow = tidx // THREADS_PER_ROW
    ccol = (tidx % THREADS_PER_ROW) * COL_SPAN
    CPJ = tuple(range(COL_SPAN // vec))

    # Bulk-store geometry: with SUB == 1 each warp's rows are contiguous
    # (2 groups x ilp_rows rows = one span), so its write-back is a single
    # shared->global bulk copy issued by lane 0.
    warp_idx = tidx // 32
    WARP_ROWS: cutlass.Constexpr[int] = 2 * ilp_rows

    smem = cutlass.utils.SmemAllocator()
    sH = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((n_stages, CHUNK_ROWS, K), stride=(CHUNK_ROWS * K, K, 1)),
        16,
    )
    sMbar = smem.allocate_tensor(
        cutlass.Int64, cute.make_layout((2 * n_stages,), stride=(1,)), 8
    )
    CHUNK_BYTES: cutlass.Constexpr[int] = CHUNK_ROWS * K * 2

    raw_slot = state_indices[i_n]

    q_src = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.Float32
    )
    k_src = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.Float32
    )
    d_src = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.Float32
    )
    q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.BFloat16
    )
    k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.BFloat16
    )
    g_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.BFloat16
    )
    dtb_f32 = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(cute.make_layout((vec,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec,), stride=(1,)), cutlass.Float32)
    r_d = cute.make_rmem_tensor(cute.make_layout((vec,), stride=(1,)), cutlass.Float32)
    r_hb = [
        [
            [
                cute.make_rmem_tensor(
                    cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
                )
                for _ in ROWS
            ]
            for _ in SUBI
        ]
        for _ in range(2)
    ]
    r_hb32 = [
        [[cute.recast_tensor(t, cutlass.Int32) for t in bufs] for bufs in par]
        for par in r_hb
    ]
    r_v_all = [
        cute.make_rmem_tensor(
            cute.make_layout((ilp_rows,), stride=(1,)), cutlass.BFloat16
        )
        for _ in range(ITERS_H * SUB)
    ]
    r_o_bf16 = cute.make_rmem_tensor(
        cute.make_layout((ilp_rows,), stride=(1,)), cutlass.BFloat16
    )

    slot_dead = (raw_slot < 0) | (cutlass.Int64(raw_slot) >= state.shape[0])
    if slot_dead:
        for r in ROWS:
            r_o_bf16[r] = cutlass.BFloat16(0.0)
        if k_lane == 0:
            for it in ITERI:
                for sub in SUBI:
                    vb = (
                        i_v * tile_v
                        + (it % ITERS_H) * CHUNK_ROWS
                        + (sub * _NUM_GROUPS + group_idx) * ilp_rows
                    )
                    ot = cute.local_tile(
                        o,
                        (1, 1, ilp_rows),
                        (i_n, i_h0 + it // ITERS_H, vb // ilp_rows),
                    )
                    cute.autovec_copy(r_o_bf16, ot)
    else:
        h_slots = [
            state[(cutlass.Int64(raw_slot), i_h0 + hh, None, None)] for hh in range(HPC)
        ]
        h_base = h_slots[0].iterator.toint()
        sh_base = cutlass.Int32(sH.iterator.toint())

        # L2 eviction steering: state writes pin as evict_last so their
        # write-back drains after kernel-end (in serving it overlaps the next
        # kernel); the read stream optionally marks evict_first.
        pol_w = cutlass.Int64(0)
        pol_r = cutlass.Int64(0)
        if cutlass.const_expr(l2_policy_mode >= 1):
            pol_w = _l2_policy_evict_last()
        if cutlass.const_expr(l2_policy_mode >= 2):
            pol_r = _l2_policy_evict_first()

        # Prologue: put the first D chunks in flight before the q/k/gate
        # staging math, which then hides their latency. In bulk-store mode one
        # ring slot stays spare so refills never wait on the newest bulk read.
        D: cutlass.Constexpr[int] = n_stages - (2 if bulk_store else 1)
        if cutlass.const_expr(tma_read):
            # TMA-1D ring: one bulk read per chunk, full/empty mbarriers
            # instead of LDGSTS groups + CTA barriers. All slots prefill.
            if tidx == 0:
                for s in range(n_stages):
                    cute.arch.mbarrier_init(sMbar.iterator + s, 1)
                    cute.arch.mbarrier_init(sMbar.iterator + n_stages + s, _NUM_THREADS)
            cute.arch.mbarrier_init_fence()
            cute.arch.barrier()
            if tidx == 0:
                for c in range(min(n_stages, ITERS)):
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        sMbar.iterator + c, CHUNK_BYTES
                    )
                    _cp_async_bulk_read(
                        sh_base + c * CHUNK_BYTES,
                        h_base + cutlass.Int64((i_v * tile_v + c * CHUNK_ROWS) * K * 2),
                        cutlass.Int32(sMbar.iterator.toint()) + c * 8,
                        CHUNK_BYTES,
                    )
        else:
            for c in range(min(D, ITERS)):
                grow_base = i_v * tile_v + c * CHUNK_ROWS + crow
                for j in CPJ:
                    elem_off = grow_base * K + ccol + j * vec
                    smem_byte = (
                        (c % n_stages) * CHUNK_ROWS * K + crow * K + ccol + j * vec
                    ) * 2
                    _cp_async_bf16x8_cg(
                        h_base, elem_off, sh_base + smem_byte, cp_l2_hint
                    )
                _cp_async_commit_group()

        # ------------------------------------------------------------------
        # Pipelined main loop: wait chunk -> barrier -> refill ring -> compute.
        # ------------------------------------------------------------------
        for it in ITERI:
            if cutlass.const_expr(it % ITERS_H == 0):
                # Head-boundary staging: q/k/gate/decay/beta/kq and this
                # head's v values. For heads after the first, the raw q/k/g/
                # dt loads were prefetched two chunks early into the (dead)
                # staging buffers, so only the staging math runs here.
                i_hh = i_h0 + it // ITERS_H
                if cutlass.const_expr(it == 0):
                    q_tile = cute.local_tile(q, (1, 1, vals), (i_n, i_hh, lane))
                    k_tile = cute.local_tile(k, (1, 1, vals), (i_n, i_hh, lane))
                    g_tile = cute.local_tile(g, (1, 1, vals), (i_n, i_hh, lane))
                    dtb_tile = cute.local_tile(dt_bias, (1, vals), (i_hh, lane))
                    cute.autovec_copy(q_tile, q_bf16)
                    cute.autovec_copy(k_tile, k_bf16)
                    cute.autovec_copy(g_tile, g_bf16)
                    cute.autovec_copy(dtb_tile, dtb_f32)

                if cutlass.const_expr(precomputed):
                    # Pre-computed convention: beta arrives already sigmoided.
                    a_exp = cutlass.Float32(0.0)
                    r_beta = cutlass.Float32(beta[(i_n, i_hh)])
                else:
                    a_exp = cute.exp(cutlass.Float32(A_log[i_hh]), fastmath=True)
                    b_logit = cutlass.Float32(beta[(i_n, i_hh)])
                    r_beta = cute.rcp(
                        cute.exp(-b_logit, fastmath=True) + 1.0, approx=True, ftz=True
                    )

                sum_q = cutlass.Float32(0.0)
                sum_k = cutlass.Float32(0.0)
                for i in VALSI:
                    q_val = cutlass.Float32(q_bf16[i])
                    k_val = cutlass.Float32(k_bf16[i])
                    q_src[i] = q_val
                    k_src[i] = k_val
                    sum_q += q_val * q_val
                    sum_k += k_val * k_val
                    if cutlass.const_expr(precomputed):
                        # g is the log-space decay: d = exp(g).
                        d_src[i] = cute.exp2(
                            cutlass.Float32(g_bf16[i]) * _LOG2_E, fastmath=True
                        )
                    else:
                        gate_x = cutlass.Float32(g_bf16[i]) + dtb_f32[i]
                        sig = cute.rcp(
                            cute.exp(-(a_exp * gate_x), fastmath=True) + 1.0,
                            approx=True,
                            ftz=True,
                        )
                        d_src[i] = cute.exp2(lb_log2e * sig, fastmath=True)

                for offset in [16, 8, 4, 2, 1]:
                    sum_q += cute.arch.shuffle_sync_bfly(
                        sum_q, offset=offset, mask=0xFFFFFFFF
                    )
                    sum_k += cute.arch.shuffle_sync_bfly(
                        sum_k, offset=offset, mask=0xFFFFFFFF
                    )
                inv_q = cute.rsqrt(sum_q + eps, fastmath=True) * scale
                inv_k = cute.rsqrt(sum_k + eps, fastmath=True)

                for i in VECI:
                    source_lane = 2 * k_lane + i // vals
                    source_value = i % vals
                    r_q[i] = (
                        cute.arch.shuffle_sync(
                            q_src[source_value], offset=source_lane, mask=0xFFFFFFFF
                        )
                        * inv_q
                    )
                    r_k[i] = (
                        cute.arch.shuffle_sync(
                            k_src[source_value], offset=source_lane, mask=0xFFFFFFFF
                        )
                        * inv_k
                    )
                    r_d[i] = cute.arch.shuffle_sync(
                        d_src[source_value], offset=source_lane, mask=0xFFFFFFFF
                    )

                # kq = (normalized q) . (normalized k), used by the fused output form
                # o = (h*d).q + vn * kq — one warp butterfly at staging instead of a
                # second dependent reduction tree in every iteration.
                kq_p = cutlass.Float32(0.0)
                for i in VALSI:
                    kq_p += q_src[i] * k_src[i]
                for offset in [16, 8, 4, 2, 1]:
                    kq_p += cute.arch.shuffle_sync_bfly(
                        kq_p, offset=offset, mask=0xFFFFFFFF
                    )
                r_kq = kq_p * (inv_q * inv_k)

                for itl in ITERI_H:
                    for sub in SUBI:
                        vb_it = (
                            i_v * tile_v
                            + itl * CHUNK_ROWS
                            + (sub * _NUM_GROUPS + group_idx) * ilp_rows
                        )
                        vt = cute.local_tile(
                            v, (1, 1, ilp_rows), (i_n, i_hh, vb_it // ilp_rows)
                        )
                        cute.autovec_copy(vt, r_v_all[itl * SUB + sub])

            if cutlass.const_expr(private_ring):
                # Per-thread deep wait: own bytes of chunk it+1 resident, so
                # the prefetched LDS below needs no CTA barrier — except
                # while PROLOGUE chunks are being consumed: the prologue uses
                # the shared crow/ccol issue geometry (better warp
                # coalescing), so chunks 0..D-1 arrive via other threads'
                # copies and each needs a publish after the matching wait.
                _cp_async_wait_group_n(
                    max(0, min(D - 2, ITERS - 2 - it)) if it + 1 < ITERS else 0
                )
                if cutlass.const_expr(it <= D - 2):
                    cute.arch.barrier()
                if cutlass.const_expr(it + D < ITERS):
                    c = it + D
                    for sub2 in SUBI:
                        for r in ROWS:
                            lrow = (sub2 * _NUM_GROUPS + group_idx) * ilp_rows + r
                            elem_off = (
                                i_v * tile_v + c * CHUNK_ROWS + lrow
                            ) * K + k_lane * vec
                            smem_byte = (
                                ((c % n_stages) * CHUNK_ROWS + lrow) * K + k_lane * vec
                            ) * 2
                            _cp_async_bf16x8_cg(
                                h_base, elem_off, sh_base + smem_byte, cp_l2_hint
                            )
                    _cp_async_commit_group()
            elif cutlass.const_expr(tma_read):
                cute.arch.mbarrier_wait(
                    sMbar.iterator + (it % n_stages), (it // n_stages) & 1
                )
            else:
                # Deep wait: chunk it+1 must also be resident so its LDS can
                # issue this iteration and cover a full compute phase.
                _cp_async_wait_group_n(
                    max(0, min(D - 2, ITERS - 2 - it)) if it + 1 < ITERS else 0
                )
                if cutlass.const_expr(bulk_store):
                    if cutlass.const_expr(it >= 2):
                        # The slot refilled below was bulk-stored two
                        # iterations ago; allow the newest bulk group to stay
                        # in flight.
                        _cp_async_bulk_wait_read(1)
                    elif cutlass.const_expr(it == 1):
                        _cp_async_bulk_wait_read(0)
                cute.arch.barrier()

                if cutlass.const_expr(it + D < ITERS):
                    c = it + D
                    grow_base = i_v * tile_v + c * CHUNK_ROWS + crow
                    for j in CPJ:
                        elem_off = grow_base * K + ccol + j * vec
                        smem_byte = (
                            (c % n_stages) * CHUNK_ROWS * K + crow * K + ccol + j * vec
                        ) * 2
                        if cutlass.const_expr(l2_policy_mode >= 2):
                            _cp_async_bf16x8_cg_hint(
                                h_base, elem_off, sh_base + smem_byte, pol_r
                            )
                        else:
                            _cp_async_bf16x8_cg(
                                h_base, elem_off, sh_base + smem_byte, cp_l2_hint
                            )
                    _cp_async_commit_group()

            h_slot = h_slots[it // ITERS_H]
            cur = it % 2
            if cutlass.const_expr(it % ITERS_H == ITERS_H - 2 and it + 2 < ITERS):
                # Prefetch the NEXT head's raw staging inputs into the dead
                # bf16 buffers; their latency rides the remaining two chunks.
                i_hn = i_h0 + it // ITERS_H + 1
                q_tile = cute.local_tile(q, (1, 1, vals), (i_n, i_hn, lane))
                k_tile = cute.local_tile(k, (1, 1, vals), (i_n, i_hn, lane))
                g_tile = cute.local_tile(g, (1, 1, vals), (i_n, i_hn, lane))
                dtb_tile = cute.local_tile(dt_bias, (1, vals), (i_hn, lane))
                cute.autovec_copy(q_tile, q_bf16)
                cute.autovec_copy(k_tile, k_bf16)
                cute.autovec_copy(g_tile, g_bf16)
                cute.autovec_copy(dtb_tile, dtb_f32)
            if cutlass.const_expr(it == 0):
                stage0 = sH[(0, None, None)]
                for sub in SUBI:
                    for r in ROWS:
                        hs = cute.local_tile(
                            stage0,
                            (1, vec),
                            (
                                (sub * _NUM_GROUPS + group_idx) * ilp_rows + r,
                                k_lane,
                            ),
                        )
                        cute.autovec_copy(hs, r_hb[0][sub][r])
            if cutlass.const_expr(it + 1 < ITERS):
                stage_n = sH[((it + 1) % n_stages, None, None)]
                for sub in SUBI:
                    for r in ROWS:
                        hs = cute.local_tile(
                            stage_n,
                            (1, vec),
                            (
                                (sub * _NUM_GROUPS + group_idx) * ilp_rows + r,
                                k_lane,
                            ),
                        )
                        cute.autovec_copy(hs, r_hb[1 - cur][sub][r])
            for sub in SUBI:
                vb = (
                    i_v * tile_v
                    + (it % ITERS_H) * CHUNK_ROWS
                    + (sub * _NUM_GROUPS + group_idx) * ilp_rows
                )
                if cutlass.const_expr(tma_read and sub == SUB - 1):
                    # Slot consumed into registers: free it and let thread 0
                    # refill immediately — the bulk read streams during the
                    # remaining compute of this iteration.
                    cute.arch.mbarrier_arrive(
                        sMbar.iterator + n_stages + (it % n_stages)
                    )
                    if cutlass.const_expr(it + n_stages < ITERS):
                        if tidx == 0:
                            cute.arch.mbarrier_wait(
                                sMbar.iterator + n_stages + (it % n_stages),
                                (it // n_stages) & 1,
                            )
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                sMbar.iterator + (it % n_stages), CHUNK_BYTES
                            )
                            _cp_async_bulk_read(
                                sh_base + (it % n_stages) * CHUNK_BYTES,
                                h_base
                                + cutlass.Int64(
                                    (i_v * tile_v + (it + n_stages) * CHUNK_ROWS)
                                    * K
                                    * 2
                                ),
                                cutlass.Int32(sMbar.iterator.toint())
                                + (it % n_stages) * 8,
                                CHUNK_BYTES,
                            )
                # Pass 1 (fused): s = (h*d).k and od = (h*d).q in one sweep;
                # their butterflies interleave (independent trees), and pass 2
                # no longer carries an output reduction.
                s_e = [cutlass.Float32(0.0) for _ in ROWS]
                s_o = [cutlass.Float32(0.0) for _ in ROWS]
                o_e = [cutlass.Float32(0.0) for _ in ROWS]
                o_o = [cutlass.Float32(0.0) for _ in ROWS]
                for i in VECP:
                    for r in ROWS:
                        h0, h1 = _bf16x2_to_f32x2(r_hb32[cur][sub][r][i // 2])
                        if cutlass.const_expr(use_packed_fma):
                            h0, h1 = cute.arch.fma_packed_f32x2(
                                src_a=(h0, h1),
                                src_b=(r_d[i], r_d[i + 1]),
                                src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                            )
                            s_e[r], s_o[r] = cute.arch.fma_packed_f32x2(
                                src_a=(h0, h1),
                                src_b=(r_k[i], r_k[i + 1]),
                                src_c=(s_e[r], s_o[r]),
                            )
                            o_e[r], o_o[r] = cute.arch.fma_packed_f32x2(
                                src_a=(h0, h1),
                                src_b=(r_q[i], r_q[i + 1]),
                                src_c=(o_e[r], o_o[r]),
                            )
                        else:
                            h0 = h0 * r_d[i]
                            h1 = h1 * r_d[i + 1]
                            s_e[r] = s_e[r] + h0 * r_k[i]
                            s_o[r] = s_o[r] + h1 * r_k[i + 1]
                            o_e[r] = o_e[r] + h0 * r_q[i]
                            o_o[r] = o_o[r] + h1 * r_q[i + 1]
                s_red = [s_e[r] + s_o[r] for r in ROWS]
                od = [o_e[r] + o_o[r] for r in ROWS]
                for offset in [8, 4, 2, 1]:
                    for r in ROWS:
                        s_red[r] += cute.arch.shuffle_sync_bfly(
                            s_red[r], offset=offset, mask=0xFFFFFFFF
                        )
                        od[r] += cute.arch.shuffle_sync_bfly(
                            od[r], offset=offset, mask=0xFFFFFFFF
                        )

                vn = [
                    (cutlass.Float32(r_v_all[(it % ITERS_H) * SUB + sub][r]) - s_red[r])
                    * r_beta
                    for r in ROWS
                ]

                # Output is already reduced: o = (h*d).q + vn * kq.
                if k_lane == 0:
                    for r in ROWS:
                        r_o_bf16[r] = cutlass.BFloat16(od[r] + vn[r] * r_kq)
                    ot = cute.local_tile(
                        o, (1, 1, ilp_rows), (i_n, i_hh, vb // ilp_rows)
                    )
                    cute.autovec_copy(r_o_bf16, ot)

                # Pass 2 (store only): h_new = h*decay + k*vn, repack bf16.
                for i in VECP:
                    for r in ROWS:
                        h0, h1 = _bf16x2_to_f32x2(r_hb32[cur][sub][r][i // 2])
                        if cutlass.const_expr(use_packed_fma):
                            h0, h1 = cute.arch.fma_packed_f32x2(
                                src_a=(h0, h1),
                                src_b=(r_d[i], r_d[i + 1]),
                                src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                            )
                            h0, h1 = cute.arch.fma_packed_f32x2(
                                src_a=(r_k[i], r_k[i + 1]),
                                src_b=(vn[r], vn[r]),
                                src_c=(h0, h1),
                            )
                        else:
                            h0 = h0 * r_d[i]
                            h1 = h1 * r_d[i + 1]
                            h0 = r_k[i] * vn[r] + h0
                            h1 = r_k[i + 1] * vn[r] + h1
                        r_hb32[cur][sub][r][i // 2] = _f32x2_to_bf16x2(h0, h1)

                if cutlass.const_expr(bulk_store):
                    # h_new goes back into this chunk's smem slot (warp-local
                    # rows), then one bulk store per warp pushes it out via the
                    # async proxy.
                    stage_v = sH[(it % n_stages, None, None)]
                    for r in ROWS:
                        hs_w = cute.local_tile(
                            stage_v,
                            (1, vec),
                            ((sub * _NUM_GROUPS + group_idx) * ilp_rows + r, k_lane),
                        )
                        cute.autovec_copy(r_hb[cur][sub][r], hs_w)
                    _sync_warp()
                    if lane == 0:
                        _fence_proxy_async()
                        row0 = it * CHUNK_ROWS + warp_idx * WARP_ROWS
                        g_addr = h_base + cutlass.Int64((i_v * tile_v + row0) * K * 2)
                        s_addr = sh_base + (
                            ((it % n_stages) * CHUNK_ROWS + warp_idx * WARP_ROWS)
                            * K
                            * 2
                        )
                        _cp_async_bulk_store(g_addr, s_addr, WARP_ROWS * K * 2)
                        _cp_async_bulk_commit()
                else:
                    if cutlass.const_expr(l2_policy_mode >= 1):
                        for r in ROWS:
                            st_addr = h_base + cutlass.Int64(
                                ((it // ITERS_H) * V * K + (vb + r) * K + k_lane * vec)
                                * 2
                            )
                            _stg_hint_v4_b32(
                                st_addr,
                                r_hb32[cur][sub][r][0],
                                r_hb32[cur][sub][r][1],
                                r_hb32[cur][sub][r][2],
                                r_hb32[cur][sub][r][3],
                                pol_w,
                            )
                    else:
                        for r in ROWS:
                            ht = cute.local_tile(h_slot, (1, vec), (vb + r, k_lane))
                            cute.autovec_copy(r_hb[cur][sub][r], ht)

        if cutlass.const_expr(bulk_store):
            # Writes must be complete (not just smem-read) before exit.
            _cp_async_bulk_wait(0)


@cute.kernel
def _kda_packed_t1_persist_kernel(
    state: cute.Tensor,  # [pool, H, V, K] bf16, padded slot stride allowed
    q: cute.Tensor,  # [B, H, K] bf16 strided view of mixed_qkv
    k: cute.Tensor,  # [B, H, K] bf16 strided view of mixed_qkv
    v: cute.Tensor,  # [B, H, V] bf16 strided view of mixed_qkv
    g: cute.Tensor,  # [B, H, K] bf16 strided view of raw_gate
    beta: cute.Tensor,  # [B, H] bf16 (row stride dynamic)
    A_log: cute.Tensor,  # [H] f32
    dt_bias: cute.Tensor,  # [H, K] f32
    o: cute.Tensor,  # [B, H, V] bf16 contiguous
    state_indices: cute.Tensor,  # [B] i32
    scale: cutlass.Constexpr[float],
    eps: cutlass.Constexpr[float],
    lb_log2e: cutlass.Constexpr[float],
    precomputed: cutlass.Constexpr[bool],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    n_stages: cutlass.Constexpr[int],
    cp_l2_hint: cutlass.Constexpr[int],
    use_packed_fma: cutlass.Constexpr[bool],
):
    """Persistent v2: the champion pipeline with items streamed per CTA.

    Composition of every proven win — fused output form, LDS double-buffer,
    barrier-free private cp.async ring — with a grid-stride item loop whose
    ring NEVER drains: chunk look-ahead and staging loads cross item
    boundaries, so the per-item fill bubble (the dominant remaining stall)
    is paid once per CTA instead of once per item. Requires
    ITERS_H % n_stages == 0 (slot/parity arithmetic stays compile-time) and
    a 16-byte-aligned pool. One item = one whole (batch, head) head block.
    """
    ilp_rows: cutlass.Constexpr[int] = 2
    CHUNK_ROWS: cutlass.Constexpr[int] = 16
    ITERS: cutlass.Constexpr[int] = V // CHUNK_ROWS  # 8 chunks per item
    P: cutlass.Constexpr[int] = n_stages - 1  # in-flight depth (2-cold refill)

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()

    B = cutlass.Int32(q.shape[0])
    num_items = B * H

    lane = tidx % 32
    k_lane = tidx % _LANES_PER_ROW
    group_idx = tidx // _LANES_PER_ROW

    vec: cutlass.Constexpr[int] = _ELEMS_PER_LANE
    vals: cutlass.Constexpr[int] = _VALS_PER_LANE
    ROWS = tuple(range(ilp_rows))
    VECI = tuple(range(vec))
    VECP = tuple(range(0, vec, 2))
    VALSI = tuple(range(vals))
    ITERI = tuple(range(ITERS))

    smem = cutlass.utils.SmemAllocator()
    sH = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((n_stages, CHUNK_ROWS, K), stride=(CHUNK_ROWS * K, K, 1)),
        16,
    )
    sh_base = cutlass.Int32(sH.iterator.toint())

    q_src = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.Float32
    )
    k_src = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.Float32
    )
    d_src = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.Float32
    )
    q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.BFloat16
    )
    k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.BFloat16
    )
    g_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.BFloat16
    )
    dtb_f32 = cute.make_rmem_tensor(
        cute.make_layout((vals,), stride=(1,)), cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(cute.make_layout((vec,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec,), stride=(1,)), cutlass.Float32)
    r_d = cute.make_rmem_tensor(cute.make_layout((vec,), stride=(1,)), cutlass.Float32)
    r_hb = [
        [
            cute.make_rmem_tensor(
                cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
            )
            for _ in ROWS
        ]
        for _ in range(2)
    ]
    r_hb32 = [[cute.recast_tensor(t, cutlass.Int32) for t in bufs] for bufs in r_hb]
    r_v_all = [
        cute.make_rmem_tensor(
            cute.make_layout((ilp_rows,), stride=(1,)), cutlass.BFloat16
        )
        for _ in ITERI
    ]
    r_o_bf16 = cute.make_rmem_tensor(
        cute.make_layout((ilp_rows,), stride=(1,)), cutlass.BFloat16
    )

    # Item metadata: current + look-ahead (byte base of the 32 KiB head
    # block; ok flags gate compute and refills of inactive/out-of-range).
    item_cur = cutlass.Int32(bidx)
    i_n_cur = item_cur // H
    i_h_cur = item_cur % H
    slot_cur = state_indices[i_n_cur]
    base_cur = cutlass.Int64(0)
    ok_cur = (slot_cur >= 0) & (cutlass.Int64(slot_cur) < state.shape[0])
    if ok_cur:
        base_cur = state[
            (cutlass.Int64(slot_cur), i_h_cur, None, None)
        ].iterator.toint()

    item_nxt = item_cur + gdim
    i_n_nxt = item_nxt // H
    i_h_nxt = item_nxt % H
    ok_nxt = item_nxt < num_items
    base_nxt = cutlass.Int64(0)
    if ok_nxt:
        slot_nxt = state_indices[i_n_nxt]
        if (slot_nxt >= 0) & (cutlass.Int64(slot_nxt) < state.shape[0]):
            base_nxt = state[
                (cutlass.Int64(slot_nxt), i_h_nxt, None, None)
            ].iterator.toint()
        else:
            ok_nxt = False

    # Prologue: first P chunks of item 0 (empty commits when inactive) —
    # self-service geometry, each thread fetches only the bytes it consumes.
    for c in range(P):
        if ok_cur:
            for r in ROWS:
                lrow = group_idx * ilp_rows + r
                elem_off = (c * CHUNK_ROWS + lrow) * K + k_lane * vec
                smem_byte = (
                    ((c % n_stages) * CHUNK_ROWS + lrow) * K + k_lane * vec
                ) * 2
                _cp_async_bf16x8_cg(base_cur, elem_off, sh_base + smem_byte, cp_l2_hint)
        _cp_async_commit_group()

    n_my_items = (num_items - item_cur + gdim - 1) // gdim
    for _j in cutlass.range(n_my_items, unroll=1):
        # ------------------------------------------------------------------
        # Per-item staging (q/k/g/dt loads land here for the FIRST item;
        # later items' loads were prefetched during the previous item and
        # only the math runs). The ring keeps streaming underneath.
        # ------------------------------------------------------------------
        if _j == 0:
            q_tile = cute.local_tile(q, (1, 1, vals), (i_n_cur, i_h_cur, lane))
            k_tile = cute.local_tile(k, (1, 1, vals), (i_n_cur, i_h_cur, lane))
            g_tile = cute.local_tile(g, (1, 1, vals), (i_n_cur, i_h_cur, lane))
            dtb_tile = cute.local_tile(dt_bias, (1, vals), (i_h_cur, lane))
            cute.autovec_copy(q_tile, q_bf16)
            cute.autovec_copy(k_tile, k_bf16)
            cute.autovec_copy(g_tile, g_bf16)
            cute.autovec_copy(dtb_tile, dtb_f32)

        if cutlass.const_expr(precomputed):
            # Pre-computed convention: beta arrives already sigmoided.
            a_exp = cutlass.Float32(0.0)
            r_beta = cutlass.Float32(beta[(i_n_cur, i_h_cur)])
        else:
            a_exp = cute.exp(cutlass.Float32(A_log[i_h_cur]), fastmath=True)
            b_logit = cutlass.Float32(beta[(i_n_cur, i_h_cur)])
            r_beta = cute.rcp(
                cute.exp(-b_logit, fastmath=True) + 1.0, approx=True, ftz=True
            )

        sum_q = cutlass.Float32(0.0)
        sum_k = cutlass.Float32(0.0)
        for i in VALSI:
            q_val = cutlass.Float32(q_bf16[i])
            k_val = cutlass.Float32(k_bf16[i])
            q_src[i] = q_val
            k_src[i] = k_val
            sum_q += q_val * q_val
            sum_k += k_val * k_val
            if cutlass.const_expr(precomputed):
                # g is the log-space decay: d = exp(g).
                d_src[i] = cute.exp2(
                    cutlass.Float32(g_bf16[i]) * _LOG2_E, fastmath=True
                )
            else:
                gate_x = cutlass.Float32(g_bf16[i]) + dtb_f32[i]
                sig = cute.rcp(
                    cute.exp(-(a_exp * gate_x), fastmath=True) + 1.0,
                    approx=True,
                    ftz=True,
                )
                d_src[i] = cute.exp2(lb_log2e * sig, fastmath=True)

        kq_p = cutlass.Float32(0.0)
        for i in VALSI:
            kq_p += q_src[i] * k_src[i]
        for offset in [16, 8, 4, 2, 1]:
            sum_q += cute.arch.shuffle_sync_bfly(sum_q, offset=offset, mask=0xFFFFFFFF)
            sum_k += cute.arch.shuffle_sync_bfly(sum_k, offset=offset, mask=0xFFFFFFFF)
            kq_p += cute.arch.shuffle_sync_bfly(kq_p, offset=offset, mask=0xFFFFFFFF)
        inv_q = cute.rsqrt(sum_q + eps, fastmath=True) * scale
        inv_k = cute.rsqrt(sum_k + eps, fastmath=True)
        r_kq = kq_p * (inv_q * inv_k)

        for i in VECI:
            source_lane = 2 * k_lane + i // vals
            source_value = i % vals
            r_q[i] = (
                cute.arch.shuffle_sync(
                    q_src[source_value], offset=source_lane, mask=0xFFFFFFFF
                )
                * inv_q
            )
            r_k[i] = (
                cute.arch.shuffle_sync(
                    k_src[source_value], offset=source_lane, mask=0xFFFFFFFF
                )
                * inv_k
            )
            r_d[i] = cute.arch.shuffle_sync(
                d_src[source_value], offset=source_lane, mask=0xFFFFFFFF
            )

        for it in ITERI:
            vt = cute.local_tile(
                v,
                (1, 1, ilp_rows),
                (
                    i_n_cur,
                    i_h_cur,
                    (it * CHUNK_ROWS + group_idx * ilp_rows) // ilp_rows,
                ),
            )
            cute.autovec_copy(vt, r_v_all[it])

        # ------------------------------------------------------------------
        # 8 chunks; refills and the LDS prefetch cross into the next item.
        # ------------------------------------------------------------------
        for it in ITERI:
            cur = it % 2
            # Per-thread wait: own bytes of chunks p and p+1 resident
            # (steady-state constant count; tail passes trivially).
            _cp_async_wait_group_n(P - 2 if P >= 2 else 0)

            # Refill chunk p+P (slot (p-1) % n: two iterations cold).
            if cutlass.const_expr(it + P < ITERS):
                c = it + P
                if ok_cur:
                    for r in ROWS:
                        lrow = group_idx * ilp_rows + r
                        elem_off = (c * CHUNK_ROWS + lrow) * K + k_lane * vec
                        smem_byte = (
                            (((it + P) % n_stages) * CHUNK_ROWS + lrow) * K
                            + k_lane * vec
                        ) * 2
                        _cp_async_bf16x8_cg(
                            base_cur, elem_off, sh_base + smem_byte, cp_l2_hint
                        )
                _cp_async_commit_group()
            else:
                c = it + P - ITERS
                if ok_nxt:
                    for r in ROWS:
                        lrow = group_idx * ilp_rows + r
                        elem_off = (c * CHUNK_ROWS + lrow) * K + k_lane * vec
                        smem_byte = (
                            (((it + P) % n_stages) * CHUNK_ROWS + lrow) * K
                            + k_lane * vec
                        ) * 2
                        _cp_async_bf16x8_cg(
                            base_nxt, elem_off, sh_base + smem_byte, cp_l2_hint
                        )
                _cp_async_commit_group()

            # First chunk of the very first item lands in buffer 0 here.
            if cutlass.const_expr(it == 0):
                if _j == 0:
                    stage0 = sH[(0, None, None)]
                    for r in ROWS:
                        hs = cute.local_tile(
                            stage0, (1, vec), (group_idx * ilp_rows + r, k_lane)
                        )
                        cute.autovec_copy(hs, r_hb[0][r])
            # LDS prefetch of chunk p+1 (crosses into the next item at it=7;
            # parity (it+1)%2 stays compile-time since ITERS is even).
            stage_n = sH[((it + 1) % n_stages, None, None)]
            for r in ROWS:
                hs = cute.local_tile(
                    stage_n, (1, vec), (group_idx * ilp_rows + r, k_lane)
                )
                cute.autovec_copy(hs, r_hb[1 - cur][r])

            # Staging prefetch for the next item, mid-item (dead buffers).
            if cutlass.const_expr(it == ITERS - 2):
                if ok_nxt:
                    q_tile = cute.local_tile(q, (1, 1, vals), (i_n_nxt, i_h_nxt, lane))
                    k_tile = cute.local_tile(k, (1, 1, vals), (i_n_nxt, i_h_nxt, lane))
                    g_tile = cute.local_tile(g, (1, 1, vals), (i_n_nxt, i_h_nxt, lane))
                    dtb_tile = cute.local_tile(dt_bias, (1, vals), (i_h_nxt, lane))
                    cute.autovec_copy(q_tile, q_bf16)
                    cute.autovec_copy(k_tile, k_bf16)
                    cute.autovec_copy(g_tile, g_bf16)
                    cute.autovec_copy(dtb_tile, dtb_f32)

            vb = it * CHUNK_ROWS + group_idx * ilp_rows
            hb32_cur = r_hb32[cur]
            r_v_it = r_v_all[it]
            if ok_cur:
                # Pass 1 (fused): s = (h*d).k and od = (h*d).q in one sweep.
                s_e = [cutlass.Float32(0.0) for _ in ROWS]
                s_o = [cutlass.Float32(0.0) for _ in ROWS]
                o_e = [cutlass.Float32(0.0) for _ in ROWS]
                o_o = [cutlass.Float32(0.0) for _ in ROWS]
                for i in VECP:
                    for r in ROWS:
                        h0, h1 = _bf16x2_to_f32x2(hb32_cur[r][i // 2])
                        h0, h1 = cute.arch.fma_packed_f32x2(
                            src_a=(h0, h1),
                            src_b=(r_d[i], r_d[i + 1]),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        s_e[r], s_o[r] = cute.arch.fma_packed_f32x2(
                            src_a=(h0, h1),
                            src_b=(r_k[i], r_k[i + 1]),
                            src_c=(s_e[r], s_o[r]),
                        )
                        o_e[r], o_o[r] = cute.arch.fma_packed_f32x2(
                            src_a=(h0, h1),
                            src_b=(r_q[i], r_q[i + 1]),
                            src_c=(o_e[r], o_o[r]),
                        )
                s = [s_e[r] + s_o[r] for r in ROWS]
                od = [o_e[r] + o_o[r] for r in ROWS]
                for offset in [8, 4, 2, 1]:
                    for r in ROWS:
                        s[r] += cute.arch.shuffle_sync_bfly(
                            s[r], offset=offset, mask=0xFFFFFFFF
                        )
                        od[r] += cute.arch.shuffle_sync_bfly(
                            od[r], offset=offset, mask=0xFFFFFFFF
                        )

                vn = [(cutlass.Float32(r_v_it[r]) - s[r]) * r_beta for r in ROWS]

                if k_lane == 0:
                    for r in ROWS:
                        r_o_bf16[r] = cutlass.BFloat16(od[r] + vn[r] * r_kq)
                    ot = cute.local_tile(
                        o, (1, 1, ilp_rows), (i_n_cur, i_h_cur, vb // ilp_rows)
                    )
                    cute.autovec_copy(r_o_bf16, ot)

                # Pass 2 (store only): h_new = h*d + k*vn, repack, STG.128.
                for i in VECP:
                    for r in ROWS:
                        h0, h1 = _bf16x2_to_f32x2(hb32_cur[r][i // 2])
                        h0, h1 = cute.arch.fma_packed_f32x2(
                            src_a=(h0, h1),
                            src_b=(r_d[i], r_d[i + 1]),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        h0, h1 = cute.arch.fma_packed_f32x2(
                            src_a=(r_k[i], r_k[i + 1]),
                            src_b=(vn[r], vn[r]),
                            src_c=(h0, h1),
                        )
                        hb32_cur[r][i // 2] = _f32x2_to_bf16x2(h0, h1)

                for r in ROWS:
                    st_addr = base_cur + cutlass.Int64(
                        ((vb + r) * K + k_lane * vec) * 2
                    )
                    _stg_v4_b32(
                        st_addr,
                        hb32_cur[r][0],
                        hb32_cur[r][1],
                        hb32_cur[r][2],
                        hb32_cur[r][3],
                    )
            else:
                for r in ROWS:
                    r_o_bf16[r] = cutlass.BFloat16(0.0)
                if k_lane == 0:
                    ot = cute.local_tile(
                        o, (1, 1, ilp_rows), (i_n_cur, i_h_cur, vb // ilp_rows)
                    )
                    cute.autovec_copy(r_o_bf16, ot)

        # Rotate item metadata.
        item_cur = item_nxt
        i_n_cur = i_n_nxt
        i_h_cur = i_h_nxt
        base_cur = base_nxt
        ok_cur = ok_nxt
        item_nxt = item_cur + gdim
        i_n_nxt = item_nxt // H
        i_h_nxt = item_nxt % H
        ok_nxt = item_nxt < num_items
        base_nxt = cutlass.Int64(0)
        if ok_nxt:
            slot_n2 = state_indices[i_n_nxt]
            if (slot_n2 >= 0) & (cutlass.Int64(slot_n2) < state.shape[0]):
                base_nxt = state[
                    (cutlass.Int64(slot_n2), i_h_nxt, None, None)
                ].iterator.toint()
            else:
                ok_nxt = False


@cute.jit
def _kda_packed_t1_persist_launch(
    state: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    A_log: cute.Tensor,
    dt_bias: cute.Tensor,
    o: cute.Tensor,
    state_indices: cute.Tensor,
    grid_ctas: cutlass.Int32,
    stream: cuda.CUstream,
    scale: cutlass.Constexpr[float],
    eps: cutlass.Constexpr[float],
    lb_log2e: cutlass.Constexpr[float],
    precomputed: cutlass.Constexpr[bool],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    n_stages: cutlass.Constexpr[int],
    cp_l2_hint: cutlass.Constexpr[int],
    use_packed_fma: cutlass.Constexpr[bool],
):
    smem_bytes = n_stages * 16 * K * 2 + 128
    _kda_packed_t1_persist_kernel(
        state,
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        o,
        state_indices,
        scale,
        eps,
        lb_log2e,
        precomputed,
        H,
        K,
        V,
        n_stages,
        cp_l2_hint,
        use_packed_fma,
    ).launch(
        grid=[grid_ctas, 1, 1],
        block=[_NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


@cute.jit
def _kda_packed_t1_smem_launch(
    state: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    A_log: cute.Tensor,
    dt_bias: cute.Tensor,
    o: cute.Tensor,
    state_indices: cute.Tensor,
    stream: cuda.CUstream,
    scale: cutlass.Constexpr[float],
    eps: cutlass.Constexpr[float],
    lb_log2e: cutlass.Constexpr[float],
    precomputed: cutlass.Constexpr[bool],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    ilp_rows: cutlass.Constexpr[int],
    n_stages: cutlass.Constexpr[int],
    chunk_rows: cutlass.Constexpr[int],
    cp_l2_hint: cutlass.Constexpr[int],
    heads_per_cta: cutlass.Constexpr[int],
    bulk_store: cutlass.Constexpr[bool],
    tma_read: cutlass.Constexpr[bool],
    l2_policy_mode: cutlass.Constexpr[int],
    private_ring: cutlass.Constexpr[bool],
    min_blocks: cutlass.Constexpr[int],
    use_packed_fma: cutlass.Constexpr[bool],
):
    num_v_tiles: cutlass.Constexpr[int] = V // tile_v
    B = cute.size(q.shape[0])
    grid_size = B * (H // heads_per_cta) * num_v_tiles
    smem_bytes = n_stages * chunk_rows * K * 2 + 2 * n_stages * 8 + 128
    _kda_packed_t1_smem_kernel(
        state,
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        o,
        state_indices,
        scale,
        eps,
        lb_log2e,
        precomputed,
        H,
        K,
        V,
        tile_v,
        num_v_tiles,
        ilp_rows,
        n_stages,
        chunk_rows,
        cp_l2_hint,
        heads_per_cta,
        bulk_store,
        tma_read,
        l2_policy_mode,
        private_ring,
        use_packed_fma,
    ).launch(
        grid=[grid_size, 1, 1],
        block=[_NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
        # Full carveout: the ring buffer is the only smem client and the
        # default split caps residency below the register limit.
        preferred_smem_carveout=100,
        # __launch_bounds__ min-blocks: bounds the register allocation so the
        # requested CTA residency is achievable (0 = compiler's choice).
        min_blocks_per_mp=min_blocks,
    )


@cute.jit
def _kda_packed_t1_launch(
    state: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    A_log: cute.Tensor,
    dt_bias: cute.Tensor,
    o: cute.Tensor,
    state_indices: cute.Tensor,
    stream: cuda.CUstream,
    scale: cutlass.Constexpr[float],
    eps: cutlass.Constexpr[float],
    lb_log2e: cutlass.Constexpr[float],
    precomputed: cutlass.Constexpr[bool],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    ilp_rows: cutlass.Constexpr[int],
    num_groups: cutlass.Constexpr[int],
    use_packed_fma: cutlass.Constexpr[bool],
    evict_first_state: cutlass.Constexpr[bool],
):
    num_v_tiles: cutlass.Constexpr[int] = V // tile_v
    B = cute.size(q.shape[0])
    grid_size = B * H * num_v_tiles
    _kda_packed_t1_kernel(
        state,
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        o,
        state_indices,
        scale,
        eps,
        lb_log2e,
        precomputed,
        H,
        K,
        V,
        tile_v,
        num_v_tiles,
        ilp_rows,
        num_groups,
        use_packed_fma,
        evict_first_state,
    ).launch(
        grid=[grid_size, 1, 1],
        block=[num_groups * _LANES_PER_ROW, 1, 1],
        stream=stream,
        # No shared memory: leave the unified cache to L1 (q/k/gate rows are
        # re-read by all warps and by every V-tile CTA of a head).
        preferred_smem_carveout=0,
    )


def _make_compile_inputs(
    qkv_div: int, gate_div: int, pool_div: int, aux_aligned: bool = True
):
    """Build symbolic compile-time tensor specs.

    ``*_div`` are element divisibility guarantees for the dynamic outer
    strides; 8 elements = 16 bytes lets autovec emit LDG.128/STG.128 on bf16,
    1 disables vectorisation (compatibility path for oddly padded callers).
    """
    B = cute.sym_int()
    N = cute.sym_int()
    H, K, V = _HEADS, _HEAD_DIM, _HEAD_DIM

    def align_for(div: int) -> int:
        return 16 if div % 8 == 0 else 2

    def mixed_view(div: int):
        return cute.runtime.make_fake_tensor(
            cute.BFloat16,
            shape=(B, H, K),
            stride=(cute.sym_int64(divisibility=div), K, 1),
            assumed_align=align_for(div),
        )

    state_fake = cute.runtime.make_fake_tensor(
        cute.BFloat16,
        shape=(N, H, V, K),
        stride=(cute.sym_int64(divisibility=pool_div), V * K, K, 1),
        assumed_align=align_for(pool_div),
    )
    gate_fake = cute.runtime.make_fake_tensor(
        cute.BFloat16,
        shape=(B, H, K),
        stride=(cute.sym_int64(divisibility=gate_div), K, 1),
        assumed_align=align_for(gate_div),
    )
    beta_fake = cute.runtime.make_fake_tensor(
        cute.BFloat16,
        shape=(B, H),
        stride=(cute.sym_int64(divisibility=1), 1),
        assumed_align=2,
    )

    def make_compact(shape, dtype=cute.BFloat16):
        # dt_bias / output may sit at any element-aligned address (e.g.
        # views shifted off a 16 B boundary); only claim 16 B when the
        # caller verified it.
        if aux_aligned:
            align = 16
        else:
            align = 4 if dtype is cute.Float32 else 2
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            assumed_align=align,
            stride_order=tuple(reversed(range(len(shape)))),
        )

    a_log_fake = cute.runtime.make_fake_compact_tensor(
        cute.Float32, (H,), assumed_align=4, stride_order=(0,)
    )
    idx_fake = cute.runtime.make_fake_compact_tensor(
        cute.Int32, (cute.sym_int(),), assumed_align=4, stride_order=(0,)
    )

    return (
        state_fake,
        mixed_view(qkv_div),
        mixed_view(qkv_div),
        mixed_view(qkv_div),
        gate_fake,
        beta_fake,
        a_log_fake,
        make_compact((H, K), dtype=cute.Float32),
        make_compact((B, H, V)),
        idx_fake,
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )


@functools.cache
def _get_compiled(
    tile_v: int,
    ilp_rows: int,
    qkv_div: int,
    gate_div: int,
    pool_div: int,
    aux_aligned: bool,
    use_packed_fma: bool,
    evict_first_state: bool,
    maxrreg: int = 0,
    n_stages: int = 0,
    num_groups: int = _NUM_GROUPS,
    cp_l2_hint: int = 0,
    chunk_rows: int = 0,
    bulk_store: bool = False,
    heads_per_cta: int = 1,
    tma_read: bool = False,
    persistent: bool = False,
    l2_policy_mode: int = 0,
    private_ring: bool = False,
    min_blocks: int = 0,
    precomputed: bool = False,
):
    options = "--enable-tvm-ffi --generate-line-info --opt-level 3"
    if maxrreg:
        options += f" --ptxas-options='-maxrregcount={maxrreg}'"
    if persistent:
        return cute.compile(
            _kda_packed_t1_persist_launch,
            *_make_compile_inputs(qkv_div, gate_div, pool_div, aux_aligned)[:-1],
            cutlass.Int32(0),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            _SCALE,
            _EPS,
            _LOWER_BOUND * _LOG2_E,
            precomputed,
            _HEADS,
            _HEAD_DIM,
            _HEAD_DIM,
            n_stages,
            cp_l2_hint,
            use_packed_fma,
            options=options,
        )
    if n_stages:
        # cp.async pipelined variant (16-byte-aligned pools, tile_v >= 64).
        return cute.compile(
            _kda_packed_t1_smem_launch,
            *_make_compile_inputs(qkv_div, gate_div, pool_div, aux_aligned),
            _SCALE,
            _EPS,
            _LOWER_BOUND * _LOG2_E,
            precomputed,
            _HEADS,
            _HEAD_DIM,
            _HEAD_DIM,
            tile_v,
            ilp_rows,
            n_stages,
            chunk_rows or _NUM_GROUPS * ilp_rows,
            cp_l2_hint,
            heads_per_cta,
            bulk_store,
            tma_read,
            l2_policy_mode,
            private_ring,
            min_blocks,
            use_packed_fma,
            options=options,
        )
    return cute.compile(
        _kda_packed_t1_launch,
        *_make_compile_inputs(qkv_div, gate_div, pool_div, aux_aligned),
        _SCALE,
        _EPS,
        _LOWER_BOUND * _LOG2_E,
        precomputed,
        _HEADS,
        _HEAD_DIM,
        _HEAD_DIM,
        tile_v,
        ilp_rows,
        num_groups,
        use_packed_fma,
        evict_first_state,
        options=options,
    )


@functools.cache
def _use_packed_fma() -> bool:
    return torch.cuda.get_device_capability(0)[0] >= 10


# tile_v thresholds on work_units = B * H: bigger tiles amortise the q/k/gate
# staging and CTA overheads, smaller tiles add CTAs so small batches still
# fill the SMs. Tuned on B200 (see benchmarks/bench_packed_kda_decode.py).


# Benchmark-override schedules: the best-known (ilp, groups, stages, evict)
# for each forced tile width, so ``tile_v=`` forcing exercises every kernel
# shape without hitting untuned/invalid combinations.
_FORCED_TILE_CONFIGS = {
    8: (8, 4, 2, 0, True),
    16: (16, 2, 8, 0, True),
    32: (32, 2, 8, 0, True),
    64: (64, 2, 8, 4, False),
    128: (128, 2, 8, 5, False),
}
_SUPPORTED_TILE_V = tuple(sorted(_FORCED_TILE_CONFIGS))


def _select_config(
    batch: int, forced_tile_v: Optional[int] = None
) -> tuple[int, int, int, int, bool]:
    """Per-batch schedule: ``(tile_v, ilp_rows, num_groups, stages, evict)``.

    ``stages > 0`` selects the cp.async pipelined kernel (128 threads);
    ``stages == 0`` selects the register-prefetch kernel with
    ``num_groups * 16`` threads. Tuned on B200 against the Cake tile8/tile16
    schedules (see PR #4378 benchmark protocol); every row was the best of a
    tile/ilp/stages/CTA-width sweep at that batch size.
    """
    if forced_tile_v is not None:
        return _FORCED_TILE_CONFIGS[forced_tile_v]
    if batch <= 11:
        cfg = (16, 2, 8, 0, True)  # latency floor: 96 CTAs/head-tile at B=1
    elif batch <= 23:
        cfg = (8, 4, 2, 0, True)  # 32-thread CTAs: finest work granularity
    elif batch <= 37:
        # Half-head tiles keep 24*B CTAs in flight through the 148-SM
        # occupancy valley; the 4-deep ring covers the shorter pipeline.
        cfg = (64, 2, 8, 4, False)
    else:
        # Whole-head CTAs, fine chunks, 5-deep ring: with the fused output
        # form, LDS double-buffering and the barrier-free private ring this
        # shape wins everywhere from B=38 up (1.17-1.26x vs Cake).
        cfg = (128, 2, 8, 5, False)

    tile_v, ilp_rows, num_groups, stages, evict = cfg
    # Looked up per call, like every other tuning override below.
    tile_v_env = os.environ.get("FLASHINFER_PACKED_KDA_TILE_V")
    if tile_v_env:
        tile_v = int(tile_v_env)
        ilp_rows = max(1, min(4, tile_v // _NUM_GROUPS))
        num_groups = _NUM_GROUPS
    ilp_env = os.environ.get("FLASHINFER_PACKED_KDA_ILP")
    if ilp_env:
        ilp_rows = int(ilp_env)
    threads_env = os.environ.get("FLASHINFER_PACKED_KDA_THREADS")
    if threads_env:
        num_groups = int(threads_env) // _LANES_PER_ROW
    if os.environ.get("FLASHINFER_PACKED_KDA_EVICT_FIRST"):
        evict = os.environ["FLASHINFER_PACKED_KDA_EVICT_FIRST"] == "1"

    # Reject schedules the kernels cannot trace correctly: the grid covers V
    # with tile_v-wide tiles, every group must own a whole number of ilp-row
    # blocks, and the DSL needs vector lengths of at least two.
    if tile_v not in _FORCED_TILE_CONFIGS:
        raise ValueError(f"tile_v must be one of {_SUPPORTED_TILE_V}, got {tile_v}")
    if num_groups <= 0 or _NUM_GROUPS % num_groups != 0:
        raise ValueError(
            f"num_groups (THREADS/16) must be a divisor of {_NUM_GROUPS}, "
            f"got {num_groups}"
        )
    rows_per_group = tile_v // num_groups
    if ilp_rows < 2 or tile_v % num_groups != 0 or rows_per_group % ilp_rows != 0:
        raise ValueError(
            f"invalid schedule tile_v={tile_v}, ilp_rows={ilp_rows}, "
            f"num_groups={num_groups}: each of the {num_groups} groups must "
            f"cover a whole number of ilp_rows>=2 blocks"
        )
    return tile_v, ilp_rows, num_groups, stages, evict


def _select_tile_v(batch: int) -> int:
    """Tile width the per-batch policy would pick (benchmark display)."""
    return _select_config(batch)[0]


def _div_class(stride_elems: int, ptr: int) -> int:
    """Largest supported element divisibility for a dynamic outer stride."""
    if stride_elems % 8 == 0 and ptr % 16 == 0:
        return 8
    return 1


def launch_packed_kda_decode_cute(
    mixed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output_view: torch.Tensor,
    forced_tile_v: Optional[int] = None,
) -> None:
    """Launch the CuTe-DSL packed KDA T=1 kernel on the current stream.

    ``output_view`` must be the ``[B, H, V]`` contiguous view of the caller's
    ``[B, 1, H, V]`` output tensor; validation happens in the public facade.
    """
    batch = mixed_qkv.shape[0]
    hk = _HEADS * _HEAD_DIM

    mixed_stride = mixed_qkv.stride(0)
    base = mixed_qkv.storage_offset()
    q = mixed_qkv.as_strided(
        (batch, _HEADS, _HEAD_DIM), (mixed_stride, _HEAD_DIM, 1), base
    )
    k = mixed_qkv.as_strided(
        (batch, _HEADS, _HEAD_DIM), (mixed_stride, _HEAD_DIM, 1), base + hk
    )
    v = mixed_qkv.as_strided(
        (batch, _HEADS, _HEAD_DIM), (mixed_stride, _HEAD_DIM, 1), base + 2 * hk
    )
    g = raw_gate.as_strided(
        (batch, _HEADS, _HEAD_DIM),
        (raw_gate.stride(0), _HEAD_DIM, 1),
        raw_gate.storage_offset(),
    )
    dtb = dt_bias.view(_HEADS, _HEAD_DIM)

    qkv_div = _div_class(mixed_stride, mixed_qkv.data_ptr())
    gate_div = _div_class(raw_gate.stride(0), raw_gate.data_ptr())
    _launch_from_views(
        q,
        k,
        v,
        g,
        raw_beta,
        A_log,
        dtb,
        state,
        state_indices,
        output_view,
        forced_tile_v,
        qkv_div,
        gate_div,
    )


def launch_unpacked_kda_decode_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output_view: torch.Tensor,
    forced_tile_v: Optional[int] = None,
    precomputed_gate: bool = False,
) -> None:
    """Launch the same T=1 kernel on separately allocated q/k/v/g tensors.

    ``precomputed_gate=True`` selects the pre-computed convention: ``g`` is
    the log-space decay (``d = exp(g)``) and ``beta`` is already sigmoided;
    ``A_log``/``dt_bias`` are ignored (pass any valid tensors).

    ``q``/``k``/``v``/``g`` are ``[B, H, K]`` bf16 views with contiguous
    inner ``[H, K]`` (any row stride); ``beta`` is ``[B, H]`` raw logits.
    The compiled kernel takes the five tensors independently, so this is the
    identical cubin the packed entry point launches -- the packed layout only
    ever existed in the view construction above.
    """
    qkv_div = min(_div_class(t.stride(0), t.data_ptr()) for t in (q, k, v))
    gate_div = _div_class(g.stride(0), g.data_ptr())
    _launch_from_views(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias.view(_HEADS, _HEAD_DIM),
        state,
        state_indices,
        output_view,
        forced_tile_v,
        qkv_div,
        gate_div,
        precomputed_gate,
    )


def _launch_from_views(
    q,
    k,
    v,
    g,
    raw_beta,
    A_log,
    dtb,
    state,
    state_indices,
    output_view,
    forced_tile_v,
    qkv_div,
    gate_div,
    precomputed_gate=False,
):
    batch = q.shape[0]
    tile_v, ilp_rows, num_groups, n_stages, evict = _select_config(batch, forced_tile_v)
    pool_div = _div_class(state.stride(0), state.data_ptr())

    # The cp.async pipeline requires a 16 B-aligned pool and at least two
    # chunks per CTA; otherwise fall back to the register-prefetch kernel
    # (which handles any alignment via scalar copies).
    chunk_rows = (
        int(os.environ.get("FLASHINFER_PACKED_KDA_CHUNKR", "0"))
        or _NUM_GROUPS * ilp_rows
    )
    # A chunk must cover at least one compute sub-iteration.
    chunk_rows = max(chunk_rows, _NUM_GROUPS * ilp_rows)
    iters = tile_v // chunk_rows
    if pool_div != 8 or iters < 2:
        n_stages = 0
    if n_stages and (
        tile_v % chunk_rows != 0 or chunk_rows % (_NUM_GROUPS * ilp_rows) != 0
    ):
        raise ValueError(
            f"FLASHINFER_PACKED_KDA_CHUNKR={chunk_rows} must divide "
            f"tile_v={tile_v} and be a multiple of "
            f"{_NUM_GROUPS * ilp_rows} rows"
        )
    if n_stages:
        # iters + 1 stages lets the prologue put ALL chunks of a short
        # pipeline in flight before the q/k/gate staging math.
        stages_env = int(os.environ.get("FLASHINFER_PACKED_KDA_STAGES", str(n_stages)))
        if stages_env == 1:
            raise ValueError(
                "FLASHINFER_PACKED_KDA_STAGES=1 is invalid: cp.async refill "
                "requires a slot to stay cold for two iterations (use 0 for "
                "the register-prefetch kernel, or >=2)"
            )
        n_stages = 0 if stages_env <= 0 else min(stages_env, iters + 1)
        num_groups = _NUM_GROUPS  # smem kernel is fixed at 128 threads
    if os.environ.get("FLASHINFER_PACKED_KDA_NO_CPASYNC", "0") == "1":
        n_stages = 0

    persistent = (
        os.environ.get("FLASHINFER_PACKED_KDA_PERSIST", "0") == "1"
        and pool_div == 8
        and n_stages >= 2
    )
    if persistent:
        # Slot/parity arithmetic requires n_stages to divide the 8 chunks
        # per item.
        n_stages = 8 if n_stages >= 8 else 4
    # The pipelined kernel has three mutually exclusive staging/wait modes
    # (private ring / TMA reads / barriered ring with optional bulk stores);
    # normalize the experimental env toggles here so the kernel never sees an
    # inconsistent combination (e.g. TMA prologue + private-ring waits).
    tma_read = os.environ.get("FLASHINFER_PACKED_KDA_TMA", "0") == "1" and n_stages > 0
    bulk_store = (
        os.environ.get("FLASHINFER_PACKED_KDA_BULK", "0") == "1"
        and n_stages > 0
        and chunk_rows == _NUM_GROUPS * ilp_rows
    )
    if tma_read and bulk_store:
        raise ValueError(
            "FLASHINFER_PACKED_KDA_TMA and FLASHINFER_PACKED_KDA_BULK are "
            "mutually exclusive"
        )
    private_ring = (
        os.environ.get("FLASHINFER_PACKED_KDA_PRIVRING", "1") == "1"
        and not tma_read
        and not bulk_store
    )
    aux_aligned = dtb.data_ptr() % 16 == 0 and output_view.data_ptr() % 16 == 0
    compiled = _get_compiled(
        tile_v,
        ilp_rows,
        qkv_div,
        gate_div,
        pool_div,
        aux_aligned,
        _use_packed_fma(),
        evict,
        int(os.environ.get("FLASHINFER_PACKED_KDA_MAXRREG", "0")),
        n_stages,
        num_groups,
        int(os.environ.get("FLASHINFER_PACKED_KDA_L2HINT", "256")),
        chunk_rows,
        bulk_store,
        (
            int(os.environ.get("FLASHINFER_PACKED_KDA_HPC", "1"))
            if n_stages > 0
            and tile_v == _HEAD_DIM
            and _HEADS % max(1, int(os.environ.get("FLASHINFER_PACKED_KDA_HPC", "1")))
            == 0
            else 1
        ),
        tma_read,
        persistent,
        int(os.environ.get("FLASHINFER_PACKED_KDA_L2POL", "0")),
        private_ring,
        int(os.environ.get("FLASHINFER_PACKED_KDA_MINBLOCKS", "0")),
        precomputed_gate,
    )
    if persistent:
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        # Balanced grid: the largest divisor of the item count that fits the
        # resident capacity gives every CTA the same item count (no tail).
        items = batch * _HEADS
        cap = sms * 6  # 6 blocks/SM at the champion's register footprint
        grid_ctas = min(items, cap)
        for div in range(cap, cap // 2, -1):
            if items % div == 0:
                grid_ctas = div
                break
        compiled(
            state,
            q,
            k,
            v,
            g,
            raw_beta,
            A_log,
            dtb,
            output_view,
            state_indices,
            grid_ctas,
        )
        return
    compiled(
        state,
        q,
        k,
        v,
        g,
        raw_beta,
        A_log,
        dtb,
        output_view,
        state_indices,
    )


def _check_cuda_tensor(name, tensor, dtype):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")


def _check_b200(device: torch.device) -> None:
    capability = get_compute_capability(device)
    if capability != (10, 0):
        raise RuntimeError(
            "experimental CuTe packed KDA requires exact compute capability "
            f"10.0 (B200), got {capability[0]}.{capability[1]}"
        )
    if not is_cuda_version_at_least("12.8"):
        raise RuntimeError(
            "experimental CuTe packed KDA on compute capability 10.0 requires "
            "CUDA 12.8 or newer"
        )


@torch.no_grad()
def run_packed_kda_decode_cute(
    mixed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    *,
    tile_v: Optional[int] = None,
) -> torch.Tensor:
    """Run the B200 CuTe packed KDA T=1 kernel.

    Packed bf16 layouts,     fp32 internal math, out-of-pool ``state_indices`` rows are inactive
    (zero output, untouched state). ``tile_v`` is a benchmark override that
    forces one of the tuned per-tile schedules; production-style calls
    should leave it as ``None`` so the per-batch policy picks the kernel.
    """
    _check_cuda_tensor("mixed_qkv", mixed_qkv, torch.bfloat16)
    _check_cuda_tensor("raw_gate", raw_gate, torch.bfloat16)
    _check_cuda_tensor("raw_beta", raw_beta, torch.bfloat16)
    _check_cuda_tensor("A_log", A_log, torch.float32)
    _check_cuda_tensor("dt_bias", dt_bias, torch.float32)
    _check_cuda_tensor("state", state, torch.bfloat16)
    _check_cuda_tensor("state_indices", state_indices, torch.int32)

    for name, tensor in (
        ("raw_gate", raw_gate),
        ("raw_beta", raw_beta),
        ("A_log", A_log),
        ("dt_bias", dt_bias),
        ("state", state),
        ("state_indices", state_indices),
    ):
        if tensor.device != mixed_qkv.device:
            raise ValueError(f"{name} must be on the same device as mixed_qkv")

    if mixed_qkv.ndim != 2 or mixed_qkv.shape[1] != _MIXED_WIDTH:
        raise ValueError(f"mixed_qkv must have shape [B, {_MIXED_WIDTH}]")
    batch = int(mixed_qkv.shape[0])
    if batch <= 0:
        raise ValueError(f"packed KDA T=1 batch must be positive, got {batch}")
    if mixed_qkv.stride(1) != 1 or mixed_qkv.stride(0) < _MIXED_WIDTH:
        raise ValueError("mixed_qkv must have contiguous, non-overlapping rows")
    gate_width = _HEADS * _HEAD_DIM
    if raw_gate.shape != (batch, gate_width) or raw_gate.stride(1) != 1:
        raise ValueError(f"raw_gate must have shape [B, {gate_width}] with stride 1")
    if raw_gate.stride(0) < gate_width:
        raise ValueError("raw_gate rows must not overlap")
    if raw_beta.shape != (batch, _HEADS) or raw_beta.stride(1) != 1:
        raise ValueError(f"raw_beta must have shape [B, {_HEADS}] with stride 1")
    if raw_beta.stride(0) < _HEADS:
        raise ValueError("raw_beta rows must not overlap")
    if A_log.shape != (_HEADS,) or not A_log.is_contiguous():
        raise ValueError(f"A_log must be contiguous with shape [{_HEADS}]")
    if dt_bias.shape != (gate_width,) or not dt_bias.is_contiguous():
        raise ValueError(f"dt_bias must be contiguous with shape [{gate_width}]")
    if state_indices.shape != (batch,) or not state_indices.is_contiguous():
        raise ValueError("state_indices must be contiguous with shape [B]")
    if (
        state.ndim != 4
        or state.shape[1:] != (_HEADS, _HEAD_DIM, _HEAD_DIM)
        or state.stride(0) < _HEADS * _HEAD_DIM * _HEAD_DIM
        or tuple(state.stride()[1:]) != (_HEAD_DIM * _HEAD_DIM, _HEAD_DIM, 1)
    ):
        raise ValueError(
            "state must have shape [N,12,128,128] with compact inner dimensions"
        )

    expected_output_shape = (batch, 1, _HEADS, _HEAD_DIM)
    if output is None:
        output = mixed_qkv.new_empty(expected_output_shape)
    else:
        _check_cuda_tensor("output", output, torch.bfloat16)
        if output.device != mixed_qkv.device:
            raise ValueError("output must be on the same device as mixed_qkv")
        if output.shape != expected_output_shape or not output.is_contiguous():
            raise ValueError("output must be contiguous with shape [B,1,12,128]")

    _check_b200(mixed_qkv.device)
    if tile_v is not None and tile_v not in _SUPPORTED_TILE_V:
        raise ValueError(f"tile_v must be one of {_SUPPORTED_TILE_V}")

    launch_packed_kda_decode_cute(
        mixed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state,
        state_indices,
        output.view(batch, _HEADS, _HEAD_DIM),
        forced_tile_v=tile_v,
    )
    return output


__all__ = [
    "launch_packed_kda_decode_cute",
    "launch_unpacked_kda_decode_cute",
    "run_packed_kda_decode_cute",
]
