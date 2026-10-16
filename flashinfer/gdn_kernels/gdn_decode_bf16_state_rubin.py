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
"""

"""
Rubin (SM107) T=1 GDN decode with BF16 state (single-token / STP).

A Rubin-tuned variant of the T=1 wide_vec kernel in ``gdn_decode_bf16_state``
(``gated_delta_rule_t1_wide_vec``). Math, FMA order and reductions are the same, so outputs and
the updated state are bitwise equal to that kernel wherever it runs (B * HV >= 512).

Why
---
Rubin's DRAM bandwidth is ~2.5x B200's, its instruction issue rate only ~1.8x. The generic kernel
loads four state rows per thread, waits, computes and repeats, so on Rubin it keeps too few bytes
in flight (its warps mostly wait on memory) and reaches ~60% of DRAM bandwidth. Requesting the
whole tile at once fixes the waiting, but the extra bookkeeping instructions then make the kernel
issue-bound. This kernel keeps the whole tile in flight and trims the instructions around it.

How
---
1. Tiles of 64 V-rows (two CTAs per head, ``tile_v=64``, 16 KB each). At entry every thread issues
   ``cp.async`` for all of its state rows into shared memory (``stage_smem``), one commit group per
   4 rows, and the main loop waits group by group. Thread group g owns rows g, g + 8, ..., so the
   copy partition and the compute mapping coincide and the state needs no CTA barrier.
2. Each 16-lane group loads and normalizes its own q/k slice and computes g/beta in registers
   (``qk_in_regs``); a warp fence keeps ptxas from hoisting those loads above the cp.async issue
   (``issue_fence``); the per-row v values are loaded at entry (``prefetch_v``).
3. ``fast_reduce``: a transpose-reduce sums the four row dot products across the 16-lane group with
   5 shuffles instead of 16 (same lane pairing, so the same bits). ``pack_dots``: the dot products
   as FFMA2 (two exact FMAs per instruction).
4. After its stores each CTA prefetches into L2 (evict_last) the state tile and q/k/v/a/b lines of
   the tile ``pf_dist`` positions ahead (default num_SMs * 25 // 3, about 1.2 residency
   generations at 7 CTAs/SM), so the CTA that later runs it finds its state in L2.
5. Instruction cuts, each a keyword argument (on by default; each turns itself off when its input
   check fails): ``pair_rows`` (row-pair FFMA2 throughout, no register regrouping), ``vn_lane`` (vn
   computed in the lane that holds the reduced dot product, then broadcast), ``o_hoist`` (output
   pointer computed once), ``st_part`` (state write-back through the copy partition: one base plus
   constant offsets), ``unpack_bits`` / ``unpack_qk`` (bf16 -> f32 by bit placement), ``vec_qk`` /
   ``static_io`` (16 B q/k loads; q/k/v/a/b/o strides as compile-time constants),
   ``pf_lite`` / ``pf_u32`` / ``pf_addr`` (cheaper prefetch bookkeeping: warp 0 alone dispatches the
   input prefetches, unsigned tile-index math, line address = tile base + line * 128 B). Together
   they remove ~38% of the warp-instructions without changing any arithmetic.

Measured on Rubin VR200, B = 512, HQ = HK = 16, HV = 64, K = V = 128 (CUPTI, cold L2): 186.0 us
with the generic kernel, 162.9 us with this one. Keeping each die's half of the batch on that die
(green contexts + locality-domain memory, set up by the caller) brings it to ~131 us, 85-86% of
the DRAM peak.

Scope: T = 1, BF16 state ``[pool, HV, V, K]``, K = V = 128, state update and output. Same arguments
and pool / index semantics as ``gdn_decode_bf16_state.gated_delta_rule`` at T = 1 (single or split
read / write indices, -1 padding redirected to slot 0). Compute capability 10.7 only.
"""

import math
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.dialects import nvvm as _nvvm
from cutlass.cutlass_dsl import dsl_user_op

from .device_target import gdn_compile_options, gdn_device_target
from .dtype_compat import as_bf16
from .gdn_decode_bf16_state import (
    ELEMS_PER_LANE,
    ILP_ROWS,
    LANES_PER_ROW,
    NUM_GROUPS,
    NUM_THREADS,
    _BF16_STATE_COMPILE_OPTS,
    _dtype_key,
    _mark_batch_dynamic,
    _mark_index_dynamic,
    _mark_slot_dynamic,
    fma_pair,
    fma_pair_mul,
)


@dsl_user_op
def _bf16x2_to_f32(w, *, loc=None, ip=None):
    """bf16x2 word -> (f32 of the low element, f32 of the high element) by bit placement.

    Exactly what cvt.f32.bf16 computes (bf16 is the top half of an f32, NaN payloads kept):
    lo = w << 16, hi = w & 0xffff0000. One integer op per element instead of ptxas'
    PRMT + IMAD.U32 for the high half (``unpack_bits`` / ``unpack_qk``).
    """
    wu = cutlass.Uint32(w)
    lo = wu << cutlass.Uint32(16)
    hi = wu & cutlass.Uint32(0xFFFF0000)
    f = cutlass.Float32.mlir_type
    return (
        cutlass.Float32(_llvm.bitcast(f, lo.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)),
        cutlass.Float32(_llvm.bitcast(f, hi.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)),
    )


@cute.jit
def _xreduce4(x0, x1, x2, x3, lane_in_group):
    """Sum 4 per-lane values across a 16-lane group with 5 shuffles instead of 16.

    Transpose-reduce: offset-8 and offset-4 stages each exchange half of the remaining
    values, then offsets 2 and 1 reduce the single survivor. Lane pairing at every stage
    is the same as the xor-butterfly (lane i adds the partner i^off), so the result is
    bitwise identical to four independent butterflies. On return the lane holds the sum
    of value j = 2*bit3(lane) + bit2(lane).
    """
    b3 = (lane_in_group & 8) != 0
    b2 = (lane_in_group & 4) != 0
    send_a = x0 if b3 else x2
    send_b = x1 if b3 else x3
    keep_a = x2 if b3 else x0
    keep_b = x3 if b3 else x1
    a = keep_a + cute.arch.shuffle_sync_bfly(
        send_a, offset=8, mask=-1, mask_and_clamp=31
    )
    b = keep_b + cute.arch.shuffle_sync_bfly(
        send_b, offset=8, mask=-1, mask_and_clamp=31
    )
    send = a if b2 else b
    keep = b if b2 else a
    r = keep + cute.arch.shuffle_sync_bfly(send, offset=4, mask=-1, mask_and_clamp=31)
    r = r + cute.arch.shuffle_sync_bfly(r, offset=2, mask=-1, mask_and_clamp=31)
    r = r + cute.arch.shuffle_sync_bfly(r, offset=1, mask=-1, mask_and_clamp=31)
    return r


@cute.kernel
def gdn_t1_rubin_kernel(
    tiled_copy_h: cute.TiledCopy,
    h0_source: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b_gate: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    h0_out_indices: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    use_packed_fma: cutlass.Constexpr[bool],
    same_pool: cutlass.Constexpr[bool],
    stage_smem: cutlass.Constexpr[bool],
    qk_in_regs: cutlass.Constexpr[bool],
    issue_fence: cutlass.Constexpr[bool],
    prefetch_v: cutlass.Constexpr[bool],
    fast_reduce: cutlass.Constexpr[bool],
    pack_dots: cutlass.Constexpr[bool],
    pf_dist: cutlass.Constexpr[int],
    pf_evict_last: cutlass.Constexpr[bool],
    pf_inputs: cutlass.Constexpr[bool],
    pair_rows: cutlass.Constexpr[bool],
    vn_lane: cutlass.Constexpr[bool],
    o_hoist: cutlass.Constexpr[bool],
    st_part: cutlass.Constexpr[bool],
    unpack_bits: cutlass.Constexpr[bool],
    unpack_qk: cutlass.Constexpr[bool],
    vec_qk: cutlass.Constexpr[bool],
    pf_lite: cutlass.Constexpr[bool],
    pf_u32: cutlass.Constexpr[bool],
    pf_addr: cutlass.Constexpr[bool],
):
    # The instruction cuts (pair_rows, vn_lane .. pf_addr) arrive normalized by the wrapper: each is
    # False unless the options it builds on are on and its input checks pass.
    tidx, _, _ = cute.arch.thread_idx()
    lane_in_warp = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    group_idx = tidx // LANES_PER_ROW
    lane_in_group = tidx % LANES_PER_ROW
    k_start = lane_in_group * ELEMS_PER_LANE
    vec: cutlass.Constexpr[int] = ELEMS_PER_LANE

    ROWS_PER_THREAD: cutlass.Constexpr[int] = tile_v // NUM_GROUPS
    ITERS: cutlass.Constexpr[int] = ROWS_PER_THREAD // ILP_ROWS

    batch_idx, _, _ = cute.arch.block_idx()
    i_v = batch_idx % num_v_tiles
    tmp = batch_idx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    cache_idx = h0_indices[i_n]
    if cache_idx < 0:
        cache_idx = cutlass.Int32(0)
    if cutlass.const_expr(same_pool):
        write_cache_idx = cache_idx
    else:
        write_cache_idx = h0_out_indices[i_n]
        if write_cache_idx < 0:
            write_cache_idx = cutlass.Int32(0)

    if cutlass.const_expr(pf_dist > 0):
        # L2 prefetch of the state tile a block ~one residency generation later will
        # need (hint only; numerics untouched). Index loaded early, used at the end.
        total_tiles = q.shape[0] * (HV * num_v_tiles)
        w_pf = batch_idx + pf_dist
        pf_ok = w_pf < total_tiles
        if not pf_ok:
            w_pf = batch_idx
        if cutlass.const_expr(pf_u32):
            # pf_u32: w_pf >= 0, so unsigned div/mod (shifts) instead of the signed
            # floor-division fix-ups
            w_pf_u = cutlass.Uint32(w_pf)
            pf_iv = cutlass.Int32(w_pf_u % cutlass.Uint32(num_v_tiles))
            pf_t_u = w_pf_u // cutlass.Uint32(num_v_tiles)
            pf_t = cutlass.Int32(pf_t_u)
            pf_hv = cutlass.Int32(pf_t_u % cutlass.Uint32(HV))
            pf_idx = h0_indices[cutlass.Int32(pf_t_u // cutlass.Uint32(HV))]
        else:
            pf_iv = w_pf % num_v_tiles
            pf_t = w_pf // num_v_tiles
            pf_hv = pf_t % HV
            pf_idx = h0_indices[pf_t // HV]
        if pf_idx < 0:
            pf_idx = cutlass.Int32(0)

    r_A_log = cutlass.Float32(A_log[i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])

    # ----- SMEM -----
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K,), stride=(1,)), 16)
    sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout((K,), stride=(1,)), 16)
    sGB = smem.allocate_tensor(cutlass.Float32, cute.make_layout((4,), stride=(1,)), 16)

    h0_slot_r = h0_source[(cutlass.Int64(cache_idx), i_hv, None, None)]
    if cutlass.const_expr(same_pool):
        h0_slot_w = h0_slot_r
    else:
        h0_slot_w = h0_source[(cutlass.Int64(write_cache_idx), i_hv, None, None)]

    # ----- (1) issue every state load of this CTA up front -----
    if cutlass.const_expr(stage_smem):
        sH = smem.allocate_tensor(
            cutlass.BFloat16, cute.make_layout((tile_v, K), stride=(K, 1)), 128
        )
        gH = cute.local_tile(h0_slot_r, (tile_v, K), (i_v, 0))
        thr_copy = tiled_copy_h.get_slice(tidx)
        tHg = thr_copy.partition_S(gH)  # (CPY, ROWS_PER_THREAD, 1)
        tHs = thr_copy.partition_D(sH)
        if cutlass.const_expr(st_part):
            # st_part: the state write-back goes through the same copy partition (one per-thread
            # base, constant row offsets; same_pool: the cp.async source registers themselves)
            # instead of per-row index math. Same bytes to the same addresses.
            if cutlass.const_expr(same_pool):
                tHw = tHg
            else:
                tHw = thr_copy.partition_S(
                    cute.local_tile(h0_slot_w, (tile_v, K), (i_v, 0))
                )
        for it in cutlass.range_constexpr(ITERS):
            for rr in cutlass.range_constexpr(ILP_ROWS):
                r = it * ILP_ROWS + rr
                cute.copy(tiled_copy_h, tHg[None, r, None], tHs[None, r, None])
            cute.arch.cp_async_commit_group()
        if cutlass.const_expr(issue_fence):
            # Scheduling fence: keep ptxas from hoisting the q/k/v/a/b loads below
            # above the cp.async issue (it otherwise issues them first and the state
            # loads only start after the q/k round trip + L2 norm).
            cute.arch.sync_warp()

    # ----- prefetch this thread's v values (one per owned row) -----
    r_vv = cute.make_rmem_tensor(
        cute.make_layout((ROWS_PER_THREAD,), stride=(1,)), cutlass.Float32
    )
    if cutlass.const_expr(vn_lane):
        # vn_lane: after _xreduce4 lane j = lane_in_group >> 2 holds row j's dot product and
        # computes vn_j itself, so it needs only that row's v: one value per iteration.
        r_vj = cute.make_rmem_tensor(
            cute.make_layout((ITERS,), stride=(1,)), cutlass.Float32
        )
        for it in cutlass.range_constexpr(ITERS):
            r_vj[it] = cutlass.Float32(
                v[
                    (
                        i_n,
                        0,
                        i_hv,
                        i_v * tile_v
                        + (it * ILP_ROWS + (lane_in_group >> 2)) * NUM_GROUPS
                        + group_idx,
                    )
                ]
            )
    if cutlass.const_expr(prefetch_v and not vn_lane):
        for r in cutlass.range_constexpr(ROWS_PER_THREAD):
            r_vv[r] = cutlass.Float32(
                v[(i_n, 0, i_hv, i_v * tile_v + r * NUM_GROUPS + group_idx)]
            )

    # ----- registers -----
    r_h = cute.make_rmem_tensor(
        cute.make_layout((ILP_ROWS, vec), stride=(vec, 1)), cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(cute.make_layout((vec,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec,), stride=(1,)), cutlass.Float32)
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_hb0 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_hb1 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_hb2 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_hb3 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )

    # ----- (2) q/k/g/beta precompute — overlaps the state loads -----
    # qk_in_regs=False: warp 0 computes into SMEM, CTA barrier (shipped scheme).
    # qk_in_regs=True : every 16-lane group loads/normalizes the 8-element q/k slice it
    #                   needs into registers and every thread computes g/beta itself:
    #                   no SMEM round-trip and no CTA barrier.
    if cutlass.const_expr(qk_in_regs):
        member = lane_in_warp % LANES_PER_ROW
        ks = member * ELEMS_PER_LANE
        q_tile = cute.local_tile(q, (1, 1, 1, vec), (i_n, 0, i_h, member))
        k_tile = cute.local_tile(k, (1, 1, 1, vec), (i_n, 0, i_h, member))
        if cutlass.const_expr(vec_qk):
            # vec_qk: one 16 B load per row slice (the wrapper checked 16 B aligned q/k rows)
            cute.autovec_copy(
                cute.make_tensor(
                    cute.make_ptr(
                        cutlass.BFloat16,
                        q_tile.iterator.toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((vec,)),
                ),
                r_q_bf16,
            )
            cute.autovec_copy(
                cute.make_tensor(
                    cute.make_ptr(
                        cutlass.BFloat16,
                        k_tile.iterator.toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((vec,)),
                ),
                r_k_bf16,
            )
        else:
            cute.autovec_copy(q_tile, r_q_bf16)
            cute.autovec_copy(k_tile, r_k_bf16)
        if cutlass.const_expr(unpack_qk):
            # unpack_qk: the same exact bf16 -> f32 conversion by bit placement
            r_qw = cute.recast_tensor(r_q_bf16, cutlass.Uint32)
            r_kw = cute.recast_tensor(r_k_bf16, cutlass.Uint32)
            for m in cutlass.range_constexpr(vec // 2):
                r_q[2 * m], r_q[2 * m + 1] = _bf16x2_to_f32(r_qw[m])
                r_k[2 * m], r_k[2 * m + 1] = _bf16x2_to_f32(r_kw[m])
        else:
            for i in cutlass.range_constexpr(vec):
                r_q[i] = cutlass.Float32(r_q_bf16[i])
                r_k[i] = cutlass.Float32(r_k_bf16[i])
        if cutlass.const_expr(use_qk_l2norm):
            sum_q = cutlass.Float32(0.0)
            sum_k = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(vec):
                sum_q += r_q[i] * r_q[i]
                sum_k += r_k[i] * r_k[i]
            for offset in [8, 4, 2, 1]:
                sum_q += cute.arch.shuffle_sync_bfly(
                    sum_q, offset=offset, mask=-1, mask_and_clamp=31
                )
                sum_k += cute.arch.shuffle_sync_bfly(
                    sum_k, offset=offset, mask=-1, mask_and_clamp=31
                )
            inv_norm_q_scaled = cute.rsqrt(sum_q + 1e-6, fastmath=True) * scale
            inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)
            for i in cutlass.range_constexpr(vec):
                r_q[i] = r_q[i] * inv_norm_q_scaled
                r_k[i] = r_k[i] * inv_norm_k
        else:
            for i in cutlass.range_constexpr(vec):
                r_q[i] = r_q[i] * scale

        r_a = cutlass.Float32(a[i_n, 0, i_hv])
        r_b = cutlass.Float32(b_gate[i_n, 0, i_hv])
        x = r_a + r_dt_bias
        beta_x = softplus_beta * x
        exp_beta_x = cute.exp(beta_x, fastmath=True)
        softplus_val = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
            cutlass.Float32(1.0) + exp_beta_x, fastmath=True
        )
        use_softplus = (
            cutlass.Float32(1.0)
            if beta_x <= softplus_threshold
            else cutlass.Float32(0.0)
        )
        softplus_x = (
            use_softplus * softplus_val + (cutlass.Float32(1.0) - use_softplus) * x
        )
        r_g_pre = cute.exp(
            -cute.exp(r_A_log, fastmath=True) * softplus_x, fastmath=True
        )
        r_beta_pre = cutlass.Float32(1.0) / (
            cutlass.Float32(1.0) + cute.exp(-r_b, fastmath=True)
        )
    else:
        if warp_idx == 0:
            member = lane_in_warp % LANES_PER_ROW
            ks = member * ELEMS_PER_LANE
            q_tile = cute.local_tile(q, (1, 1, 1, vec), (i_n, 0, i_h, member))
            k_tile = cute.local_tile(k, (1, 1, 1, vec), (i_n, 0, i_h, member))
            cute.autovec_copy(q_tile, r_q_bf16)
            cute.autovec_copy(k_tile, r_k_bf16)
            for i in cutlass.range_constexpr(vec):
                r_q[i] = cutlass.Float32(r_q_bf16[i])
                r_k[i] = cutlass.Float32(r_k_bf16[i])
            if cutlass.const_expr(use_qk_l2norm):
                sum_q = cutlass.Float32(0.0)
                sum_k = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(vec):
                    sum_q += r_q[i] * r_q[i]
                    sum_k += r_k[i] * r_k[i]
                for offset in [8, 4, 2, 1]:
                    sum_q += cute.arch.shuffle_sync_bfly(
                        sum_q, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    sum_k += cute.arch.shuffle_sync_bfly(
                        sum_k, offset=offset, mask=-1, mask_and_clamp=31
                    )
                inv_norm_q_scaled = cute.rsqrt(sum_q + 1e-6, fastmath=True) * scale
                inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)
                for i in cutlass.range_constexpr(vec):
                    r_q[i] = r_q[i] * inv_norm_q_scaled
                    r_k[i] = r_k[i] * inv_norm_k
            else:
                for i in cutlass.range_constexpr(vec):
                    r_q[i] = r_q[i] * scale
            for i in cutlass.range_constexpr(vec):
                sQ[ks + i] = r_q[i]
                sK[ks + i] = r_k[i]

            r_a = cutlass.Float32(a[i_n, 0, i_hv])
            r_b = cutlass.Float32(b_gate[i_n, 0, i_hv])
            x = r_a + r_dt_bias
            beta_x = softplus_beta * x
            exp_beta_x = cute.exp(beta_x, fastmath=True)
            softplus_val = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                cutlass.Float32(1.0) + exp_beta_x, fastmath=True
            )
            use_softplus = (
                cutlass.Float32(1.0)
                if beta_x <= softplus_threshold
                else cutlass.Float32(0.0)
            )
            softplus_x = (
                use_softplus * softplus_val + (cutlass.Float32(1.0) - use_softplus) * x
            )
            r_g_pre = cute.exp(
                -cute.exp(r_A_log, fastmath=True) * softplus_x, fastmath=True
            )
            r_beta_pre = cutlass.Float32(1.0) / (
                cutlass.Float32(1.0) + cute.exp(-r_b, fastmath=True)
            )
            if True:
                if lane_in_warp == 0:
                    sGB[0] = r_g_pre
                    sGB[1] = r_beta_pre

    if cutlass.const_expr(qk_in_regs):
        r_g = r_g_pre
        r_beta = r_beta_pre
    if cutlass.const_expr(not qk_in_regs):
        cute.arch.barrier()
        r_g = sGB[0]
        r_beta = sGB[1]

    if cutlass.const_expr(o_hoist):
        # o_hoist: this lane's output pointer once (row (it * 4 + j) * 8 + g of the tile,
        # j = lane_in_group >> 2); each iteration is then a predicated store at +32 rows * it.
        o_t = cute.local_tile(o[(i_n, 0, i_hv, None)], (tile_v,), (i_v,))
        o_lane = cute.make_tensor(
            o_t.iterator + ((lane_in_group >> 2) * NUM_GROUPS + group_idx),
            cute.make_layout((ITERS,), stride=(ILP_ROWS * NUM_GROUPS,)),
        )
        o_pred = (lane_in_group & 3) == 0

    # ----- (3) main loop: 4 interleaved rows per iteration -----
    for it in cutlass.range_constexpr(ITERS):
        rl0 = (it * ILP_ROWS + 0) * NUM_GROUPS + group_idx  # row within the tile
        rl1 = (it * ILP_ROWS + 1) * NUM_GROUPS + group_idx
        rl2 = (it * ILP_ROWS + 2) * NUM_GROUPS + group_idx
        rl3 = (it * ILP_ROWS + 3) * NUM_GROUPS + group_idx
        v0 = i_v * tile_v + rl0
        v1 = i_v * tile_v + rl1
        v2 = i_v * tile_v + rl2
        v3 = i_v * tile_v + rl3

        if cutlass.const_expr(stage_smem):
            # group it is complete once at most the ITERS - 1 - it groups after it are pending
            cute.arch.cp_async_wait_group(ITERS - 1 - it)
        if cutlass.const_expr(unpack_bits):
            # unpack_bits: LDS.128 as 4 words per row, exact bf16 -> f32 by bit placement
            r_w0 = cute.make_rmem_tensor(
                cute.make_layout((1, vec // 2), stride=(vec // 2, 1)), cutlass.Uint32
            )
            r_w1 = cute.make_rmem_tensor(
                cute.make_layout((1, vec // 2), stride=(vec // 2, 1)), cutlass.Uint32
            )
            r_w2 = cute.make_rmem_tensor(
                cute.make_layout((1, vec // 2), stride=(vec // 2, 1)), cutlass.Uint32
            )
            r_w3 = cute.make_rmem_tensor(
                cute.make_layout((1, vec // 2), stride=(vec // 2, 1)), cutlass.Uint32
            )
            cute.autovec_copy(
                cute.recast_tensor(
                    cute.local_tile(sH, (1, vec), (rl0, lane_in_group)), cutlass.Uint32
                ),
                r_w0,
            )
            cute.autovec_copy(
                cute.recast_tensor(
                    cute.local_tile(sH, (1, vec), (rl1, lane_in_group)), cutlass.Uint32
                ),
                r_w1,
            )
            cute.autovec_copy(
                cute.recast_tensor(
                    cute.local_tile(sH, (1, vec), (rl2, lane_in_group)), cutlass.Uint32
                ),
                r_w2,
            )
            cute.autovec_copy(
                cute.recast_tensor(
                    cute.local_tile(sH, (1, vec), (rl3, lane_in_group)), cutlass.Uint32
                ),
                r_w3,
            )
            for m in cutlass.range_constexpr(vec // 2):
                r_h[0, 2 * m], r_h[0, 2 * m + 1] = _bf16x2_to_f32(r_w0[0, m])
                r_h[1, 2 * m], r_h[1, 2 * m + 1] = _bf16x2_to_f32(r_w1[0, m])
                r_h[2, 2 * m], r_h[2, 2 * m + 1] = _bf16x2_to_f32(r_w2[0, m])
                r_h[3, 2 * m], r_h[3, 2 * m + 1] = _bf16x2_to_f32(r_w3[0, m])
        else:
            if cutlass.const_expr(stage_smem):
                cute.autovec_copy(
                    cute.local_tile(sH, (1, vec), (rl0, lane_in_group)), r_hb0
                )
                cute.autovec_copy(
                    cute.local_tile(sH, (1, vec), (rl1, lane_in_group)), r_hb1
                )
                cute.autovec_copy(
                    cute.local_tile(sH, (1, vec), (rl2, lane_in_group)), r_hb2
                )
                cute.autovec_copy(
                    cute.local_tile(sH, (1, vec), (rl3, lane_in_group)), r_hb3
                )
            else:
                cute.autovec_copy(
                    cute.local_tile(h0_slot_r, (1, vec), (v0, lane_in_group)), r_hb0
                )
                cute.autovec_copy(
                    cute.local_tile(h0_slot_r, (1, vec), (v1, lane_in_group)), r_hb1
                )
                cute.autovec_copy(
                    cute.local_tile(h0_slot_r, (1, vec), (v2, lane_in_group)), r_hb2
                )
                cute.autovec_copy(
                    cute.local_tile(h0_slot_r, (1, vec), (v3, lane_in_group)), r_hb3
                )
            for i in cutlass.range_constexpr(vec):
                r_h[0, i] = cutlass.Float32(r_hb0[i])
                r_h[1, i] = cutlass.Float32(r_hb1[i])
                r_h[2, i] = cutlass.Float32(r_hb2[i])
                r_h[3, i] = cutlass.Float32(r_hb3[i])

        # decay + s = h . k
        s0 = cutlass.Float32(0.0)
        s1 = cutlass.Float32(0.0)
        s2 = cutlass.Float32(0.0)
        s3 = cutlass.Float32(0.0)
        if cutlass.const_expr(pair_rows):
            # Row-pair grouping (rows 0/1, 2/3 at column i) for every FFMA2: the packed dot products need these
            # pairs, so doing the decay and update on them too removes the register regrouping MOVs. Same scalar
            # FMAs (operands and per-row order over i), so bitwise identical to the column-pair form below.
            for i in cutlass.range_constexpr(vec):
                kv = r_k[i]
                zero2 = (cutlass.Float32(0.0), cutlass.Float32(0.0))
                r_h[0, i], r_h[1, i] = cute.arch.fma_packed_f32x2(
                    src_a=(r_h[0, i], r_h[1, i]), src_b=(r_g, r_g), src_c=zero2
                )
                r_h[2, i], r_h[3, i] = cute.arch.fma_packed_f32x2(
                    src_a=(r_h[2, i], r_h[3, i]), src_b=(r_g, r_g), src_c=zero2
                )
                s0, s1 = cute.arch.fma_packed_f32x2(
                    src_a=(r_h[0, i], r_h[1, i]), src_b=(kv, kv), src_c=(s0, s1)
                )
                s2, s3 = cute.arch.fma_packed_f32x2(
                    src_a=(r_h[2, i], r_h[3, i]), src_b=(kv, kv), src_c=(s2, s3)
                )
        else:
            for i in cutlass.range_constexpr(0, vec, 2):
                if cutlass.const_expr(qk_in_regs):
                    kv0 = r_k[i]
                    kv1 = r_k[i + 1]
                else:
                    kv0 = sK[k_start + i]
                    kv1 = sK[k_start + i + 1]
                if cutlass.const_expr(use_packed_fma):
                    zero2 = (cutlass.Float32(0.0), cutlass.Float32(0.0))
                    r_h[0, i], r_h[0, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[0, i], r_h[0, i + 1]), src_b=(r_g, r_g), src_c=zero2
                    )
                    r_h[1, i], r_h[1, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[1, i], r_h[1, i + 1]), src_b=(r_g, r_g), src_c=zero2
                    )
                    r_h[2, i], r_h[2, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[2, i], r_h[2, i + 1]), src_b=(r_g, r_g), src_c=zero2
                    )
                    r_h[3, i], r_h[3, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[3, i], r_h[3, i + 1]), src_b=(r_g, r_g), src_c=zero2
                    )
                else:
                    r_h[0, i], r_h[0, i + 1] = fma_pair_mul(
                        r_h[0, i], r_h[0, i + 1], r_g, r_g
                    )
                    r_h[1, i], r_h[1, i + 1] = fma_pair_mul(
                        r_h[1, i], r_h[1, i + 1], r_g, r_g
                    )
                    r_h[2, i], r_h[2, i + 1] = fma_pair_mul(
                        r_h[2, i], r_h[2, i + 1], r_g, r_g
                    )
                    r_h[3, i], r_h[3, i + 1] = fma_pair_mul(
                        r_h[3, i], r_h[3, i + 1], r_g, r_g
                    )
                if cutlass.const_expr(pack_dots):
                    # two independent exact FMAs per FFMA2: per-row order unchanged (bitwise)
                    s0, s1 = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[0, i], r_h[1, i]), src_b=(kv0, kv0), src_c=(s0, s1)
                    )
                    s2, s3 = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[2, i], r_h[3, i]), src_b=(kv0, kv0), src_c=(s2, s3)
                    )
                    s0, s1 = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[0, i + 1], r_h[1, i + 1]),
                        src_b=(kv1, kv1),
                        src_c=(s0, s1),
                    )
                    s2, s3 = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[2, i + 1], r_h[3, i + 1]),
                        src_b=(kv1, kv1),
                        src_c=(s2, s3),
                    )
                else:
                    s0 = s0 + r_h[0, i] * kv0 + r_h[0, i + 1] * kv1
                    s1 = s1 + r_h[1, i] * kv0 + r_h[1, i + 1] * kv1
                    s2 = s2 + r_h[2, i] * kv0 + r_h[2, i + 1] * kv1
                    s3 = s3 + r_h[3, i] * kv0 + r_h[3, i + 1] * kv1
        if cutlass.const_expr(fast_reduce):
            sj = _xreduce4(s0, s1, s2, s3, lane_in_group)
            seg = lane_in_warp & 16
            if cutlass.const_expr(vn_lane):
                # vn_lane: the same expression as below, evaluated once in the lane that holds
                # s_j; the broadcast then carries vn_j instead of s_j (bitwise the same vn).
                vnj = (r_vj[it] - sj) * r_beta
                vn0 = cute.arch.shuffle_sync(
                    vnj, offset=seg | 0, mask=-1, mask_and_clamp=31
                )
                vn1 = cute.arch.shuffle_sync(
                    vnj, offset=seg | 4, mask=-1, mask_and_clamp=31
                )
                vn2 = cute.arch.shuffle_sync(
                    vnj, offset=seg | 8, mask=-1, mask_and_clamp=31
                )
                vn3 = cute.arch.shuffle_sync(
                    vnj, offset=seg | 12, mask=-1, mask_and_clamp=31
                )
            else:
                s0 = cute.arch.shuffle_sync(
                    sj, offset=seg | 0, mask=-1, mask_and_clamp=31
                )
                s1 = cute.arch.shuffle_sync(
                    sj, offset=seg | 4, mask=-1, mask_and_clamp=31
                )
                s2 = cute.arch.shuffle_sync(
                    sj, offset=seg | 8, mask=-1, mask_and_clamp=31
                )
                s3 = cute.arch.shuffle_sync(
                    sj, offset=seg | 12, mask=-1, mask_and_clamp=31
                )
        else:
            for offset in [8, 4, 2, 1]:
                s0 += cute.arch.shuffle_sync_bfly(
                    s0, offset=offset, mask=-1, mask_and_clamp=31
                )
                s1 += cute.arch.shuffle_sync_bfly(
                    s1, offset=offset, mask=-1, mask_and_clamp=31
                )
                s2 += cute.arch.shuffle_sync_bfly(
                    s2, offset=offset, mask=-1, mask_and_clamp=31
                )
                s3 += cute.arch.shuffle_sync_bfly(
                    s3, offset=offset, mask=-1, mask_and_clamp=31
                )

        if cutlass.const_expr(not vn_lane):  # vn_lane: vn0..vn3 broadcast above
            if cutlass.const_expr(prefetch_v):
                vn0 = (r_vv[it * ILP_ROWS + 0] - s0) * r_beta
                vn1 = (r_vv[it * ILP_ROWS + 1] - s1) * r_beta
                vn2 = (r_vv[it * ILP_ROWS + 2] - s2) * r_beta
                vn3 = (r_vv[it * ILP_ROWS + 3] - s3) * r_beta
            else:
                vn0 = (cutlass.Float32(v[(i_n, 0, i_hv, v0)]) - s0) * r_beta
                vn1 = (cutlass.Float32(v[(i_n, 0, i_hv, v1)]) - s1) * r_beta
                vn2 = (cutlass.Float32(v[(i_n, 0, i_hv, v2)]) - s2) * r_beta
                vn3 = (cutlass.Float32(v[(i_n, 0, i_hv, v3)]) - s3) * r_beta

        # rank-1 update + o = h . q
        o0 = cutlass.Float32(0.0)
        o1 = cutlass.Float32(0.0)
        o2 = cutlass.Float32(0.0)
        o3 = cutlass.Float32(0.0)
        if cutlass.const_expr(pair_rows):
            for i in cutlass.range_constexpr(vec):
                kv = r_k[i]
                qv = r_q[i]
                r_h[0, i], r_h[1, i] = cute.arch.fma_packed_f32x2(
                    src_a=(kv, kv), src_b=(vn0, vn1), src_c=(r_h[0, i], r_h[1, i])
                )
                r_h[2, i], r_h[3, i] = cute.arch.fma_packed_f32x2(
                    src_a=(kv, kv), src_b=(vn2, vn3), src_c=(r_h[2, i], r_h[3, i])
                )
                o0, o1 = cute.arch.fma_packed_f32x2(
                    src_a=(r_h[0, i], r_h[1, i]), src_b=(qv, qv), src_c=(o0, o1)
                )
                o2, o3 = cute.arch.fma_packed_f32x2(
                    src_a=(r_h[2, i], r_h[3, i]), src_b=(qv, qv), src_c=(o2, o3)
                )
        else:
            for i in cutlass.range_constexpr(0, vec, 2):
                if cutlass.const_expr(qk_in_regs):
                    qv0 = r_q[i]
                    qv1 = r_q[i + 1]
                    kv0 = r_k[i]
                    kv1 = r_k[i + 1]
                else:
                    qv0 = sQ[k_start + i]
                    qv1 = sQ[k_start + i + 1]
                    kv0 = sK[k_start + i]
                    kv1 = sK[k_start + i + 1]
                if cutlass.const_expr(use_packed_fma):
                    r_h[0, i], r_h[0, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(kv0, kv1),
                        src_b=(vn0, vn0),
                        src_c=(r_h[0, i], r_h[0, i + 1]),
                    )
                    r_h[1, i], r_h[1, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(kv0, kv1),
                        src_b=(vn1, vn1),
                        src_c=(r_h[1, i], r_h[1, i + 1]),
                    )
                    r_h[2, i], r_h[2, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(kv0, kv1),
                        src_b=(vn2, vn2),
                        src_c=(r_h[2, i], r_h[2, i + 1]),
                    )
                    r_h[3, i], r_h[3, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(kv0, kv1),
                        src_b=(vn3, vn3),
                        src_c=(r_h[3, i], r_h[3, i + 1]),
                    )
                else:
                    r_h[0, i], r_h[0, i + 1] = fma_pair(
                        kv0, kv1, vn0, vn0, r_h[0, i], r_h[0, i + 1]
                    )
                    r_h[1, i], r_h[1, i + 1] = fma_pair(
                        kv0, kv1, vn1, vn1, r_h[1, i], r_h[1, i + 1]
                    )
                    r_h[2, i], r_h[2, i + 1] = fma_pair(
                        kv0, kv1, vn2, vn2, r_h[2, i], r_h[2, i + 1]
                    )
                    r_h[3, i], r_h[3, i + 1] = fma_pair(
                        kv0, kv1, vn3, vn3, r_h[3, i], r_h[3, i + 1]
                    )
                if cutlass.const_expr(pack_dots):
                    # two independent exact FMAs per FFMA2: per-row order unchanged (bitwise)
                    o0, o1 = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[0, i], r_h[1, i]), src_b=(qv0, qv0), src_c=(o0, o1)
                    )
                    o2, o3 = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[2, i], r_h[3, i]), src_b=(qv0, qv0), src_c=(o2, o3)
                    )
                    o0, o1 = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[0, i + 1], r_h[1, i + 1]),
                        src_b=(qv1, qv1),
                        src_c=(o0, o1),
                    )
                    o2, o3 = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[2, i + 1], r_h[3, i + 1]),
                        src_b=(qv1, qv1),
                        src_c=(o2, o3),
                    )
                else:
                    o0 = o0 + r_h[0, i] * qv0 + r_h[0, i + 1] * qv1
                    o1 = o1 + r_h[1, i] * qv0 + r_h[1, i + 1] * qv1
                    o2 = o2 + r_h[2, i] * qv0 + r_h[2, i + 1] * qv1
                    o3 = o3 + r_h[3, i] * qv0 + r_h[3, i + 1] * qv1
        if cutlass.const_expr(fast_reduce):
            oj = _xreduce4(o0, o1, o2, o3, lane_in_group)
            if cutlass.const_expr(o_hoist):
                if o_pred:
                    o_lane[it] = cutlass.BFloat16(oj)
            else:
                if (lane_in_group & 3) == 0:
                    j = lane_in_group >> 2
                    o[
                        (
                            i_n,
                            0,
                            i_hv,
                            i_v * tile_v + (it * ILP_ROWS + j) * NUM_GROUPS + group_idx,
                        )
                    ] = cutlass.BFloat16(oj)
        else:
            for offset in [8, 4, 2, 1]:
                o0 += cute.arch.shuffle_sync_bfly(
                    o0, offset=offset, mask=-1, mask_and_clamp=31
                )
                o1 += cute.arch.shuffle_sync_bfly(
                    o1, offset=offset, mask=-1, mask_and_clamp=31
                )
                o2 += cute.arch.shuffle_sync_bfly(
                    o2, offset=offset, mask=-1, mask_and_clamp=31
                )
                o3 += cute.arch.shuffle_sync_bfly(
                    o3, offset=offset, mask=-1, mask_and_clamp=31
                )
            if lane_in_group == 0:
                o[(i_n, 0, i_hv, v0)] = cutlass.BFloat16(o0)
                o[(i_n, 0, i_hv, v1)] = cutlass.BFloat16(o1)
                o[(i_n, 0, i_hv, v2)] = cutlass.BFloat16(o2)
                o[(i_n, 0, i_hv, v3)] = cutlass.BFloat16(o3)

        # state write-back
        for i in cutlass.range_constexpr(vec):
            r_hb0[i] = cutlass.BFloat16(r_h[0, i])
            r_hb1[i] = cutlass.BFloat16(r_h[1, i])
            r_hb2[i] = cutlass.BFloat16(r_h[2, i])
            r_hb3[i] = cutlass.BFloat16(r_h[3, i])
        if cutlass.const_expr(st_part):
            cute.autovec_copy(
                r_hb0,
                cute.make_tensor(
                    tHw[(None, it * ILP_ROWS + 0, None)].iterator,
                    cute.make_layout((vec,)),
                ),
            )
            cute.autovec_copy(
                r_hb1,
                cute.make_tensor(
                    tHw[(None, it * ILP_ROWS + 1, None)].iterator,
                    cute.make_layout((vec,)),
                ),
            )
            cute.autovec_copy(
                r_hb2,
                cute.make_tensor(
                    tHw[(None, it * ILP_ROWS + 2, None)].iterator,
                    cute.make_layout((vec,)),
                ),
            )
            cute.autovec_copy(
                r_hb3,
                cute.make_tensor(
                    tHw[(None, it * ILP_ROWS + 3, None)].iterator,
                    cute.make_layout((vec,)),
                ),
            )
        else:
            cute.autovec_copy(
                r_hb0, cute.local_tile(h0_slot_w, (1, vec), (v0, lane_in_group))
            )
            cute.autovec_copy(
                r_hb1, cute.local_tile(h0_slot_w, (1, vec), (v1, lane_in_group))
            )
            cute.autovec_copy(
                r_hb2, cute.local_tile(h0_slot_w, (1, vec), (v2, lane_in_group))
            )
            cute.autovec_copy(
                r_hb3, cute.local_tile(h0_slot_w, (1, vec), (v3, lane_in_group))
            )

    if cutlass.const_expr(pf_dist > 0 and pf_lite):
        # pf_lite: the same prefetches with fewer instructions: the per-lane input dispatch runs
        # in warp 0 only (warp-uniform branch; no jump table in warps 1-3) and each thread's
        # state line is tile base + line * 128 B (dense rows, checked by the wrapper).
        if pf_ok:
            pf_tile = cute.local_tile(
                h0_source[(cutlass.Int64(pf_idx), pf_hv, None, None)],
                (tile_v, K),
                (pf_iv, 0),
            )
            if cutlass.const_expr(pf_inputs):
                if warp_idx == 0:
                    if cutlass.const_expr(pf_u32):
                        pf_n = cutlass.Int32(cutlass.Uint32(pf_t) // cutlass.Uint32(HV))
                        pf_h = cutlass.Int32(
                            cutlass.Uint32(pf_hv) // cutlass.Uint32(HV // H)
                        )
                    else:
                        pf_n = pf_t // HV
                        pf_h = pf_hv // (HV // H)
                    if lane_in_warp < 2:
                        cute.arch.prefetch(
                            cute.local_tile(
                                q[(pf_n, 0, pf_h, None)], (64,), (lane_in_warp,)
                            ).iterator.llvm_ptr,
                            cache_level="L2",
                        )
                    if lane_in_warp >= 2 and lane_in_warp < 4:
                        cute.arch.prefetch(
                            cute.local_tile(
                                k[(pf_n, 0, pf_h, None)], (64,), (lane_in_warp - 2,)
                            ).iterator.llvm_ptr,
                            cache_level="L2",
                        )
                    if lane_in_warp == 4:
                        cute.arch.prefetch(
                            cute.local_tile(
                                v[(pf_n, 0, pf_hv, None)], (tile_v,), (pf_iv,)
                            ).iterator.llvm_ptr,
                            cache_level="L2",
                        )
                    if lane_in_warp == 5:
                        cute.arch.prefetch(
                            a[(pf_n, 0, None)].iterator.llvm_ptr, cache_level="L2"
                        )
                    if lane_in_warp == 6:
                        cute.arch.prefetch(
                            b_gate[(pf_n, 0, None)].iterator.llvm_ptr, cache_level="L2"
                        )
            PF_LINES: cutlass.Constexpr[int] = (
                tile_v * K * 2 // 128
            )  # a multiple of NUM_THREADS
            for li in cutlass.range_constexpr(PF_LINES // NUM_THREADS):
                if cutlass.const_expr(li == 0):
                    line = tidx
                else:
                    line = li * NUM_THREADS + tidx
                if cutlass.const_expr(pf_addr):
                    # pf_addr: uniform 64-bit tile base + line * 128 B
                    pl_ptr = cute.make_ptr(
                        cutlass.BFloat16,
                        pf_tile.iterator.toint()
                        + cutlass.Int64(line) * cutlass.Int64(128),
                        cute.AddressSpace.gmem,
                        assumed_align=128,
                    )
                else:
                    pf_lines = cute.make_tensor(
                        pf_tile.iterator,
                        cute.make_layout((PF_LINES, 64), stride=(64, 1)),
                    )
                    pl_ptr = cute.local_tile(pf_lines, (1, 64), (line, 0)).iterator
                if cutlass.const_expr(pf_evict_last):
                    cute.arch.prefetch(
                        pl_ptr.llvm_ptr,
                        cache_level="L2",
                        evict_priority=_nvvm.CacheEvictionPriority.EvictLast,
                    )
                else:
                    cute.arch.prefetch(pl_ptr.llvm_ptr, cache_level="L2")
    if cutlass.const_expr(pf_dist > 0 and not pf_lite):
        if pf_ok:
            pf_tile = cute.local_tile(
                h0_source[(cutlass.Int64(pf_idx), pf_hv, None, None)],
                (tile_v, K),
                (pf_iv, 0),
            )
            if cutlass.const_expr(pf_inputs):
                # small per-tile inputs of the future tile: q, k rows (2 lines each),
                # v slice (1 line), a and b (the line holding [n, hv])
                if cutlass.const_expr(pf_u32):
                    pf_n = cutlass.Int32(cutlass.Uint32(pf_t) // cutlass.Uint32(HV))
                    pf_h = cutlass.Int32(
                        cutlass.Uint32(pf_hv) // cutlass.Uint32(HV // H)
                    )
                else:
                    pf_n = pf_t // HV
                    pf_h = pf_hv // (HV // H)
                if tidx < 2:
                    cute.arch.prefetch(
                        cute.local_tile(
                            q[(pf_n, 0, pf_h, None)], (64,), (tidx,)
                        ).iterator.llvm_ptr,
                        cache_level="L2",
                    )
                if tidx >= 2 and tidx < 4:
                    cute.arch.prefetch(
                        cute.local_tile(
                            k[(pf_n, 0, pf_h, None)], (64,), (tidx - 2,)
                        ).iterator.llvm_ptr,
                        cache_level="L2",
                    )
                if tidx == 4:
                    cute.arch.prefetch(
                        cute.local_tile(
                            v[(pf_n, 0, pf_hv, None)], (tile_v,), (pf_iv,)
                        ).iterator.llvm_ptr,
                        cache_level="L2",
                    )
                if tidx == 5:
                    cute.arch.prefetch(
                        a[(pf_n, 0, None)].iterator.llvm_ptr, cache_level="L2"
                    )
                if tidx == 6:
                    cute.arch.prefetch(
                        b_gate[(pf_n, 0, None)].iterator.llvm_ptr, cache_level="L2"
                    )
            # one 128 B line per thread (tile_v * K * 2 B / 128 B lines)
            LINES: cutlass.Constexpr[int] = tile_v * K * 2 // 128
            for li in cutlass.range_constexpr((LINES + NUM_THREADS - 1) // NUM_THREADS):
                line = li * NUM_THREADS + tidx
                if line < LINES:
                    if cutlass.const_expr(pf_addr):
                        # pf_addr: uniform 64-bit tile base + line * 128 B (dense rows, checked by the wrapper)
                        pl_ptr = cute.make_ptr(
                            cutlass.BFloat16,
                            pf_tile.iterator.toint()
                            + cutlass.Int64(line) * cutlass.Int64(128),
                            cute.AddressSpace.gmem,
                            assumed_align=128,
                        )
                    else:
                        pl_ptr = cute.local_tile(
                            pf_tile, (1, 64), (line // 2, line % 2)
                        ).iterator
                    if cutlass.const_expr(pf_evict_last):
                        cute.arch.prefetch(
                            pl_ptr.llvm_ptr,
                            cache_level="L2",
                            evict_priority=_nvvm.CacheEvictionPriority.EvictLast,
                        )
                    else:
                        cute.arch.prefetch(pl_ptr.llvm_ptr, cache_level="L2")


@cute.jit
def _run_t1_rubin(
    h0_source: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b_gate: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    h0_out_indices: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    use_packed_fma: cutlass.Constexpr[bool],
    same_pool: cutlass.Constexpr[bool],
    stage_smem: cutlass.Constexpr[bool],
    qk_in_regs: cutlass.Constexpr[bool],
    issue_fence: cutlass.Constexpr[bool],
    prefetch_v: cutlass.Constexpr[bool],
    min_blocks: cutlass.Constexpr[int],
    fast_reduce: cutlass.Constexpr[bool],
    pack_dots: cutlass.Constexpr[bool],
    pf_dist: cutlass.Constexpr[int],
    pf_evict_last: cutlass.Constexpr[bool],
    pf_inputs: cutlass.Constexpr[bool],
    pair_rows: cutlass.Constexpr[bool],
    vn_lane: cutlass.Constexpr[bool],
    o_hoist: cutlass.Constexpr[bool],
    st_part: cutlass.Constexpr[bool],
    unpack_bits: cutlass.Constexpr[bool],
    unpack_qk: cutlass.Constexpr[bool],
    vec_qk: cutlass.Constexpr[bool],
    pf_lite: cutlass.Constexpr[bool],
    pf_u32: cutlass.Constexpr[bool],
    pf_addr: cutlass.Constexpr[bool],
    io_layouts: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    if cutlass.const_expr(io_layouts is not None):
        # static_io: q/k/v/a/b/o rebuilt with this call's static strides (they are in the wrapper's
        # cache key); only the batch extent stays dynamic. Same addresses, shorter address math.
        (lq, sq), (lk, sk), (lv, sv), (la, sa), (lb, sb), (lo, so) = io_layouts
        q = cute.make_tensor(
            q.iterator, cute.make_layout((q.shape[0],) + lq, stride=sq)
        )
        k = cute.make_tensor(
            k.iterator, cute.make_layout((k.shape[0],) + lk, stride=sk)
        )
        v = cute.make_tensor(
            v.iterator, cute.make_layout((v.shape[0],) + lv, stride=sv)
        )
        a = cute.make_tensor(
            a.iterator, cute.make_layout((a.shape[0],) + la, stride=sa)
        )
        b_gate = cute.make_tensor(
            b_gate.iterator, cute.make_layout((b_gate.shape[0],) + lb, stride=sb)
        )
        o = cute.make_tensor(
            o.iterator, cute.make_layout((o.shape[0],) + lo, stride=so)
        )
    num_v_tiles: cutlass.Constexpr[int] = V // tile_v
    B = cute.size(q.shape[0])
    grid_size = B * HV * num_v_tiles

    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )
    thr_layout = cute.make_layout(
        (NUM_GROUPS, LANES_PER_ROW), stride=(LANES_PER_ROW, 1)
    )
    val_layout = cute.make_layout((1, ELEMS_PER_LANE))
    tiled_copy_h = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

    smem_bytes = 4 * K * 2 + 4 * 4 + 256
    if cutlass.const_expr(stage_smem):
        smem_bytes = smem_bytes + 2 * tile_v * K + 128

    gdn_t1_rubin_kernel(
        tiled_copy_h,
        h0_source,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b_gate,
        o,
        h0_indices,
        h0_out_indices,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        H,
        K,
        V,
        tile_v,
        num_v_tiles,
        use_qk_l2norm,
        use_packed_fma,
        same_pool,
        stage_smem,
        qk_in_regs,
        issue_fence,
        prefetch_v,
        fast_reduce,
        pack_dots,
        pf_dist,
        pf_evict_last,
        pf_inputs,
        pair_rows,
        vn_lane,
        o_hoist,
        st_part,
        unpack_bits,
        unpack_qk,
        vec_qk,
        pf_lite,
        pf_u32,
        pf_addr,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
        min_blocks_per_mp=min_blocks,
    )


_compiled_t1_rubin: dict = {}


def _io_layout(torch_t: torch.Tensor):
    # static_io: (shape after the batch mode, strides) baked into the build and held in the cache key.
    # A size-1 mode is only ever indexed at 0, so its stride is normalized to the next inner mode's
    # extent x stride (what torch's view() gives it): B=1 slices of fused tensors then share the
    # compact layout's build instead of failing a compact-layout check, and every layout (also
    # non-contiguous views of a fused QKV buffer) gets static strides.
    st = list(torch_t.stride())
    for d in reversed(range(torch_t.dim())):
        if torch_t.shape[d] == 1:
            st[d] = st[d + 1] * torch_t.shape[d + 1] if d + 1 < torch_t.dim() else 1
    return tuple(torch_t.shape[1:]), tuple(st)


def gated_delta_rule_t1_rubin(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    initial_state_source: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    output_state_indices: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    tile_v: int = 64,
    stage_smem: bool = True,
    qk_in_regs: bool = True,
    issue_fence: bool = True,
    prefetch_v: bool = True,
    min_blocks: int = 0,
    fast_reduce: bool = True,
    pack_dots: bool = True,
    pf_dist: Optional[int] = None,
    pf_evict_last: bool = True,
    pf_inputs: bool = True,
    pair_rows: bool = True,
    vn_lane: bool = True,
    o_hoist: bool = True,
    st_part: bool = True,
    unpack_bits: bool = True,
    unpack_qk: bool = True,
    vec_qk: bool = True,
    static_io: bool = True,
    pf_lite: bool = True,
    pf_u32: bool = True,
    pf_addr: bool = True,
) -> torch.Tensor:
    """T=1 GDN decode with BF16 state on Rubin (compute capability 10.7): state update + output.

    Same arguments and semantics as ``gdn_decode_bf16_state.gated_delta_rule`` at T=1: q/k/v
    ``[B, 1, H|HV, 128]``, state pool ``initial_state_source`` ``[pool, HV, V=128, K=128]`` bf16
    updated in place at ``output_state_indices`` (default: ``initial_state_indices``), output
    ``[B, 1, HV, 128]`` bf16. Bitwise equal to that kernel's T=1 wide_vec path.

    The remaining keyword arguments are tuning options (see the module docstring); their defaults
    are the measured best on Rubin. Raises ``ValueError`` on any other compute capability.
    """
    assert q is not None and k is not None and v is not None
    assert b is not None and initial_state_source is not None
    q, k, v, a, b = as_bf16(q, k, v, a, b)
    B_val, T_val, H_val, K_val = q.shape
    HV_val = v.shape[2]
    V_val = v.shape[3]
    assert T_val == 1, f"T=1 kernel, got T={T_val}"
    assert K_val == 128 and V_val == 128
    assert initial_state_source.dtype == torch.bfloat16
    assert tile_v in (32, 64, 128) and (tile_v // NUM_GROUPS) % ILP_ROWS == 0
    target = gdn_device_target(q.device)
    if (target.major, target.minor) != (10, 7):
        raise ValueError(
            "gated_delta_rule_t1_rubin needs compute capability 10.7 (Rubin), got "
            f"{target.major}.{target.minor}"
        )
    if scale is None:
        scale = 1.0 / math.sqrt(K_val)
    if pf_dist is None:
        # Cross-CTA L2 prefetch distance in tiles: ~1.2 residency generations at tile_v=64
        # (7 CTAs/SM). At tile_v=128 (6 CTAs/SM) it overshoots; pass pf_dist explicitly there.
        pf_dist = (
            torch.cuda.get_device_properties(q.device).multi_processor_count * 25 // 3
        )

    if output_state_indices is not None and output_state_indices.dtype != torch.int32:
        output_state_indices = output_state_indices.to(
            torch.int32
        )  # as the generic kernel
    use_packed_fma = target.use_packed_fma
    same_pool = (
        output_state_indices is None or output_state_indices is initial_state_indices
    )
    if initial_state_indices is None:
        initial_state_indices = torch.arange(B_val, dtype=torch.int32, device=q.device)
    if output_state_indices is None:
        output_state_indices = initial_state_indices
    if output is None:
        output = torch.empty(
            B_val, 1, HV_val, V_val, device=q.device, dtype=torch.bfloat16
        )
    stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)

    # Instruction cuts: reduced to the effective value, so a cut is off unless the options it
    # builds on are on and its input checks pass (equal builds share one cache entry; the
    # kernel relies on these preconditions).
    pf_late = pf_dist > 0
    # the pf_lite / pf_addr line address (tile base + line * 128 B) needs dense 256 B state rows
    dense_rows = (
        initial_state_source.stride(-1) == 1
        and initial_state_source.stride(-2) == K_val
    )
    # pair_rows reads q / k from registers in every warp: with qk_in_regs=False only warp 0 has them
    pair_rows = pair_rows and use_packed_fma and pack_dots and qk_in_regs
    vn_lane = vn_lane and fast_reduce and prefetch_v
    o_hoist = o_hoist and fast_reduce and output.stride(-1) == 1
    st_part = st_part and stage_smem
    unpack_bits = unpack_bits and stage_smem
    unpack_qk = unpack_qk and qk_in_regs
    vec_qk = (
        vec_qk
        and qk_in_regs
        and all(
            st[-1] == 1 and all(x % 8 == 0 for x in st[:-1]) and t.data_ptr() % 16 == 0
            for t, st in ((t, _io_layout(t)[1]) for t in (q, k))
        )
    )
    # static_io: the per-token tensors' own strides (size-1 modes normalized), baked in and keyed
    io_layouts = (
        tuple(_io_layout(t) for t in (q, k, v, a, b, output)) if static_io else None
    )
    pf_lite = (
        pf_lite
        and pf_late
        and dense_rows
        and (tile_v * K_val * 2 // 128) % NUM_THREADS == 0
    )
    pf_u32 = pf_u32 and pf_dist > 0
    pf_addr = pf_addr and pf_late and dense_rows

    contiguous_pool = initial_state_source.is_contiguous()
    cache_key = (
        T_val,
        H_val,
        HV_val,
        K_val,
        V_val,
        contiguous_pool,
        None if contiguous_pool else tuple(initial_state_source.stride()),
        tile_v,
        stage_smem,
        qk_in_regs,
        issue_fence,
        prefetch_v,
        min_blocks,
        fast_reduce,
        pack_dots and use_packed_fma,
        pf_dist,
        pf_evict_last,
        pf_inputs,
        pair_rows,
        (vn_lane, o_hoist, st_part, unpack_bits, unpack_qk, vec_qk, io_layouts),
        (pf_lite, pf_u32, pf_addr),
        use_qk_l2norm_in_kernel,
        scale,
        softplus_beta,
        softplus_threshold,
        use_packed_fma,
        same_pool,
        _dtype_key(A_log, dt_bias, initial_state_indices),
        target.compile_key,
    )
    if cache_key not in _compiled_t1_rubin:
        if contiguous_pool:
            h_ = _mark_slot_dynamic(initial_state_source)
        else:
            h_ = from_dlpack(
                initial_state_source, assumed_align=32, enable_tvm_ffi=True
            )
        _compiled_t1_rubin[cache_key] = cute.compile[
            gdn_compile_options(q.device, *_BF16_STATE_COMPILE_OPTS)
        ](
            _run_t1_rubin,
            h_,
            from_dlpack(A_log, assumed_align=32, enable_tvm_ffi=True),
            _mark_batch_dynamic(a),
            from_dlpack(dt_bias, assumed_align=32, enable_tvm_ffi=True),
            _mark_batch_dynamic(q),
            _mark_batch_dynamic(k),
            _mark_batch_dynamic(v),
            _mark_batch_dynamic(b),
            _mark_batch_dynamic(output),
            _mark_index_dynamic(initial_state_indices),
            _mark_index_dynamic(output_state_indices),
            softplus_beta,
            softplus_threshold,
            scale,
            HV_val,
            H_val,
            K_val,
            V_val,
            tile_v,
            use_qk_l2norm_in_kernel,
            use_packed_fma,
            same_pool,
            stage_smem,
            qk_in_regs,
            issue_fence,
            prefetch_v,
            min_blocks,
            fast_reduce,
            pack_dots and use_packed_fma,
            pf_dist,
            pf_evict_last,
            pf_inputs,
            pair_rows,
            vn_lane,
            o_hoist,
            st_part,
            unpack_bits,
            unpack_qk,
            vec_qk,
            pf_lite,
            pf_u32,
            pf_addr,
            io_layouts,
            stream,
        )
    _compiled_t1_rubin[cache_key](
        initial_state_source,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        output,
        initial_state_indices,
        output_state_indices,
        stream,
    )
    return output
