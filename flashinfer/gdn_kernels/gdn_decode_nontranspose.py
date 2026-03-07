"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Gated Delta Rule Decode Kernel - Nontranspose (K-major) Layout
===============================================================

CuTe-DSL implementation of GDN decode with state layout [B*HV, K, V].
Uses TMA pipelining for T=1 decode with K-major state (no transpose needed).
"""

import functools

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

# ============================================================================
# Constants for NONTRANSPOSE version ([pool, HV, K, V])
# ============================================================================
TILE_K_NT = 128
TILE_V_NT = 32
TILE_V_PADDED_NT = 36
TILE_V_SMALL_NT = 16
TILE_V_SMALL_PADDED_NT = 20
NUM_STAGES_NT = 2
NUM_THREADS_NT = 128
NUM_BLOCKS_PER_STATE_SMALL_NT = 8
NUM_THREADS_LARGE_NT = 256
NUM_WARPS_LARGE_NT = 8
V_PER_WARP_NT = 4
ROWS_PER_ITER_NT = 8
NUM_K_ITERS_NT = TILE_K_NT // ROWS_PER_ITER_NT
SMALL_BATCH_THRESHOLD_NT = 32


@cute.kernel
def gdn_decode_kernel_small_batch_nontranspose(
    tiled_copy_load: cute.TiledCopy,
    h0_source: cute.Tensor,
    smem_layout_staged: cute.Layout,
    num_v_tiles: cutlass.Constexpr[int],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    a: cute.Tensor,
    b: cute.Tensor,
    A_log: cute.Tensor,
    dt_bias: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
):
    """Small batch kernel for (N, 1, ...) format with K-major state layout."""
    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    block_idx, _, _ = cute.arch.block_idx()

    NUM_WARPS_SMALL = 4
    V_PER_WARP_SMALL = TILE_V_SMALL_NT // NUM_WARPS_SMALL
    ROWS_PER_ITER_SMALL = 32 // V_PER_WARP_SMALL
    NUM_K_ITERS_SMALL = TILE_K_NT // ROWS_PER_ITER_SMALL

    batch_idx = block_idx // NUM_BLOCKS_PER_STATE_SMALL_NT
    batch_inner = block_idx % NUM_BLOCKS_PER_STATE_SMALL_NT
    num_v_tiles_per_block = num_v_tiles // NUM_BLOCKS_PER_STATE_SMALL_NT
    start_v_tile = batch_inner * num_v_tiles_per_block

    i_n = batch_idx // HV
    i_hv = batch_idx % HV
    i_h = i_hv // (HV // H)

    pool_idx = h0_indices[i_n]

    if pool_idx >= 0:
        k_local = in_warp_tid // V_PER_WARP_SMALL
        v_local = in_warp_tid % V_PER_WARP_SMALL
        v_base = warp_idx * V_PER_WARP_SMALL
        v_idx = v_base + v_local

        smem = cutlass.utils.SmemAllocator()
        sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
        smem_o_layout = cute.make_layout((TILE_V_SMALL_NT,), stride=(1,))
        smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
        smem_k_layout = cute.make_layout((TILE_K_NT,), stride=(1,))
        smem_q_layout = cute.make_layout((TILE_K_NT,), stride=(1,))
        sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
        sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)

        if tidx < TILE_K_NT:
            sK[tidx] = cutlass.Float32(k[i_n, 0, i_h, tidx])
            sQ[tidx] = cutlass.Float32(q[i_n, 0, i_h, tidx])

        # Compute flat index for flattened state [B*HV, K, V]
        flat_idx = pool_idx * HV + i_hv
        gSrc_batch = h0_source[(flat_idx, None, None)]
        gSrc = cute.local_tile(gSrc_batch, (TILE_K_NT, TILE_V_SMALL_NT), (0, None))
        thr_copy_load = tiled_copy_load.get_slice(tidx)

        prefetch_count = cutlass.min(NUM_STAGES_NT - 1, num_v_tiles_per_block)
        for v_tile_offset in range(prefetch_count):
            v_tile = start_v_tile + v_tile_offset
            stage = v_tile_offset % NUM_STAGES_NT
            gSrc_tile = gSrc[(None, None, v_tile)]
            sData_stage = sData[(None, None, stage)]
            thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
            thr_sData = thr_copy_load.partition_D(sData_stage)
            cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
            cute.arch.cp_async_commit_group()

        r_A_log = cutlass.Float32(A_log[i_hv])
        r_dt_bias = cutlass.Float32(dt_bias[i_hv])
        r_a = cutlass.Float32(a[i_n, 0, i_hv])
        r_b = cutlass.Float32(b[i_n, 0, i_hv])

        r_g = 0.0
        r_beta = 0.0
        if in_warp_tid == 0:
            x = r_a + r_dt_bias
            beta_x = softplus_beta * x
            softplus_x = 0.0
            if beta_x <= softplus_threshold:
                exp_beta_x = cute.exp(beta_x, fastmath=True)
                log_input = cutlass.Float32(1.0 + exp_beta_x)
                log_result = cutlass.Float32(cute.log(log_input, fastmath=True))
                softplus_x = cutlass.Float32(
                    (cutlass.Float32(1.0) / softplus_beta) * log_result
                )
            else:
                softplus_x = x
            r_g_value = -cute.exp(r_A_log, fastmath=True) * softplus_x
            r_beta = 1.0 / (1.0 + cute.exp(-r_b, fastmath=True))
            r_g = cute.exp(r_g_value, fastmath=True)

        r_g = cute.arch.shuffle_sync(r_g, 0)
        r_beta = cute.arch.shuffle_sync(r_beta, 0)

        cute.arch.barrier()

        if use_qk_l2norm:
            sum_q_partial = 0.0
            sum_k_partial = 0.0
            if tidx < TILE_K_NT:
                q_val = sQ[tidx]
                k_val = sK[tidx]
                sum_q_partial = q_val * q_val
                sum_k_partial = k_val * k_val

            for offset in [16, 8, 4, 2, 1]:
                sum_q_partial += cute.arch.shuffle_sync_bfly(
                    sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31
                )
                sum_k_partial += cute.arch.shuffle_sync_bfly(
                    sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31
                )

            if in_warp_tid == 0:
                smem_o[warp_idx] = sum_q_partial
                smem_o[warp_idx + 4] = sum_k_partial
            cute.arch.barrier()

            inv_norm_q = 0.0
            inv_norm_k = 0.0
            if warp_idx == 0:
                local_sum_q = 0.0
                local_sum_k = 0.0
                if in_warp_tid < NUM_WARPS_SMALL:
                    local_sum_q = smem_o[in_warp_tid]
                    local_sum_k = smem_o[in_warp_tid + 4]
                for offset in [2, 1]:
                    local_sum_q += cute.arch.shuffle_sync_bfly(
                        local_sum_q, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    local_sum_k += cute.arch.shuffle_sync_bfly(
                        local_sum_k, offset=offset, mask=-1, mask_and_clamp=31
                    )
                if in_warp_tid == 0:
                    smem_o[0] = cute.rsqrt(local_sum_q + 1e-6, fastmath=True)
                    smem_o[1] = cute.rsqrt(local_sum_k + 1e-6, fastmath=True)
            cute.arch.barrier()

            inv_norm_q = smem_o[0]
            inv_norm_k = smem_o[1]

            if tidx < TILE_K_NT:
                sK[tidx] = sK[tidx] * inv_norm_k
                sQ[tidx] = sQ[tidx] * scale * inv_norm_q
            cute.arch.barrier()
        else:
            if tidx < TILE_K_NT:
                sQ[tidx] = sQ[tidx] * scale
            cute.arch.barrier()

        for v_tile_offset in range(num_v_tiles_per_block):
            v_tile = start_v_tile + v_tile_offset
            stage = v_tile_offset % NUM_STAGES_NT

            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            next_v_tile_offset = v_tile_offset + prefetch_count
            if next_v_tile_offset < num_v_tiles_per_block:
                next_v_tile = start_v_tile + next_v_tile_offset
                next_stage = next_v_tile_offset % NUM_STAGES_NT
                gSrc_next = gSrc[(None, None, next_v_tile)]
                sData_next = sData[(None, None, next_stage)]
                thr_gSrc = thr_copy_load.partition_S(gSrc_next)
                thr_sData = thr_copy_load.partition_D(sData_next)
                cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                cute.arch.cp_async_commit_group()

            v_global = v_tile * TILE_V_SMALL_NT + v_idx
            r_v = cutlass.Float32(v[i_n, 0, i_hv, v_global])

            sum_hk = 0.0
            for k_iter in range(NUM_K_ITERS_SMALL, unroll=16):  # type: ignore[call-overload]
                k_base = k_iter * ROWS_PER_ITER_SMALL
                k_idx = k_base + k_local
                h_val = sData[(k_idx, v_idx, stage)] * r_g
                r_k_val = sK[k_idx]
                sum_hk += h_val * r_k_val

            for offset in [4, 2, 1]:
                sum_hk += cute.arch.shuffle_sync_bfly(
                    sum_hk,
                    offset=offset * V_PER_WARP_SMALL,
                    mask=-1,
                    mask_and_clamp=31,
                )

            v_new = (r_v - sum_hk) * r_beta
            v_new = cute.arch.shuffle_sync(v_new, v_local)

            sum_hq = 0.0
            for k_iter in range(NUM_K_ITERS_SMALL, unroll=16):  # type: ignore[call-overload]
                k_base = k_iter * ROWS_PER_ITER_SMALL
                k_idx = k_base + k_local
                h_old = sData[(k_idx, v_idx, stage)] * r_g
                r_k_val = sK[k_idx]
                r_q_val = sQ[k_idx]
                h_new = h_old + r_k_val * v_new
                sData[(k_idx, v_idx, stage)] = h_new
                sum_hq += h_new * r_q_val

            for offset in [4, 2, 1]:
                sum_hq += cute.arch.shuffle_sync_bfly(
                    sum_hq,
                    offset=offset * V_PER_WARP_SMALL,
                    mask=-1,
                    mask_and_clamp=31,
                )

            if k_local == 0:
                v_global_out = v_tile * TILE_V_SMALL_NT + v_idx
                o[(i_n, 0, i_hv, v_global_out)] = cutlass.BFloat16(sum_hq)

            cute.arch.barrier()

            for k_iter in cutlass.range_constexpr(NUM_K_ITERS_SMALL):
                flat_tid = tidx + k_iter * 128
                k_write = flat_tid // TILE_V_SMALL_NT
                v_write = flat_tid % TILE_V_SMALL_NT
                if k_write < TILE_K_NT:
                    h_val = sData[(k_write, v_write, stage)]
                    v_global_write = v_tile * TILE_V_SMALL_NT + v_write
                    # Use flat index for flattened state [B*HV, K, V]
                    h0_source[(flat_idx, k_write, v_global_write)] = h_val

            cute.arch.barrier()


@cute.kernel
def gdn_decode_kernel_big_batch_nontranspose(
    tiled_copy_load: cute.TiledCopy,
    h0_source: cute.Tensor,
    smem_layout_staged: cute.Layout,
    num_v_tiles: cutlass.Constexpr[int],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    a: cute.Tensor,
    b: cute.Tensor,
    A_log: cute.Tensor,
    dt_bias: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
):
    """Large batch kernel for (N, 1, ...) format with K-major state layout."""
    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    batch_idx, _, _ = cute.arch.block_idx()
    i_n = batch_idx // HV
    i_hv = batch_idx % HV
    i_h = i_hv // (HV // H)

    pool_idx = h0_indices[i_n]

    if pool_idx >= 0:
        k_local = in_warp_tid // V_PER_WARP_NT
        v_local = in_warp_tid % V_PER_WARP_NT
        v_base = warp_idx * V_PER_WARP_NT
        v_idx = v_base + v_local

        smem = cutlass.utils.SmemAllocator()
        sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
        smem_o_layout = cute.make_layout((TILE_V_NT,), stride=(1,))
        smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
        smem_k_layout = cute.make_layout((TILE_K_NT,), stride=(1,))
        smem_q_layout = cute.make_layout((TILE_K_NT,), stride=(1,))
        sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
        sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)

        if tidx < TILE_K_NT:
            sK[tidx] = cutlass.Float32(k[i_n, 0, i_h, tidx])
            sQ[tidx] = cutlass.Float32(q[i_n, 0, i_h, tidx])

        # Compute flat index for flattened state [B*HV, K, V]
        flat_idx = pool_idx * HV + i_hv
        gSrc_batch = h0_source[(flat_idx, None, None)]
        gSrc = cute.local_tile(gSrc_batch, (TILE_K_NT, TILE_V_NT), (0, None))
        thr_copy_load = tiled_copy_load.get_slice(tidx)

        prefetch_count = cutlass.min(NUM_STAGES_NT - 1, num_v_tiles)
        for v_tile in range(prefetch_count):
            stage = v_tile % NUM_STAGES_NT
            gSrc_tile = gSrc[(None, None, v_tile)]
            sData_stage = sData[(None, None, stage)]
            thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
            thr_sData = thr_copy_load.partition_D(sData_stage)
            cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
            cute.arch.cp_async_commit_group()

        r_A_log = cutlass.Float32(A_log[i_hv])
        r_dt_bias = cutlass.Float32(dt_bias[i_hv])
        r_a = cutlass.Float32(a[i_n, 0, i_hv])
        r_b = cutlass.Float32(b[i_n, 0, i_hv])

        r_g = 0.0
        r_beta = 0.0
        if in_warp_tid == 0:
            x = r_a + r_dt_bias
            beta_x = softplus_beta * x
            softplus_x = 0.0
            if beta_x <= softplus_threshold:
                exp_beta_x = cute.exp(beta_x, fastmath=True)
                log_input = cutlass.Float32(1.0 + exp_beta_x)
                log_result = cutlass.Float32(cute.log(log_input, fastmath=True))
                softplus_x = cutlass.Float32(
                    (cutlass.Float32(1.0) / softplus_beta) * log_result
                )
            else:
                softplus_x = x
            r_g_value = -cute.exp(r_A_log, fastmath=True) * softplus_x
            r_beta = 1.0 / (1.0 + cute.exp(-r_b, fastmath=True))
            r_g = cute.exp(r_g_value, fastmath=True)

        r_g = cute.arch.shuffle_sync(r_g, 0)
        r_beta = cute.arch.shuffle_sync(r_beta, 0)

        cute.arch.barrier()

        if use_qk_l2norm:
            sum_q_partial = 0.0
            sum_k_partial = 0.0
            if tidx < TILE_K_NT:
                q_val = sQ[tidx]
                k_val = sK[tidx]
                sum_q_partial = q_val * q_val
                sum_k_partial = k_val * k_val

            for offset in [16, 8, 4, 2, 1]:
                sum_q_partial += cute.arch.shuffle_sync_bfly(
                    sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31
                )
                sum_k_partial += cute.arch.shuffle_sync_bfly(
                    sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31
                )

            if in_warp_tid == 0:
                smem_o[warp_idx] = sum_q_partial
                smem_o[warp_idx + 8] = sum_k_partial
            cute.arch.barrier()

            inv_norm_q = 0.0
            inv_norm_k = 0.0
            if warp_idx == 0:
                local_sum_q = 0.0
                local_sum_k = 0.0
                if in_warp_tid < NUM_WARPS_LARGE_NT:
                    local_sum_q = smem_o[in_warp_tid]
                    local_sum_k = smem_o[in_warp_tid + 8]
                for offset in [4, 2, 1]:
                    local_sum_q += cute.arch.shuffle_sync_bfly(
                        local_sum_q, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    local_sum_k += cute.arch.shuffle_sync_bfly(
                        local_sum_k, offset=offset, mask=-1, mask_and_clamp=31
                    )
                if in_warp_tid == 0:
                    smem_o[0] = cute.rsqrt(local_sum_q + 1e-6, fastmath=True)
                    smem_o[1] = cute.rsqrt(local_sum_k + 1e-6, fastmath=True)
            cute.arch.barrier()

            inv_norm_q = smem_o[0]
            inv_norm_k = smem_o[1]

            if tidx < TILE_K_NT:
                sK[tidx] = sK[tidx] * inv_norm_k
                sQ[tidx] = sQ[tidx] * scale * inv_norm_q
            cute.arch.barrier()
        else:
            if tidx < TILE_K_NT:
                sQ[tidx] = sQ[tidx] * scale
            cute.arch.barrier()

        for v_tile in range(num_v_tiles):
            stage = v_tile % NUM_STAGES_NT

            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            next_v_tile = v_tile + prefetch_count
            if next_v_tile < num_v_tiles:
                next_stage = next_v_tile % NUM_STAGES_NT
                gSrc_next = gSrc[(None, None, next_v_tile)]
                sData_next = sData[(None, None, next_stage)]
                thr_gSrc = thr_copy_load.partition_S(gSrc_next)
                thr_sData = thr_copy_load.partition_D(sData_next)
                cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                cute.arch.cp_async_commit_group()

            v_global = v_tile * TILE_V_NT + v_idx
            r_v = cutlass.Float32(v[i_n, 0, i_hv, v_global])

            sum_hk = 0.0
            for k_iter in range(NUM_K_ITERS_NT, unroll=8):  # type: ignore[call-overload]
                k_base = k_iter * ROWS_PER_ITER_NT
                k_idx = k_base + k_local
                h_val = sData[(k_idx, v_idx, stage)] * r_g
                r_k_val = sK[k_idx]
                sum_hk += h_val * r_k_val

            for offset in [4, 2, 1]:
                sum_hk += cute.arch.shuffle_sync_bfly(
                    sum_hk, offset=offset * V_PER_WARP_NT, mask=-1, mask_and_clamp=31
                )

            v_new = (r_v - sum_hk) * r_beta
            v_new = cute.arch.shuffle_sync(v_new, v_local)

            sum_hq = 0.0
            for k_iter in range(NUM_K_ITERS_NT, unroll=8):  # type: ignore[call-overload]
                k_base = k_iter * ROWS_PER_ITER_NT
                k_idx = k_base + k_local
                h_old = sData[(k_idx, v_idx, stage)] * r_g
                r_k_val = sK[k_idx]
                r_q_val = sQ[k_idx]
                h_new = h_old + r_k_val * v_new
                sData[(k_idx, v_idx, stage)] = h_new
                sum_hq += h_new * r_q_val

            for offset in [4, 2, 1]:
                sum_hq += cute.arch.shuffle_sync_bfly(
                    sum_hq, offset=offset * V_PER_WARP_NT, mask=-1, mask_and_clamp=31
                )

            if k_local == 0:
                v_global_out = v_tile * TILE_V_NT + v_idx
                o[(i_n, 0, i_hv, v_global_out)] = cutlass.BFloat16(sum_hq)

            cute.arch.barrier()

            for k_iter in cutlass.range_constexpr(NUM_K_ITERS_NT):
                flat_tid = tidx + k_iter * 256
                k_write = flat_tid // TILE_V_NT
                v_write = flat_tid % TILE_V_NT
                if k_write < TILE_K_NT:
                    h_val = sData[(k_write, v_write, stage)]
                    v_global_write = v_tile * TILE_V_NT + v_write
                    # Use flat index for flattened state [B*HV, K, V]
                    h0_source[(flat_idx, k_write, v_global_write)] = h_val

            cute.arch.barrier()


@cute.jit
def run_gdn_decode_kernel_small_batch_nontranspose(
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    a: cute.Tensor,
    b: cute.Tensor,
    A_log: cute.Tensor,
    dt_bias: cute.Tensor,
    h0_source: cute.Tensor,
    h0_indices: cute.Tensor,
    o: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    # h0_source is flattened to [B*HV, K, V] to ensure proper alignment for SIMT async copy
    batch_hv_dim, k_dim, v_dim = h0_source.layout.shape
    h0_indices.layout.shape[0]
    batch_size = batch_hv_dim  # batch_hv_dim = B * HV

    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    num_v_tiles_small = cute.ceil_div(v_dim, TILE_V_SMALL_NT)
    smem_layout_small = cute.make_layout(
        (TILE_K_NT, TILE_V_SMALL_NT, NUM_STAGES_NT),
        stride=(TILE_V_SMALL_PADDED_NT, 1, TILE_K_NT * TILE_V_SMALL_PADDED_NT),
    )
    thread_layout_small = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_small = cute.make_layout((1, 4))
    tiled_copy_load_small = cute.make_tiled_copy_tv(
        copy_atom, thread_layout_small, val_layout_small
    )
    smem_bytes_small = (
        4 * TILE_K_NT * TILE_V_SMALL_PADDED_NT * NUM_STAGES_NT
        + 4 * TILE_V_SMALL_NT
        + 4 * TILE_K_NT * 2
        + 64
    )

    gdn_decode_kernel_small_batch_nontranspose(
        tiled_copy_load_small,
        h0_source,
        smem_layout_small,
        num_v_tiles_small,
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        o,
        h0_indices,
        softplus_beta,
        softplus_threshold,
        scale,
        H,
        HV,
        use_qk_l2norm,
    ).launch(
        grid=(batch_size * NUM_BLOCKS_PER_STATE_SMALL_NT, 1, 1),
        block=[NUM_THREADS_NT, 1, 1],
        smem=smem_bytes_small,
        stream=stream,
    )


@cute.jit
def run_gdn_decode_kernel_big_batch_nontranspose(
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    a: cute.Tensor,
    b: cute.Tensor,
    A_log: cute.Tensor,
    dt_bias: cute.Tensor,
    h0_source: cute.Tensor,
    h0_indices: cute.Tensor,
    o: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    # h0_source is flattened to [B*HV, K, V] to ensure proper alignment for SIMT async copy
    batch_hv_dim, k_dim, v_dim = h0_source.layout.shape
    h0_indices.layout.shape[0]
    batch_size = batch_hv_dim  # batch_hv_dim = B * HV

    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    num_v_tiles = cute.ceil_div(v_dim, TILE_V_NT)
    base_smem_layout = cute.make_layout(
        (TILE_K_NT, TILE_V_NT, NUM_STAGES_NT),
        stride=(TILE_V_PADDED_NT, 1, TILE_K_NT * TILE_V_PADDED_NT),
    )
    thread_layout = cute.make_layout((32, 8), stride=(8, 1))
    val_layout = cute.make_layout((1, 4))
    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)
    smem_bytes = (
        4 * TILE_K_NT * TILE_V_PADDED_NT * NUM_STAGES_NT
        + 4 * TILE_V_NT
        + 4 * TILE_K_NT * 2
        + 64
    )

    gdn_decode_kernel_big_batch_nontranspose(
        tiled_copy_load,
        h0_source,
        base_smem_layout,
        num_v_tiles,
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        o,
        h0_indices,
        softplus_beta,
        softplus_threshold,
        scale,
        H,
        HV,
        use_qk_l2norm,
    ).launch(
        grid=(batch_size, 1, 1),
        block=[NUM_THREADS_LARGE_NT, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


# ============================================================================
# Compilation cache and public entry point
# ============================================================================


@functools.cache
def _get_compiled_decode_kernel_nontranspose(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    scale: float,
    use_qk_l2norm: bool,
):
    """Cache compiled kernel for given configuration (nontranspose version)."""
    # This will be populated on first call
    return {}


def run_nontranspose_decode(
    h0_source: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    scale: float,
    use_qk_l2norm: bool,
):
    """Compile and execute the nontranspose decode kernel.

    Args:
        h0_source: State tensor reshaped to [B*HV, K, V].
        A_log, a, dt_bias, q, k, v, b: Input tensors.
        output: Pre-allocated output tensor [B, T, HV, V].
        B, T, H, HV, K, V: Dimension sizes.
        scale: Query scale factor.
        use_qk_l2norm: Whether to apply L2 normalization.
    """
    # Compile kernel with TVM FFI (cached)
    cache_key = (B, T, H, HV, K, V, q.dtype, scale, use_qk_l2norm)
    cache = _get_compiled_decode_kernel_nontranspose(*cache_key)

    # Get or create h0_indices and cu_seqlens (cached per config)
    if "h0_indices" not in cache or cache["h0_indices"].device != q.device:
        cache["h0_indices"] = torch.arange(B, dtype=torch.int32, device=q.device)
        cache["cu_seqlens"] = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
    h0_indices = cache["h0_indices"]
    cu_seqlens = cache["cu_seqlens"]

    if "compiled" not in cache:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        # Choose kernel based on batch size
        use_small_batch = B < SMALL_BATCH_THRESHOLD_NT

        if use_small_batch:
            run_func = run_gdn_decode_kernel_small_batch_nontranspose
        else:
            run_func = run_gdn_decode_kernel_big_batch_nontranspose

        # Convert tensors to CuTe format for compilation only
        h0_source_tensor = from_dlpack(h0_source, assumed_align=16)
        A_log_tensor = from_dlpack(A_log, assumed_align=16)
        a_tensor = from_dlpack(a, assumed_align=16)
        dt_bias_tensor = from_dlpack(dt_bias, assumed_align=16)
        q_tensor = from_dlpack(q, assumed_align=16)
        k_tensor = from_dlpack(k, assumed_align=16)
        v_tensor = from_dlpack(v, assumed_align=16)
        b_tensor = from_dlpack(b, assumed_align=16)
        o_tensor = from_dlpack(output, assumed_align=16)
        h0_indices_tensor = from_dlpack(h0_indices, assumed_align=16)
        cu_seqlens_tensor = from_dlpack(cu_seqlens, assumed_align=16)

        # Use TVM FFI to reduce runtime overhead
        compiled = cute.compile(
            run_func,
            cu_seqlens_tensor,
            q_tensor,
            k_tensor,
            v_tensor,
            a_tensor,
            b_tensor,
            A_log_tensor,
            dt_bias_tensor,
            h0_source_tensor,
            h0_indices_tensor,
            o_tensor,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            scale=scale,
            B=B,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            use_initial_state=True,
            use_qk_l2norm=use_qk_l2norm,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        cache["compiled"] = compiled
    else:
        compiled = cache["compiled"]

    # Run kernel directly with PyTorch tensors (no from_dlpack needed)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        cu_seqlens,
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        h0_source,
        h0_indices,
        output,
        stream,
    )
