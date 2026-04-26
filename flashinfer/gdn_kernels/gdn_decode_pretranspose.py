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

Gated Delta Rule Decode Kernel - Pretranspose (V-major / K-last) Layout
========================================================================

CuTe-DSL implementation of GDN decode with state layout [B*HV, V, K].
Uses TMA pipelining and vectorized memory access for T=1 decode.
"""

import functools
from typing import Optional

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

# ============================================================================
# Constants for PRETRANSPOSE version ([B*HV, V, K])
# ============================================================================
TILE_V = 8
TILE_K = 128
NUM_STAGES = 2
NUM_THREADS = 128  # 4 warps
NUM_BLOCKS_PER_STATE = 8


@cute.kernel
def gdn_decode_kernel_small_batch_pretranspose(
    tiled_copy_load: cute.TiledCopy,
    h0_source: cute.Tensor,
    smem_layout_staged: cute.Layout,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    A_log: cute.Tensor,  # [HV]
    a: cute.Tensor,  # [B, T, HV]
    dt_bias: cute.Tensor,  # [HV]
    q: cute.Tensor,  # [B, T, H, K]
    k: cute.Tensor,  # [B, T, H, K]
    v: cute.Tensor,  # [B, T, HV, V]
    b: cute.Tensor,  # [B, T, HV]
    o: cute.Tensor,  # [B, T, HV, V] - output
    h0_indices: cute.Tensor,  # [B] - initial state indices (read)
    h0_out_indices: cute.Tensor,  # [B] - output state indices (write)
    cu_seqlens: cute.Tensor,  # [B+1] - cumulative sequence lengths (for varlen)
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
    use_pool_indexing: cutlass.Constexpr[bool] = False,
):
    """Each block uses pipeline to load one batch and vectorized writeback"""

    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    block_idx, _, _ = cute.arch.block_idx()
    batch_idx = block_idx // NUM_BLOCKS_PER_STATE
    batch_inner = block_idx % NUM_BLOCKS_PER_STATE
    num_v_tiles_per_block = num_v_tiles // NUM_BLOCKS_PER_STATE
    i_n = batch_idx // HV
    i_hv = batch_idx % HV
    i_h = i_hv // (HV // H)
    i_t = 0

    smem = cutlass.utils.SmemAllocator()

    # ===================================================================
    # Allocate shared memory (using passed-in layout)
    # ===================================================================
    sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)

    # Allocate shared memory for output (size V) - use BFloat16 to match SGLang
    sOutput = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((V,)), 16)

    # Allocate shared memory for v values (size K, to reduce register usage)
    sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout((V,)), 16)

    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    # r_v moved to shared memory (sV)
    r_h = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    # BF16 register tensors for vectorized q, k, v loading
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_v_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )

    # Compute k_start for contiguous access pattern
    k_start = lane_id * vec_size

    # Read gate values from GMEM EARLY (before barrier, latency hidden during sync)
    r_A_log = cutlass.Float32(A_log[i_hv])
    r_a = cutlass.Float32(a[i_n, i_t, i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])
    r_b = cutlass.Float32(b[i_n, i_t, i_hv])

    cute.arch.barrier()

    # Compute state index: use pool indexing if enabled.
    if cutlass.const_expr(use_pool_indexing):
        pool_idx = h0_indices[i_n]
        out_pool_idx = h0_out_indices[i_n]
        # Redirect negative write indices to null buffer (slot 0)
        if out_pool_idx < 0:
            out_pool_idx = cutlass.Int32(0)
    else:
        pool_idx = 0
        out_pool_idx = 0

    if pool_idx >= 0:
        # Get current batch
        if cutlass.const_expr(use_pool_indexing):
            # h0_source layout: [pool_size, HV, V, K] (supports non-contiguous page stride)
            gSrc_batch = h0_source[(pool_idx, i_hv, None, None)]  # (V, K)
            gDst = cute.local_tile(
                h0_source, (1, 1, TILE_V, TILE_K), (out_pool_idx, i_hv, None, 0)
            )
        else:
            # h0_source layout: [B*HV, V, K]
            state_idx = batch_idx
            gSrc_batch = h0_source[(state_idx, None, None)]  # (V, K)
            gDst = cute.local_tile(h0_source, (1, TILE_V, TILE_K), (state_idx, None, 0))
        # Tile along V dimension
        gSrc = cute.local_tile(
            gSrc_batch, (TILE_V, TILE_K), (None, 0)
        )  # (TILE_V, TILE_K, num_v_tiles)

        # Partition for load
        thr_copy_load = tiled_copy_load.get_slice(tidx)

        # ===================================================================
        # Prefetch: All threads participate in cp.async load
        # ===================================================================
        start_v_tiles = batch_inner * num_v_tiles_per_block
        prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles_per_block)
        for v_tiles in range(start_v_tiles, start_v_tiles + prefetch_count):
            stage = (v_tiles - start_v_tiles) % NUM_STAGES

            gSrc_tile = gSrc[(None, None, v_tiles)]
            sData_stage = sData[(None, None, stage)]

            thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
            thr_sData = thr_copy_load.partition_D(sData_stage)

            cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
            cute.arch.cp_async_commit_group()

        # Load q, k into BF16 registers using autovec_copy (contiguous pattern)
        q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
        k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
        cute.autovec_copy(q_tile, r_q_bf16)
        cute.autovec_copy(k_tile, r_k_bf16)

        # Convert BF16 to FP32
        for i in cutlass.range_constexpr(vec_size):
            r_q[i] = cutlass.Float32(r_q_bf16[i])
            r_k[i] = cutlass.Float32(r_k_bf16[i])

        # Load v into BF16 registers using autovec_copy, convert to FP32, store to sV
        v_tile = cute.local_tile(v, (1, 1, 1, vec_size), (i_n, i_t, i_hv, lane_id))
        cute.autovec_copy(v_tile, r_v_bf16)
        for i in cutlass.range_constexpr(vec_size):
            sV[k_start + i] = cutlass.Float32(r_v_bf16[i])

        cute.arch.barrier()  # Ensure all threads finish writing to sV

        # ===================================================================
        # Compute g and beta (scalar values)
        # ===================================================================
        r_g = 0.0
        r_beta = 0.0
        if lane_id == 0:
            x = r_a + r_dt_bias
            beta_x = softplus_beta * x
            softplus_x = 0.0

            if beta_x <= softplus_threshold:
                # softplus(x) = (1/beta) * log(1 + exp(beta*x))
                # Compute in Float32
                exp_beta_x = cute.exp(beta_x, fastmath=True)
                log_input = cutlass.Float32(1.0 + exp_beta_x)
                log_result = cutlass.Float32(cute.log(log_input, fastmath=True))
                softplus_x = cutlass.Float32(
                    (cutlass.Float32(1.0) / softplus_beta) * log_result
                )
            else:
                softplus_x = x

            # Compute g = exp(A_log) * softplus_x
            r_g_value = -cute.exp(r_A_log, fastmath=True) * softplus_x

            # Compute beta = 1 / (1 + exp(-b))
            r_beta = 1.0 / (1.0 + cute.exp(-r_b, fastmath=True))

            # Store to scalar (Float32)
            r_g = cute.exp(r_g_value, fastmath=True)

        r_g = cute.arch.shuffle_sync(r_g, 0)
        r_beta = cute.arch.shuffle_sync(r_beta, 0)

        if use_qk_l2norm:
            # Compute L2 norm of q and k
            sum_q = 0.0
            sum_k = 0.0
            for i in cutlass.range_constexpr(vec_size):
                sum_q += r_q[i] * r_q[i]
                sum_k += r_k[i] * r_k[i]
            # Warp-level reduction using butterfly shuffle
            for offset in [16, 8, 4, 2, 1]:
                sum_q += cute.arch.shuffle_sync_bfly(
                    sum_q, offset=offset, mask=-1, mask_and_clamp=31
                )
                sum_k += cute.arch.shuffle_sync_bfly(
                    sum_k, offset=offset, mask=-1, mask_and_clamp=31
                )

            inv_norm_q = cute.rsqrt(sum_q + 1e-6, fastmath=True)
            inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)
            for i in cutlass.range_constexpr(vec_size):
                r_q[i] = r_q[i] * inv_norm_q
                r_k[i] = r_k[i] * inv_norm_k

        # Apply scaling in Float32
        for i in cutlass.range_constexpr(vec_size):
            r_q[i] = r_q[i] * scale

        # ===================================================================
        # Mainloop: All threads participate
        # ===================================================================
        end_v_tiles = start_v_tiles + num_v_tiles_per_block
        for v_tiles in range(start_v_tiles, end_v_tiles):
            stage = (v_tiles - start_v_tiles) % NUM_STAGES

            # Step 1: Wait for current stage to complete
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            # Step 2: Issue async load for next tile (after compute)
            next_v_tiles = v_tiles + prefetch_count
            if next_v_tiles < end_v_tiles:
                next_stage = (next_v_tiles - start_v_tiles) % NUM_STAGES

                gSrc_next = gSrc[(None, None, next_v_tiles)]
                sData_next = sData[(None, None, next_stage)]

                thr_gSrc = thr_copy_load.partition_S(gSrc_next)
                thr_sData = thr_copy_load.partition_D(sData_next)

                cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                cute.arch.cp_async_commit_group()

            # Step 3: Compute using data from current stage (contiguous access pattern)
            for row in cutlass.range_constexpr(0, TILE_V, 4):
                row_offset = tidx // 32
                sum_hk = 0.0

                # Load h from sData using 3D local_tile + autovec_copy (contiguous in K)
                sData_tile = cute.local_tile(
                    sData, (1, vec_size, 1), (row + row_offset, lane_id, stage)
                )
                cute.autovec_copy(sData_tile, r_h)

                for i in cutlass.range_constexpr(vec_size):
                    r_h[i] = r_h[i] * r_g
                    sum_hk += r_h[i] * r_k[i]

                for offset in [16, 8, 4, 2, 1]:
                    sum_hk += cute.arch.shuffle_sync_bfly(
                        sum_hk, offset=offset, mask=-1, mask_and_clamp=31
                    )

                v_new = sV[v_tiles * TILE_V + row + row_offset] - sum_hk
                v_new = v_new * r_beta

                sum_hq = 0.0
                for i in cutlass.range_constexpr(vec_size):
                    r_h[i] += r_k[i] * v_new
                    sum_hq += r_h[i] * r_q[i]

                # Write h to gDst using 4D local_tile + autovec_copy (contiguous in K)
                if cutlass.const_expr(use_pool_indexing):
                    gDst_tile = cute.local_tile(
                        gDst,
                        (1, 1, 1, vec_size, 1),
                        (0, 0, row + row_offset, lane_id, v_tiles),
                    )
                else:
                    gDst_tile = cute.local_tile(
                        gDst,
                        (1, 1, vec_size, 1),
                        (0, row + row_offset, lane_id, v_tiles),
                    )
                cute.autovec_copy(r_h, gDst_tile)

                for offset in [16, 8, 4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(
                        sum_hq, offset=offset, mask=-1, mask_and_clamp=31
                    )

                o_idx = v_tiles * TILE_V + row + row_offset
                if lane_id == 0 and o_idx < V:
                    sOutput[o_idx] = cutlass.BFloat16(sum_hq)

        # ===================================================================
        # Final writeback: Copy output from shared memory to global memory
        # All threads write (V=128, NUM_THREADS=128)
        # ===================================================================
        cute.arch.barrier()  # Ensure all writes to sOutput are complete
        if tidx >= start_v_tiles * TILE_V and tidx < end_v_tiles * TILE_V:
            o[(i_n, i_t, i_hv, tidx)] = sOutput[tidx]

    else:
        start_v_tiles = batch_inner * num_v_tiles_per_block
        end_v_tiles = start_v_tiles + num_v_tiles_per_block
        if tidx >= start_v_tiles * TILE_V and tidx < end_v_tiles * TILE_V:
            o[(i_n, i_t, i_hv, tidx)] = cutlass.BFloat16(0.0)


@cute.kernel
def gdn_decode_kernel_big_batch_pretranspose(
    tiled_copy_load: cute.TiledCopy,
    h0_source: cute.Tensor,
    smem_layout_staged: cute.Layout,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    A_log: cute.Tensor,  # [HV]
    a: cute.Tensor,  # [B, T, HV]
    dt_bias: cute.Tensor,  # [HV]
    q: cute.Tensor,  # [B, T, H, K]
    k: cute.Tensor,  # [B, T, H, K]
    v: cute.Tensor,  # [B, T, HV, V]
    b: cute.Tensor,  # [B, T, HV]
    o: cute.Tensor,  # [B, T, HV, V] - output
    h0_indices: cute.Tensor,  # [B] - initial state indices (read)
    h0_out_indices: cute.Tensor,  # [B] - output state indices (write)
    cu_seqlens: cute.Tensor,  # [B+1] - cumulative sequence lengths (for varlen)
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
    use_pool_indexing: cutlass.Constexpr[bool] = False,
):
    """Each block uses pipeline to load one batch and vectorized writeback"""

    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    batch_idx, _, _ = cute.arch.block_idx()
    i_n = batch_idx // HV
    i_hv = batch_idx % HV
    i_h = i_hv // (HV // H)
    i_t = 0

    r_A_log = cutlass.Float32(A_log[i_hv])
    r_a = cutlass.Float32(a[i_n, i_t, i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])
    r_b = cutlass.Float32(b[i_n, i_t, i_hv])

    smem = cutlass.utils.SmemAllocator()

    # ===================================================================
    # Allocate shared memory (using passed-in layout)
    # ===================================================================
    sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)

    # Allocate shared memory for output (size V) - use BFloat16 to match SGLang
    sOutput = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((V,)), 16)

    # Allocate shared memory for v values (size K, to reduce register usage)
    sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout((V,)), 16)

    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    # r_v moved to shared memory (sV)
    r_h = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    # BF16 register tensors for vectorized q, k, v loading
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_v_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )

    # Compute k_start for contiguous access pattern
    k_start = lane_id * vec_size

    cute.arch.barrier()

    # Compute state index: use pool indexing if enabled.
    if cutlass.const_expr(use_pool_indexing):
        pool_idx = h0_indices[i_n]
        out_pool_idx = h0_out_indices[i_n]
        # Redirect negative write indices to null buffer (slot 0)
        if out_pool_idx < 0:
            out_pool_idx = cutlass.Int32(0)
    else:
        pool_idx = 0
        out_pool_idx = 0

    if pool_idx >= 0:
        # Get current state slice.
        if cutlass.const_expr(use_pool_indexing):
            # h0_source layout: [pool_size, HV, V, K] (supports non-contiguous page stride)
            gSrc_batch = h0_source[(pool_idx, i_hv, None, None)]  # (V, K)
            gDst = cute.local_tile(
                h0_source, (1, 1, TILE_V, TILE_K), (out_pool_idx, i_hv, None, 0)
            )
        else:
            # h0_source layout: [B*HV, V, K]
            state_idx = batch_idx
            gSrc_batch = h0_source[(state_idx, None, None)]  # (V, K)
            gDst = cute.local_tile(h0_source, (1, TILE_V, TILE_K), (state_idx, None, 0))
        # Tile along V dimension
        gSrc = cute.local_tile(
            gSrc_batch, (TILE_V, TILE_K), (None, 0)
        )  # (TILE_V, TILE_K, num_v_tiles)

        # Partition for load
        thr_copy_load = tiled_copy_load.get_slice(tidx)

        # ===================================================================
        # Prefetch: All threads participate in cp.async load
        # ===================================================================
        prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles)
        for v_tiles in range(prefetch_count):
            stage = v_tiles % NUM_STAGES

            gSrc_tile = gSrc[(None, None, v_tiles)]
            sData_stage = sData[(None, None, stage)]

            thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
            thr_sData = thr_copy_load.partition_D(sData_stage)

            cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
            cute.arch.cp_async_commit_group()

        # Load q, k into BF16 registers using autovec_copy (contiguous pattern)
        q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
        k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
        cute.autovec_copy(q_tile, r_q_bf16)
        cute.autovec_copy(k_tile, r_k_bf16)

        # Convert BF16 to FP32
        for i in cutlass.range_constexpr(vec_size):
            r_q[i] = cutlass.Float32(r_q_bf16[i])
            r_k[i] = cutlass.Float32(r_k_bf16[i])

        # Load v into BF16 registers using autovec_copy, convert to FP32, store to sV
        v_tile = cute.local_tile(v, (1, 1, 1, vec_size), (i_n, i_t, i_hv, lane_id))
        cute.autovec_copy(v_tile, r_v_bf16)
        for i in cutlass.range_constexpr(vec_size):
            sV[k_start + i] = cutlass.Float32(r_v_bf16[i])

        cute.arch.barrier()  # Ensure all threads finish writing to sV

        # ===================================================================
        # Compute g and beta (scalar values)
        # ===================================================================
        r_g = 0.0
        r_beta = 0.0
        if lane_id == 0:
            x = r_a + r_dt_bias
            beta_x = softplus_beta * x
            softplus_x = 0.0

            if beta_x <= softplus_threshold:
                # softplus(x) = (1/beta) * log(1 + exp(beta*x))
                # Compute in Float32
                exp_beta_x = cute.exp(beta_x, fastmath=True)
                log_input = cutlass.Float32(1.0 + exp_beta_x)
                log_result = cutlass.Float32(cute.log(log_input, fastmath=True))
                softplus_x = cutlass.Float32(
                    (cutlass.Float32(1.0) / softplus_beta) * log_result
                )
            else:
                softplus_x = x

            # Compute g = exp(A_log) * softplus_x
            r_g_value = -cute.exp(r_A_log, fastmath=True) * softplus_x

            # Compute beta = 1 / (1 + exp(-b))
            r_beta = 1.0 / (1.0 + cute.exp(-r_b, fastmath=True))

            # Store to scalar (Float32)
            r_g = cute.exp(r_g_value, fastmath=True)

        r_g = cute.arch.shuffle_sync(r_g, 0)
        r_beta = cute.arch.shuffle_sync(r_beta, 0)

        if use_qk_l2norm:
            # Compute L2 norm of q and k
            sum_q = 0.0
            sum_k = 0.0
            for i in cutlass.range_constexpr(vec_size):
                sum_q += r_q[i] * r_q[i]
                sum_k += r_k[i] * r_k[i]
            # Warp-level reduction using butterfly shuffle
            for offset in [16, 8, 4, 2, 1]:
                sum_q += cute.arch.shuffle_sync_bfly(
                    sum_q, offset=offset, mask=-1, mask_and_clamp=31
                )
                sum_k += cute.arch.shuffle_sync_bfly(
                    sum_k, offset=offset, mask=-1, mask_and_clamp=31
                )

            inv_norm_q = cute.rsqrt(sum_q + 1e-6, fastmath=True)
            inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)
            for i in cutlass.range_constexpr(vec_size):
                r_q[i] = r_q[i] * inv_norm_q
                r_k[i] = r_k[i] * inv_norm_k

        # Apply scaling in Float32
        for i in cutlass.range_constexpr(vec_size):
            r_q[i] = r_q[i] * scale

        # ===================================================================
        # Mainloop: All threads participate
        # ===================================================================
        for v_tiles in range(num_v_tiles):
            stage = v_tiles % NUM_STAGES

            # Step 1: Wait for current stage to complete
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            # Step 2: Issue async load for next tile (after compute)
            next_v_tiles = v_tiles + prefetch_count
            if next_v_tiles < num_v_tiles:
                next_stage = next_v_tiles % NUM_STAGES

                gSrc_next = gSrc[(None, None, next_v_tiles)]
                sData_next = sData[(None, None, next_stage)]

                thr_gSrc = thr_copy_load.partition_S(gSrc_next)
                thr_sData = thr_copy_load.partition_D(sData_next)

                cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                cute.arch.cp_async_commit_group()

            # Step 3: Compute using data from current stage (contiguous access pattern)
            for row in cutlass.range_constexpr(0, TILE_V, 4):
                row_offset = tidx // 32
                sum_hk = 0.0

                # Load h from sData using 3D local_tile + autovec_copy (contiguous in K)
                sData_tile = cute.local_tile(
                    sData, (1, vec_size, 1), (row + row_offset, lane_id, stage)
                )
                cute.autovec_copy(sData_tile, r_h)

                for i in cutlass.range_constexpr(vec_size):
                    r_h[i] = r_h[i] * r_g
                    sum_hk += r_h[i] * r_k[i]

                for offset in [16, 8, 4, 2, 1]:
                    sum_hk += cute.arch.shuffle_sync_bfly(
                        sum_hk, offset=offset, mask=-1, mask_and_clamp=31
                    )

                v_new = sV[v_tiles * TILE_V + row + row_offset] - sum_hk
                v_new = v_new * r_beta

                sum_hq = 0.0
                for i in cutlass.range_constexpr(vec_size):
                    r_h[i] += r_k[i] * v_new
                    sum_hq += r_h[i] * r_q[i]

                # Write h back to state.
                if cutlass.const_expr(use_pool_indexing):
                    gDst_tile = cute.local_tile(
                        gDst,
                        (1, 1, 1, vec_size, 1),
                        (0, 0, row + row_offset, lane_id, v_tiles),
                    )
                else:
                    gDst_tile = cute.local_tile(
                        gDst,
                        (1, 1, vec_size, 1),
                        (0, row + row_offset, lane_id, v_tiles),
                    )
                cute.autovec_copy(r_h, gDst_tile)

                for offset in [16, 8, 4, 2, 1]:
                    sum_hq += cute.arch.shuffle_sync_bfly(
                        sum_hq, offset=offset, mask=-1, mask_and_clamp=31
                    )

                o_idx = v_tiles * TILE_V + row + row_offset
                if lane_id == 0 and o_idx < V:
                    sOutput[o_idx] = cutlass.BFloat16(sum_hq)

        # ===================================================================
        # Final writeback: Copy output from shared memory to global memory
        # All threads write (V=128, NUM_THREADS=128)
        # ===================================================================
        cute.arch.barrier()  # Ensure all writes to sOutput are complete

        if tidx < V:
            o[(i_n, i_t, i_hv, tidx)] = sOutput[tidx]

    else:
        if tidx < V:
            o[(i_n, i_t, i_hv, tidx)] = cutlass.BFloat16(0.0)


@cute.jit
def run_gdn_decode_kernel_small_batch_pretranspose(
    h0_source: cute.Tensor,  # [B*HV, K, V]
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    h0_out_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
    use_pool_indexing: cutlass.Constexpr[bool] = False,
    stream: cuda.CUstream = None,
):
    """Launch original pipelined kernel for small batch pretranspose."""
    # h0_source:
    # - non-pool: (B*HV, V, K)
    # - pool: (pool_size, HV, V, K)
    if cutlass.const_expr(use_pool_indexing):
        v_dim = h0_source.layout.shape[2]
        k_dim = h0_source.layout.shape[3]
    else:
        v_dim = h0_source.layout.shape[1]
        k_dim = h0_source.layout.shape[2]
    # Grid size: use B*HV (actual batch) not h0_source.shape[0] (which may be pool_size*HV)
    grid_batch = B * HV

    # Create cp.async copy with cache-global mode (bypass L1)
    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,  # 4 elements per copy
    )

    # Thread layout: 4 rows × 32 threads/row = 128 threads
    thread_layout = cute.make_layout(
        (4, 32),  # 4 rows, 32 threads/row
        stride=(32, 1),
    )
    val_layout = cute.make_layout((1, 4))  # Each thread handles 4 elements

    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)

    num_v_tiles = cute.ceil_div(v_dim, TILE_V)
    v_dim * k_dim * 4 / 1024 / 1024

    vec_size = (
        TILE_K // 32
    )  # Each thread in a warp processes this many elements (always 4 for TILE_K=128)

    # Create SMEM layout
    smem_layout_staged = cute.make_layout(
        (TILE_V, TILE_K, NUM_STAGES), stride=(TILE_K, 1, TILE_V * TILE_K)
    )

    # sData: TILE_V * TILE_K * NUM_STAGES * 4 bytes (Float32)
    # sV: K * 4 bytes (Float32)
    # sOutput: V * 2 bytes (BFloat16)
    smem_bytes = 4 * TILE_V * TILE_K * NUM_STAGES + 4 * k_dim + 2 * v_dim + 32

    gdn_decode_kernel_small_batch_pretranspose(
        tiled_copy_load,
        h0_source,
        smem_layout_staged,
        vec_size,
        num_v_tiles,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        h0_indices,
        h0_out_indices,
        cu_seqlens,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        B,
        T,
        H,
        K,
        V,
        use_initial_state,
        use_qk_l2norm,
        is_varlen,
        use_pool_indexing,
    ).launch(
        grid=(grid_batch * NUM_BLOCKS_PER_STATE, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


@cute.jit
def run_gdn_decode_kernel_big_batch_pretranspose(
    h0_source: cute.Tensor,  # [B*HV, K, V]
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    h0_out_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
    use_pool_indexing: cutlass.Constexpr[bool] = False,
    stream: cuda.CUstream = None,
):
    if cutlass.const_expr(use_pool_indexing):
        v_dim = h0_source.layout.shape[2]
        k_dim = h0_source.layout.shape[3]
    else:
        v_dim = h0_source.layout.shape[1]
        k_dim = h0_source.layout.shape[2]
    grid_batch = B * HV

    # Create cp.async copy with cache-global mode (bypass L1)
    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,  # 4 elements per copy
    )

    # Thread layout: 4 rows × 32 threads/row = 128 threads
    thread_layout = cute.make_layout(
        (4, 32),  # 4 rows, 32 threads/row
        stride=(32, 1),
    )
    val_layout = cute.make_layout((1, 4))  # Each thread handles 4 elements

    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)

    num_v_tiles = cute.ceil_div(v_dim, TILE_V)
    v_dim * k_dim * 4 / 1024 / 1024

    vec_size = (
        TILE_K // 32
    )  # Each thread in a warp processes this many elements (always 4 for TILE_K=128)

    # Create SMEM layout
    smem_layout_staged = cute.make_layout(
        (TILE_V, TILE_K, NUM_STAGES), stride=(TILE_K, 1, TILE_V * TILE_K)
    )

    # sData: TILE_V * TILE_K * NUM_STAGES * 4 bytes (Float32)
    # sV: K * 4 bytes (Float32)
    # sOutput: V * 2 bytes (BFloat16)
    smem_bytes = 4 * TILE_V * TILE_K * NUM_STAGES + 4 * k_dim + 2 * v_dim + 32

    gdn_decode_kernel_big_batch_pretranspose(
        tiled_copy_load,
        h0_source,
        smem_layout_staged,
        vec_size,
        num_v_tiles,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        h0_indices,
        h0_out_indices,
        cu_seqlens,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        B,
        T,
        H,
        K,
        V,
        use_initial_state,
        use_qk_l2norm,
        is_varlen,
        use_pool_indexing,
    ).launch(
        grid=(grid_batch, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


# ============================================================================
# Compilation cache and public entry point
# ============================================================================


@functools.cache
def _get_compiled_decode_kernel(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    scale: float,
    use_qk_l2norm: bool,
    use_pool_indexing: bool = False,
    pool_size: int = 0,
    stride0: int = 0,
    stride1: int = 0,
    stride2: int = 0,
    stride3: int = 0,
):
    """Cache compiled kernel for given configuration (pretranspose version)."""
    # This will be populated on first call
    return {}


def run_pretranspose_decode(
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
    use_pool_indexing: bool = False,
    initial_state_indices: Optional[torch.Tensor] = None,
    output_state_indices: Optional[torch.Tensor] = None,
):
    """Compile and execute the pretranspose decode kernel.

    Args:
        h0_source: State tensor of shape [B*HV, V, K], or [pool_size, HV, V, K]
                   when use_pool_indexing=True.
        A_log, a, dt_bias, q, k, v, b: Input tensors.
        output: Pre-allocated output tensor [B, T, HV, V].
        B, T, H, HV, K, V: Dimension sizes.
        scale: Query scale factor.
        use_qk_l2norm: Whether to apply L2 normalization.
        use_pool_indexing: Whether to use pool-based indirect state indexing.
        initial_state_indices: Int32 indices into state pool, shape [B].
            Negative values indicate padding (kernel writes zeros).
        output_state_indices: Optional int32 indices for write destination, shape [B].
            When None, writes go to the same slot as initial_state_indices.
    """
    # Compile kernel with TVM FFI (cached)
    if use_pool_indexing:
        pool_size = int(h0_source.shape[0])
        stride0, stride1, stride2, stride3 = tuple(int(x) for x in h0_source.stride())
    else:
        pool_size = stride0 = stride1 = stride2 = stride3 = 0
    cache_key = (
        B,
        T,
        H,
        HV,
        K,
        V,
        q.dtype,
        scale,
        use_qk_l2norm,
        use_pool_indexing,
        pool_size,
        stride0,
        stride1,
        stride2,
        stride3,
    )
    cache = _get_compiled_decode_kernel(*cache_key)

    # Get or create h0_indices and cu_seqlens (cached per config)
    if "h0_indices" not in cache or cache["h0_indices"].device != q.device:
        cache["h0_indices"] = torch.zeros(B, dtype=torch.int32, device=q.device)
        cache["cu_seqlens"] = torch.zeros(B + 1, dtype=torch.int32, device=q.device)

    if use_pool_indexing and initial_state_indices is not None:
        h0_indices = initial_state_indices.to(torch.int32)
    else:
        h0_indices = cache["h0_indices"]
    # Resolve output indices: default to same as read indices
    if use_pool_indexing and output_state_indices is not None:
        h0_out_indices = output_state_indices.to(torch.int32)
    else:
        h0_out_indices = h0_indices
    cu_seqlens = cache["cu_seqlens"]

    if "compiled" not in cache:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        # Convert tensors to CuTe format for compilation only
        # Use the actual tensor view so strided pool layouts are preserved.
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
        h0_out_indices_tensor = from_dlpack(h0_out_indices, assumed_align=16)
        cu_seqlens_tensor = from_dlpack(cu_seqlens, assumed_align=16)

        # Always use 8-CTA architecture (benchmarks show it's better for all batch sizes)
        run_func = run_gdn_decode_kernel_small_batch_pretranspose

        # Use TVM FFI to reduce runtime overhead
        compiled = cute.compile(
            run_func,
            h0_source_tensor,
            A_log_tensor,
            a_tensor,
            dt_bias_tensor,
            q_tensor,
            k_tensor,
            v_tensor,
            b_tensor,
            o_tensor,
            h0_indices_tensor,
            h0_out_indices_tensor,
            cu_seqlens_tensor,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            scale=scale,
            HV=HV,
            B=B,
            T=T,
            H=H,
            K=K,
            V=V,
            use_initial_state=True,
            use_qk_l2norm=use_qk_l2norm,
            use_pool_indexing=use_pool_indexing,
            is_varlen=False,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        cache["compiled"] = compiled
    else:
        compiled = cache["compiled"]

    # Run kernel directly with PyTorch tensors (no from_dlpack needed)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    cache["compiled"](
        h0_source,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        output,
        h0_indices,
        h0_out_indices,
        cu_seqlens,
        stream,
    )
