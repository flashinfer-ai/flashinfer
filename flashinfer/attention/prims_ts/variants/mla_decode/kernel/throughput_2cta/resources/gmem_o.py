# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Output epilogue and GMEM O stores of the 2CTA MLA kernel."""

import cutlass.cute as cute
import cutlass
from cutlass import Float32, Int32, Int64
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    consumer_work,
    producer_work,
)
from dataclasses import dataclass, field
from ......device.packing import pack_float4_to_fp8_e4m3
from ...device.constants import (
    BF16_OUTPUT_VECTOR_ELEMENTS,
    EPILOGUE_COLUMN_GROUP_SHIFT,
    EPILOGUE_ROW_MASK,
    EPILOGUE_THREAD_TILE_MASK,
    EPILOGUE_THREAD_TILE_THREADS,
    FP8_OUTPUT_VECTOR_ELEMENTS,
    LN_2,
    LOG2_E,
    PACKED_FP8_OUTPUT_REGS,
    TCGEN05_32B_REGS_PER_LOAD,
    TCGEN05_32B_SHAPE,
    WARP_LANE_SHIFT,
)
from ...device.math import final_lse, mul_packed_f32x2, output_dtype
from ......device.packing import pack_float2_to_bf16
from ...device.nvfp4_kv import store_shared_u32x4
from ...device.ops import fp8_p_scale_rcp
from ...device.query import flat_query_row_state
from ..config import MlaDecodeConfig, OUTPUT_TMA_STAGE_BYTES
from .common import HighThroughputMlaResource
from typing import Any


# =====================================================================
# GmemOResource — Output to GMEM (no pipeline)
# =====================================================================


@dataclass(kw_only=True)
class GmemOResource(HighThroughputMlaResource):
    """Final output: direct stores or staged quantization with a TMA consumer."""

    # Static configuration
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    logical_num_heads_q: cutlass.Constexpr[int] = 128
    lse_in_natural_log: cutlass.Constexpr[bool] = False
    output_quant: cutlass.Constexpr = None

    # Linked resources
    tmem_corr_ref: Any = None  # Reference to TmemCorrResource for correction data
    tmem_o_ref: Any = None  # Reference to TmemOResource for tmem_base_addr

    # Runtime scalars
    logical_seq_len_q: Int32 = 1

    # Per-task state
    attention_sinks: Any = None
    cu_seqlens_q: Any = None
    lse: Any = None
    output: Any = None
    output_quant_args: Any = None
    output_scale: Any = None
    page_offsets: Any = None
    partial_lse: Any = None
    partial_output: Any = None
    scale_softmax_log2: Any = None
    smem_exchange: Any = None  # SMEM for row_sum exchange (as Int32 base addr)
    smem_o: Any = None
    split_kv: Any = None
    tma_o_desc: Any = None

    @producer_work
    @cute.jit
    def epilogue_store(self, stage_info: StageInfo) -> None:
        """Full epilogue: load O from TMEM, normalize by row_sum, store to GMEM.

        Steps:
        1. Exchange row_sum across warp pairs via SMEM
        2. Load O from TMEM
        3. Normalize by output_scale / row_sum
        4. Store to GMEM
        5. Compute and store LSE
        """
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        # Use work_tile's blk_coord (updated by persistent loop) instead of
        # self.blk_coord (captured at construction time) so that each work
        # tile writes to the correct output location.
        blk_coord = stage_info.work_tile.tile_idx

        row_sum = self.tmem_corr_ref.final_row_stats[0]
        row_max = self.tmem_corr_ref.final_row_stats[1]

        # Exchange row_sum between warp pairs (0,2) and (1,3) via SMEM
        # Use local thread index within the 4-warp correction group,
        # matching bare-metal: tidx % (num_compute_warps * threads_per_warp)
        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads
        smem_ex_ptr = cutlass.inttoptr(
            self.smem_exchange + local_tidx * 4,
            3,
            Float32,
        )
        smem_ex_ptr.store(row_sum)
        prims.barrier_cta_sync(
            cfg.epilogue_sync_bar_id, thread_count=cfg.epilogue_sync_threads
        )
        peer_idx = (local_tidx + 64) % num_compute_threads
        peer_ptr = cutlass.inttoptr(
            self.smem_exchange + peer_idx * 4,
            3,
            Float32,
        )
        row_sum = row_sum + peer_ptr.load()
        prims.barrier_cta_sync(
            cfg.epilogue_sync_bar_id, thread_count=cfg.epilogue_sync_threads
        )

        # TMEM address for O — use local tidx within 4-warp correction group
        tmem_base_addr = self.tmem_o_ref.tmem_base_addr
        tmem_warp_row_id = (
            tmem_base_addr + (local_tidx >> WARP_LANE_SHIFT) * TCGEN05_32B_REGS_PER_LOAD
        )
        tmem_raw_addr = (tmem_warp_row_id << 16) | cfg.tmem_o_offset

        t2r_shape = TCGEN05_32B_SHAPE
        num_tmem_loads = 4  # 4 x 32 = 128 elements per iter_n

        # Per-thread O indexing — use local tidx
        tile_h = cfg.mma_pv_tiler[0] // cfg.num_mma_ctas  # 64
        tile_d = cfg.mma_pv_tiler[1]  # 256
        tidx_g = local_tidx & EPILOGUE_THREAD_TILE_MASK
        g_i = tidx_g & EPILOGUE_ROW_MASK
        g_j = (tidx_g >> EPILOGUE_COLUMN_GROUP_SHIFT) * EPILOGUE_THREAD_TILE_THREADS

        # Public O/LSE remain in logical coordinates. Split-KV partials retain
        # physical flat-tile coordinates until final reduction.
        logical_num_heads_q = Int32(self.logical_num_heads_q)
        physical_tile_rows = Int32(cfg.mma_qk_tiler[0])
        D = cfg.latent_dim
        head_tile_idx = blk_coord[0]
        seq_q_idx = blk_coord[1]
        batch_idx = blk_coord[2]
        split_kv_idx = blk_coord[3]
        row_in_tile = head_tile_idx * tile_h + g_i
        (
            storage_flat_query_row,
            logical_head_idx,
            logical_q_idx,
            _,
            query_is_valid,
        ) = self.query_row_state(row_in_tile, seq_q_idx, batch_idx)

        if cutlass.const_expr(
            self.output_quant is not None and self.partial_output is None
        ):
            self.prefetch_output_rope(
                storage_flat_query_row,
                batch_idx,
                local_tidx,
                logical_head_idx,
                row_in_tile < physical_tile_rows and query_is_valid,
            )

        # Fully masked split rows can occur when physical tail rows or earlier
        # causal query rows have no visible K values in this split. Store zero
        # O and -inf LSE so split-KV reduction gives those rows zero weight.
        row_has_values = row_sum > Float32(0)
        norm_scale = self.normalization_scale(
            row_sum, row_max, local_tidx, blk_coord, stage_info
        )

        for iter_n in cutlass.range_constexpr(cfg.iterations_pv_n):
            # Load O from TMEM
            qk_acc_regs = cutlass.Array(Float32, 128, space=cutlass.AddressSpace.rmem)
            tmem_raw_addr_n = tmem_raw_addr + (
                iter_n * (cfg.mma_pv_tiler[1] // cfg.warps_in_n)
            )
            if cutlass.const_expr(
                self.output_quant is not None
                and self.partial_output is None
                and iter_n + 1 == cfg.iterations_pv_n
            ):
                qk_acc_regs = self.load_rotary_slice(
                    tmem_raw_addr_n,
                    norm_scale,
                    storage_flat_query_row,
                    batch_idx,
                    iter_n * tile_d + g_j,
                    row_in_tile < physical_tile_rows and query_is_valid,
                )
            else:
                for load_idx in cutlass.range_constexpr(num_tmem_loads):
                    curr_addr = tmem_raw_addr_n + load_idx * TCGEN05_32B_REGS_PER_LOAD
                    tmem_ptr = prims.make_tmem_ptr(curr_addr, Float32)
                    loaded = prims.tcgen05_ld(
                        t2r_shape, tmem_ptr, num=TCGEN05_32B_REGS_PER_LOAD
                    )
                    qk_acc_regs.store(loaded, load_idx * TCGEN05_32B_REGS_PER_LOAD)

                # Normalize: O = O * output_scale / row_sum
                for i in cutlass.range_constexpr(0, 128, 2):
                    scaled = mul_packed_f32x2(
                        (qk_acc_regs[i], qk_acc_regs[i + 1]),
                        (norm_scale, norm_scale),
                    )
                    qk_acc_regs[i] = scaled[0]
                    qk_acc_regs[i + 1] = scaled[1]

            # Store O to GMEM
            if row_in_tile < physical_tile_rows and query_is_valid:
                if cutlass.const_expr(self.partial_output is not None):
                    # Split-KV partial O uses BF16 workspace storage.  LSE and
                    # the eventual cross-split accumulation remain FP32.
                    S_q = (
                        cutlass.Int32(self.partial_output.shape[3])
                        if self.partial_output is not None
                        else Int32(1)
                    )
                    o_base_ptr = (
                        self.partial_output.iterator.raw_ptr()
                        + Int64(row_in_tile) * Int64(self.split_kv) * Int64(D)
                        + Int64(split_kv_idx) * Int64(D)
                        + Int64(seq_q_idx)
                        * Int64(self.split_kv)
                        * Int64(physical_tile_rows)
                        * Int64(D)
                        + Int64(batch_idx)
                        * Int64(physical_tile_rows)
                        * Int64(self.split_kv)
                        * Int64(S_q)
                        * Int64(D)
                    )
                    output_base = o_base_ptr + iter_n * tile_d + g_j
                    for load_idx in cutlass.range_constexpr(num_tmem_loads):
                        for j in cutlass.range_constexpr(4):
                            offset = (
                                load_idx * TCGEN05_32B_REGS_PER_LOAD
                                + j * BF16_OUTPUT_VECTOR_ELEMENTS
                            )
                            vec_f32 = qk_acc_regs.load(
                                offset, BF16_OUTPUT_VECTOR_ELEMENTS
                            )
                            vec_partial = vec_f32.to(cutlass.BFloat16)
                            (
                                output_base
                                + load_idx * TCGEN05_32B_REGS_PER_LOAD
                                + j * BF16_OUTPUT_VECTOR_ELEMENTS
                            ).nvvm_store_ext(
                                vec_partial,
                                evict="noallocate",
                            )
                elif cutlass.const_expr(self.output_quant is not None):
                    self.store_quantized_slice(
                        qk_acc_regs,
                        storage_flat_query_row,
                        batch_idx,
                        iter_n * tile_d + g_j,
                    )
                else:
                    # 16-bit output (split_kv == 1, direct output)
                    if cutlass.const_expr(self.cu_seqlens_q is not None):
                        o_base_ptr = self.output.iterator.raw_ptr() + Int64(
                            storage_flat_query_row
                        ) * Int64(D)
                    else:
                        S_q = (
                            cutlass.Int32(self.output.shape[2])
                            if self.output is not None
                            else Int32(1)
                        )
                        o_base_ptr = (
                            self.output.iterator.raw_ptr()
                            + Int64(logical_head_idx) * Int64(D)
                            + Int64(logical_q_idx)
                            * Int64(logical_num_heads_q)
                            * Int64(D)
                            + Int64(batch_idx)
                            * Int64(logical_num_heads_q)
                            * Int64(D)
                            * Int64(S_q)
                        )
                    output_base = o_base_ptr + iter_n * tile_d + g_j
                    for load_idx in cutlass.range_constexpr(num_tmem_loads):
                        for j in cutlass.range_constexpr(2):
                            offset = (
                                load_idx * TCGEN05_32B_REGS_PER_LOAD
                                + j * FP8_OUTPUT_VECTOR_ELEMENTS
                            )
                            vec_f32 = qk_acc_regs.load(
                                offset, FP8_OUTPUT_VECTOR_ELEMENTS
                            )
                            if cutlass.const_expr(cfg.use_fp8_output == 1):
                                packed_o = cutlass.Array(
                                    Int32,
                                    PACKED_FP8_OUTPUT_REGS,
                                    space=cutlass.AddressSpace.rmem,
                                )
                                for pack_idx in cutlass.range_constexpr(
                                    PACKED_FP8_OUTPUT_REGS
                                ):
                                    pack_offset = pack_idx * PACKED_FP8_OUTPUT_REGS
                                    packed_o[pack_idx] = pack_float4_to_fp8_e4m3(
                                        vec_f32[pack_offset],
                                        vec_f32[pack_offset + 1],
                                        vec_f32[pack_offset + 2],
                                        vec_f32[pack_offset + 3],
                                    )
                                raw_ptr = cutlass.inttoptr(
                                    (
                                        output_base
                                        + load_idx * TCGEN05_32B_REGS_PER_LOAD
                                        + j * FP8_OUTPUT_VECTOR_ELEMENTS
                                    ).toint(Int64),
                                    mem_space=1,
                                    dtype=Int32,
                                )
                                raw_ptr.store(
                                    packed_o.load(0, PACKED_FP8_OUTPUT_REGS),
                                    alignment=16,
                                )
                            else:
                                vec_o = vec_f32.to(output_dtype(self.cfg))
                                (
                                    output_base
                                    + load_idx * TCGEN05_32B_REGS_PER_LOAD
                                    + j * FP8_OUTPUT_VECTOR_ELEMENTS
                                ).nvvm_store_ext(
                                    vec_o,
                                    evict="noallocate",
                                )

        # Compute and store LSE in the same row-sum domain used by P.
        lse_row_sum = row_sum
        if cutlass.const_expr(cfg.is_fp8_qkv()):
            lse_row_sum = lse_row_sum * fp8_p_scale_rcp(cfg)
        lse = (
            cute.math.log2(lse_row_sum, fastmath=True)
            + self.scale_softmax_log2 * row_max
            if row_has_values
            else Float32(-Float32.inf)
        )

        if cutlass.const_expr(self.lse_in_natural_log):
            lse *= Float32(LN_2)
        # Use local_tidx (0..127 within correction warpgroup) for LSE
        # indexing, not global tidx (which is 128..255 for correction warps).
        lse_tidx = local_tidx
        if lse_tidx < tile_h:
            lse_row_in_tile = head_tile_idx * tile_h + lse_tidx
            (
                storage_flat_lse_row,
                logical_lse_head_idx,
                logical_lse_q_idx,
                _,
                lse_query_is_valid,
            ) = self.query_row_state(lse_row_in_tile, seq_q_idx, batch_idx)
            if lse_row_in_tile < physical_tile_rows and lse_query_is_valid:
                if cutlass.const_expr(self.partial_lse is not None):
                    S_q = (
                        cutlass.Int32(self.partial_lse.shape[2])
                        if self.partial_lse is not None
                        else Int32(1)
                    )
                    lse_base_ptr = (
                        self.partial_lse.iterator.raw_ptr()
                        + Int64(lse_row_in_tile) * Int64(self.split_kv)
                        + Int64(split_kv_idx)
                        + Int64(seq_q_idx)
                        * Int64(physical_tile_rows)
                        * Int64(self.split_kv)
                        + Int64(batch_idx)
                        * Int64(physical_tile_rows)
                        * Int64(self.split_kv)
                        * Int64(S_q)
                    )
                    lse_base_ptr.store(lse)
                elif cutlass.const_expr(self.lse is not None):
                    if cutlass.const_expr(self.cu_seqlens_q is not None):
                        lse_base_ptr = (
                            self.lse.iterator.raw_ptr() + storage_flat_lse_row
                        )
                    else:
                        S_q = (
                            cutlass.Int32(self.lse.shape[1])
                            if self.lse is not None
                            else Int32(1)
                        )
                        lse_base_ptr = (
                            self.lse.iterator.raw_ptr()
                            + Int64(logical_lse_head_idx)
                            + Int64(logical_lse_q_idx) * Int64(logical_num_heads_q)
                            + Int64(batch_idx) * Int64(logical_num_heads_q) * Int64(S_q)
                        )
                    lse_base_ptr.store(final_lse(lse, cfg.lse_base_on_e))

        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        cute.arch.fence_view_async_tmem_load()

    @producer_work
    @cute.jit
    def epilogue_store_slice(
        self,
        stage_info: StageInfo,
        *,
        row_sum,
        row_max,
        iter_n: cutlass.Constexpr[int],
    ) -> None:
        """Store one output-N slice after its O pipeline token is ready."""
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        blk_coord = stage_info.work_tile.tile_idx

        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads

        tmem_base_addr = self.tmem_o_ref.tmem_base_addr
        tmem_warp_row_id = (
            tmem_base_addr + (local_tidx >> WARP_LANE_SHIFT) * TCGEN05_32B_REGS_PER_LOAD
        )
        tmem_raw_addr = (tmem_warp_row_id << 16) | (
            cfg.tmem_o_offset + iter_n * (cfg.mma_pv_tiler[1] // cfg.warps_in_n)
        )

        t2r_shape = TCGEN05_32B_SHAPE
        num_tmem_loads = 4
        qk_acc_regs = cutlass.Array(Float32, 128, space=cutlass.AddressSpace.rmem)
        row_has_values = row_sum > Float32(0)
        norm_scale = self.normalization_scale(
            row_sum, row_max, local_tidx, blk_coord, stage_info
        )

        tile_h = cfg.mma_pv_tiler[0] // cfg.num_mma_ctas
        tile_d = cfg.mma_pv_tiler[1]
        tidx_g = local_tidx & EPILOGUE_THREAD_TILE_MASK
        g_i = tidx_g & EPILOGUE_ROW_MASK
        g_j = (tidx_g >> EPILOGUE_COLUMN_GROUP_SHIFT) * EPILOGUE_THREAD_TILE_THREADS

        logical_num_heads_q = Int32(self.logical_num_heads_q)
        physical_tile_rows = Int32(cfg.mma_qk_tiler[0])
        D = cfg.latent_dim
        head_tile_idx = blk_coord[0]
        seq_q_idx = blk_coord[1]
        batch_idx = blk_coord[2]
        split_kv_idx = blk_coord[3]
        row_in_tile = head_tile_idx * tile_h + g_i
        (
            storage_flat_query_row,
            logical_head_idx,
            logical_q_idx,
            _,
            query_is_valid,
        ) = self.query_row_state(row_in_tile, seq_q_idx, batch_idx)

        if cutlass.const_expr(
            self.output_quant is not None
            and self.partial_output is None
            and iter_n == 0
        ):
            self.prefetch_output_rope(
                storage_flat_query_row,
                batch_idx,
                local_tidx,
                logical_head_idx,
                row_in_tile < physical_tile_rows and query_is_valid,
            )

        if cutlass.const_expr(
            self.output_quant is not None
            and self.partial_output is None
            and iter_n + 1 == cfg.iterations_pv_n
        ):
            qk_acc_regs = self.load_rotary_slice(
                tmem_raw_addr,
                norm_scale,
                storage_flat_query_row,
                batch_idx,
                iter_n * tile_d + g_j,
                row_in_tile < physical_tile_rows and query_is_valid,
            )
        else:
            for load_idx in cutlass.range_constexpr(num_tmem_loads):
                curr_addr = tmem_raw_addr + load_idx * TCGEN05_32B_REGS_PER_LOAD
                tmem_ptr = prims.make_tmem_ptr(curr_addr, Float32)
                loaded = prims.tcgen05_ld(
                    t2r_shape, tmem_ptr, num=TCGEN05_32B_REGS_PER_LOAD
                )
                qk_acc_regs.store(loaded, load_idx * TCGEN05_32B_REGS_PER_LOAD)
            for i in cutlass.range_constexpr(0, 128, 2):
                scaled = mul_packed_f32x2(
                    (qk_acc_regs[i], qk_acc_regs[i + 1]),
                    (norm_scale, norm_scale),
                )
                qk_acc_regs[i] = scaled[0]
                qk_acc_regs[i + 1] = scaled[1]

        if cutlass.const_expr(cfg.stage_bf16_output):
            if cutlass.const_expr(self.cu_seqlens_q is not None):
                staged_row = storage_flat_query_row
            else:
                staged_row = (
                    logical_head_idx
                    + logical_q_idx * logical_num_heads_q
                    + batch_idx * logical_num_heads_q * Int32(self.output.shape[2])
                )
            self.store_staged_bf16_slice(
                qk_acc_regs,
                staged_row,
                iter_n * tile_d + g_j,
                row_in_tile < physical_tile_rows and query_is_valid,
                local_tidx,
            )
        direct_store = Int32(0 if cfg.stage_bf16_output else 1)
        if row_in_tile < physical_tile_rows and query_is_valid and direct_store != 0:
            if cutlass.const_expr(self.partial_output is not None):
                S_q = (
                    cutlass.Int32(self.partial_output.shape[3])
                    if self.partial_output is not None
                    else Int32(1)
                )
                o_base_ptr = (
                    self.partial_output.iterator.raw_ptr()
                    + Int64(row_in_tile) * Int64(self.split_kv) * Int64(D)
                    + Int64(split_kv_idx) * Int64(D)
                    + Int64(seq_q_idx)
                    * Int64(self.split_kv)
                    * Int64(physical_tile_rows)
                    * Int64(D)
                    + Int64(batch_idx)
                    * Int64(physical_tile_rows)
                    * Int64(self.split_kv)
                    * Int64(S_q)
                    * Int64(D)
                )
                output_base = o_base_ptr + iter_n * tile_d + g_j
                for load_idx in cutlass.range_constexpr(num_tmem_loads):
                    for j in cutlass.range_constexpr(4):
                        offset = (
                            load_idx * TCGEN05_32B_REGS_PER_LOAD
                            + j * BF16_OUTPUT_VECTOR_ELEMENTS
                        )
                        vec_f32 = qk_acc_regs.load(offset, BF16_OUTPUT_VECTOR_ELEMENTS)
                        vec_partial = vec_f32.to(cutlass.BFloat16)
                        (
                            output_base
                            + load_idx * TCGEN05_32B_REGS_PER_LOAD
                            + j * BF16_OUTPUT_VECTOR_ELEMENTS
                        ).nvvm_store_ext(
                            vec_partial,
                            evict="noallocate",
                        )
            elif cutlass.const_expr(self.output_quant is not None):
                self.store_quantized_slice(
                    qk_acc_regs,
                    storage_flat_query_row,
                    batch_idx,
                    iter_n * tile_d + g_j,
                    stage_idx=stage_info.stage_idx
                    if cutlass.const_expr(cfg.use_tma_output)
                    else 0,
                )
            else:
                if cutlass.const_expr(self.cu_seqlens_q is not None):
                    o_base_ptr = self.output.iterator.raw_ptr() + Int64(
                        storage_flat_query_row
                    ) * Int64(D)
                else:
                    S_q = (
                        cutlass.Int32(self.output.shape[2])
                        if self.output is not None
                        else Int32(1)
                    )
                    o_base_ptr = (
                        self.output.iterator.raw_ptr()
                        + Int64(logical_head_idx) * Int64(D)
                        + Int64(logical_q_idx) * Int64(logical_num_heads_q) * Int64(D)
                        + Int64(batch_idx)
                        * Int64(logical_num_heads_q)
                        * Int64(D)
                        * Int64(S_q)
                    )
                output_base = o_base_ptr + iter_n * tile_d + g_j
                for load_idx in cutlass.range_constexpr(num_tmem_loads):
                    for j in cutlass.range_constexpr(2):
                        offset = (
                            load_idx * TCGEN05_32B_REGS_PER_LOAD
                            + j * FP8_OUTPUT_VECTOR_ELEMENTS
                        )
                        vec_f32 = qk_acc_regs.load(offset, FP8_OUTPUT_VECTOR_ELEMENTS)
                        if cutlass.const_expr(cfg.use_fp8_output == 1):
                            packed_o = cutlass.Array(
                                Int32,
                                PACKED_FP8_OUTPUT_REGS,
                                space=cutlass.AddressSpace.rmem,
                            )
                            for pack_idx in cutlass.range_constexpr(
                                PACKED_FP8_OUTPUT_REGS
                            ):
                                pack_offset = pack_idx * PACKED_FP8_OUTPUT_REGS
                                packed_o[pack_idx] = pack_float4_to_fp8_e4m3(
                                    vec_f32[pack_offset],
                                    vec_f32[pack_offset + 1],
                                    vec_f32[pack_offset + 2],
                                    vec_f32[pack_offset + 3],
                                )
                            raw_ptr = cutlass.inttoptr(
                                (
                                    output_base
                                    + load_idx * TCGEN05_32B_REGS_PER_LOAD
                                    + j * FP8_OUTPUT_VECTOR_ELEMENTS
                                ).toint(Int64),
                                mem_space=1,
                                dtype=Int32,
                            )
                            raw_ptr.store(
                                packed_o.load(0, PACKED_FP8_OUTPUT_REGS),
                                alignment=16,
                            )
                        else:
                            vec_o = vec_f32.to(output_dtype(self.cfg))
                            (
                                output_base
                                + load_idx * TCGEN05_32B_REGS_PER_LOAD
                                + j * FP8_OUTPUT_VECTOR_ELEMENTS
                            ).nvvm_store_ext(
                                vec_o,
                                evict="noallocate",
                            )

        if cutlass.const_expr(iter_n == 0):
            lse_row_sum = row_sum
            if cutlass.const_expr(cfg.is_fp8_qkv()):
                lse_row_sum = lse_row_sum * fp8_p_scale_rcp(cfg)
            lse = (
                cute.math.log2(lse_row_sum, fastmath=True)
                + self.scale_softmax_log2 * row_max
                if row_has_values
                else Float32(-Float32.inf)
            )
            if cutlass.const_expr(self.lse_in_natural_log):
                lse *= Float32(LN_2)
            lse_tidx = local_tidx
            if lse_tidx < tile_h:
                lse_row_in_tile = head_tile_idx * tile_h + lse_tidx
                (
                    storage_flat_lse_row,
                    logical_lse_head_idx,
                    logical_lse_q_idx,
                    _,
                    lse_query_is_valid,
                ) = self.query_row_state(lse_row_in_tile, seq_q_idx, batch_idx)
                if lse_row_in_tile < physical_tile_rows and lse_query_is_valid:
                    if cutlass.const_expr(self.partial_lse is not None):
                        S_q = (
                            cutlass.Int32(self.partial_lse.shape[2])
                            if self.partial_lse is not None
                            else Int32(1)
                        )
                        lse_base_ptr = (
                            self.partial_lse.iterator.raw_ptr()
                            + Int64(lse_row_in_tile) * Int64(self.split_kv)
                            + Int64(split_kv_idx)
                            + Int64(seq_q_idx)
                            * Int64(physical_tile_rows)
                            * Int64(self.split_kv)
                            + Int64(batch_idx)
                            * Int64(physical_tile_rows)
                            * Int64(self.split_kv)
                            * Int64(S_q)
                        )
                        lse_base_ptr.store(lse)
                    elif cutlass.const_expr(self.lse is not None):
                        if cutlass.const_expr(self.cu_seqlens_q is not None):
                            lse_base_ptr = (
                                self.lse.iterator.raw_ptr() + storage_flat_lse_row
                            )
                        else:
                            S_q = (
                                cutlass.Int32(self.lse.shape[1])
                                if self.lse is not None
                                else Int32(1)
                            )
                            lse_base_ptr = (
                                self.lse.iterator.raw_ptr()
                                + Int64(logical_lse_head_idx)
                                + Int64(logical_lse_q_idx) * Int64(logical_num_heads_q)
                                + Int64(batch_idx)
                                * Int64(logical_num_heads_q)
                                * Int64(S_q)
                            )
                        lse_base_ptr.store(final_lse(lse, cfg.lse_base_on_e))

        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        cute.arch.fence_view_async_tmem_load()

    @cute.jit
    def load_rotary_slice(
        self, tmem_addr, norm_scale, flat_row, batch_idx, column, valid
    ):
        """Rotate the upper half before the other 64 accumulators become live."""
        row = Int64(flat_row)
        if cutlass.const_expr(self.cu_seqlens_q is None):
            row += Int64(batch_idx) * self.logical_num_heads_q * self.logical_seq_len_q
        result = cutlass.Array(Float32, 128, space=cutlass.AddressSpace.rmem)
        # The correction warp's register budget cannot keep 128 accumulators
        # and a batch of RoPE coefficients live together without serial loads.
        for half in cutlass.range_constexpr(2):
            offset = cutlass.const_expr(64 if half == 0 else 0)
            values = cutlass.Array(Float32, 64, space=cutlass.AddressSpace.rmem)
            for chunk in cutlass.range_constexpr(2):
                ptr = prims.make_tmem_ptr(tmem_addr + offset + chunk * 32, Float32)
                values.store(
                    prims.tcgen05_ld(TCGEN05_32B_SHAPE, ptr, num=32), chunk * 32
                )
            for i in cutlass.range_constexpr(0, 64, 2):
                scaled = mul_packed_f32x2(
                    (values[i], values[i + 1]), (norm_scale, norm_scale)
                )
                values[i], values[i + 1] = scaled[0], scaled[1]
            if cutlass.const_expr(half == 0):
                if valid:
                    self.output_quant.rotate(
                        self.output_quant_args, values, row, column + offset
                    )
            result.store(values.load(0, 64), offset)
        return result

    # Normalize one output row, including its head's sink mass and any FP8 probability
    # scale.
    @cute.jit
    def normalization_scale(self, row_sum, row_max, local_tidx, blk_coord, stage_info):
        """Compute output normalization including sink mass and the final source's V units.

        Row sums retain the FP8 probability scale. Apply tensor-FP8 V descale
        only to O, after denominator/sink normalization; it must not change LSE."""
        safe_sum = row_sum if row_sum > Float32(0) else Float32(1)
        scale = self.output_scale * cute.math.rcp(safe_sum, approx=True)
        if cutlass.const_expr(self.attention_sinks is not None):
            head = (
                Int32(blk_coord[1]) * self.cfg.mma_qk_tiler[0]
                + Int32(blk_coord[0])
                * (self.cfg.mma_qk_tiler[0] // self.cfg.num_mma_ctas)
                + (local_tidx & Int32(63))
            )
            if cutlass.const_expr(
                (
                    isinstance(self.logical_seq_len_q, int)
                    and self.logical_seq_len_q == 1
                )
            ):
                # Preserve the cheap single-query mapping, including padded heads.
                head = cute.math.min(head, Int32(self.logical_num_heads_q - 1))
            else:
                row = Int32(blk_coord[0]) * (
                    self.cfg.mma_qk_tiler[0] // self.cfg.num_mma_ctas
                ) + (local_tidx & Int32(63))
                _, head, _, _, _ = self.query_row_state(
                    row, Int32(blk_coord[1]), Int32(blk_coord[2])
                )
            sink = Float32(self.attention_sinks[head])
            scale = Float32(0)
            if row_sum > Float32(0) and sink != Float32(Float32.inf):
                max_log2 = row_max * self.scale_softmax_log2
                sink_log2 = sink * Float32(LOG2_E)
                norm_max = cute.math.max(max_log2, sink_log2)
                rescale = cute.math.exp2(max_log2 - norm_max, fastmath=True)
                inv_quant = Float32(
                    fp8_p_scale_rcp(self.cfg) if self.cfg.is_fp8_qkv() else 1.0
                )
                denominator = row_sum * inv_quant * rescale + cute.math.exp2(
                    sink_log2 - norm_max, fastmath=True
                )
                scale = (
                    self.output_scale
                    * inv_quant
                    * rescale
                    * cute.math.rcp(denominator, approx=True)
                )
        if cutlass.const_expr(self.cfg.has_fp8_primary):
            work = stage_info.work_tile
            last = (work.k_index_base + work.k_tile_count - Int32(1)) * Int32(
                self.cfg.mma_qk_tiler[1]
            )
            scale *= self.page_offsets.tensor_scale(last, work.tile_idx[2])
        return scale

    @cute.jit
    def prefetch_output_rope(self, flat_row, batch_idx, local_tidx, head, valid):
        """Fetch rotary coefficients while the nonrotary output is finalized."""
        tile_h = self.cfg.mma_pv_tiler[0] // self.cfg.num_mma_ctas
        # One lane per logical query in this CTA; the first row may start
        # partway through a head group. The other column group duplicates it.
        if local_tidx < tile_h and valid and (head == 0 or local_tidx == 0):
            row = Int64(flat_row)
            if cutlass.const_expr(self.cu_seqlens_q is None):
                row += (
                    Int64(batch_idx) * self.logical_num_heads_q * self.logical_seq_len_q
                )
            _, positions, cos_sin, _ = self.output_quant_args
            position = positions[row // self.logical_num_heads_q]
            cache_row = cos_sin.iterator.raw_ptr() + Int64(position) * 64
            prims.prefetch_l1(cache_row)
            prims.prefetch_l1(cache_row + 32)

    @cute.jit
    def query_row_state(self, row_in_tile, query_tile_idx, batch_idx):
        """Map one physical flat-tile row to public storage."""
        return flat_query_row_state(
            row_in_tile,
            query_tile_idx,
            self.cfg.mma_qk_tiler[0],
            self.logical_num_heads_q,
            self.logical_seq_len_q,
            self.cu_seqlens_q,
            batch_idx,
        )

    @cute.jit
    def store_quantized_slice(
        self,
        values,
        flat_row,
        batch_idx,
        column,
        stage_idx=0,
    ):
        row = Int64(flat_row)
        if cutlass.const_expr(self.cu_seqlens_q is None):
            row += Int64(batch_idx) * self.logical_num_heads_q * self.logical_seq_len_q
        shared_output = None
        if cutlass.const_expr(self.cfg.use_tma_output):
            shared_output = self.smem_o.data_ptr(
                stage_idx * OUTPUT_TMA_STAGE_BYTES
            ).toint(Int32)
        self.output_quant.store(
            self.output_quant_args,
            values,
            row,
            column,
            rotary=False,
            shared_output=shared_output,
        )
        if cutlass.const_expr(self.cfg.use_tma_output):
            cute.arch.fence_view_async_shared()

    @cute.jit
    def store_staged_bf16_slice(self, values, flat_row, column, valid, local_tidx):
        """TMA-store one warp's normalized 32-row x 128-column BF16 slice.

        Each correction warp owns two 4 KiB boxes (32 rows x 64 BF16, 128-byte
        swizzle) that it reuses across slices and work tiles. Its elected lane
        issues and retires the bulk groups; a box is rewritten only after the
        group that last read it has finished reading SMEM.
        """
        lane = local_tidx & Int32(31)
        warp = local_tidx >> Int32(5)
        # Consecutive lanes own consecutive flattened rows.
        first_row = Int32(flat_row) - lane
        for box in cutlass.range_constexpr(2):
            slot = (warp * Int32(2) + Int32(box)) * Int32(4096)
            if prims.elect_sync():
                prims.cp_async_bulk_wait_group(1, read=True)
            cute.arch.sync_warp()
            for chunk in cutlass.range_constexpr(8):
                base = box * 64 + chunk * 8
                words = [
                    pack_float2_to_bf16(values[base + 2 * w], values[base + 2 * w + 1])
                    for w in range(4)
                ]
                offset = (
                    slot
                    + lane * Int32(128)
                    + ((Int32(chunk) ^ (lane & Int32(7))) * Int32(16))
                )
                store_shared_u32x4(
                    self.smem_o, offset, words[0], words[1], words[2], words[3]
                )
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            if valid:
                if prims.elect_sync():
                    prims.cp_async_bulk_tensor_global_shared_cta(
                        self.tma_o_desc,
                        self.smem_o.subview(slot),
                        (column + box * 64, first_row),
                    )
                    prims.cp_async_bulk_commit_group()

    @consumer_work
    @cute.jit
    def tma_store_output(
        self, stage_info: StageInfo, *, iter_n: cutlass.Constexpr[int]
    ):
        # Sparse launch flattens queries into batch rows; each CTA owns 64 heads.
        head_tile_idx, _, token, _ = stage_info.work_tile.tile_idx
        # A tile can span groups or lie inside the single H128 group.
        heads_per_group = cutlass.const_expr(
            self.output_quant.num_heads // self.output_quant.num_groups
        )
        head_start = head_tile_idx * Int32(64)
        group = head_start // heads_per_group
        head_in_group = head_start % heads_per_group
        if prims.elect_sync():
            for half in cutlass.range_constexpr(2):
                prims.cp_async_bulk_tensor_global_shared_cta(
                    self.tma_o_desc,
                    self.smem_o.subview(
                        stage_info.stage_idx * OUTPUT_TMA_STAGE_BYTES + half * 8192
                    ),
                    (iter_n * 256 + half * 128, head_in_group, token, group),
                )
        prims.cp_async_bulk_commit_group()
        # Release the stage only after TMA has consumed its shared source.
        # The store task waits for global completion before retiring the CTA.
        prims.cp_async_bulk_wait_group(0, read=True)
