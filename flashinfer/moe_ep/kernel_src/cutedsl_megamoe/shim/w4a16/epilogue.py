# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""BF16 MegaMoE epilogue with NVFP4 per-expert FP32 weight scaling."""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.typing import Float32
from cutlass.cutlass_dsl import Int64
from src.iket_compat import iket
from common.moe_utils import fmin, fmax, quant_sfd_row
from moe_bf16_glu.epilogue_bf16 import GluBf16Epilogue
from moe_mxfp8_glu.epilogue_mxfp8 import (
    Fc1GateUpInterleave,
    EpilogueTileN,
    WarpThreadCount,
    EpiWarpCount,
    Fc2OutputDest,
)
from moe_nvfp4_swapab.epilogue import _red_add_relaxed_sys_v2_bf16x2


class W4A16Epilogue(GluBf16Epilogue):
    """Preserve weight scaling before the FC1 activation and FC2 cast."""

    @cute.jit
    def _run_fc1_subtile(
        self,
        subtile_idx,
        tmem_gate_tensor: cute.Tensor,
        tmem_up_tensor: cute.Tensor,
        real_fc1_output: cute.Tensor,
        real_fc1_output_sf: cute.Tensor,
        real_topk_scores: cute.Tensor,
        work_tile_info,
        smem_fc1_output_buffer: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        r2s_copy_atom: cute.CopyAtom,
        warp_idx: int,
        tidx,
        alpha,
        norm_const,
        rmem_sf: cute.Tensor,
        acc_is_release: bool,
        acc_pipeline,
        acc_consumer_state,
        smem_c_buffer: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        gmem_c_subtile_view: cute.Tensor,
        c_buffer_idx,
        c_pipeline,
    ) -> None:
        """Scale the FP32 FC1 accumulator, then activate and cast to BF16."""
        iket.range_push("fc1_epilogue_subtile")

        r_layout = cute.make_layout((((Fc1GateUpInterleave,), 1),), stride=(((1,), 0),))
        r_gate = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
        r_up = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)

        atom_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32),
            self.acc_dtype,
        )
        cute.copy(atom_t2r, tmem_gate_tensor, r_gate)
        cute.copy(atom_t2r, tmem_up_tensor, r_up)

        if cutlass.const_expr(self._overlapping_accum):
            self._acc_pipeline_consumer_release(
                acc_pipeline, acc_consumer_state, acc_is_release
            )

        # ── generate_c: store raw gate+up to GMEM C tensor via SMEM staging ──
        if cutlass.const_expr(self._generate_c):
            self._store_fc1_c_subtile(
                r_gate=r_gate,
                r_up=r_up,
                smem_c_buffer=smem_c_buffer,
                tma_atom_c=tma_atom_c,
                gmem_c_subtile_view=gmem_c_subtile_view,
                c_buffer_idx=c_buffer_idx,
                work_tile_info=work_tile_info,
                warp_idx=warp_idx,
                tidx=tidx,
                c_pipeline=c_pipeline,
            )

        weight_alpha = alpha[work_tile_info.expert_idx]
        r_gate.store(r_gate.load() * weight_alpha)
        r_up.store(r_up.load() * weight_alpha)

        if cutlass.const_expr(self.glu_clamp is not None):
            for i in cutlass.range_constexpr(cute.size(r_up)):
                r_gate[i] = fmin(r_gate[i], self.glu_clamp)
                r_up[i] = fmin(r_up[i], self.glu_clamp)
                r_up[i] = fmax(r_up[i], -self.glu_clamp)

        topk = None
        if cutlass.const_expr(self._apply_topk_in_fc1):
            thread_in_warp = tidx % cutlass.Int32(WarpThreadCount)
            token_in_tile = (
                work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m)
                + cutlass.Int32(warp_idx * WarpThreadCount)
                + thread_in_warp
            )
            topk = Float32(real_topk_scores[token_in_tile])

        swiglu = cute.make_rmem_tensor(r_layout.shape, self.acc_dtype)
        self._swiglu_act(swiglu, r_up, r_gate, topk)

        c = cute.make_rmem_tensor(r_layout.shape, self.fc1_output_dtype)
        if cutlass.const_expr(self.fc1_output_dtype.width == 8):
            # Quantized hand-off: fp8 data + E8M0 block scale.
            qpvscale = quant_sfd_row(
                swiglu,
                c,
                norm_const,
                self._sf_vec_size,
                self.sf_dtype,
                self.fc1_output_dtype,
            )
            if subtile_idx == 0:
                rmem_sf[0] = qpvscale
            elif subtile_idx == 1:
                rmem_sf[1] = qpvscale
            elif subtile_idx == 2:
                rmem_sf[2] = qpvscale
            elif subtile_idx == 3:
                rmem_sf[3] = qpvscale
        else:
            # Plain-data hand-off: direct cast to the fc1 output dtype (the
            # fc1_output workspace is reloaded as fc2's A operand).
            c.store(swiglu.load().to(self.fc1_output_dtype))

        thread_in_warp = tidx % WarpThreadCount
        if cutlass.const_expr(self._use_stg_fc1):
            # Direct STG.256 to GMEM — no SMEM staging or TMA store needed.
            token_in_tile = cutlass.Int32(warp_idx * EpilogueTileN) + thread_in_warp
            if token_in_tile < work_tile_info.valid_tokens_in_cta_tile:
                abs_token = (
                    work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m)
                    + cutlass.Int32(warp_idx * EpilogueTileN)
                    + thread_in_warp
                )
                # absolute column start (element index in the intermediate axis)
                col_elem = (
                    work_tile_info.tile_n_idx * cutlass.Int32(self._subtile_cnt)
                    + subtile_idx
                ) * cutlass.Int32(Fc1GateUpInterleave)
                # (1,1,1) tile gives pointer to element at (abs_token, col_elem, 0).
                g_base = cute.local_tile(
                    real_fc1_output,
                    (1, 1, 1),
                    (abs_token, col_elem, cutlass.Int32(0)),
                )
                stg_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.fc1_output_dtype,
                    num_bits_per_copy=256,
                )
                # col_elem is always a multiple of Fc1GateUpInterleave=32 (FP8 elements),
                # so the pointer is 32-byte aligned — matching STG.256 requirement.
                aligned_iter = cute.make_ptr(
                    self.fc1_output_dtype,
                    g_base.iterator.toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=32,
                )
                g_vec = cute.make_tensor(
                    aligned_iter, cute.make_layout(Fc1GateUpInterleave)
                )
                cute.copy(stg_atom, cute.coalesce(c), g_vec)
        else:
            sC_stage = cute.slice_(smem_fc1_output_buffer, (None, None, warp_idx))
            sC_thread_row = cute.local_tile(
                sC_stage, (1, Fc1GateUpInterleave), (thread_in_warp, subtile_idx)
            )
            cute.copy(r2s_copy_atom, cute.coalesce(c), cute.coalesce(sC_thread_row))

        if cutlass.const_expr(self._generate_c):
            if warp_idx == 0:
                c_pipeline.producer_acquire()
            c_store_bar = pipeline.NamedBarrier(
                barrier_id=self._CStoreBarId,
                num_threads=EpiWarpCount * WarpThreadCount,
            )
            c_store_bar.arrive_and_wait()

        iket.range_pop()

    @cute.jit
    def _run_fc2_subtile(
        self,
        subtile_idx,
        tmem_subtile_tensor: cute.Tensor,
        real_fc2_output: cute.Tensor,
        work_tile_info,
        valid_hidden,
        warp_idx: int,
        tidx,
        token_comm_args=None,
        rmem_sf_fc2=None,
        *,
        preload_acc=None,
    ) -> None:
        """fc2 subtile: LDTM + encode + STG.

        Encoding follows ``self._combine_format``:
          * BF16 (default): fp32->bf16 conversion, 256-bit STG × 2 half-tiles.
          * MXFP8: quant_sfd_row fp32->e4m3 (32 elems = one block), single
            256-bit STG for data; E8M0 scale written to local fc2_output_sf.

        When ``token_comm_args`` is not None (MegaMoE path), the data STG is
        routed to ``combine_output[src_token, src_topk, :]`` on the source rank
        via ``Fc2OutputDest``; for token_back_by_dispatch the epilogue writes to
        the local ``fc2_output_workspace`` pool instead.
        """
        iket.range_push("fc2_epilogue_subtile")

        fc2_subtile_cnt = self._cta_tile_n // EpilogueTileN  # = 8
        hidden_group = (
            work_tile_info.tile_n_idx * cutlass.Int32(fc2_subtile_cnt) + subtile_idx
        )
        hidden_col_start = work_tile_info.tile_n_idx * cutlass.Int32(
            self._cta_tile_n
        ) + subtile_idx * cutlass.Int32(EpilogueTileN)
        r_acc_layout = cute.make_layout((((EpilogueTileN,), 1),), stride=(((1,), 0),))
        atom_t2r = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32),
            self.acc_dtype,
        )
        r_acc = cute.make_rmem_tensor(r_acc_layout.shape, self.acc_dtype)
        cute.copy(atom_t2r, tmem_subtile_tensor, r_acc)
        weight_alpha = token_comm_args.fc2_alpha[work_tile_info.expert_idx]
        r_acc.store(r_acc.load() * weight_alpha)
        thread_in_warp = tidx % WarpThreadCount
        token_row_in_cta = cutlass.Int32(warp_idx * WarpThreadCount) + thread_in_warp
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile
        if token_row_in_cta < valid_tokens and hidden_col_start < valid_hidden:
            if cutlass.const_expr(
                token_comm_args is not None
                and not self._token_back_by_dispatch
                and self._combine_mxfp8
            ):
                # MegaMoE Form A, quantized combine:
                # 1. Quantize fp32 → fp8 + compute E8M0 block scale.
                # 2. STG fp8 data to peer's combine_output.
                # 3. Write E8M0 scale to local fc2_output_sf for token-back push.
                fp8_dtype = self._combine_format.act_dtype
                r_fp8 = cute.make_rmem_tensor(r_acc_layout.shape, fp8_dtype)
                qpvscale = quant_sfd_row(
                    r_acc,
                    r_fp8,
                    1.0,
                    EpilogueTileN,
                    cutlass.Float8E8M0FNU,
                    fp8_dtype,
                )
                pool_token_global = (
                    work_tile_info.cumulative_data_physical_row
                    + work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m)
                    + token_row_in_cta
                )
                metadata_u32 = cute.recast_tensor(
                    token_comm_args.token_src_metadata,
                    cutlass.Uint32,
                )
                fc2_output_dest = Fc2OutputDest(
                    tensor=token_comm_args.combine_output,
                    metadata=metadata_u32,
                    peer_rank_ptr_mapper=token_comm_args.peer_rank_ptr_mapper,
                )
                dest_row = fc2_output_dest.resolve_token_row(pool_token_global)
                # STG 32 fp8 elements = 256 bits in one shot.
                r_fp8_flat = cute.make_tensor(r_fp8.iterator, cute.make_layout(32))
                stg_fp8_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    fp8_dtype,
                    num_bits_per_copy=256,
                )
                dest_fp8_ptr = cute.make_ptr(
                    fp8_dtype,
                    dest_row.iterator.toint() + Int64(hidden_col_start),
                    cute.AddressSpace.gmem,
                    assumed_align=32,
                )
                cute.copy(
                    stg_fp8_atom,
                    r_fp8_flat,
                    cute.make_tensor(dest_fp8_ptr, cute.make_layout(32)),
                )
                # Buffer the E8M0 scale; the whole task tile's 8 scales are
                # flushed together by _stg_sf_fc2 (single stg.64 when aligned).
                self._write_sf_fc2_buffer(rmem_sf_fc2, subtile_idx, qpvscale)
            elif cutlass.const_expr(
                self._token_back_by_dispatch and self._combine_mxfp8
            ):
                # MegaMoE token-back-by-dispatch + quantized combine:
                # Epilogue writes fp8 data to local pool; dispatch warps push
                # both data (fc2_output_workspace) and SF (fc2_output_sf) to peers.
                pool_token_global = (
                    work_tile_info.cumulative_data_physical_row
                    + work_tile_info.tile_m_idx * cutlass.Int32(self._cta_tile_m)
                    + token_row_in_cta
                )
                fp8_dtype = self._combine_format.act_dtype
                r_fp8 = cute.make_rmem_tensor(r_acc_layout.shape, fp8_dtype)
                qpvscale = quant_sfd_row(
                    r_acc,
                    r_fp8,
                    1.0,
                    EpilogueTileN,
                    cutlass.Float8E8M0FNU,
                    fp8_dtype,
                )
                # Write 32 fp8 elements to local fc2_output_workspace pool.
                fp8_byte_addr = (
                    token_comm_args.fc2_output_workspace.iterator.toint()
                    + Int64(pool_token_global) * Int64(self._hidden_fc2)
                    + Int64(hidden_col_start)
                )
                stg_fp8_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    fp8_dtype,
                    num_bits_per_copy=256,
                )
                aligned_fp8_iter = cute.make_ptr(
                    fp8_dtype,
                    fp8_byte_addr,
                    cute.AddressSpace.gmem,
                    assumed_align=32,
                )
                r_fp8_flat = cute.make_tensor(
                    r_fp8.iterator, cute.make_layout(EpilogueTileN)
                )
                cute.copy(
                    stg_fp8_atom,
                    r_fp8_flat,
                    cute.make_tensor(aligned_fp8_iter, cute.make_layout(EpilogueTileN)),
                )
                # Buffer the E8M0 scale; flushed together by _stg_sf_fc2 after
                # the subtile loop (single stg.64 when hidden-aligned).
                self._write_sf_fc2_buffer(rmem_sf_fc2, subtile_idx, qpvscale)
            else:
                # BF16 path (default): fp32->bf16, two 256-bit STGs.
                r_bf16 = cute.make_rmem_tensor(r_acc_layout.shape, cutlass.BFloat16)
                r_bf16.store(r_acc.load().to(cutlass.BFloat16))
                stg_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    cutlass.BFloat16,
                    num_bits_per_copy=256,
                )
                for stg_half in cutlass.range_constexpr(EpilogueTileN // 16):
                    reg_view = cute.make_tensor(
                        r_bf16.iterator + stg_half * 16,
                        cute.make_layout(16),
                    )
                    if cutlass.const_expr(
                        token_comm_args is not None and not self._token_back_by_dispatch
                    ):
                        metadata_u32 = cute.recast_tensor(
                            token_comm_args.token_src_metadata,
                            cutlass.Uint32,
                        )
                        fc2_output_dest = Fc2OutputDest(
                            tensor=token_comm_args.combine_output,
                            metadata=metadata_u32,
                            peer_rank_ptr_mapper=token_comm_args.peer_rank_ptr_mapper,
                            reduce_topk_in_kernel=self._fc2_in_kernel_topk_reduce,
                        )
                        pool_token_global = (
                            work_tile_info.cumulative_data_physical_row
                            + work_tile_info.tile_m_idx
                            * cutlass.Int32(self._cta_tile_m)
                            + token_row_in_cta
                        )
                        dest_row = fc2_output_dest.resolve_token_row(pool_token_global)
                        hidden_off = hidden_col_start + cutlass.Int32(stg_half * 16)
                        dest_ptr = cute.make_ptr(
                            cutlass.BFloat16,
                            dest_row.iterator.toint() + hidden_off * cutlass.Int64(2),
                            cute.AddressSpace.gmem,
                            assumed_align=32,
                        )
                        if cutlass.const_expr(self._fc2_in_kernel_topk_reduce):
                            reg_u32 = cute.recast_tensor(reg_view, cutlass.Uint32)
                            for pair in cutlass.range_constexpr(16 // 4):
                                _red_add_relaxed_sys_v2_bf16x2(
                                    dest_ptr + cutlass.Int32(pair * 4),
                                    cutlass.Uint32(reg_u32[pair * 2]),
                                    cutlass.Uint32(reg_u32[pair * 2 + 1]),
                                )
                        else:
                            cute.copy(
                                stg_atom,
                                reg_view,
                                cute.make_tensor(dest_ptr, cute.make_layout(16)),
                            )
                    else:
                        g_fc2_output_tile = cute.local_tile(
                            real_fc2_output,
                            (self._cta_tile_m, EpilogueTileN, 1),
                            (work_tile_info.tile_m_idx, hidden_group, 0),
                        )
                        g_fc2_slice = cute.slice_(g_fc2_output_tile, (None, None, 0))
                        g_thread_row = cute.local_tile(
                            g_fc2_slice,
                            (1, 16),
                            (token_row_in_cta, stg_half),
                        )
                        g_flat = cute.coalesce(g_thread_row)
                        aligned_iter = cute.make_ptr(
                            cutlass.BFloat16,
                            g_flat.iterator.toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=32,
                        )
                        cute.copy(
                            stg_atom,
                            reg_view,
                            cute.make_tensor(aligned_iter, g_flat.layout),
                        )

        iket.range_pop()
