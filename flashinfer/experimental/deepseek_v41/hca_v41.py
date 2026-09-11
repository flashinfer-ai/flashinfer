# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Derived from flashinfer/cute_dsl/attention/dsa/hca_fp8.py (DeepSeek V4 HCA decode,
# CuTe DSL, SM100) by Mengyu Guo <mengyug@nvidia.com>, flashinfer PRs #3943 and
# #4368. This file adapts that kernel to the DeepSeek V4.1 mixed-cache decode
# contract: BF16 query, MXFP8 (E4M3 + per-32 E8M0) sliding-window KV and FP4
# (E2M1 + per-16 E4M3) compressed KV, both dequantized to BF16 for kind::f16
# MMA, plus the V4.1 per-head sink. The single-CTA head64 structure of DeepSeek
# FlashMLA (csrc/kernels/sm100/decode/sparse/head64, MIT) was used as a design
# reference for the SMEM/TMEM budget; no FlashMLA code is copied.
# Work in progress: see the module docstring for the current state.

import math
from typing import Type, Tuple, Optional
from types import SimpleNamespace

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims
from . import hca_v41_primitives as hw
from cutlass.cute.nvgpu import tcgen05, OperandMajorMode
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.arch import Arch
from cutlass.cute.arch import nvvm_wrappers as _nvw
from cutlass.cutlass_dsl import BaseDSL

from ...cute_dsl.attention.dsa.hca_helpers import (
    ceil_div,
    MAX_SPLITS,
    LOG2_E,
    HCAStaticTileScheduler,
    HCAStaticTileSchedulerParams,
    create_hca_static_tile_scheduler,
    create_hca_static_tile_scheduler_params,
)

"""SM100 H64/S1 DS4.1 mixed-cache decode, adapted from Mengyu Guo's HCA.

The public wrapper selects the M64 single-CTA primitives specialization.
Queries, dequantized KV and probabilities use BF16; accumulation and softmax
use FP32. Physical slot -1 is masked without reading cache storage. The sink
is applied once at final normalization, preserving natural-log, sink-exclusive
public LSE even when the sink dominates all real keys. Split intermediates
retain log2 LSE for merging. See examples/deepseek_v41/decode.py.
"""


class BlackwellV41MixedCacheDecode:
    arch_str: str = "sm_100"
    arch_name: str = "Blackwell SM100"

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        lse_dtype: Type[cutlass.Numeric],
        mma_qk_tiler_mn: Tuple[int, int],
        mma_pv_tiler_mn: Tuple[int, int],
        max_active_clusters: int,
        page_size_cmp: int,
        skip_correction_threshold: float,
        is_persistent: bool,
        is_var_seq: bool,
        is_var_split_kv: bool,
        is_causal: bool = False,
        seq_len_q: int = 1,
        hca_compress_ratio: int = 128,
        window_page_size: int = 64,
        compressed_page_size: int = 64,
        compressed_is_fp4: bool = True,
        max_topk: int = 512,
    ):
        """Initializes the configuration for a Blackwell Heavily Compressed Attention (HCA) kernel.

        :param acc_dtype: Data type for accumulation S and O
        :type acc_dtype: Type[cutlass.Numeric]
        :param lse_dtype: Data type for output LSE
        :type lse_dtype: Type[cutlass.Numeric]
        :param mma_s_tiler: The (H, K) tile shape of the MMA instruction for S
        :type mma_s_tiler: Tuple[int, int]
        :param mma_p_tiler: The (H, D) tile shape of the MMA instruction for P
        :type mma_p_tiler: Tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: int
        :param page_size_cmp: Page size of the compressed-KV page table
        :type page_size_cmp: int
        :param skip_correction_threshold: Threshold to skip correction
        :type skip_correction_threshold: float
        :param is_persistent: Whether to use persistent kernel mode
        :type is_persistent: bool
        :param is_var_seq: Whether to use variable sequence length
        :type is_var_seq: bool
        :param is_var_split_kv: Whether to use variable split KV
        :type is_var_split_kv: bool
        :param is_causal: Whether to apply HCA sparse causal masking
        :type is_causal: bool
        :param seq_len_q: Sequence length of Q, used for causal visibility
        :type seq_len_q: int
        :param hca_compress_ratio: Number of raw KV tokens represented by one compressed HCA slot
        :type hca_compress_ratio: int
        """

        # latent_dim is the FULL per-head depth for HCA (head_dim).
        # The last `qk_rope_head_dim` of these are assumed to be already
        # RoPE-rotated by the caller; the kernel does not see rope as a
        # separate path.
        self.latent_dim = 512
        self.acc_dtype = acc_dtype
        self.lse_dtype = lse_dtype
        self.mma_qk_tiler_mn = mma_qk_tiler_mn
        self.mma_pv_tiler_mn = mma_pv_tiler_mn
        self.max_active_clusters = max_active_clusters
        self.skip_correction_threshold = skip_correction_threshold
        self.is_persistent = is_persistent
        self.page_size_cmp = page_size_cmp
        self.is_var_seq = is_var_seq
        self.is_var_split_kv = is_var_split_kv
        self.is_causal = is_causal
        self.seq_len_q = seq_len_q
        self.hca_compress_ratio = hca_compress_ratio
        # V4.1 raw cache formats. Window rows: 512 B E4M3 + 16 B E8M0 (per-32).
        # Compressed rows: 256 B E2M1 + 32 B E4M3 (per-16) when fp4, else as
        # the window format. Pages hold all data rows, then all scale rows.
        self.window_page_size = window_page_size
        self.compressed_page_size = compressed_page_size
        self.compressed_is_fp4 = compressed_is_fp4
        # Upper bound on selected compressed keys per row; sizes the SMEM slot cache.
        self.max_topk = max_topk
        self.use_2cta_instrs = mma_qk_tiler_mn[0] == 128
        self.use_primitives = not self.use_2cta_instrs
        self.cluster_shape_mnk = (2 if self.use_2cta_instrs else 1, 1, 1)
        if not self.use_2cta_instrs:
            assert mma_qk_tiler_mn == (64, 64)
            assert mma_pv_tiler_mn == (64, 128)
        # When using 2 CTAs with m=128: warps 0-1 handle accumulation for first half [0, n/2),
        # while warps 2-3 handle accumulation for second half [n/2, n)
        self.warps_in_n = 2
        self.num_compute_warps = 4
        self.threads_per_warp = 32
        mma_qk_tiler_k = 128
        self.mma_qk_tiler = (
            self.mma_qk_tiler_mn[0],
            self.mma_qk_tiler_mn[1],
            mma_qk_tiler_k,
        )
        self.mma_pv_tiler = (
            self.mma_pv_tiler_mn[0],
            self.mma_pv_tiler_mn[1],
            self.mma_qk_tiler[1] * self.mma_qk_tiler[2] // self.mma_pv_tiler_mn[1],
        )
        self.iterations_qk_latent = self.latent_dim // self.mma_qk_tiler[2]
        self.iterations_qk = self.iterations_qk_latent
        self.iterations_pv_k = self.mma_qk_tiler[1] // self.mma_pv_tiler[2]
        self.iterations_pv_n = self.latent_dim // self.mma_pv_tiler[1]

        # Set specialized warp ids.
        # Compute (softmax) warp groups: g0 = warps 0-3 (even k-tiles), g1 =
        # warps 8-11 (odd k-tiles). Correction warps 4-7. Each 128-thread
        # compute group must start at a multiple of 4 warps because the
        # compute code derives its lane from tidx % 128. MMA split: W12 issues
        # QK only, W14 issues PV only; W13 loads Q by TMA (K/V come from the
        # dequant warps). The original TMA-V warp is gone; its registers go to
        # the dequant group.
        self.compute_warp_ids = (0, 1, 2, 3)
        self.correction_warp_ids = (4, 5, 6, 7)
        self.second_compute_warp_ids = (8, 9, 10, 11)
        self.mma_qk_warp_id = 12
        self.load_tma_k_warp_id = 13
        self.mma_pv_warp_id = 14
        # Dequant group: gathers raw FP8/FP4 rows with plain loads and writes
        # the BF16 K and V tiles that the UMMA warps consume.
        self.dequant_warp_ids = (15, 16, 17, 18, 19, 20, 21, 22)
        # Register allocation rounds the CTA up to a multiple of 4 warps, so a
        # 23-warp launch gets the 24-warp budget (80/lane) but only 23 shares of
        # it. Launch the 24th warp as an idle one at the minimum allocation to
        # keep its 80 - 24 registers in the pool.
        self.idle_warp_id = 23
        self.num_total_compute_warps = self.num_compute_warps + len(
            self.second_compute_warp_ids
        )
        self.threads_per_cta = self.threads_per_warp * len(
            (
                self.mma_qk_warp_id,
                self.load_tma_k_warp_id,
                *self.compute_warp_ids,
                *self.second_compute_warp_ids,
                *self.correction_warp_ids,
                self.mma_pv_warp_id,
                *self.dequant_warp_ids,
                self.idle_warp_id,
            )
        )

        # 24 warps launch with 80 registers per lane (65536 / 768 rounded down
        # to the 8-register granule; ptxas reports REG accordingly), so the
        # setmaxnreg targets must sum to at most 24 * 80 = 1920:
        # 8*96 + 4*136 + 3*40 + 1*24 + 8*56 = 1904. A larger sum blocks the
        # increases forever; a "decrease" above the launch value is an illegal
        # instruction. With 64-key tiles each softmax thread holds 32 scores,
        # which is what lets softmax live in 96.
        # M64/PV128 halves the correction fragment. Its MMA/Q-load warps
        # fit in 32 registers, allowing dequant to use 80 without exceeding
        # the pool: 8*96 + 4*96 + 3*32 + 24 + 8*80 = 1912 <= 1920.
        # Retain the larger load phase to expose independent global loads.
        self.softmax_reg_num = 96
        self.correction_reg_num = 136 if self.use_2cta_instrs else 96
        self.other_reg_num = 40 if self.use_2cta_instrs else 32
        self.idle_reg_num = 24
        self.dequant_reg_num = 56 if self.use_2cta_instrs else 80
        # Raw 16-byte chunks each dequant thread keeps in flight per phase.
        self.dequant_phase_chunks = 8
        # Timing-attribution switch (results are wrong when != 0): 1 = dequant
        # warps only run the pipeline, 2 = loads only, 3 = convert+store only.
        self.dbg_mode = 0
        # The V4.1 sliding window (128 keys) sits at the head of the key
        # sequence and spans window_tiles key tiles; window_indices has shape
        # [window_len, rows].
        self.window_len = 128
        assert self.window_len % self.mma_qk_tiler[1] == 0
        self.window_tiles = self.window_len // self.mma_qk_tiler[1]
        # Named barriers synchronize strict warp subsets. Emit non-aligned
        # CUTLASS primitives at every arrive/sync site; legacy bar.sync and
        # bar.arrive incorrectly promise whole-CTA convergence here. Retain
        # every slot, participant count, and producer/consumer phase.
        self.tmem_ptr_sync_bar = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=(
                self.threads_per_warp * 2
                + self.threads_per_warp * self.num_total_compute_warps
                + self.threads_per_warp * self.num_compute_warps
            ),
        )
        self.softmax_exchange_sync_bar_0 = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=(self.threads_per_warp * self.num_compute_warps),
        )
        self.softmax_exchange_sync_bar_1 = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=(self.threads_per_warp * self.num_compute_warps),
        )
        self.epilogue_exchange_sync_bar = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=(self.threads_per_warp * self.num_compute_warps),
        )
        self.softmax_order_bar_0 = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=(self.threads_per_warp * self.num_total_compute_warps),
        )
        self.softmax_order_bar_1 = pipeline.NamedBarrier(
            barrier_id=6,
            num_threads=(self.threads_per_warp * self.num_total_compute_warps),
        )
        self.softmax_warps_initial_sync_bar = pipeline.NamedBarrier(
            barrier_id=7,
            num_threads=(self.threads_per_warp * self.num_total_compute_warps),
        )
        # W12 acquires both window stages; W9/W10 join the handoff so their
        # canonical states stay aligned while W12-W15 issue K/V quarters.
        self.dequant_sync_bar = pipeline.NamedBarrier(
            barrier_id=9,
            num_threads=self.threads_per_warp * len(self.dequant_warp_ids),
        )
        self.init_row_max = -float("inf")
        self.tmem_corr_stage_cols = 4
        self.tmem_s_offset = 0 if self.use_2cta_instrs else (16 << 16)

    def _setup_attributes(self):
        """Set up configurations and parameters for the HCA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the multi-head latent attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        self.load_q_stage = 1
        # V4.1 derivation, milestone 1a: BF16 operands double every tile, so K/V
        # staging drops to one stage to stay under the 227 KB SMEM budget.
        # Two K/V stages: with 64-key tiles a stage is 32 KB + 32 KB, so
        # Q 64K + K/V 128K + P 16K fits the 227 KB budget.
        self.load_k_stage = 2
        self.load_v_stage = 2
        self.mma_s_stage = 2
        self.p_mma_stage = 2
        self.p_cor_stage = 2
        self.mma_o_stage = 2

        self.tmem_o_offset = (
            (self.mma_s_stage * self.mma_qk_tiler[1] // self.warps_in_n)
            if self.use_2cta_instrs
            else 0
        )
        self.correction_factor_offset = (
            self.tmem_o_offset + self.latent_dim // self.warps_in_n
        )

    @cute.jit
    def __call__(
        self,
        q_latent: cute.Tensor,
        c_latent_win: cute.Tensor,
        c_latent_cmp: cute.Tensor,
        window_indices: cute.Tensor,
        compressed_indices: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        workspace: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: Optional[cute.Tensor],
        block_split_kvs: Optional[cute.Tensor],
        sparse_mla_topk_lens: cute.Tensor,
        window_valid_lens: cute.Tensor,
        softmax_scale: cutlass.Float32,
        output_scale: cutlass.Float32,
        attn_sink_unscaled: cute.Tensor,
        stream: cuda.CUstream,
        debug_kv: Optional[cute.Tensor] = None,
    ):
        """Execute the Multi-Head Latent Attention operation on the provided tensors.

        The method handles:
        1. Initialization of workspace for temporary split KV buffers
        2. Validation of tensor data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch(split KV kernel and reduction kernel) with appropriate parameters

        :param q_latent: The query tensor with shape [num_head, latent_dim, seq_len_q, batch_size]
            (last `qk_rope_head_dim` lanes pre-rotated by caller).
        :type q_latent: cute.Tensor
        :param c_latent_win: Flat sliding-window KV pool with shape
            [window_pool_rows, latent_dim]
        :type c_latent_win: cute.Tensor
        :param c_latent_cmp: Flat compressed KV pool with shape
            [compressed_pool_rows, latent_dim]; rows are gathered by
            ``compressed_indices`` (V4.1 top-k selection is token-sparse).
        :type c_latent_cmp: cute.Tensor
        :param window_indices: Token-level absolute row indices into
            ``c_latent_win`` with shape [mma_qk_tiler[1], table_rows]. Every
            nonnegative entry must be a legal backing row; ``-1`` is masked.
        :type window_indices: cute.Tensor
        :param compressed_indices: Token-level absolute row indices into
            ``c_latent_cmp`` with shape [compressed_topk, table_rows]. Every
            nonnegative entry must be a legal backing row; ``-1`` is masked.
        :type compressed_indices: cute.Tensor
        :param o: The output tensor with shape [num_head, latent_dim, seq_len_q, batch_size]
        :type o: cute.Tensor
        :param lse: The LSE tensor with shape [num_head, seq_len_q, batch_size]
        :type lse: cute.Tensor
        :param workspace: The workspace tensor with 1-d shape prepared for acc_o and acc_lse
        :type workspace: cute.Tensor
        :param split_kv: The scalar factor for split KV
        :type split_kv: cutlass.Int32
        :param cache_seqs: The cache sequences tensor with shape [batch_size]
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: The block split KV tensor with shape [batch_size]
        :type block_split_kvs: cute.Tensor
        :param sparse_mla_topk_lens: Per-query valid sparse HCA length with shape
            [batch_size * seq_len_q] for causal mode, or [batch_size] otherwise.
            A zero-length row requires a zero window-valid length plus legal
            window indices and backing rows for the fully masked sink-only tile.
        :type sparse_mla_topk_lens: cute.Tensor
        :param window_valid_lens: Per-query valid length within the sliding-window
            pool, using the same row layout as sparse_mla_topk_lens.
        :type window_valid_lens: cute.Tensor
        :param softmax_scale: The scale factor for softmax
        :type softmax_scale: cutlass.Float32
        :param output_scale: The scale factor for the output
        :type output_scale: cutlass.Float32
        :param attn_sink_unscaled: Per-head attention-sink logit in scaled
            score space. Shape [num_heads]. Added once at final normalization
            as a virtual extra logit with V=0; excluded from returned LSE.
        :type attn_sink_unscaled: cute.Tensor
        :param stream: The CUDA stream to execute the kernel on
        :type stream: cuda.CUstream

        :raises TypeError: If tensor data types don't match or aren't supported
        """

        # Prefetch the unsplit M64 route, where one CTA consumes many key
        # tiles. Retain producer ordering for split and two-CTA routes. The
        # workspace optional type is static even with symbolic tensor shapes.
        self.prefetch_raw = not self.use_2cta_instrs and workspace is None

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q_latent.element_type
        # K/V are dequantized to the query dtype inside the kernel.
        self.k_dtype = self.q_dtype
        self.v_dtype = self.q_dtype
        self.o_dtype = o.element_type

        if cutlass.const_expr(self.q_dtype != cutlass.BFloat16):
            raise TypeError("V4.1 decode expects a BF16 query")
        if cutlass.const_expr(
            c_latent_win.element_type != cutlass.Uint8
            or c_latent_cmp.element_type != cutlass.Uint8
        ):
            raise TypeError("V4.1 caches must be uint8 page pools")

        # The FlashInfer TVM-FFI boundary uses standard contiguous PyTorch
        # layouts. Reinterpret those layouts without moving data into the
        # internal HCA layouts used by the original DKG kernel.
        def _reinterpret_4d(tensor):
            # [B, S_q, H, D] -> [H, D, S_q, B]
            return cute.make_tensor(
                tensor.iterator,
                cute.make_layout(
                    (
                        tensor.shape[2],
                        tensor.shape[3],
                        tensor.shape[1],
                        tensor.shape[0],
                    ),
                    stride=(
                        tensor.stride[2],
                        tensor.stride[3],
                        tensor.stride[1],
                        tensor.stride[0],
                    ),
                ),
            )

        def _reinterpret_3d_kv(tensor):
            # [num_pages, page_size, D] -> [page_size, D, num_pages]
            return cute.make_tensor(
                tensor.iterator,
                cute.make_layout(
                    (tensor.shape[1], tensor.shape[2], tensor.shape[0]),
                    stride=(tensor.stride[1], tensor.stride[2], tensor.stride[0]),
                ),
            )

        def _reinterpret_row_table(tensor):
            # [rows, columns] -> [columns, rows]
            return cute.make_tensor(
                tensor.iterator,
                cute.make_layout(
                    (tensor.shape[1], tensor.shape[0]),
                    stride=(tensor.stride[1], tensor.stride[0]),
                ),
            )

        q_latent = _reinterpret_4d(q_latent)
        window_indices = _reinterpret_row_table(window_indices)
        compressed_indices = _reinterpret_row_table(compressed_indices)
        o = _reinterpret_4d(o)
        # [B, S_q, H] -> [H, S_q, B]
        lse = cute.make_tensor(
            lse.iterator,
            cute.make_layout(
                (lse.shape[2], lse.shape[1], lse.shape[0]),
                stride=(lse.stride[2], lse.stride[1], lse.stride[0]),
            ),
        )

        if cutlass.const_expr(
            len(c_latent_win.shape) != 2 or len(c_latent_cmp.shape) != 2
        ):
            raise ValueError(
                "c_latent_win and c_latent_cmp must be [pages, page_bytes]"
            )
        if cutlass.const_expr(
            len(window_indices.shape) != 2 or len(compressed_indices.shape) != 2
        ):
            raise ValueError(
                "window_indices and compressed_indices must be rank-2 tensors"
            )

        # check leading dimensions of input/output
        if cutlass.const_expr(q_latent.stride[1] != 1):
            raise ValueError("q_latent must have leading dimension 1")
        if cutlass.const_expr(c_latent_cmp.stride[1] != 1):
            raise ValueError("c_latent_cmp must have leading dimension 1")
        if cutlass.const_expr(c_latent_win.stride[1] != 1):
            raise ValueError("c_latent_win must have leading dimension 1")
        if cutlass.const_expr(window_indices.stride[0] != 1):
            raise ValueError("window_indices must have leading dimension 0")
        if cutlass.const_expr(o.stride[1] != 1):
            raise ValueError("o must have leading dimension 1")
        if cutlass.const_expr(lse.stride[0] != 1):
            raise ValueError("lse must have leading dimension 0")

        acc_o, acc_lse = self.initialize_workspace(
            q_latent.shape[0],
            q_latent.shape[1],
            q_latent.shape[2],
            q_latent.shape[3],
            split_kv,
            self.acc_dtype,
            workspace,
        )

        self.q_major_mode = OperandMajorMode.K
        self.k_major_mode = OperandMajorMode.K
        self.v_major_mode = OperandMajorMode.MN

        self._setup_attributes()

        cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        # the intermediate tensor p is from smem & k-major
        p_major_mode = OperandMajorMode.K
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.acc_dtype,
            cta_group,
            self.mma_qk_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.acc_dtype,
            cta_group,
            self.mma_pv_tiler[:2],
        )

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        self.epi_tile = self.mma_pv_tiler[:2]

        q_latent_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.mma_qk_tiler,
            self.q_dtype,
            (self.iterations_qk_latent * self.load_q_stage),
        )
        q_latent_smem_layout_staged = cute.logical_divide(
            q_latent_smem_layout_staged, (None, None, None, self.iterations_qk_latent)
        )

        kc_latent_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.mma_qk_tiler,
            self.k_dtype,
            (self.iterations_qk_latent * self.load_k_stage),
        )
        kc_latent_smem_layout_staged = cute.logical_divide(
            kc_latent_smem_layout_staged, (None, None, None, self.iterations_qk_latent)
        )

        p_smem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.mma_pv_tiler,
            self.q_dtype,
            (self.iterations_pv_k * self.p_mma_stage),
        )
        p_smem_layout_staged = cute.logical_divide(
            p_smem_layout_staged, (None, None, None, self.iterations_pv_k)
        )

        vc_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.mma_pv_tiler,
            self.v_dtype,
            (self.iterations_pv_k * self.iterations_pv_n * self.load_v_stage),
        )
        vc_smem_layout_staged = cute.logical_divide(
            cute.logical_divide(
                vc_smem_layout_staged,
                (None, None, None, self.iterations_pv_k * self.iterations_pv_n),
            ),
            (None, None, None, (self.iterations_pv_n, None)),
        )
        # TMA load for Q latent
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)

        q_smem_layout = cute.select(q_latent_smem_layout_staged, mode=[0, 1, 2])

        tma_atom_q_latent, tma_tensor_q_latent = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q_latent,
            q_smem_layout,
            self.mma_qk_tiler,
            qk_tiled_mma,
            cta_layout_vmnk.shape,
        )
        q_latent_copy_size = (
            cute.size_in_bytes(self.q_dtype, q_smem_layout)
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_latent
        )
        self.tma_copy_q_bytes = q_latent_copy_size

        tile_sched_params, grid = self._compute_grid(
            o,
            split_kv,
            self.cluster_shape_mnk,
            self.max_active_clusters,
            self.is_persistent,
        )

        @cute.struct
        class SplitKVKernelSharedStorage:
            # Pipeline barriers
            load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_q_stage * 2]
            load_k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_k_stage * 2]
            load_v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_v_stage * 2]
            mma_s_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_s_stage * 2]
            p_mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.p_mma_stage * 2]
            p_cor_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.p_cor_stage * 2]
            mma_o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_o_stage * 2]

            # Smem tensors
            smem_p: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(p_smem_layout_staged)],
                1024,
            ]
            smem_kc_latent: cute.struct.Align[
                cute.struct.MemRange[
                    self.k_dtype, cute.cosize(kc_latent_smem_layout_staged)
                ],
                1024,
            ]
            smem_q_latent: cute.struct.Align[
                cute.struct.MemRange[
                    self.q_dtype, cute.cosize(q_latent_smem_layout_staged)
                ],
                1024,
            ]
            smem_vc: cute.struct.Align[
                cute.struct.MemRange[
                    self.v_dtype,
                    cute.cosize(vc_smem_layout_staged) if self.use_2cta_instrs else 0,
                ],
                1024,
            ]
            softmax_smem_exchange: cute.struct.MemRange[
                self.acc_dtype, 2 * self.num_compute_warps * self.threads_per_warp
            ]
            # Page-table slots of this row (window keys, then compressed keys),
            # filled once per work tile by the dequant warps.
            smem_slots: cute.struct.MemRange[
                cutlass.Int32, self.window_len + self.max_topk
            ]
            epilogue_smem_exchange: cute.struct.MemRange[
                self.acc_dtype, self.num_compute_warps * self.threads_per_warp
            ]

            smem_metadata: cute.struct.MemRange[
                self.acc_dtype, 0 if self.use_2cta_instrs else 1024
            ]

            # Tmem dealloc cluster barrier
            tmem_dealloc_mbar: cutlass.Int64

            # Tmem holding buffer
            tmem_holding_buf: cutlass.Int32

        softmax_scale_log2 = softmax_scale * LOG2_E

        self.split_kv_kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q_latent,
            tma_tensor_q_latent,
            c_latent_win,
            c_latent_cmp,
            window_indices,
            compressed_indices,
            o,
            lse,
            acc_o,
            acc_lse,
            split_kv,
            cache_seqs,
            block_split_kvs,
            sparse_mla_topk_lens,
            window_valid_lens,
            softmax_scale_log2,
            output_scale,
            attn_sink_unscaled,
            debug_kv,
            q_latent_smem_layout_staged,
            kc_latent_smem_layout_staged,
            p_smem_layout_staged,
            vc_smem_layout_staged,
            cta_layout_vmnk,
            tile_sched_params,
            SplitKVKernelSharedStorage,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )
        if cutlass.const_expr(acc_o is not None):
            # Specialize the merge IO layout before launch; a branch inside
            # one GPU kernel regressed larger-batch merge timings.
            if (
                q_latent.shape[3] <= 4
                and q_latent.shape[0] == 64
                and q_latent.shape[2] == 1
            ):
                self.reduction_kernel(
                    o,
                    lse,
                    acc_o,
                    acc_lse,
                    split_kv,
                    cache_seqs,
                    block_split_kvs,
                    attn_sink_unscaled,
                    True,
                ).launch(
                    grid=(q_latent.shape[0], q_latent.shape[2], q_latent.shape[3]),
                    block=[self.threads_per_warp * self.num_compute_warps, 1, 1],
                    stream=stream,
                    min_blocks_per_mp=1,
                )
            else:
                self.reduction_kernel(
                    o,
                    lse,
                    acc_o,
                    acc_lse,
                    split_kv,
                    cache_seqs,
                    block_split_kvs,
                    attn_sink_unscaled,
                    False,
                ).launch(
                    grid=(q_latent.shape[0], q_latent.shape[2], q_latent.shape[3]),
                    block=[self.threads_per_warp * self.num_compute_warps, 1, 1],
                    stream=stream,
                    min_blocks_per_mp=1,
                )

    @cute.kernel
    def split_kv_kernel(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tma_atom_q_latent: Optional[cute.CopyAtom],
        mQL: cute.Tensor,
        mWinPool: cute.Tensor,
        mCmpPool: cute.Tensor,
        mWindowIndices: cute.Tensor,
        mCmpIndices: cute.Tensor,
        mO: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mAccO: Optional[cute.Tensor],
        mAccLSE: Optional[cute.Tensor],
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        sparse_mla_topk_lens: cute.Tensor,
        window_valid_lens: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        attn_sink_unscaled: cute.Tensor,
        mDebugKV: Optional[cute.Tensor],
        q_latent_smem_layout_staged: cute.ComposedLayout,
        kc_latent_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        vc_smem_layout_staged: cute.ComposedLayout,
        cta_layout_vmnk: cute.Layout,
        tile_sched_params: HCAStaticTileSchedulerParams,
        SharedStorage: cutlass.Constexpr,
    ):
        """The device split_kv kernel implementation of the Heavily Compressed Attention (HCA).

        This kernel coordinates multiple specialized warps to perform different phases of the HCA computation:
        1. Load warp: Loads Q/C latent data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Compute warps: Compute softmax and do rescaling on accumulators, and store the intermediate/final results
        to global memory

        The kernel produces either intermediate or final results of the HCA computation based on the split_kv parameter.
        When split_kv is 1, the kernel generates the final results directly. Otherwise, it produces intermediate results
        that will later be combined by a reduction kernel.

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases.

        :param tiled_mma_qk: Tiled MMA for Q*K^T
        :type tiled_mma_qk: cute.TiledMma
        :param tiled_mma_pv: Tiled MMA for P*V
        :type tiled_mma_pv: cute.TiledMma
        :param tma_atom_q_latent: TMA copy atom for query latent tensor
        :type tma_atom_q_latent: cute.CopyAtom
        :param mQL: query latent tensor
        :type mQL: cute.Tensor
        :param mCL_win: Window-stream key tensor
        :type mCL_win: cute.Tensor
        :param mCL_cmp: Compressed-stream key tensor
        :type mCL_cmp: cute.Tensor
        :param mCLT_win: Window-stream V transpose tensor
        :type mCLT_win: cute.Tensor
        :param mCLT_cmp: Compressed-stream V transpose tensor
        :type mCLT_cmp: cute.Tensor
        :param mWindowIndices: Token-level absolute window-row indices
        :type mWindowIndices: cute.Tensor
        :param mCmpIndices: Token-level absolute compressed-row indices
        :type mCmpIndices: cute.Tensor
        :param mO: Output tensor
        :type mO: cute.Tensor
        :param mLSE: Log-sum-exp tensor
        :type mLSE: cute.Tensor
        :param mAccO: Intermediate accumulator output tensor
        :type mAccO: cute.Tensor
        :param mAccLSE: Intermediate accumulator log-sum-exp tensor
        :type mAccLSE: cute.Tensor
        :param split_kv: The split_kv parameter
        :type split_kv: cutlass.Int32
        :param cache_seqs: The variable sequence length tensor
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: The per-block split_kv values tensor
        :type block_split_kvs: cute.Tensor
        :param sparse_mla_topk_lens: Per-query valid sparse HCA length tensor
        :type sparse_mla_topk_lens: cute.Tensor
        :param window_valid_lens: Per-query valid sliding-window length tensor
        :type window_valid_lens: cute.Tensor
        :param softmax_scale_log2: The log2 scale factor for softmax
        :type softmax_scale_log2: cutlass.Float32
        :param output_scale: The scale factor for the output
        :type output_scale: cutlass.Float32
        :param q_latent_smem_layout_staged: Shared memory layout for query tensor
        :type q_latent_smem_layout_staged: cute.ComposedLayout
        :param kc_latent_smem_layout_staged: Shared memory layout for key tensor
        :type kc_latent_smem_layout_staged: cute.ComposedLayout
        :param p_smem_layout_staged: Shared memory layout for probability matrix
        :type p_smem_layout_staged: cute.ComposedLayout
        :param vc_smem_layout_staged: Shared memory layout for value tensor
        :type vc_smem_layout_staged: cute.ComposedLayout
        :param cta_layout_vmnk: Layout for compute threads
        :type cta_layout_vmnk: cute.Layout
        :param tile_sched_params: Scheduling parameters for work distribution
        :type tile_sched_params: HCAStaticTileSchedulerParams
        :param SharedStorage: Shared storage for the kernel
        :type SharedStorage: cutlass.Constexpr
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # Prefetch tma descriptor
        if warp_idx == self.mma_qk_warp_id:
            cpasync.prefetch_descriptor(tma_atom_q_latent)

        # Alloc
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_ptr_sync_bar,
            allocator_warp_id=self.mma_pv_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch=self.arch_str,
        )

        load_q_pipeline = self.make_and_init_load_qkv_pipeline(
            storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            self.load_q_stage,
            self.tma_copy_q_bytes,
        )
        # K/V stages are produced by the dequant warps (plain stores), so
        # these pipelines are thread-produced / UMMA-consumed.
        load_k_pipeline = self.make_and_init_kv_pipeline(
            storage.load_k_mbar_ptr.data_ptr(), cta_layout_vmnk, self.load_k_stage
        )
        load_v_pipeline = self.make_and_init_kv_pipeline(
            storage.load_v_mbar_ptr.data_ptr(), cta_layout_vmnk, self.load_v_stage
        )
        mma_s_pipeline = self.make_and_init_mma_s_pipeline(
            storage.mma_s_mbar_ptr.data_ptr(), cta_layout_vmnk
        )
        p_mma_pipeline = self.make_and_init_p_mma_pipeline(
            storage.p_mma_mbar_ptr.data_ptr(), cta_layout_vmnk
        )
        p_cor_pipeline = self.make_and_init_p_cor_pipeline(
            storage.p_cor_mbar_ptr.data_ptr()
        )
        mma_o_pipeline = self.make_and_init_mma_o_pipeline(
            storage.mma_o_mbar_ptr.data_ptr(), cta_layout_vmnk
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk, is_relaxed=True)

        # Generate smem tensor Q/KC/VC/exchange
        # (MMA, MMA_H, MMA_R, PIPE)
        sQ = storage.smem_q_latent.get_tensor(
            q_latent_smem_layout_staged.outer, swizzle=q_latent_smem_layout_staged.inner
        )
        # (MMA, MMA_K, MMA_R, PIPE)
        sKC = storage.smem_kc_latent.get_tensor(
            kc_latent_smem_layout_staged.outer,
            swizzle=kc_latent_smem_layout_staged.inner,
        )
        # Raw element pointers for the dequant writers (K-major / MN-major
        # SW128 tiles, addressed with explicit swizzle math).
        sKC_base = storage.smem_kc_latent.data_ptr()
        # (MMA, MMA_D, MMA_K, PIPE)
        if cutlass.const_expr(self.use_2cta_instrs):
            sVC = storage.smem_vc.get_tensor(
                vc_smem_layout_staged.outer, swizzle=vc_smem_layout_staged.inner
            )
            sVC_base = storage.smem_vc.data_ptr()
        else:
            # M64 single CTA: QK's K-major (64 keys, 128 latent) tile and
            # PV's MN-major (128 latent, 64 keys) tile use the same SW128
            # bytes. Both consumers retain the stage until PV completes.
            sVC = storage.smem_kc_latent.get_tensor(
                vc_smem_layout_staged.outer, swizzle=vc_smem_layout_staged.inner
            )
            sVC_base = storage.smem_kc_latent.data_ptr()
        sSlots = storage.smem_slots.get_tensor(
            cute.make_layout(self.window_len + self.max_topk)
        )
        # (MMA, MMA_H, MMA_K)
        sP = storage.smem_p.get_tensor(
            p_smem_layout_staged.outer, swizzle=p_smem_layout_staged.inner
        )
        # (compute_threads,) doubled for 2-softmax.
        softmax_smem_exchange = storage.softmax_smem_exchange.get_tensor(
            cute.make_layout(2 * self.num_compute_warps * self.threads_per_warp)
        )
        epilogue_smem_exchange = storage.epilogue_smem_exchange.get_tensor(
            cute.make_layout(self.num_compute_warps * self.threads_per_warp)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk)

        sMeta = None
        if cutlass.const_expr(not self.use_2cta_instrs):
            sMeta = storage.smem_metadata.get_tensor(
                cute.make_layout((128, 4, 2), stride=(1, 128, 512))
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Load warps, including window indices, compressed page table, and data
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_tma_k_warp_id:
            prims.setmaxregister(
                self.other_reg_num, prims.SetMaxRegisterAction.DECREASE
            )
            load_q_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_q_stage
            )
            tile_sched = create_hca_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, _ = self.get_k_tile_count(
                    split_kv,
                    cache_seqs,
                    block_split_kvs,
                    blk_coord,
                )
                if k_tile_count > 0:
                    tma_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        load_q_pipeline=load_q_pipeline,
                    )
                    tma_qk_params = SimpleNamespace(
                        tiled_mma_qk=tiled_mma_qk,
                        tma_atom_q_latent=tma_atom_q_latent,
                        mQL=mQL,
                        sQ=sQ,
                    )
                    load_q_producer_state = self.load_tma_q(
                        tma_common_params, tma_qk_params, load_q_producer_state
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            load_q_pipeline.producer_tail(load_q_producer_state)

        if warp_idx == self.idle_warp_id:
            prims.setmaxregister(self.idle_reg_num, prims.SetMaxRegisterAction.DECREASE)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Dequant warps (W15-W22): raw FP8/FP4 rows -> BF16 K and V tiles.
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.dequant_warp_ids[0]
            and warp_idx <= self.dequant_warp_ids[-1]
        ):
            prims.setmaxregister(
                self.dequant_reg_num, prims.SetMaxRegisterAction.DECREASE
            )
            load_k_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_k_stage
            )
            load_v_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_v_stage
            )
            tile_sched = create_hca_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, _ = self.get_k_tile_count(
                    split_kv,
                    cache_seqs,
                    block_split_kvs,
                    blk_coord,
                )
                if k_tile_count > 0:
                    tidx_g = tidx - self.dequant_warp_ids[0] * self.threads_per_warp
                    table_row = self.get_page_table_row(blk_coord)
                    n_tile = self.mma_qk_tiler[1]
                    # One cooperative gather of this split's page-table slots into
                    # SMEM; the per-tile dequant then resolves rows with an LDS
                    # instead of a dependent global round trip.
                    prims.barrier_cta_sync(
                        self.dequant_sync_bar.barrier_id,
                        thread_count=self.dequant_sync_bar.num_threads,
                    )
                    for key in cutlass.range(
                        k_index * n_tile + tidx_g,
                        (k_index + k_tile_count) * n_tile,
                        self.threads_per_warp * len(self.dequant_warp_ids),
                    ):
                        slot = cutlass.Int32(-1)
                        bound = cutlass.Int32(0)
                        if key < self.window_len:
                            slot = mWindowIndices[key, table_row]
                            bound = mWinPool.shape[0] * self.window_page_size
                        else:
                            slot = mCmpIndices[key - self.window_len, table_row]
                            bound = mCmpPool.shape[0] * self.compressed_page_size
                        sSlots[key] = cutlass.select_(
                            (slot >= 0) & (slot < bound), slot, cutlass.Int32(-1)
                        )
                    prims.barrier_cta_sync(
                        self.dequant_sync_bar.barrier_id,
                        thread_count=self.dequant_sync_bar.num_threads,
                    )
                    dq_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        tidx_g=tidx_g,
                        sSlots=sSlots,
                        cta_in_cluster=blk_coord[0] % self.cluster_shape_mnk[0],
                        table_row=table_row,
                        load_k_pipeline=load_k_pipeline,
                        load_v_pipeline=load_v_pipeline,
                        mWinPool=mWinPool,
                        mCmpPool=mCmpPool,
                        mWindowIndices=mWindowIndices,
                        mCmpIndices=mCmpIndices,
                        sKC_base=sKC_base,
                        sVC_base=sVC_base,
                        mDebugKV=mDebugKV,
                        bidx=bidx,
                    )
                    while k_tile_count > 0:
                        load_k_producer_state, load_v_producer_state = (
                            self.dequant_tile(
                                dq_params,
                                k_index,
                                load_k_producer_state,
                                load_v_producer_state,
                            )
                        )
                        k_index += 1
                        k_tile_count -= 1
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            load_k_pipeline.producer_tail(load_k_producer_state)
            load_v_pipeline.producer_tail(load_v_producer_state)

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA-QK warp (W8): issues all Q*K^T MMA and produces S.
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_qk_warp_id:
            prims.setmaxregister(
                self.other_reg_num, prims.SetMaxRegisterAction.DECREASE
            )
            prims.barrier_cta_sync(
                self.tmem_ptr_sync_bar.barrier_id,
                thread_count=self.tmem_ptr_sync_bar.num_threads,
            )
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            load_q_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_q_stage
            )
            load_k_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_k_stage
            )
            mma_s_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_s_stage
            )
            tile_sched = create_hca_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv, cache_seqs, block_split_kvs, blk_coord
                )
                if k_tile_count > 0:
                    mma_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        local_split_kv=local_split_kv,
                        load_q_pipeline=load_q_pipeline,
                        load_k_pipeline=load_k_pipeline,
                        tmem_ptr=tmem_ptr,
                        sMeta=sMeta,
                        is_leader_cta=is_leader_cta,
                        L=self.latent_dim,
                    )
                    mma_qk_params = SimpleNamespace(
                        mma_s_pipeline=mma_s_pipeline,
                        sQ=sQ,
                        sKC=sKC,
                    )
                    (
                        tiled_mma_qk,
                        load_q_consumer_state,
                        load_k_consumer_state,
                        mma_s_producer_state,
                    ) = self.mma_qk_warp_body(
                        mma_common_params,
                        mma_qk_params,
                        k_tile_count,
                        tiled_mma_qk,
                        load_q_consumer_state,
                        load_k_consumer_state,
                        mma_s_producer_state,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mma_s_pipeline.producer_tail(mma_s_producer_state)

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA-PV warp (W11): owns TMEM lifetime and issues all P*V MMA.
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_pv_warp_id:
            prims.setmaxregister(
                self.other_reg_num, prims.SetMaxRegisterAction.DECREASE
            )
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols(self.arch_str))
            prims.barrier_cta_sync(
                self.tmem_ptr_sync_bar.barrier_id,
                thread_count=self.tmem_ptr_sync_bar.num_threads,
            )
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            load_v_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_v_stage
            )
            p_mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.p_mma_stage
            )
            mma_o_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_o_stage
            )
            tile_sched = create_hca_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv, cache_seqs, block_split_kvs, blk_coord
                )
                if k_tile_count > 0:
                    mma_pv_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        local_split_kv=local_split_kv,
                        load_v_pipeline=load_v_pipeline,
                        tmem_ptr=tmem_ptr,
                        sMeta=sMeta,
                        is_leader_cta=is_leader_cta,
                        L=self.latent_dim,
                    )
                    mma_pv_params = SimpleNamespace(
                        p_mma_pipeline=p_mma_pipeline,
                        mma_o_pipeline=mma_o_pipeline,
                        sP=sP,
                        sVC=sVC,
                    )
                    (
                        tiled_mma_pv,
                        load_v_consumer_state,
                        p_mma_consumer_state,
                        mma_o_producer_state,
                    ) = self.mma_pv_warp_body(
                        mma_pv_common_params,
                        mma_pv_params,
                        k_tile_count,
                        tiled_mma_pv,
                        load_v_consumer_state,
                        p_mma_consumer_state,
                        mma_o_producer_state,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mma_o_pipeline.producer_tail(mma_o_producer_state)
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Compute warp
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.compute_warp_ids[0]
            and warp_idx <= self.compute_warp_ids[-1]
        ):
            prims.setmaxregister(
                self.softmax_reg_num, prims.SetMaxRegisterAction.INCREASE
            )
            mma_s_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_s_stage
            )
            p_mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.p_mma_stage
            )
            p_cor_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.p_cor_stage
            )
            mma_o_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_o_stage
            )
            prims.barrier_cta_sync(
                self.tmem_ptr_sync_bar.barrier_id,
                thread_count=self.tmem_ptr_sync_bar.num_threads,
            )
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            tile_sched = create_hca_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv, cache_seqs, block_split_kvs, blk_coord
                )
                if k_tile_count > 0:
                    compute_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        split_kv=split_kv,
                        local_split_kv=local_split_kv,
                        smem_exchange=softmax_smem_exchange,
                        sSlots=sSlots,
                        mAccO=mAccO,
                        mO=mO,
                        k_tile_total=cute.ceil_div(
                            cache_seqs[blk_coord[2]], self.mma_qk_tiler[1]
                        ),
                        K_valid=self.get_effective_hca_k(
                            sparse_mla_topk_lens, blk_coord
                        ),
                        window_valid_len=self.get_window_valid_len(
                            window_valid_lens, blk_coord
                        ),
                        L=self.latent_dim,
                        tmem_ptr=tmem_ptr,
                        sMeta=sMeta,
                        tidx=tidx,
                        p_cor_pipeline=p_cor_pipeline,
                        attn_sink_unscaled=attn_sink_unscaled,
                    )
                    compute_softmax_params = SimpleNamespace(
                        tiled_mma_qk=tiled_mma_qk,
                        sP=sP,
                        mma_s_pipeline=mma_s_pipeline,
                        p_mma_pipeline=p_mma_pipeline,
                        softmax_scale_log2=softmax_scale_log2,
                    )
                    mma_s_consumer_state, p_mma_producer_state, p_cor_producer_state = (
                        self.compute(
                            compute_common_params,
                            compute_softmax_params,
                            k_index=k_index,
                            k_tile_count=k_tile_count,
                            mma_s_consumer_state=mma_s_consumer_state,
                            p_mma_producer_state=p_mma_producer_state,
                            p_cor_producer_state=p_cor_producer_state,
                            is_second_compute_warp=False,
                        )
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Compute warp - second group (g1, warps 8-11, odd k-tiles).
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.second_compute_warp_ids[0]
            and warp_idx <= self.second_compute_warp_ids[-1]
        ):
            prims.setmaxregister(
                self.softmax_reg_num, prims.SetMaxRegisterAction.INCREASE
            )
            mma_s_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_s_stage
            )
            p_mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.p_mma_stage
            )
            p_cor_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.p_cor_stage
            )
            mma_o_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_o_stage
            )
            prims.barrier_cta_sync(
                self.tmem_ptr_sync_bar.barrier_id,
                thread_count=self.tmem_ptr_sync_bar.num_threads,
            )
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            tile_sched = create_hca_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            mma_s_consumer_state.advance()
            p_mma_producer_state.advance()
            p_cor_producer_state.advance()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv, cache_seqs, block_split_kvs, blk_coord
                )
                if k_tile_count > 0:
                    compute_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        split_kv=split_kv,
                        local_split_kv=local_split_kv,
                        smem_exchange=softmax_smem_exchange,
                        sSlots=sSlots,
                        mAccO=mAccO,
                        mO=mO,
                        k_tile_total=cute.ceil_div(
                            cache_seqs[blk_coord[2]], self.mma_qk_tiler[1]
                        ),
                        K_valid=self.get_effective_hca_k(
                            sparse_mla_topk_lens, blk_coord
                        ),
                        window_valid_len=self.get_window_valid_len(
                            window_valid_lens, blk_coord
                        ),
                        L=self.latent_dim,
                        tmem_ptr=tmem_ptr,
                        sMeta=sMeta,
                        tidx=tidx,
                        p_cor_pipeline=p_cor_pipeline,
                        attn_sink_unscaled=attn_sink_unscaled,
                    )
                    compute_softmax_params = SimpleNamespace(
                        tiled_mma_qk=tiled_mma_qk,
                        sP=sP,
                        mma_s_pipeline=mma_s_pipeline,
                        p_mma_pipeline=p_mma_pipeline,
                        softmax_scale_log2=softmax_scale_log2,
                    )
                    mma_s_consumer_state, p_mma_producer_state, p_cor_producer_state = (
                        self.compute(
                            compute_common_params,
                            compute_softmax_params,
                            k_index=k_index,
                            k_tile_count=k_tile_count,
                            mma_s_consumer_state=mma_s_consumer_state,
                            p_mma_producer_state=p_mma_producer_state,
                            p_cor_producer_state=p_cor_producer_state,
                            is_second_compute_warp=True,
                        )
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction warp
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.correction_warp_ids[0]
            and warp_idx <= self.correction_warp_ids[-1]
        ):
            prims.setmaxregister(
                self.correction_reg_num, prims.SetMaxRegisterAction.INCREASE
            )
            p_cor_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.p_cor_stage
            )
            mma_o_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_o_stage
            )
            # sync with mma warp before retrieving tmem ptr
            prims.barrier_cta_sync(
                self.tmem_ptr_sync_bar.barrier_id,
                thread_count=self.tmem_ptr_sync_bar.num_threads,
            )

            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            tile_sched = create_hca_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv, cache_seqs, block_split_kvs, blk_coord
                )
                if k_tile_count > 0:
                    compute_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        split_kv=split_kv,
                        local_split_kv=local_split_kv,
                        smem_exchange=epilogue_smem_exchange,
                        attn_sink=attn_sink_unscaled,
                        mAccO=mAccO,
                        mO=mO,
                        L=self.latent_dim,
                        H=mQL.shape[0],
                        tmem_ptr=tmem_ptr,
                        sMeta=sMeta,
                        tidx=tidx,
                        tiled_mma_pv=tiled_mma_pv,
                        p_cor_pipeline=p_cor_pipeline,
                        mma_o_pipeline=mma_o_pipeline,
                    )
                    compute_epilogue_params = SimpleNamespace(
                        output_scale=output_scale,
                        softmax_scale_log2=softmax_scale_log2,
                        mAccLSE=mAccLSE,
                        mLSE=mLSE,
                    )
                    p_cor_consumer_state, mma_o_consumer_state = self.correction(
                        compute_common_params,
                        compute_epilogue_params,
                        k_tile_count=k_tile_count,
                        p_cor_consumer_state=p_cor_consumer_state,
                        mma_o_consumer_state=mma_o_consumer_state,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        if cutlass.const_expr(self.use_2cta_instrs):
            # A CTA must stay alive until its peer has finished all DSMEM
            # V stores, including tiles whose query heads are padding.
            prims.barrier_cluster_arrive()
            prims.barrier_cluster_wait()
        return

    @cute.kernel
    def reduction_kernel(
        self,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mAccO: cute.Tensor,
        mAccLSE: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        attn_sink: cute.Tensor,
        vector_io: cutlass.Constexpr,
    ):
        """The reduction kernel for Heavily Compressed Attention (HCA) that combines intermediate results
        from multiple split_kv blocks into final outputs.

        :param mO: Output tensor for storing final results
        :type mO: cute.Tensor
        :param mLSE: Log-sum-exp tensor for storing final LSE values
        :type mLSE: cute.Tensor
        :param mAccO: Accumulated output tensor from split_kv blocks
        :type mAccO: cute.Tensor
        :param mAccLSE: Accumulated LSE tensor from split_kv blocks
        :type mAccLSE: cute.Tensor
        :param split_kv: Number of split_kv blocks
        :type split_kv: cutlass.Int32
        :param cache_seqs: Cache sequence lengths tensor
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: Per-block split_kv values tensor (for variable split_kv)
        :type block_split_kvs: cute.Tensor
        :param vector_io: Compile contiguous four-element IO for the
            measured H64/S1 small-batch route. Split accumulation
            order and output/LSE arithmetic remain identical.
        :type vector_io: cutlass.Constexpr
        """
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        blk_coord = (bidx, bidy, bidz)
        local_split_kv = (
            block_split_kvs[blk_coord[2]] if self.is_var_split_kv else split_kv
        )
        k_tile_total = cute.ceil_div(cache_seqs[blk_coord[2]], self.mma_qk_tiler[1])
        k_tile_per_cta = cute.ceil_div(k_tile_total, local_split_kv)
        local_split_kv = cute.ceil_div(k_tile_total, k_tile_per_cta)

        # No split can contain less than one key tile. The slot-cache capacity
        # bounds key tiles, so round that bound to a warp instead of always
        # expanding all 256 split slots. Padding still contributes exact zeros.
        reduction_splits = min(
            MAX_SPLITS,
            (
                (self.window_len + self.max_topk + self.mma_qk_tiler[1] - 1)
                // self.mma_qk_tiler[1]
                + 31
            )
            // 32
            * 32,
        )
        # Alloc shared memory
        smem = utils.SmemAllocator()
        storage = smem.allocate(reduction_splits * self.acc_dtype.width // 8, 16)
        lse_scale_ptr = cute.recast_ptr(storage, dtype=self.acc_dtype)
        smem_lse_scale = cute.make_tensor(
            lse_scale_ptr, cute.make_layout(reduction_splits)
        )

        gLSE = mAccLSE[blk_coord[0], None, blk_coord[1], blk_coord[2]]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            # calculate the global lse and exp ^ (local_lse - global_lse)
            lse_per_thread = cute.ceil_div(reduction_splits, self.threads_per_warp)

            local_lse = cute.make_rmem_tensor(
                cute.make_layout(lse_per_thread), self.lse_dtype
            )
            lse_max = -self.lse_dtype.inf
            # find the max lse
            for i in cutlass.range_constexpr(lse_per_thread):
                split_kv_idx = tidx + i * self.threads_per_warp
                local_lse[i] = (
                    gLSE[split_kv_idx]
                    if cute.elem_less(split_kv_idx, local_split_kv)
                    else -self.lse_dtype.inf
                )
                # reduce the local lse
                lse_max = cute.arch.fmax(lse_max, local_lse[i])
            lse_max = cute.arch.warp_reduction_max(lse_max)
            lse_max = lse_max if lse_max != -self.lse_dtype.inf else 0.0
            # calculate sum_lse
            sum_lse = 0.0
            for i in cutlass.range_constexpr(lse_per_thread):
                sum_lse += cute.math.exp2(local_lse[i] - lse_max, fastmath=True)
            sum_lse = cute.arch.warp_reduction_sum(sum_lse)
            has_reduction_mass = sum_lse != self.lse_dtype(0.0) or sum_lse != sum_lse
            # calculate the global_lse
            global_lse = (
                lse_max + cute.math.log2(sum_lse, fastmath=True)
                if has_reduction_mass
                else -self.lse_dtype.inf
            )
            if tidx == 0:
                mLSE[blk_coord[0], blk_coord[1], blk_coord[2]] = global_lse / LOG2_E
            # Add one sink to the denominator, never to the returned LSE.
            sink_log2 = attn_sink[blk_coord[0]] * LOG2_E
            denominator_max = cute.arch.fmax(global_lse, sink_log2)
            denominator_max = (
                denominator_max if denominator_max != -self.lse_dtype.inf else 0.0
            )
            denominator_lse = denominator_max + cute.math.log2(
                cute.math.exp2(global_lse - denominator_max, fastmath=True)
                + cute.math.exp2(sink_log2 - denominator_max, fastmath=True),
                fastmath=True,
            )
            # store the scale to shared memory
            for i in cutlass.range_constexpr(lse_per_thread):
                split_kv_idx = tidx + i * self.threads_per_warp
                if cute.elem_less(split_kv_idx, local_split_kv):
                    smem_lse_scale[split_kv_idx] = (
                        cute.math.exp2(local_lse[i] - denominator_lse, fastmath=True)
                        if has_reduction_mass
                        else self.acc_dtype(0.0)
                    )

        pipeline.sync(barrier_id=4)

        elements_per_thread = cute.ceil_div(
            self.latent_dim, self.threads_per_warp * self.num_compute_warps
        )
        gAccO = mAccO[blk_coord[0], None, None, blk_coord[1], blk_coord[2]]
        rAccO = cute.make_rmem_tensor(
            cute.make_layout(elements_per_thread), self.acc_dtype
        )
        rO = cute.make_rmem_tensor(cute.make_layout(elements_per_thread), self.o_dtype)
        rAccO.fill(0.0)
        # Paired SM100 measurements favor vector IO for B1/B4 and the
        # strided lane mapping for larger batches. Both visit splits in
        # the same order and produce identical output/LSE bytes.
        if cutlass.const_expr(vector_io):
            # Four adjacent latent values per lane enable one aligned vector load
            # per split and one vector output store. Accumulate splits in the same
            # order and retain the same FP32 arithmetic and final BF16 conversion.
            for i in range(local_split_kv):
                src_ptr = cute.make_ptr(
                    self.acc_dtype,
                    gAccO.iterator.toint()
                    + (i * gAccO.stride[0] + tidx * elements_per_thread)
                    * (self.acc_dtype.width // 8),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                values = cute.make_tensor(
                    src_ptr, cute.make_layout(elements_per_thread)
                )
                rAccO.store(rAccO.load() + values.load() * smem_lse_scale[i])
            rO.store(rAccO.load().to(self.o_dtype))
            gOut = mO[blk_coord[0], None, blk_coord[1], blk_coord[2]]
            dst_ptr = cute.make_ptr(
                self.o_dtype,
                gOut.iterator.toint()
                + tidx * elements_per_thread * self.o_dtype.width // 8,
                cute.AddressSpace.gmem,
                assumed_align=8,
            )
            cute.make_tensor(dst_ptr, cute.make_layout(elements_per_thread)).store(
                rO.load()
            )
        else:
            for i in range(local_split_kv):
                for j in cutlass.range_constexpr(elements_per_thread):
                    element_idx = (
                        tidx + j * self.threads_per_warp * self.num_compute_warps
                    )
                    rAccO[j] += gAccO[i, element_idx] * smem_lse_scale[i]
            rO.store(rAccO.load().to(self.o_dtype))
            for j in cutlass.range_constexpr(elements_per_thread):
                element_idx = tidx + j * self.threads_per_warp * self.num_compute_warps
                mO[blk_coord[0], element_idx, blk_coord[1], blk_coord[2]] = rO[j]
        return

    @staticmethod
    def get_split_kv(
        B: int, S: int, K: int, mma_qk_tiler_mn: tuple, max_active_blocks: int
    ) -> int:
        """Get the proper split_kv value for the HCA kernel based on parameters.

        :param B: Batch size
        :type B: int
        :param S: Sequence length
        :type S: int
        :param K: Sequence length
        :type K: int
        :param mma_qk_tiler_mn: HCA QK tiling parameters
        :type mma_qk_tiler_mn: tuple
        :param max_active_blocks: Maximum number of active blocks
        :type max_active_blocks: int
        :return: Split_kv value
        :rtype: int
        """
        max_splits = ceil_div(K, mma_qk_tiler_mn[1])
        blocks_per_batch = max(1, max_active_blocks // B // (S * 2))
        split_heur = min(max_splits, blocks_per_batch)
        # {$nv-internal-release begin}
        # TODO: figure out the error of make_tile with dynamic int_tuple
        # {$nv-internal-release end}
        k_waves = ceil_div(max_splits, split_heur)
        split_wave_aware = ceil_div(max_splits, k_waves)
        max_split_kv = 32
        return min(split_wave_aware, max_split_kv)

    @cute.jit
    def get_effective_hca_k(
        self,
        sparse_mla_topk_lens: cute.Tensor,
        blk_coord: cute.Coord,
    ) -> cutlass.Int32:
        return sparse_mla_topk_lens[self.get_page_table_row(blk_coord)]

    @cute.jit
    def get_window_valid_len(
        self,
        window_valid_lens: cute.Tensor,
        blk_coord: cute.Coord,
    ) -> cutlass.Int32:
        return window_valid_lens[self.get_page_table_row(blk_coord)]

    @cute.jit
    def get_page_table_row(self, blk_coord: cute.Coord) -> cutlass.Int32:
        row = blk_coord[2]
        if cutlass.const_expr(self.is_causal):
            row = blk_coord[2] * self.seq_len_q + blk_coord[1]
        return row

    @cute.jit
    def get_k_tile_count(
        self,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        blk_coord: cute.Coord,
    ) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        """Get the current k_index, k_tile_count, and local split_kv value for the HCA kernel.

        :param split_kv: Split_kv value
        :type split_kv: cutlass.Int32
        :param cache_seqs: Cache sequence lengths tensor
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: Per-block split_kv values tensor
        :type block_split_kvs: cute.Tensor
        :param blk_coord: Block coordinate
        :type blk_coord: cute.Coord
        :return: k_index, k_tile_count, split_kv
        :rtype: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
        """
        K = cache_seqs[blk_coord[2]]
        if cutlass.const_expr(self.is_var_split_kv):
            split_kv = block_split_kvs[blk_coord[2]]

        k_tile_total = cute.ceil_div(K, self.mma_qk_tiler[1])
        # {$nv-internal-release begin}
        # TODO: figure out the error of make_tile with dynamic int_tuple
        # {$nv-internal-release end}
        k_tile_per_cta = cute.ceil_div(k_tile_total, split_kv)
        k_index = blk_coord[3] * k_tile_per_cta
        k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index)
        return k_index, k_tile_count, split_kv

    def make_and_init_kv_pipeline(
        self, mbar_ptr, cta_layout_vmnk, num_stages
    ) -> pipeline.PipelineAsyncUmma:
        """K/V stage pipeline: produced by the dequant warps of both CTAs,
        consumed by one UMMA-issuing warp."""
        producer_thread_size = (
            self.threads_per_warp
            * len(self.dequant_warp_ids)
            * self.cluster_shape_mnk[0]
        )
        producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, producer_thread_size
        )
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=mbar_ptr,
            num_stages=num_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    @cute.jit
    def load_tma_q(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        load_q_producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Load the Q tile of one work tile with TMA (single Q stage)."""
        mma_qk_tiler_mk = cute.select(self.mma_qk_tiler, mode=[0, 2])
        gQL = cute.flat_divide(qk_params.mQL, mma_qk_tiler_mk)
        thr_mma_qk = qk_params.tiled_mma_qk.get_slice(
            common_params.blk_coord[0] % cute.size(qk_params.tiled_mma_qk.thr_id)
        )
        tSgQL = thr_mma_qk.partition_A(gQL)
        tQsQ, tQLgQL_mkl = cpasync.tma_partition(
            qk_params.tma_atom_q_latent,
            0,
            cute.make_layout(1),
            cute.group_modes(qk_params.sQ, 0, 3),
            cute.group_modes(tSgQL, 0, 3),
        )
        tQLgQL = tQLgQL_mkl[
            None, None, None, common_params.blk_coord[1], common_params.blk_coord[2]
        ]
        load_q_pipeline = common_params.load_q_pipeline
        tma_bar_ptr = load_q_pipeline.producer_get_barrier(load_q_producer_state)
        load_q_pipeline.producer_acquire(load_q_producer_state)
        for i in cutlass.range_constexpr(self.iterations_qk_latent):
            cute.copy(
                qk_params.tma_atom_q_latent,
                tQLgQL[None, 0, i],
                tQsQ[None, (i, 0)],
                tma_bar_ptr=tma_bar_ptr,
            )
        load_q_producer_state.advance()
        return load_q_producer_state

    @cute.jit
    def _swz(self, e: cutlass.Int32) -> cutlass.Int32:
        # SW128 on byte addresses: the 16-byte lane (byte bits [4,7)) is XORed
        # with the row inside the 8-row atom (byte bits [7,10)). In 16-bit
        # element units that is bits [3,6) ^= bits [6,9). Verified against the
        # canonical smem tensor indexing (probe_smem_write.py).
        return e ^ (((e >> 6) & 7) << 3)

    @cute.jit
    def _e8m0_to_f32(self, byte: cutlass.Int32) -> cutlass.Float32:
        bits = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Int32)
        bits[0] = (byte & 255) << 23
        return cute.make_tensor(
            cute.recast_ptr(bits.iterator, dtype=cutlass.Float32), bits.layout
        )[0]

    @cute.jit
    def _e4m3_to_f32(self, byte: cutlass.Int32) -> cutlass.Float32:
        b = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Uint8)
        b[0] = byte.to(cutlass.Uint8)
        return cute.make_tensor(
            cute.recast_ptr(b.iterator, dtype=cutlass.Float8E4M3FN), b.layout
        )[0].to(cutlass.Float32)

    @cute.jit
    def _load_chunk(self, pool: cute.Tensor, byte_off: cutlass.Int64):
        """One 16-byte vector load (ld.global.v4.b32) of raw cache bytes.

        The pool pointer plus a dynamic offset loses its alignment fact, so
        rebuild the pointer with assumed_align=16; without it the DSL emits
        four scalar loads."""
        ptr = cute.make_ptr(
            cutlass.Int32,
            pool.iterator.toint() + byte_off,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        return hw.load_cache_vector(ptr)

    @cute.jit
    def _load_u16(self, pool: cute.Tensor, byte_off: cutlass.Int64) -> cutlass.Int32:
        ptr = cute.make_ptr(
            cutlass.Uint16,
            pool.iterator.toint() + byte_off,
            cute.AddressSpace.gmem,
            assumed_align=2,
        )
        return hw.load_cache_scale(ptr, cutlass.Uint16)

    @cute.jit
    def _load_u8(self, pool: cute.Tensor, byte_off: cutlass.Int64) -> cutlass.Int32:
        return hw.load_cache_scale(pool.iterator + byte_off, cutlass.Uint8)

    @cute.jit
    def _fp4x8_to_f32x8(self, word: cutlass.Int32):
        """8 packed E2M1 (low nibble first) -> 8 f32 via cvt.rn.f16x2.e2m1x2."""
        w = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Int32)
        w[0] = word
        v = cute.make_tensor(
            cute.recast_ptr(w.iterator, dtype=cutlass.Float4E2M1FN),
            cute.make_layout(8),
        ).load()
        f16 = _nvw.cvt_f4e2m1x8_to_f16x8(v)
        return cute.TensorSSA(f16, (8,), cutlass.Float16).to(cutlass.Float32)

    @cute.jit
    def _to_bf16_frag(self, vals):
        frag = cute.make_rmem_tensor(cute.make_layout(8), cutlass.BFloat16)
        frag.store(vals.to(cutlass.BFloat16))
        return frag

    @cute.jit
    def _smem_dst(self, base_ptr, elem_off: cutlass.Int32):
        # The swizzle XOR hides the 16-byte alignment from the DSL, which then
        # emits st.shared.b16; rebuild the address with the alignment fact so
        # each copy is one STS.128.
        return cute.make_ptr(
            cutlass.BFloat16,
            base_ptr.toint() + self._swz(elem_off) * 2,
            cute.AddressSpace.smem,
            assumed_align=16,
        )

    @cute.jit
    def _store_frag(self, frag, base_ptr, elem_off: cutlass.Int32) -> None:
        hw.store_bf16_fragment(frag, self._smem_dst(base_ptr, elem_off))

    @cute.jit
    def _store_frag_dsmem(
        self, frag, base_ptr, elem_off: cutlass.Int32, rank: cutlass.Int32
    ) -> None:
        """Same store into the tile of cluster CTA `rank` (may be this CTA)
        through the shared::cluster window."""
        if cutlass.const_expr(self.use_2cta_instrs):
            dst = cute.arch.map_dsmem_ptr(self._smem_dst(base_ptr, elem_off), rank)
            cute.autovec_copy(frag, cute.make_tensor(dst, cute.make_layout(8)))

    @cute.jit
    def _store_fp4_chunk(
        self,
        chunk,
        sc_word: cutlass.Int32,
        k_base,
        e0k: cutlass.Int32,
        v_base,
        e0v: cutlass.Int32,
        owner: cutlass.Int32,
    ) -> None:
        """32 E2M1 elements (two 16-groups; E4M3 scale bytes in sc_word[7:0],
        [15:8]) -> four 8-element BF16 fragments, each written to this CTA's K
        tile and to the V tile of CTA `owner`."""
        s0 = self._e4m3_to_f32(sc_word & 255)
        s1 = self._e4m3_to_f32((sc_word >> 8) & 255)
        for q in cutlass.range_constexpr(4):
            if cutlass.const_expr(q < 2):
                sc = s0
            else:
                sc = s1
            frag = self._to_bf16_frag(self._fp4x8_to_f32x8(chunk[q]) * sc)
            self._store_frag(frag, k_base, e0k + q * 8)
            self._store_frag_dsmem(frag, v_base, e0v + q * 8, owner)

    @cute.jit
    def _store_fp8_chunk(
        self,
        chunk,
        sc_byte: cutlass.Int32,
        k_base,
        e0k: cutlass.Int32,
        v_base,
        e0v: cutlass.Int32,
        owner: cutlass.Int32,
    ) -> None:
        """16 E4M3 elements with one E8M0 scale -> two 8-element fragments."""
        sf = self._e8m0_to_f32(sc_byte)
        r8 = cute.make_tensor(
            cute.recast_ptr(chunk.iterator, dtype=cutlass.Float8E4M3FN),
            cute.make_layout(16),
        )
        for h in cutlass.range_constexpr(2):
            vals = (
                cute.make_tensor(r8.iterator + 8 * h, cute.make_layout(8))
                .load()
                .to(cutlass.Float32)
            )
            frag = self._to_bf16_frag(vals * sf)
            self._store_frag(frag, k_base, e0k + 8 * h)
            self._store_frag_dsmem(frag, v_base, e0v + 8 * h, owner)

    @cute.jit
    def _lane_map(self, l: cutlass.Int32, rpi: cutlass.Constexpr):
        """Lane -> (row within the instruction's row group, chunk within the
        instruction's chunk group). rpi rows x (32 // rpi) consecutive 16-byte
        chunks per warp instruction: the global read covers 32//rpi * 16
        contiguous bytes per row, and each 8-lane STS.128 phase touches either
        8 consecutive swizzle rows (rpi = 8) or 4 rows x 2 chunks of different
        swizzle quads (rpi = 4), so the SMEM writes are conflict-free."""
        return l % rpi, l // rpi

    @cute.jit
    def _row_offsets(
        self,
        slots: cute.Tensor,
        key: cutlass.Int32,
        table_row: cutlass.Int32,
        page_bytes: cutlass.Int32,
        page_size: cutlass.Constexpr,
        row_bytes: cutlass.Constexpr,
        scale_bytes: cutlass.Constexpr,
    ):
        slot = slots[key]
        page = cutlass.Int64(slot // page_size)
        local = slot % page_size
        data_off = cutlass.select_(
            slot >= 0, page * page_bytes + local * row_bytes, cutlass.Int64(-1)
        )
        scale_off = page * page_bytes + page_size * row_bytes + local * scale_bytes
        return data_off, scale_off

    @cute.jit
    def _dequant_rows(
        self,
        dq: SimpleNamespace,
        pool: cute.Tensor,
        slots: cute.Tensor,
        key_base: cutlass.Int32,
        k_base,
        v_base,
        page_size: cutlass.Constexpr,
        is_fp4: cutlass.Constexpr,
        load_k_producer_state: pipeline.PipelineState,
        load_v_producer_state: pipeline.PipelineState,
    ) -> None:
        """This CTA's n_cta keys x 512 latent of one key tile, gathered and
        converted once, written twice: into this CTA's K tile (K-major B
        operand of QK) and into the V tile of the CTA that owns the latent
        range (MN-major B operand of PV), the peer's through the cluster
        shared-memory window. So each element of the tile is converted once
        per cluster instead of once for K and once for V.

        V placement (verified element-wise against the TMA-written tile of the
        original kernel): grouped by PV K-block of kpk keys, then output slice
        j, then 64-latent block, then key row, then latent. All loads of a
        phase are issued before the first conversion."""
        w = dq.tidx_g // self.threads_per_warp
        l = dq.tidx_g % self.threads_per_warp
        cta = dq.cta_in_cluster
        n_cta = self.mma_qk_tiler[1] // self.cluster_shape_mnk[0]
        kpk = self.mma_pv_tiler[2]
        rows_per_warp = n_cta // len(self.dequant_warp_ids)
        # Four rows give eight lanes (128 contiguous bytes) per row.
        # M64 processes two row groups; stores stay SW128 conflict-free.
        rpi = min(4, rows_per_warp)
        octets = rows_per_warp // rpi
        lpr = self.threads_per_warp // rpi
        chunk_elems = 32 if is_fp4 else 16
        cpb = 64 // chunk_elems  # chunks per 64-latent swizzle row
        cpo = 128 // chunk_elems  # chunks per owner segment
        cps = 256 // chunk_elems  # chunks per output slice
        row_bytes = self.latent_dim // (2 if is_fp4 else 1)
        scale_bytes = self.latent_dim // 16 if is_fp4 else self.latent_dim // 32
        groups = (row_bytes // 16) // lpr
        iters = octets * groups
        page_bytes = pool.shape[1]
        pc = min(self.dequant_phase_chunks, iters)
        row, lane_chunk = self._lane_map(l, rpi)
        offs = cute.make_rmem_tensor(cute.make_layout(2 * octets), cutlass.Int64)
        for o in cutlass.range_constexpr(octets):
            n = w * rows_per_warp + o * rpi + row
            d, sc = self._row_offsets(
                slots,
                key_base + cta * n_cta + n,
                dq.table_row,
                page_bytes,
                page_size,
                row_bytes,
                scale_bytes,
            )
            offs[2 * o] = d
            offs[2 * o + 1] = sc
        for ph in cutlass.range_constexpr(iters // pc):
            chunks = cute.make_rmem_tensor(cute.make_layout(4 * pc), cutlass.Int32)
            scs = cute.make_rmem_tensor(cute.make_layout(pc), cutlass.Int32)
            for ii in cutlass.range_constexpr(pc):
                i = ph * pc + ii
                cq = (i // octets) * lpr + lane_chunk
                data_off = offs[2 * (i % octets)]
                scale_off = offs[2 * (i % octets) + 1]
                # Invalid slots do not read a backing row, including its scales.
                for q4 in cutlass.range_constexpr(4):
                    chunks[4 * ii + q4] = 0
                scs[ii] = 0
                if data_off >= 0:
                    cute.make_tensor(
                        chunks.iterator + 4 * ii, cute.make_layout(4)
                    ).store(self._load_chunk(pool, data_off + cq * 16))
                    if cutlass.const_expr(is_fp4):
                        scs[ii] = self._load_u16(pool, scale_off + cq * 2)
                    else:
                        scs[ii] = self._load_u8(pool, scale_off + cq // 2)
            # Raw registers do not alias the K/V stage. Issue the existing
            # full-phase loads while the prior tile can still own that stage,
            # then wait before the first conversion/store. There is no new
            # register buffer, no new stage, and no early shared-memory write.
            if cutlass.const_expr(ph == 0 and self.prefetch_raw):
                dq.load_k_pipeline.producer_acquire(load_k_producer_state)
                dq.load_v_pipeline.producer_acquire(load_v_producer_state)
            for ii in cutlass.range_constexpr(pc):
                i = ph * pc + ii
                n = w * rows_per_warp + (i % octets) * rpi + row
                cq = (i // octets) * lpr + lane_chunk
                # Every term is a provable multiple of 8 elements (bit ops, no
                # '%'), otherwise the DSL scalarizes the 16-byte SMEM stores.
                blk64 = cq // cpb
                within = (cq & (cpb - 1)) * chunk_elems
                e0k = (
                    (blk64 // 2) * (n_cta * 128)
                    + (blk64 & 1) * (n_cta * 64)
                    + n * 64
                    + within
                )
                nt = cta * n_cta + n  # key row within the tile
                j = cq // cps
                owner = (cq // cpo) & 1
                e0v = (
                    (nt // kpk) * (kpk * 256)
                    + j * (kpk * 128)
                    + (blk64 & 1) * (kpk * 64)
                    + (nt & (kpk - 1)) * 64
                    + within
                )
                chunk = cute.make_tensor(chunks.iterator + 4 * ii, cute.make_layout(4))
                if cutlass.const_expr(self.dbg_mode == 2):
                    sink = cute.make_tensor(
                        cute.recast_ptr(k_base, dtype=cutlass.Int32) + dq.tidx_g,
                        cute.make_layout(1),
                    )
                    sink[0] = chunk[0] ^ chunk[1] ^ chunk[2] ^ chunk[3] ^ scs[ii]
                elif cutlass.const_expr(is_fp4):
                    self._store_fp4_chunk(
                        chunk, scs[ii], k_base, e0k, v_base, e0v, owner
                    )
                else:
                    self._store_fp8_chunk(
                        chunk, scs[ii], k_base, e0k, v_base, e0v, owner
                    )

    @cute.jit
    def dequant_tile(
        self,
        dq: SimpleNamespace,
        k_index: cutlass.Int32,
        load_k_producer_state: pipeline.PipelineState,
        load_v_producer_state: pipeline.PipelineState,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState]:
        """Produce this CTA's share of the BF16 K tile and V tile of one key
        tile (see _dequant_rows)."""
        # Per-stage tile sizes: K = this CTA's keys x 512; V = all keys of the
        # tile x this CTA's 256 latent columns.
        k_stage_elems = (
            self.mma_qk_tiler[1] // self.cluster_shape_mnk[0]
        ) * self.latent_dim
        v_stage_elems = self.mma_qk_tiler[1] * (
            self.latent_dim // self.cluster_shape_mnk[0]
        )
        n_tile = self.mma_qk_tiler[1]
        is_win = k_index < self.window_tiles
        key_base = k_index * n_tile  # absolute key: window keys, then compressed

        # Both stages of a tile are freed by the same UMMA completions in both
        # CTAs (tcgen05.commit multicasts to the pair), so once this CTA's
        # stage is free the peer's is too and the remote V writes are safe.
        # The selected M64 route acquires after its raw-load phase.
        # Keep the peer-CTA lifetime protocol and no-load diagnostic unchanged.
        if cutlass.const_expr(not self.prefetch_raw or self.dbg_mode == 1):
            dq.load_k_pipeline.producer_acquire(load_k_producer_state)
            dq.load_v_pipeline.producer_acquire(load_v_producer_state)
        # Rebuild the stage pointers with their alignment fact: a dynamic stage
        # index times the stage size loses it and the DSL then halves the
        # width of every SMEM store below.
        k_base = cute.make_ptr(
            self.k_dtype,
            dq.sKC_base.toint()
            + load_k_producer_state.index * (k_stage_elems * self.k_dtype.width // 8),
            cute.AddressSpace.smem,
            assumed_align=1024,
        )
        v_base = cute.make_ptr(
            self.v_dtype,
            dq.sVC_base.toint()
            + load_v_producer_state.index * (v_stage_elems * self.v_dtype.width // 8),
            cute.AddressSpace.smem,
            assumed_align=1024,
        )
        if cutlass.const_expr(self.dbg_mode == 1):
            pass
        elif is_win:
            self._dequant_rows(
                dq,
                dq.mWinPool,
                dq.sSlots,
                key_base,
                k_base,
                v_base,
                self.window_page_size,
                False,
                load_k_producer_state,
                load_v_producer_state,
            )
        else:
            self._dequant_rows(
                dq,
                dq.mCmpPool,
                dq.sSlots,
                key_base,
                k_base,
                v_base,
                self.compressed_page_size,
                self.compressed_is_fp4,
                load_k_producer_state,
                load_v_producer_state,
            )
        # Publish local writes to UMMA. Only the two-CTA path also writes
        # peer V tiles and needs the cluster-scope fence; M64 aliases K/V
        # locally, so a second fence would unnecessarily drain its pipeline.
        prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
        if cutlass.const_expr(self.use_2cta_instrs):
            prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cluster)
        if cutlass.const_expr(dq.mDebugKV is not None):
            # Debug: dump this CTA's K and V stages of tile 0 (first cluster) as
            # raw element order: mDebugKV[0/1 (K/V), cta*32768 + elem].
            # (diagnostic only; intended for a single-tile, single-cluster run)
            for i in cutlass.range_constexpr(256):
                e = dq.tidx_g * 256 + i
                kv = cute.make_tensor(k_base + e, cute.make_layout(1))
                vv = cute.make_tensor(v_base + e, cute.make_layout(1))
                dq.mDebugKV[0, dq.cta_in_cluster * 32768 + e] = kv[0]
                dq.mDebugKV[1, dq.cta_in_cluster * 32768 + e] = vv[0]
        dq.load_k_pipeline.producer_commit(load_k_producer_state)
        load_k_producer_state.advance()
        dq.load_v_pipeline.producer_commit(load_v_producer_state)
        load_v_producer_state.advance()
        return load_k_producer_state, load_v_producer_state

    @cute.jit
    def _score_fragment_layout(self, tiled_mma_qk: cute.TiledMma):
        shape = tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        fragment = tiled_mma_qk.make_fragment_C(cute.append(shape, self.mma_s_stage))
        layout = fragment.layout
        if cutlass.const_expr(not self.use_2cta_instrs):
            # M64 SS fragments otherwise pack stages into the unused 16
            # datapaths. Keep both score stages in the odd datapath groups;
            # the even groups hold the live output accumulator.
            layout = cute.make_layout(
                fragment.shape,
                stride=(*fragment.stride[:3], self.mma_qk_tiler[1]),
            )
        return layout

    @cute.jit
    def mma_qk_warp_body(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        k_tile_count: cutlass.Int32,
        tiled_mma_qk: cute.TiledMma,
        load_q_consumer_state: pipeline.PipelineState,
        load_k_consumer_state: pipeline.PipelineState,
        mma_s_producer_state: pipeline.PipelineState,
    ) -> tuple[
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        """QK-only MMA warp body for W8."""
        tSrQ = tiled_mma_qk.make_fragment_A(qk_params.sQ)
        tSrKC = tiled_mma_qk.make_fragment_B(qk_params.sKC)

        tStS_staged = cute.make_tensor(
            common_params.tmem_ptr + self.tmem_s_offset,
            self._score_fragment_layout(tiled_mma_qk),
        )

        qk_params.tSrQ = tSrQ
        qk_params.tSrKC = tSrKC
        qk_params.tStS_staged = tStS_staged

        load_q_pipeline = common_params.load_q_pipeline
        if common_params.is_leader_cta:
            load_q_release_state = load_q_consumer_state.clone()
            load_q_pipeline.consumer_wait(load_q_consumer_state)
            load_q_consumer_state.advance()
            while k_tile_count > 0:
                (
                    tiled_mma_qk,
                    load_q_consumer_state,
                    load_k_consumer_state,
                    mma_s_producer_state,
                ) = self.mma_qk(
                    common_params,
                    qk_params,
                    tiled_mma_qk,
                    load_q_consumer_state,
                    load_k_consumer_state,
                    mma_s_producer_state,
                    wait_q=False,
                )
                k_tile_count -= 1
            load_q_pipeline.consumer_release(load_q_release_state)
            load_q_release_state.advance()

        return (
            tiled_mma_qk,
            load_q_consumer_state,
            load_k_consumer_state,
            mma_s_producer_state,
        )

    @cute.jit
    def mma_pv_warp_body(
        self,
        common_params: SimpleNamespace,
        pv_params: SimpleNamespace,
        k_tile_count: cutlass.Int32,
        tiled_mma_pv: cute.TiledMma,
        load_v_consumer_state: pipeline.PipelineState,
        p_mma_consumer_state: pipeline.PipelineState,
        mma_o_producer_state: pipeline.PipelineState,
    ) -> tuple[
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        """PV-only MMA warp body for W11."""
        tOrP = tiled_mma_pv.make_fragment_A(pv_params.sP)
        tOrVC = tiled_mma_pv.make_fragment_B(pv_params.sVC)

        tOtO_shape = tiled_mma_pv.partition_shape_C(
            cute.select(self.mma_pv_tiler, mode=[0, 1])
        )
        tOtO = tiled_mma_pv.make_fragment_C(tOtO_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                common_params.L // self.mma_pv_tiler[1],
                stride=self.mma_pv_tiler[1]
                // (self.warps_in_n if self.use_2cta_instrs else 1),
            ),
        )
        tOtO_staged = cute.make_tensor(
            common_params.tmem_ptr + self.tmem_o_offset, tOtO_layout
        )

        pv_params.tOrP = tOrP
        pv_params.tOrVC = tOrVC
        pv_params.tOtO_staged = tOtO_staged

        tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, False)
        if common_params.is_leader_cta:
            while k_tile_count > 0:
                (
                    tiled_mma_pv,
                    load_v_consumer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                ) = self.mma_pv(
                    common_params,
                    pv_params,
                    tiled_mma_pv,
                    load_v_consumer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                )
                k_tile_count -= 1

        return (
            tiled_mma_pv,
            load_v_consumer_state,
            p_mma_consumer_state,
            mma_o_producer_state,
        )

    @cute.jit
    def mma_qk(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        tiled_mma_qk: cute.TiledMma,
        load_q_consumer_state: pipeline.PipelineState,
        load_k_consumer_state: pipeline.PipelineState,
        mma_s_producer_state: pipeline.PipelineState,
        wait_q: bool,
    ) -> tuple[
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        """Compute one k-tile of mma for Q*K^T. Updates the tiled MMA QK and pipeline states.

        :param qk_params: The qk parameters
        :type qk_params: SimpleNamespace
        :param tiled_mma_qk: The tiled mma qk
        :type tiled_mma_qk: cute.TiledMma
        :param load_q_consumer_state: The load q consumer state
        :type load_q_consumer_state: pipeline.PipelineState
        :param load_k_consumer_state: The load k consumer state
        :type load_k_consumer_state: pipeline.PipelineState
        :param mma_s_producer_state: The mma s producer state
        :type mma_s_producer_state: pipeline.PipelineState

        :return: The tiled mma qk, the load q consumer state, the load k consumer state, and the mma s producer state
        :rtype: tuple[cute.TiledMma, pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]
        """
        tStS = qk_params.tStS_staged[None, None, None, mma_s_producer_state.index]

        qk_params.mma_s_pipeline.producer_acquire(mma_s_producer_state)
        tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
        load_q_pipeline = common_params.load_q_pipeline
        load_k_pipeline = common_params.load_k_pipeline
        if cutlass.const_expr(wait_q):
            load_q_pipeline.consumer_wait(load_q_consumer_state)
        load_k_pipeline.consumer_wait(load_k_consumer_state)
        if cutlass.const_expr(self.use_primitives):
            hw.qk_mma(
                qk_params.sQ, qk_params.sKC, tStS.iterator, load_k_consumer_state.index
            )
            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
        else:
            for q_stage in range(self.iterations_qk_latent):
                kc_stage = load_k_consumer_state.index
                for k_block in cutlass.range_constexpr(
                    cute.size(qk_params.tSrQ.shape[2])
                ):
                    cute.gemm(
                        tiled_mma_qk,
                        tStS,
                        qk_params.tSrQ[None, None, k_block, (q_stage, 0)],
                        qk_params.tSrKC[None, None, k_block, (q_stage, kc_stage)],
                        tStS,
                    )
                    tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
        load_k_pipeline.consumer_release(load_k_consumer_state)
        load_k_consumer_state.advance()
        if cutlass.const_expr(wait_q):
            load_q_consumer_state.advance()

        qk_params.mma_s_pipeline.producer_commit(mma_s_producer_state)
        mma_s_producer_state.advance()
        return (
            tiled_mma_qk,
            load_q_consumer_state,
            load_k_consumer_state,
            mma_s_producer_state,
        )

    @cute.jit
    def mma_pv(
        self,
        common_params: SimpleNamespace,
        pv_params: SimpleNamespace,
        tiled_mma_pv: cute.TiledMma,
        load_v_consumer_state: pipeline.PipelineState,
        p_mma_consumer_state: pipeline.PipelineState,
        mma_o_producer_state: pipeline.PipelineState,
    ) -> tuple[
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        """Compute one k-tile of mma for P*V. Updates the tiled mma pv and pipeline states.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param pv_params: The pv parameters
        :type pv_params: SimpleNamespace
        :param tiled_mma_pv: The tiled mma pv
        :type tiled_mma_pv: cute.TiledMma
        :param load_v_consumer_state: The load v consumer state
        :type load_v_consumer_state: pipeline.PipelineState
        :param p_mma_consumer_state: The P MMA consumer state
        :type p_mma_consumer_state: pipeline.PipelineState
        :param mma_o_producer_state: The MMA o producer state
        :type mma_o_producer_state: pipeline.PipelineState

        :return: The tiled mma pv, the load v consumer state, the P MMA consumer state, and the MMA o producer state
        :rtype: tuple[cute.TiledMma, pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]
        """

        pv_params.p_mma_pipeline.consumer_wait(p_mma_consumer_state)
        load_v_pipeline = common_params.load_v_pipeline
        accumulate_flag = tiled_mma_pv.get(tcgen05.Field.ACCUMULATE)
        mma_o_pipeline = pv_params.mma_o_pipeline

        load_v_pipeline.consumer_wait(load_v_consumer_state)
        vc_stage = load_v_consumer_state.index
        for acc_stage in range(self.iterations_pv_n):
            mma_o_pipeline.producer_acquire(mma_o_producer_state)
            tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, accumulate_flag)
            if cutlass.const_expr(self.use_primitives):
                tOtO = pv_params.tOtO_staged[None, None, None, acc_stage]
                hw.pv_mma(
                    pv_params.sP,
                    pv_params.sVC,
                    tOtO.iterator,
                    p_mma_consumer_state.index,
                    vc_stage,
                    acc_stage,
                    accumulate_flag,
                )
                tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
            else:
                for p_stage in range(self.iterations_pv_k):
                    tOtO = pv_params.tOtO_staged[None, None, None, acc_stage]
                    for k_block in cutlass.range_constexpr(pv_params.tOrP.shape[2]):
                        cute.gemm(
                            tiled_mma_pv,
                            tOtO,
                            pv_params.tOrP[
                                None,
                                None,
                                k_block,
                                (p_stage, p_mma_consumer_state.index),
                            ],
                            pv_params.tOrVC[
                                None, None, k_block, ((acc_stage, p_stage), vc_stage)
                            ],
                            tOtO,
                        )
                        tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
            mma_o_pipeline.producer_commit(mma_o_producer_state)
            mma_o_producer_state.advance()
        load_v_pipeline.consumer_release(load_v_consumer_state)
        load_v_consumer_state.advance()
        pv_params.p_mma_pipeline.consumer_release(p_mma_consumer_state)
        p_mma_consumer_state.advance()

        return (
            tiled_mma_pv,
            load_v_consumer_state,
            p_mma_consumer_state,
            mma_o_producer_state,
        )

    @cute.jit
    def correction(
        self,
        common_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        k_tile_count: cutlass.Int32,
        p_cor_consumer_state: pipeline.PipelineState,
        mma_o_consumer_state: pipeline.PipelineState,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState]:
        """Compute warp to compute the result of softmax, rescale, and epilogue. Updates the related pipeline states.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param epilogue_params: The epilogue parameters
        :type epilogue_params: SimpleNamespace
        :param k_index: The index of the k-tile
        :type k_index: cutlass.Int32
        :param k_tile_count: The number of k-tiles
        :type k_tile_count: cutlass.Int32
        :param p_cor_consumer_state: The P correction consumer state
        :type p_cor_consumer_state: pipeline.PipelineState
        :param mma_o_consumer_state: The MMA o consumer state
        :type mma_o_consumer_state: pipeline.PipelineState

        :return: The P correction consumer state, and the MMA o consumer state
        :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState]
        """

        k_tile_count_init = k_tile_count
        row_sum = self.acc_dtype(0.0)
        row_max = self.acc_dtype(self.init_row_max)
        while k_tile_count > 0:
            p_cor_consumer_state, row_sum, row_max, correction_factor, no_correction = (
                self.get_correction_factor(
                    common_params, p_cor_consumer_state, k_tile_count == 1
                )
            )
            if k_tile_count_init != k_tile_count:
                mma_o_consumer_state = self.rescale(
                    common_params,
                    mma_o_consumer_state,
                    correction_factor,
                    no_correction,
                )
            k_tile_count = k_tile_count - 1

        # The caller guarantees at least one K tile, so row_sum and row_max
        # contain the final correction metadata here. Keeping the epilogue
        # outside the dynamic loop prevents its global LSE address from being
        # hoisted across every rescale iteration.
        mma_o_consumer_state = self.epilogue(
            common_params,
            epilogue_params,
            mma_o_consumer_state,
            row_sum,
            row_max,
        )
        return p_cor_consumer_state, mma_o_consumer_state

    @cute.jit
    def exchange_p_cor_metadata(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        correction_factor: cutlass.Float32,
        row_sum: cutlass.Float32,
        row_max: cutlass.Float32,
        row_max_new: cutlass.Float32,
        tAcc: cute.Tensor,
        tidx: cutlass.Int32,
        p_cor_producer_state: pipeline.PipelineState,
    ) -> tuple[pipeline.PipelineState, cutlass.Float32]:
        """Compute the correction factor for the last k tile."""
        if cutlass.const_expr(not self.use_2cta_instrs):
            no_correction = cutlass.Int32(0)
            if (
                row_max_new - row_max
            ) * softmax_params.softmax_scale_log2 <= self.skip_correction_threshold:
                no_correction = cutlass.Int32(1)
                row_max_new = row_max
            stage = p_cor_producer_state.index
            common_params.sMeta[tidx, 0, stage] = row_sum
            common_params.sMeta[tidx, 1, stage] = row_max_new
            common_params.sMeta[tidx, 2, stage] = correction_factor
            common_params.sMeta[tidx, 3, stage] = no_correction.to(cutlass.Float32)
            common_params.p_cor_pipeline.producer_commit(p_cor_producer_state)
            p_cor_producer_state.advance()
            return p_cor_producer_state, row_max_new
        else:
            no_correction = 0
            if (
                row_max_new - row_max
            ) * softmax_params.softmax_scale_log2 <= self.skip_correction_threshold:
                no_correction = 1
                row_max_new = row_max

            # pad for 4x32b
            corr_layout = cute.make_layout(
                (tAcc.shape[0], (4, tAcc.shape[1][1]), self.mma_s_stage),
                stride=(tAcc.stride[0], (1, tAcc.stride[1][1]), 4),
            )
            tCor = cute.make_tensor(
                common_params.tmem_ptr + self.correction_factor_offset,
                corr_layout,
            )
            cCor = cute.make_identity_tensor(tCor.shape)
            corr_tmem_store_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(4)), self.acc_dtype
            )
            corr_tmem_store_tiled_copy = tcgen05.make_tmem_copy(
                corr_tmem_store_atom, tCor
            )
            corr_tmem_store_thr_copy = corr_tmem_store_tiled_copy.get_slice(tidx)
            cCor_for_copy = corr_tmem_store_thr_copy.partition_S(cCor)
            tCor_for_copy = corr_tmem_store_thr_copy.partition_D(tCor)
            rCor = cute.make_fragment_like(
                cCor_for_copy[None, None, None, 0], self.acc_dtype
            )
            rCor_int = cute.make_tensor(
                cute.recast_ptr(rCor.iterator, dtype=cutlass.Int32), rCor.layout
            )
            rCor[0] = row_sum
            rCor[1] = row_max_new
            rCor[2] = correction_factor
            rCor_int[3] = no_correction

            cute.copy(
                corr_tmem_store_tiled_copy,
                rCor,
                tCor_for_copy[None, None, None, p_cor_producer_state.index],
            )
            # fence between tmem store and correction warp
            prims.tcgen05_wait("store")
            common_params.p_cor_pipeline.producer_commit(p_cor_producer_state)
            p_cor_producer_state.advance()
            return p_cor_producer_state, row_max_new

    @cute.jit
    def softmax_advance_to_next_group(
        self,
        common_params: SimpleNamespace,
        p_mma_producer_state: pipeline.PipelineState,
        mma_s_consumer_state: pipeline.PipelineState,
        p_cor_producer_state: pipeline.PipelineState,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]:
        p_mma_producer_state.advance()
        mma_s_consumer_state.advance()
        p_cor_producer_state.advance()
        common_params.p_cor_pipeline.producer_acquire(p_cor_producer_state)
        return p_mma_producer_state, mma_s_consumer_state, p_cor_producer_state

    @cute.jit
    def compute(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        mma_s_consumer_state: pipeline.PipelineState,
        p_mma_producer_state: pipeline.PipelineState,
        p_cor_producer_state: pipeline.PipelineState,
        is_second_compute_warp: bool,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]:
        k_tile_total = common_params.k_tile_total

        # Accumulate only real keys. Applying the sink at final normalization
        # avoids subtracting nearly equal masses to recover sink-exclusive LSE.
        sink_row_max = self.acc_dtype(self.init_row_max)
        sink_row_sum = self.acc_dtype(0.0)

        row_max = self.acc_dtype(self.init_row_max)
        row_sum = self.acc_dtype(0.0)
        correction_factor = self.acc_dtype(1)
        odd_k_tile = k_tile_count % 2 == 1
        if cutlass.const_expr(is_second_compute_warp):
            k_index = k_index + 1
            k_tile_count = k_tile_count // 2
        else:
            k_tile_count = (k_tile_count + 1) // 2
        # Only the first even tile uses the window-stream valid length. Keep
        # that value off the loop backedge; after the first iteration every
        # remaining tile is bounded by the compressed-stream valid length.
        tile_valid_len = common_params.K_valid
        if cutlass.const_expr(self.is_causal):
            # Window tiles (both compute groups own one when the window spans
            # two 64-key tiles) are bounded by the window valid length.
            if k_index < self.window_tiles:
                tile_valid_len = common_params.window_valid_len
        valid_k_tile_count = k_tile_count > 0
        prims.barrier_cta_sync(
            self.softmax_warps_initial_sync_bar.barrier_id,
            thread_count=self.softmax_warps_initial_sync_bar.num_threads,
        )
        common_params.p_cor_pipeline.producer_acquire(p_cor_producer_state)

        if cutlass.const_expr(is_second_compute_warp):
            self.init_p_cor_metadata(
                common_params,
                softmax_params,
                p_cor_producer_state,
                sink_row_max,
                sink_row_sum,
            )
            prims.barrier_cta_arrive(
                self.softmax_order_bar_0.barrier_id,
                self.softmax_order_bar_0.num_threads,
            )

        while k_tile_count > 0:
            is_global_last_tile = k_index == k_tile_total - 1
            apply_mask = is_global_last_tile
            if cutlass.const_expr(self.is_causal):
                apply_mask = cutlass.Boolean(True)
            is_local_last_tile = (
                k_tile_count == 1
                if cutlass.const_expr(common_params.mAccO is not None)
                else is_global_last_tile
            )
            (
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
            ) = self.softmax(
                common_params,
                softmax_params,
                k_index,
                tile_valid_len,
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
                is_second_compute_warp,
                apply_mask,
                is_local_last_tile,
            )
            tile_valid_len = common_params.K_valid
            k_index = k_index + 2
            k_tile_count = k_tile_count - 1
            if k_tile_count > 0:
                p_mma_producer_state, mma_s_consumer_state, p_cor_producer_state = (
                    self.softmax_advance_to_next_group(
                        common_params,
                        p_mma_producer_state,
                        mma_s_consumer_state,
                        p_cor_producer_state,
                    )
                )

        if odd_k_tile and valid_k_tile_count:
            if cutlass.const_expr(is_second_compute_warp):
                p_mma_producer_state.advance()
                mma_s_consumer_state.advance()
                p_cor_producer_state.advance()
                p_mma_producer_state.advance()
                mma_s_consumer_state.advance()
                p_cor_producer_state.advance()
        else:
            p_mma_producer_state.advance()
            mma_s_consumer_state.advance()
            p_cor_producer_state.advance()
        if cutlass.const_expr(is_second_compute_warp):
            if odd_k_tile:
                prims.barrier_cta_sync(
                    self.softmax_order_bar_1.barrier_id,
                    thread_count=self.softmax_order_bar_1.num_threads,
                )
        else:
            if not odd_k_tile:
                prims.barrier_cta_sync(
                    self.softmax_order_bar_0.barrier_id,
                    thread_count=self.softmax_order_bar_0.num_threads,
                )
        return mma_s_consumer_state, p_mma_producer_state, p_cor_producer_state

    @cute.jit
    def init_p_cor_metadata(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        p_cor_producer_state: pipeline.PipelineState,
        init_row_max: cutlass.Float32,
        init_row_sum: cutlass.Float32,
    ) -> None:
        if cutlass.const_expr(not self.use_2cta_instrs):
            tidx = common_params.tidx % 128
            stage = p_cor_producer_state.index
            common_params.sMeta[tidx, 0, stage] = init_row_sum
            common_params.sMeta[tidx, 1, stage] = init_row_max
            common_params.sMeta[tidx, 2, stage] = self.acc_dtype(1.0)
            common_params.sMeta[tidx, 3, stage] = self.acc_dtype(1.0)
        else:
            init_tidx = common_params.tidx % (
                self.num_compute_warps * self.threads_per_warp
            )
            init_tStS_shape = softmax_params.tiled_mma_qk.partition_shape_C(
                cute.select(self.mma_qk_tiler, mode=[0, 1])
            )
            init_tStS_layout = softmax_params.tiled_mma_qk.make_fragment_C(
                cute.append(init_tStS_shape, self.mma_s_stage)
            ).layout
            init_tStS = cute.make_tensor(
                common_params.tmem_ptr + self.tmem_s_offset, init_tStS_layout
            )
            init_tAcc = init_tStS[(None, None), 0, 0, 0]

            init_corr_layout = cute.make_layout(
                (init_tAcc.shape[0], 4, self.mma_s_stage),
                stride=(init_tAcc.stride[0], 1, self.tmem_corr_stage_cols),
            )
            init_tCor = cute.make_tensor(
                common_params.tmem_ptr + self.correction_factor_offset,
                init_corr_layout,
            )
            init_cCor = cute.make_identity_tensor(init_tCor.shape)
            init_store_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(4)), self.acc_dtype
            )
            init_tiled_copy = tcgen05.make_tmem_copy(init_store_atom, init_tCor)
            init_thr_copy = init_tiled_copy.get_slice(init_tidx)
            init_cCor_part = init_thr_copy.partition_S(init_cCor)
            init_tCor_part = init_thr_copy.partition_D(init_tCor)
            init_rCor = cute.make_fragment_like(
                init_cCor_part[None, None, None, 0], self.acc_dtype
            )
            init_rCor_int = cute.make_tensor(
                cute.recast_ptr(init_rCor.iterator, dtype=cutlass.Int32),
                init_rCor.layout,
            )
            init_rCor[0] = init_row_sum
            init_rCor[1] = init_row_max
            init_rCor[2] = self.acc_dtype(1.0)
            init_rCor_int[3] = cutlass.Int32(1)
            cute.copy(
                init_tiled_copy,
                init_rCor,
                init_tCor_part[None, None, None, p_cor_producer_state.index],
            )
            prims.tcgen05_wait("store")

    @cute.jit
    def load_other_group_metadata(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        p_cor_producer_state: pipeline.PipelineState,
    ) -> tuple[cutlass.Float32, cutlass.Float32]:
        if cutlass.const_expr(not self.use_2cta_instrs):
            tidx = common_params.tidx % 128
            stage = (p_cor_producer_state.index + 1) % self.mma_s_stage
            return common_params.sMeta[tidx, 1, stage], common_params.sMeta[
                tidx, 0, stage
            ]
        else:
            other_stage = (p_cor_producer_state.index + 1) % self.mma_s_stage
            tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
            tStS_shape = softmax_params.tiled_mma_qk.partition_shape_C(
                cute.select(self.mma_qk_tiler, mode=[0, 1])
            )
            tStS_layout = softmax_params.tiled_mma_qk.make_fragment_C(
                cute.append(tStS_shape, self.mma_s_stage)
            ).layout
            tStS = cute.make_tensor(
                common_params.tmem_ptr + self.tmem_s_offset, tStS_layout
            )
            tAcc = tStS[(None, None), 0, 0, 0]
            corr_layout = cute.make_layout(
                (tAcc.shape[0], 4, self.mma_s_stage),
                stride=(tAcc.stride[0], 1, self.tmem_corr_stage_cols),
            )
            tCor = cute.make_tensor(
                common_params.tmem_ptr + self.correction_factor_offset, corr_layout
            )
            cCor = cute.make_identity_tensor(tCor.shape)
            load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(4)), self.acc_dtype
            )
            load_tiled_copy = tcgen05.make_tmem_copy(load_atom, tCor)
            load_thr_copy = load_tiled_copy.get_slice(tidx)
            tCor_part = load_thr_copy.partition_S(tCor)
            cCor_part = load_thr_copy.partition_D(cCor)
            rCor = cute.make_fragment_like(
                cCor_part[None, None, None, 0], self.acc_dtype
            )
            cute.copy(load_tiled_copy, tCor_part[None, None, None, other_stage], rCor)
            return rCor[1], rCor[0]

    @cute.jit
    def store_p_cor_row_sum(
        self,
        common_params: SimpleNamespace,
        row_sum: cutlass.Float32,
        saved_stage_idx: cutlass.Int32,
        tAcc: cute.Tensor,
        tidx: cutlass.Int32,
    ) -> None:
        if cutlass.const_expr(not self.use_2cta_instrs):
            common_params.sMeta[tidx, 0, saved_stage_idx] = row_sum
        else:
            corr_layout_1 = cute.make_layout(
                (tAcc.shape[0], 1, self.mma_s_stage),
                stride=(tAcc.stride[0], 1, self.tmem_corr_stage_cols),
            )
            tCor = cute.make_tensor(
                common_params.tmem_ptr + self.correction_factor_offset,
                corr_layout_1,
            )
            cCor = cute.make_identity_tensor(tCor.shape)
            store_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(1)), self.acc_dtype
            )
            tiled_copy = tcgen05.make_tmem_copy(store_atom, tCor)
            thr_copy = tiled_copy.get_slice(tidx)
            cCor_for_copy = thr_copy.partition_S(cCor)
            tCor_for_copy = thr_copy.partition_D(tCor)
            rCor = cute.make_fragment_like(
                cCor_for_copy[None, None, None, 0], self.acc_dtype
            )
            rCor[0] = row_sum
            cute.copy(
                tiled_copy, rCor, tCor_for_copy[None, None, None, saved_stage_idx]
            )
            prims.tcgen05_wait("store")

    @cute.jit
    def softmax(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        k_index: cutlass.Int32,
        tile_valid_len: cutlass.Int32,
        mma_s_consumer_state: pipeline.PipelineState,
        p_mma_producer_state: pipeline.PipelineState,
        p_cor_producer_state: pipeline.PipelineState,
        row_max: cutlass.Float32,
        row_sum: cutlass.Float32,
        correction_factor: cutlass.Float32,
        is_second_compute_warp: bool,
        apply_mask: bool,
        is_local_last_tile: cutlass.Boolean,
    ) -> tuple[
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Float32,
    ]:
        softmax_exchange_sync_bar = (
            self.softmax_exchange_sync_bar_1
            if is_second_compute_warp
            else self.softmax_exchange_sync_bar_0
        )

        softmax_params.mma_s_pipeline.consumer_wait(mma_s_consumer_state)

        tStS_staged = cute.make_tensor(
            common_params.tmem_ptr + self.tmem_s_offset,
            self._score_fragment_layout(softmax_params.tiled_mma_qk),
        )
        tStS = tStS_staged[None, None, None, mma_s_consumer_state.index]

        tAcc = tStS[(None, None), 0, 0]
        cta_qk_tiler = (
            self.mma_qk_tiler[0] // self.cluster_shape_mnk[0],
            self.mma_qk_tiler[1],
            self.mma_qk_tiler[2],
        )
        cS = cute.make_identity_tensor(cute.select(cta_qk_tiler, mode=[0, 1]))

        tmem_load_atom = cute.make_copy_atom(
            (
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32))
                if self.use_2cta_instrs
                else tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(32))
            ),
            self.acc_dtype,
        )
        tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc)

        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc = tmem_thr_copy.partition_S(tAcc)
        tTR_tS = tmem_thr_copy.partition_D(cS)
        tTR_rAcc = cute.make_fragment_like(tTR_tS, self.acc_dtype)

        row_max_new = row_max
        arch = BaseDSL._get_dsl().get_arch_enum()
        if cutlass.const_expr(
            not self.use_2cta_instrs or (arch >= Arch.sm_100 and arch <= Arch.sm_100f)
        ):
            if cutlass.const_expr(self.use_primitives):
                hw.load_tmem_m64(tAcc.iterator, tTR_rAcc)
            else:
                cute.copy(tmem_tiled_copy, tTR_tAcc, tTR_rAcc)
            for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                sparse_k_idx = tTR_tS[i][1] + self.mma_qk_tiler[1] * k_index
                is_valid = (common_params.sSlots[sparse_k_idx] >= 0) & (
                    cute.elem_less(sparse_k_idx, tile_valid_len)
                )
                tTR_rAcc[i] = tTR_rAcc[i] if is_valid else -self.acc_dtype.inf
            row_max_new = tTR_rAcc.load().reduce(cute.ReductionOp.MAX, row_max_new, 0)
        elif cutlass.const_expr(arch >= Arch.sm_103 and arch <= Arch.sm_103f):
            tmem_load_red_atom = cute.make_copy_atom(
                tcgen05.copy.LdRed32x32bOp(
                    tcgen05.copy.Repetition(64), redOp=tcgen05.TmemLoadRedOp.MAX
                ),
                self.acc_dtype,
            )
            tmem_red_tiled_copy = tcgen05.make_tmem_copy(tmem_load_red_atom, tAcc)
            tmem_red_thr_copy = tmem_red_tiled_copy.get_slice(tidx)
            tTR_tAcc_red = tmem_red_thr_copy.partition_S(tAcc)
            tTR_tS_red = tmem_red_thr_copy.partition_D(cS)
            tTR_rAcc_red = cute.make_fragment_like(tTR_tS_red, self.acc_dtype)
            tTR_rMax = cute.make_rmem_tensor(
                cute.make_layout((1, tTR_tS_red.shape[1], tTR_tS_red.shape[2])),
                self.acc_dtype,
            )
            cute.copy(tmem_red_tiled_copy, tTR_tAcc_red, (tTR_rAcc_red, tTR_rMax))
            tTR_rAcc = cute.make_tensor(tTR_rAcc_red.iterator, tTR_rAcc.layout)
            if apply_mask:
                for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                    sparse_k_idx = tTR_tS[i][1] + self.mma_qk_tiler[1] * k_index
                    is_valid = cute.elem_less(sparse_k_idx, tile_valid_len)
                    tTR_rAcc[i] = tTR_rAcc[i] if is_valid else -self.acc_dtype.inf
                row_max_new = tTR_rAcc.load().reduce(
                    cute.ReductionOp.MAX, row_max_new, 0
                )
            else:
                row_max_new = cute.arch.fmax(row_max_new, tTR_rMax[0])

        # The primitives helper already waits before returning its fragment.
        if cutlass.const_expr(not self.use_primitives):
            prims.tcgen05_wait("load")
        softmax_params.mma_s_pipeline.consumer_release(mma_s_consumer_state)

        group_offset = self.num_compute_warps * self.threads_per_warp
        if cutlass.const_expr(is_second_compute_warp):
            my_base = group_offset
        else:
            my_base = 0
        if cutlass.const_expr(self.warps_in_n == 2):
            common_params.smem_exchange[my_base + tidx] = row_max_new
            prims.barrier_cta_sync(
                softmax_exchange_sync_bar.barrier_id,
                thread_count=softmax_exchange_sync_bar.num_threads,
            )
            row_max_new = cute.arch.fmax(
                row_max_new,
                common_params.smem_exchange[
                    my_base + (tidx ^ (64 if self.use_2cta_instrs else 16))
                ],
            )

        if cutlass.const_expr(is_second_compute_warp):
            prims.barrier_cta_sync(
                self.softmax_order_bar_1.barrier_id,
                thread_count=self.softmax_order_bar_1.num_threads,
            )
        else:
            prims.barrier_cta_sync(
                self.softmax_order_bar_0.barrier_id,
                thread_count=self.softmax_order_bar_0.num_threads,
            )

        other_row_max, other_row_sum = self.load_other_group_metadata(
            common_params, softmax_params, p_cor_producer_state
        )
        row_max_new = cute.arch.fmax(row_max_new, other_row_max)
        row_max = other_row_max
        row_sum = other_row_sum

        # A split can begin with a completely masked window tile and no
        # sink. Preserve its -inf maximum for a later nonempty tile, while
        # avoiding -inf - -inf in the rescale and exponentiation.
        empty_row = row_max_new == -self.acc_dtype.inf
        correction_factor = cute.math.exp2(
            cutlass.select_(
                empty_row,
                self.acc_dtype(0.0),
                (row_max - row_max_new) * softmax_params.softmax_scale_log2,
            ),
            fastmath=True,
        )
        saved_p_cor_idx = p_cor_producer_state.index
        if not is_local_last_tile:
            p_cor_producer_state, row_max_new = self.exchange_p_cor_metadata(
                common_params,
                softmax_params,
                correction_factor,
                row_sum,
                row_max,
                row_max_new,
                tAcc,
                tidx,
                p_cor_producer_state,
            )

        fma_b = softmax_params.softmax_scale_log2
        fma_c = cutlass.select_(
            empty_row,
            self.acc_dtype(0.0),
            (0.0 - row_max_new) * softmax_params.softmax_scale_log2,
        )
        for i in cutlass.range(cute.size(tTR_rAcc), vectorize=True, unroll_full=True):
            tTR_rAcc[i] = tTR_rAcc[i] * fma_b + fma_c
            tTR_rAcc[i] = cute.math.exp2(tTR_rAcc[i], fastmath=True)

        tTR_rS = cute.make_fragment_like(tTR_tS, self.q_dtype)
        tTR_rS.store(tTR_rAcc.load().to(self.q_dtype))

        sP = softmax_params.sP[None, None, None, (None, p_mma_producer_state.index)]
        sP_mk_view = cute.make_tensor(
            sP.iterator,
            cute.make_layout(
                (
                    (sP.shape[0][0], sP.shape[1]),
                    (sP.shape[0][1], sP.shape[2], sP.shape[3]),
                ),
                stride=(
                    (sP.stride[0][0], sP.stride[1]),
                    (sP.stride[0][1], sP.stride[2], sP.stride[3]),
                ),
            ),
        )
        sP_wo_swizzle_iter = cute.recast_ptr(sP.iterator, swizzle_=None)
        swizzle_bits = (
            int(math.log2(self.mma_pv_tiler[2] * self.q_dtype.width // 8 // 32)) + 1
        )
        swizzle_base = 3 if self.q_dtype.width == 16 else 4
        sP_swizzle = cute.make_swizzle(swizzle_bits, swizzle_base, 3)
        sP_mk_view = cute.make_tensor(
            sP_wo_swizzle_iter,
            cute.make_composed_layout(sP_swizzle, 0, sP_mk_view.layout),
        )
        universal_copy_bits = 128
        smem_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.q_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        smem_tiled_copy = cute.make_tiled_copy_D(smem_copy_atom, tmem_tiled_copy)
        smem_thr_copy = smem_tiled_copy.get_slice(tidx)
        rP_copy_view = smem_thr_copy.retile(tTR_rS)
        sP_copy_view = smem_thr_copy.partition_D(sP_mk_view)

        softmax_params.p_mma_pipeline.producer_acquire(p_mma_producer_state)
        cute.copy(smem_tiled_copy, rP_copy_view, sP_copy_view)
        prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)
        softmax_params.p_mma_pipeline.producer_commit(p_mma_producer_state)
        p_mma_producer_state.advance()

        row_sum = row_sum * correction_factor
        row_sum_vec = (0.0, 0.0)
        for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc), 2):
            row_sum_vec = cute.arch.add_packed_f32x2(
                row_sum_vec, (tTR_rAcc[i], tTR_rAcc[i + 1])
            )
        row_sum = row_sum_vec[0] + row_sum_vec[1] + row_sum

        if not is_local_last_tile:
            self.store_p_cor_row_sum(
                common_params,
                row_sum,
                saved_p_cor_idx,
                tAcc,
                tidx,
            )

        if is_local_last_tile:
            p_cor_producer_state, row_max_new = self.exchange_p_cor_metadata(
                common_params,
                softmax_params,
                correction_factor,
                row_sum,
                row_max,
                row_max_new,
                tAcc,
                tidx,
                p_cor_producer_state,
            )
        if cutlass.const_expr(is_second_compute_warp):
            prims.barrier_cta_arrive(
                self.softmax_order_bar_0.barrier_id,
                self.softmax_order_bar_0.num_threads,
            )
        else:
            prims.barrier_cta_arrive(
                self.softmax_order_bar_1.barrier_id,
                self.softmax_order_bar_1.num_threads,
            )

        mma_s_consumer_state.advance()
        return (
            mma_s_consumer_state,
            p_mma_producer_state,
            p_cor_producer_state,
            row_max_new,
            row_sum,
            correction_factor,
        )

    @cute.jit
    def _tmem_load_partition(
        self, common_params: SimpleNamespace, tiled_mma_pv: cute.TiledMma, iter_n: int
    ) -> tuple[
        cute.TiledMma,
        cute.TiledMma,
        cute.TiledMma,
        cute.TiledMma,
        cute.TiledMma,
        cute.TiledMma,
    ]:
        """Tensor memory load partition for rescale and epilogue.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param tiled_mma_pv: The tiled mma pv
        :type tiled_mma_pv: cute.TiledMma
        :param iter_n: The iteration number
        :type iter_n: int

        :return: The tiled copy and five partitioned tensor views.
        :rtype: tuple[cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma]
        """

        tOtO_shape = tiled_mma_pv.partition_shape_C(
            cute.select(self.mma_pv_tiler, mode=[0, 1])
        )
        tOtO = tiled_mma_pv.make_fragment_C(tOtO_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                common_params.L // self.mma_pv_tiler[1],
                stride=self.mma_pv_tiler[1]
                // (self.warps_in_n if self.use_2cta_instrs else 1),
            ),
        )
        tOtO = cute.make_tensor(
            common_params.tmem_ptr + self.tmem_o_offset, tOtO_layout
        )
        tOtO = tOtO[None, None, None, iter_n]

        tAcc = tOtO[(None, None), 0, 0]

        tmem_load_atom = cute.make_copy_atom(
            (
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32))
                if self.use_2cta_instrs
                else tcgen05.copy.Ld16x32bx2Op(tcgen05.copy.Repetition(32))
            ),
            self.acc_dtype,
        )
        tmem_load_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc)
        # {$nv-internal-release begin}
        # TODO: supports size() on tiled copy.
        # {$nv-internal-release end}
        tmem_load_thr_copy = tmem_load_tiled_copy.get_slice(
            common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        )

        cta_pv_tiler = (
            self.mma_pv_tiler[0] // self.cluster_shape_mnk[0],
            self.mma_pv_tiler[1],
            self.mma_pv_tiler[2],
        )
        # Flatten divide and partition global tensors for O
        cta_pv_tiler_mn = cute.select(cta_pv_tiler, mode=[0, 1])

        gO = None
        if cutlass.const_expr(common_params.mAccO is not None):
            gO = cute.local_tile(
                common_params.mAccO[None, common_params.blk_coord[3], None, None, None],
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
            cO = cute.local_tile(
                cute.make_identity_tensor(
                    common_params.mAccO[
                        None, common_params.blk_coord[3], None, None, None
                    ].shape
                ),
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
        else:
            gO = cute.local_tile(
                common_params.mO,
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
            cO = cute.local_tile(
                cute.make_identity_tensor(common_params.mO.shape),
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    iter_n,
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
            )
        tTR_tAcc = tmem_load_thr_copy.partition_S(tAcc)
        tTR_gO = tmem_load_thr_copy.partition_D(gO)
        tTR_cO = tmem_load_thr_copy.partition_D(cO)
        tTR_rAcc = cute.make_fragment_like(tTR_gO, self.acc_dtype)
        return tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc

    @cute.jit
    def get_correction_factor(
        self,
        common_params: SimpleNamespace,
        p_cor_consumer_state: pipeline.PipelineState,
        is_last_tile: cutlass.Boolean,
    ) -> tuple[
        pipeline.PipelineState,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Int32,
    ]:
        """Get the correction factor from the P correction consumer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param p_cor_consumer_state: The P correction consumer state
        :type p_cor_consumer_state: pipeline.PipelineState

        :return: The P correction consumer state, the row_sum, the row_max, and the correction factor
        :rtype: tuple[pipeline.PipelineState, cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Int32]
        """
        if cutlass.const_expr(not self.use_2cta_instrs):
            common_params.p_cor_pipeline.consumer_wait(p_cor_consumer_state)
            tidx = common_params.tidx % 128
            stage = p_cor_consumer_state.index
            # Non-final tiles publish correction metadata before computing
            # the updated sum for the next softmax group. That sum slot is
            # still being written; correction only needs it for epilogue.
            row_sum = self.acc_dtype(0.0)
            if is_last_tile:
                row_sum = common_params.sMeta[tidx, 0, stage]
            row_max = common_params.sMeta[tidx, 1, stage]
            correction_factor = common_params.sMeta[tidx, 2, stage]
            no_correction = common_params.sMeta[tidx, 3, stage].to(cutlass.Int32)
            common_params.p_cor_pipeline.consumer_release(p_cor_consumer_state)
            p_cor_consumer_state.advance()
            return (
                p_cor_consumer_state,
                row_sum,
                row_max,
                correction_factor,
                no_correction,
            )
        else:
            common_params.p_cor_pipeline.consumer_wait(p_cor_consumer_state)
            tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
            # load correction factor
            _, tAcc, _, _, _, _ = self._tmem_load_partition(
                common_params, common_params.tiled_mma_pv, 0
            )
            corr_layout = cute.make_layout(
                (tAcc.shape[0], (4, tAcc.shape[1][1]), self.p_cor_stage),
                stride=(tAcc.stride[0], (1, tAcc.stride[1][1]), 4),
            )
            tCor = cute.make_tensor(
                common_params.tmem_ptr + self.correction_factor_offset, corr_layout
            )
            cCor = cute.make_identity_tensor(tCor.shape)
            corr_tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(4)), self.acc_dtype
            )
            corr_tmem_load_tiled_copy = tcgen05.make_tmem_copy(
                corr_tmem_load_atom, tCor
            )
            corr_tmem_load_thr_copy = corr_tmem_load_tiled_copy.get_slice(tidx)
            tCor_for_copy = corr_tmem_load_thr_copy.partition_S(tCor)
            cCor_for_copy = corr_tmem_load_thr_copy.partition_D(cCor)
            rCor = cute.make_fragment_like(
                cCor_for_copy[None, None, None, 0], self.acc_dtype
            )
            rCor_int = cute.make_tensor(
                cute.recast_ptr(rCor.iterator, dtype=cutlass.Int32), rCor.layout
            )
            cute.copy(
                corr_tmem_load_tiled_copy,
                tCor_for_copy[None, None, None, p_cor_consumer_state.index],
                rCor,
            )
            row_sum = rCor[0]
            row_max = rCor[1]
            correction_factor = rCor[2]
            no_correction = rCor_int[3]

            common_params.p_cor_pipeline.consumer_release(p_cor_consumer_state)
            p_cor_consumer_state.advance()
            return (
                p_cor_consumer_state,
                row_sum,
                row_max,
                correction_factor,
                no_correction,
            )

    @cute.jit
    def rescale(
        self,
        common_params: SimpleNamespace,
        mma_o_consumer_state: pipeline.PipelineState,
        correction_factor: cutlass.Float32,
        no_correction: cutlass.Int32,
    ) -> pipeline.PipelineState:
        """Rescale for one k-tile. Updates the related pipeline state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param mma_o_consumer_state: The mma o consumer state
        :type mma_o_consumer_state: pipeline.PipelineState
        :param correction_factor: The correction factor
        :type correction_factor: cutlass.Float32
        :param no_correction: Whether to apply correction factor
        :type no_correction: cutlass.Int32

        :return: The MMA o consumer state
        :rtype: pipeline.PipelineState
        """
        skip_correction = cute.arch.vote_all_sync(no_correction == 1)
        for iter_n in cutlass.range_constexpr(self.iterations_pv_n):
            common_params.mma_o_pipeline.consumer_wait(mma_o_consumer_state)
            if not skip_correction:
                # tmem load tiled copy and partition results.
                tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc = (
                    self._tmem_load_partition(
                        common_params, common_params.tiled_mma_pv, iter_n
                    )
                )

                # tmem store tiled copy
                tmem_store_atom = cute.make_copy_atom(
                    (
                        tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32))
                        if self.use_2cta_instrs
                        else tcgen05.copy.St16x32bx2Op(tcgen05.copy.Repetition(32))
                    ),
                    self.acc_dtype,
                )
                tmem_store_tiled_copy = tcgen05.make_tmem_copy(tmem_store_atom, tAcc)

                # load o
                if cutlass.const_expr(self.use_primitives):
                    hw.load_tmem_m64(tAcc.iterator, tTR_rAcc)
                else:
                    cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)
                # rescale, using `mul_packed_f32x2` to reduce the number of instructions
                for i in cutlass.range(
                    cute.size(tTR_rAcc), vectorize=True, unroll_full=True
                ):
                    tTR_rAcc[i] = tTR_rAcc[i] * correction_factor

                # store o to tensor memory for next k tile
                if cutlass.const_expr(self.use_primitives):
                    hw.store_tmem_m64(tTR_rAcc, tAcc.iterator)
                else:
                    cute.copy(tmem_store_tiled_copy, tTR_rAcc, tTR_tAcc)

            prims.tcgen05_wait("store")
            common_params.mma_o_pipeline.consumer_release(mma_o_consumer_state)
            mma_o_consumer_state.advance()

        return mma_o_consumer_state

    @cute.jit
    def epilogue(
        self,
        common_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        mma_o_consumer_state: pipeline.PipelineState,
        row_sum: cutlass.Float32,
        row_max: cutlass.Float32,
    ) -> pipeline.PipelineState:
        """Normalize and store the output after all local K tiles are consumed.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param epilogue_params: The epilogue parameters
        :type epilogue_params: SimpleNamespace
        :param mma_o_consumer_state: The mma o consumer state
        :type mma_o_consumer_state: pipeline.PipelineState
        :param row_sum: The row sum
        :type row_sum: cutlass.Float32
        :param row_max: The row max
        :type row_max: cutlass.Float32

        :return: The MMA o consumer state
        :rtype: pipeline.PipelineState
        """

        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)

        # exchange row_sum between warps (0, 1) and (2, 3)
        if cutlass.const_expr(self.warps_in_n == 2):
            common_params.smem_exchange[tidx] = row_sum
            prims.barrier_cta_sync(
                self.epilogue_exchange_sync_bar.barrier_id,
                thread_count=self.epilogue_exchange_sync_bar.num_threads,
            )
            # (64, 2)
            row_sum = (
                row_sum
                + common_params.smem_exchange[
                    (tidx ^ (64 if self.use_2cta_instrs else 16))
                ]
            )

            # LSE is shared by every PV-N output tile. Store it once before
            # materializing the large O fragment so its global pointer and
            # scalar temporaries do not overlap the epilogue's peak live set.
            cta_pv_tiler = (
                self.mma_pv_tiler[0] // self.cluster_shape_mnk[0],
                self.mma_pv_tiler[1],
                self.mma_pv_tiler[2],
            )
            if cutlass.const_expr(epilogue_params.mAccLSE is None):
                gLSE = cute.local_tile(
                    epilogue_params.mLSE,
                    (cta_pv_tiler[0], 1, 1),
                    (
                        common_params.blk_coord[0],
                        common_params.blk_coord[1],
                        common_params.blk_coord[2],
                    ),
                    (1, 1, 1),
                )
                cLSE = cute.local_tile(
                    cute.make_identity_tensor(epilogue_params.mLSE.shape),
                    (cta_pv_tiler[0], 1, 1),
                    (
                        common_params.blk_coord[0],
                        common_params.blk_coord[1],
                        common_params.blk_coord[2],
                    ),
                    (1, 1, 1),
                )
            else:
                gLSE = cute.local_tile(
                    epilogue_params.mAccLSE[
                        None, common_params.blk_coord[3], None, None
                    ],
                    (cta_pv_tiler[0], 1, 1),
                    (
                        common_params.blk_coord[0],
                        common_params.blk_coord[1],
                        common_params.blk_coord[2],
                    ),
                    (1, 1, 1),
                )
                cLSE = cute.local_tile(
                    cute.make_identity_tensor(
                        epilogue_params.mAccLSE[
                            None, common_params.blk_coord[3], None, None
                        ].shape
                    ),
                    (cta_pv_tiler[0], 1, 1),
                    (
                        common_params.blk_coord[0],
                        common_params.blk_coord[1],
                        common_params.blk_coord[2],
                    ),
                    (1, 1, 1),
                )
            lse = (
                cute.math.log2(row_sum, fastmath=True)
                + epilogue_params.softmax_scale_log2 * row_max
            )
            lse_row = tidx if self.use_2cta_instrs else ((tidx // 32) * 16 + tidx % 16)
            if cute.elem_less(cLSE[lse_row][0], common_params.H):
                if self.use_2cta_instrs or tidx % 32 < 16:
                    if cutlass.const_expr(epilogue_params.mAccLSE is None):
                        gLSE[lse_row] = lse / LOG2_E
                    else:
                        gLSE[lse_row] = lse

        sink_scale = self.acc_dtype(1.0)
        if cutlass.const_expr(common_params.mAccO is None):
            head = common_params.blk_coord[0] * cta_pv_tiler[0] + lse_row
            sink_log2 = common_params.attn_sink[head] * LOG2_E
            denominator_max = cute.arch.fmax(lse, sink_log2)
            denominator_max = (
                denominator_max if denominator_max != -self.lse_dtype.inf else 0.0
            )
            denominator_lse = denominator_max + cute.math.log2(
                cute.math.exp2(lse - denominator_max, fastmath=True)
                + cute.math.exp2(sink_log2 - denominator_max, fastmath=True),
                fastmath=True,
            )
            sink_scale = (
                cute.math.exp2(lse - denominator_lse, fastmath=True)
                if row_sum != self.acc_dtype(0.0)
                else self.acc_dtype(0.0)
            )

        # mma_o pipeline consumer wait
        for iter_n in cutlass.range_constexpr(self.iterations_pv_n):
            common_params.mma_o_pipeline.consumer_wait(mma_o_consumer_state)
            # tmem load tiled copy and partition results.
            tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc = (
                self._tmem_load_partition(
                    common_params, common_params.tiled_mma_pv, iter_n
                )
            )

            # load o
            if cutlass.const_expr(self.use_primitives):
                hw.load_tmem_m64(tAcc.iterator, tTR_rAcc)
            else:
                cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)

            # Keep the vectorized epilogue branch-free. An empty causal split
            # has a zero accumulator and selects a zero reciprocal, while its
            # row_sum remains zero so the LSE contribution stays -inf.
            inv_row_sum = cutlass.select_(
                row_sum == self.acc_dtype(0.0),
                self.acc_dtype(0.0),
                cute.arch.rcp_approx(row_sum),
            )
            for i in cutlass.range(
                cute.size(tTR_rAcc), vectorize=True, unroll_full=True
            ):
                tTR_rAcc[i] = (
                    tTR_rAcc[i]
                    * epilogue_params.output_scale
                    * inv_row_sum
                    * sink_scale
                )

            # store o to global memory
            tR2G_rO_src = None
            tR2G_rO_dst = tTR_gO
            if cutlass.const_expr(common_params.mAccO is None):
                tR2G_rO_src = cute.make_fragment_like(tTR_gO, self.o_dtype)
                # using final output dtype for o
                tR2G_rO_src.store(tTR_rAcc.load().to(self.o_dtype))
            else:
                # using accumulate dtype for o
                tR2G_rO_src = tTR_rAcc

            if cute.elem_less(tTR_cO[0][0], common_params.H):
                cute.autovec_copy(
                    tR2G_rO_src,
                    tR2G_rO_dst,
                    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                )

            # The primitives helper already completed this TMEM load.
            if cutlass.const_expr(not self.use_primitives):
                prims.tcgen05_wait("load")
            common_params.mma_o_pipeline.consumer_release(mma_o_consumer_state)
            mma_o_consumer_state.advance()

        return mma_o_consumer_state

    def make_and_init_load_qkv_pipeline(
        self, load_qkv_mbar_ptr, cta_layout_vmnk, load_stages, tx_count
    ) -> pipeline.PipelineTmaUmma:
        """Create and initialize the tma load qkv pipeline.

        :param load_qkv_mbar_ptr: The load qkv mbar pointer
        :type load_qkv_mbar_ptr: cute.Tensor
        :param cta_layout_vmnk: The cta layout vmnk
        :type cta_layout_vmnk: tuple[int, int, int]
        :param load_stages: The load stages
        :type load_stages: list[int]
        :param tx_count: The tx count
        :type tx_count: int

        :return: The tma load qkv pipeline
        :rtype: pipeline.PipelineTmaUmma
        """
        load_qkv_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.load_tma_k_warp_id])
        )
        load_qkv_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_qk_warp_id])
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_qkv_mbar_ptr,
            num_stages=load_stages,
            producer_group=load_qkv_producer_group,
            consumer_group=load_qkv_consumer_group,
            tx_count=tx_count,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_mma_s_pipeline(
        self, mma_s_mbar_ptr, cta_layout_vmnk
    ) -> pipeline.PipelineUmmaAsync:
        """Create and initialize the mma s pipeline.

        :param mma_s_mbar_ptr: The mma s mbar pointer
        :type mma_s_mbar_ptr: cute.Tensor
        :param cta_layout_vmnk: The cta layout vmnk
        :type cta_layout_vmnk: tuple[int, int, int]

        :return: The mma s pipeline
        :rtype: pipeline.PipelineUmmaAsync
        """

        mma_s_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_qk_warp_id])
        )
        consumer_thread_size = (
            self.threads_per_warp
            * len(self.compute_warp_ids)
            * self.cluster_shape_mnk[0]
        )
        mma_s_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            consumer_thread_size,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_s_mbar_ptr,
            num_stages=self.mma_s_stage,
            producer_group=mma_s_producer_group,
            consumer_group=mma_s_consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_p_mma_pipeline(
        self, p_mma_mbar_ptr, cta_layout_vmnk
    ) -> pipeline.PipelineAsyncUmma:
        """Create and initialize the p mma pipeline.

        :param p_mma_mbar_ptr: The p mma mbar pointer
        :type p_mma_mbar_ptr: cute.Tensor
        :param cta_layout_vmnk: The cta layout vmnk
        :type cta_layout_vmnk: tuple[int, int, int]

        :return: The p mma pipeline
        :rtype: pipeline.PipelineAsyncUmma
        """

        producer_thread_size = (
            self.threads_per_warp
            * len(self.compute_warp_ids)
            * self.cluster_shape_mnk[0]
        )
        p_mma_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            producer_thread_size,
        )
        p_mma_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_pv_warp_id])
        )
        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=p_mma_mbar_ptr,
            num_stages=self.p_mma_stage,
            producer_group=p_mma_producer_group,
            consumer_group=p_mma_consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_p_cor_pipeline(
        self, p_cor_mbar_ptr
    ) -> pipeline.PipelineAsyncUmma:
        """Create and initialize the p correction pipeline.

        :param p_cor_mbar_ptr: The p correction mbar pointer
        :type p_cor_mbar_ptr: cute.Tensor

        :return: The p correction pipeline
        :rtype: pipeline.PipelineAsyncUmma
        """

        producer_thread_size = self.threads_per_warp * len(self.compute_warp_ids)
        p_cor_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            producer_thread_size,
        )
        p_cor_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            producer_thread_size,
        )
        return pipeline.PipelineAsync.create(
            barrier_storage=p_cor_mbar_ptr,
            num_stages=self.p_cor_stage,
            producer_group=p_cor_producer_group,
            consumer_group=p_cor_consumer_group,
            defer_sync=True,
        )

    def make_and_init_mma_o_pipeline(
        self, mma_o_mbar_ptr, cta_layout_vmnk
    ) -> pipeline.PipelineUmmaAsync:
        """Create and initialize the mma o pipeline.

        :param mma_o_mbar_ptr: The mma o mbar pointer
        :type mma_o_mbar_ptr: cute.Tensor
        :param cta_layout_vmnk: The cta layout vmnk
        :type cta_layout_vmnk: tuple[int, int, int]

        :return: The mma o pipeline
        :rtype: pipeline.PipelineUmmaAsync
        """

        mma_o_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_pv_warp_id])
        )
        consumer_thread_size = (
            self.threads_per_warp
            * len(self.compute_warp_ids)
            * self.cluster_shape_mnk[0]
        )
        mma_o_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            consumer_thread_size,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_o_mbar_ptr,
            num_stages=self.mma_o_stage,
            producer_group=mma_o_producer_group,
            consumer_group=mma_o_consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    @staticmethod
    def _compute_grid(
        o: cute.Tensor,
        split_kv: cutlass.Int32,
        cluster_shape_mnk: Tuple[int, int, int],
        max_active_clusters: int,
        is_persistent: bool,
    ) -> Tuple[HCAStaticTileSchedulerParams, Tuple[int, int, int]]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]

        :return: Tile scheduler parameters and grid shape.
        :rtype: tuple[HCAStaticTileSchedulerParams, tuple[int, int, int]]
        """
        o_shape = o.shape
        tile_sched_params = create_hca_static_tile_scheduler_params(
            is_persistent,
            cute.size(o_shape[3]),
            cute.size(o_shape[2]),
            cluster_shape_mnk,
            split_kv,
        )
        grid = HCAStaticTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def get_workspace_size(
        H: int,
        S: int,
        D: int,
        B: int,
        split_kv: int,
        acc_dtype: Type[cutlass.Numeric],
    ) -> int:
        """Get the extra workspace(device memory) size for the HCA kernel when split_kv is not 1.

        :param H: The height of the output tensor C
        :type H: int
        :param S: The sequence length of the output tensor C
        :type S: int
        :param D: The depth of the output tensor C
        :type D: int
        :param B: The batch size of the output tensor C
        :type B: int
        :param split_kv: The split key-value of the output tensor C
        :type split_kv: int
        :param acc_dtype: The data type of the output tensor C
        :type acc_dtype: Type[cutlass.Numeric]

        :return: The workspace size for the HCA kernel
        :rtype: int
        """
        if split_kv == 1:
            return 0
        return B * H * S * split_kv * (D + 1) * acc_dtype.width // 8

    @cute.jit
    def initialize_workspace(
        self,
        H: cutlass.Int32,
        D: cutlass.Int32,
        S: cutlass.Int32,
        B: cutlass.Int32,
        split_kv: cutlass.Int32,
        acc_dtype: Type[cutlass.Numeric],
        workspace: cute.Tensor,
    ) -> tuple[cute.Tensor, cute.Tensor]:
        """Initialize the workspace for the HCA kernel. Construct the intermediate tensors
        acc_o and acc_lse.

        :param H: The height of the output tensor C
        :type H: cutlass.Int32
        :param D: The depth of the output tensor C
        :type D: cutlass.Int32
        :param S: The sequence length of the output tensor C
        :type S: cutlass.Int32
        :param B: The batch size of the output tensor C
        :type B: cutlass.Int32
        :param split_kv: The split key-value of the output tensor C
        :type split_kv: cutlass.Int32
        :param acc_dtype: The data type of the output tensor C
        :type acc_dtype: Type[cutlass.Numeric]
        :param workspace: The workspace tensor
        :type workspace: cute.Tensor

        :return: The output tensor C and the workspace tensor
        :rtype: tuple[cute.Tensor, cute.Tensor]
        """
        acc_o, acc_lse = None, None
        if cutlass.const_expr(workspace is not None):
            align = 256 // self.q_dtype.width
            acc_o_layout = cute.make_layout(
                (H, split_kv, D, S, B),
                stride=(
                    cute.assume(split_kv * D, align),
                    cute.assume(D, align),
                    1,
                    cute.assume(split_kv * H * D, align),
                    cute.assume(H * split_kv * S * D, align),
                ),
            )
            acc_o_iter = cute.recast_ptr(workspace.iterator, dtype=acc_dtype)
            acc_o = cute.make_tensor(acc_o_iter, acc_o_layout)
            acc_lse_layout = cute.make_layout(
                (H, split_kv, S, B),
                stride=(split_kv, 1, H * split_kv, H * split_kv * S),
            )
            acc_lse_iter = cute.recast_ptr(
                workspace.iterator + cute.cosize(acc_o_layout) * acc_dtype.width // 8,
                dtype=acc_dtype,
            )
            acc_lse = cute.make_tensor(acc_lse_iter, acc_lse_layout)
        return acc_o, acc_lse

    @staticmethod
    def can_implement(
        B: int,
        S: int,
        K: int,
        H: int,
        L: int,
        in_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        lse_dtype: Type[cutlass.Numeric],
        mma_qk_tiler_mn: Tuple[int, int],
        mma_pv_tiler_mn: Tuple[int, int],
        split_kv: int,
        is_persistent: bool,
        is_var_seq: bool,
        is_var_split_kv: bool,
        page_size_cmp: int,
    ) -> bool:
        """Check if the HCA kernel can be implemented.

        :param B: The batch size of the output tensor C
        :type B: int
        :param S: The sequence length of the output tensor C
        :type S: int
        :param K: The width of the output tensor KV
        :type K: int
        :param H: The number of heads of the output tensor C
        :type H: int
        :param L: The full per-head depth (head_dim) of the tensor KV
            (last `qk_rope_head_dim` lanes assumed pre-rotated by caller)
        :type L: int
        :param in_dtype: The data type of the input tensor
        :type in_dtype: Type[cutlass.Numeric]
        :param out_dtype: The data type of the output tensor
        :type out_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param lse_dtype: The data type of the log-sum-exp
        :type lse_dtype: Type[cutlass.Numeric]
        :param mma_qk_tiler_mn: The tile shape of the query-key matrix multiplication
        :type mma_qk_tiler_mn: Tuple[int, int]
        :param mma_pv_tiler_mn: The tile shape of the probability-value matrix multiplication
        :type mma_pv_tiler_mn: Tuple[int, int]
        :param split_kv: The split key-value of the output tensor C
        :type split_kv: int
        :param is_persistent: Whether to use persistent kernel optimization
        :type is_persistent: bool
        :param is_var_seq: Whether to use variable sequence length
        :type is_var_seq: bool
        :param is_var_split_kv: Whether to use variable split_kv
        :type is_var_split_kv: bool
        :param page_size_cmp: Page size for compressed-KV stream
        :type page_size_cmp: int

        :return: Whether the HCA kernel can be implemented
        :rtype: bool
        """
        if L != 512:
            return False
        if in_dtype not in [cutlass.Float8E4M3FN, cutlass.BFloat16]:
            return False
        if out_dtype not in [cutlass.Float8E4M3FN, cutlass.BFloat16]:
            return False
        if acc_dtype != cutlass.Float32 or lse_dtype != cutlass.Float32:
            return False
        if split_kv < 1 or split_kv > MAX_SPLITS:
            return False
        # The compressed stream remains paged and must satisfy the TMA 128B
        # alignment requirement. Window gather4 splits the sequence tile across
        # two CTAs, with each CTA issuing groups of four indices.
        if page_size_cmp <= 1 or mma_qk_tiler_mn[1] % page_size_cmp != 0:
            return False
        if page_size_cmp & (page_size_cmp - 1) != 0:
            return False
        if mma_qk_tiler_mn[1] % 8 != 0:
            return False
        if mma_qk_tiler_mn[0] != mma_pv_tiler_mn[0]:
            return False
        if mma_qk_tiler_mn[0] == 64:
            # The shared K/V and TMEM layouts below specialize H64 decode.
            if (
                mma_qk_tiler_mn != (64, 64)
                or mma_pv_tiler_mn != (64, 128)
                or H != 64
                or S != 1
                or is_persistent
                or is_var_split_kv
                or in_dtype != cutlass.BFloat16
                or out_dtype != cutlass.BFloat16
            ):
                return False
        elif mma_qk_tiler_mn[0] != 128:
            return False
        if is_var_split_kv and not is_var_seq:
            return False
        if H <= 0 or H > 128:
            return False
        if H < 128 and split_kv != 1:
            # The H64 split epilogue predicates padded heads and the merge
            # launches only real heads. Other partial-head shapes retain
            # the original split restriction.
            if (
                H != 64
                or S != 1
                or is_persistent
                or is_var_split_kv
                or mma_qk_tiler_mn[1] != 64
                or mma_pv_tiler_mn[1] != (128 if mma_qk_tiler_mn[0] == 64 else 256)
                or in_dtype != cutlass.BFloat16
                or out_dtype != cutlass.BFloat16
            ):
                return False
        if S <= 0:
            return False
        if not is_persistent and B * S > 65535:
            return False
        if K <= 0:
            return False
        return True
