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

import math
from typing import Type, Tuple, Optional
from types import SimpleNamespace

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05, OperandMajorMode
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.arch import Arch
from cutlass.cutlass_dsl import BaseDSL

from .hca_helpers import (
    ceil_div,
    MAX_SPLITS,
    LOG2_E,
    HCAStaticTileScheduler,
    HCAStaticTileSchedulerParams,
    create_hca_static_tile_scheduler,
    create_hca_static_tile_scheduler_params,
)

"""
A Heavily Compressed Attention (HCA) kernel using FP8 input and FP8 or BF16
output for the NVIDIA Blackwell SM100 architecture using CuTe DSL.

This example demonstrates an implementation of inference of heavily compressed attention using a TMA + Blackwell
SM100 TensorCore warp-specialized persistent kernel. The implementation integrates the (Qc + Qr)*(Kc + Kr)^T
matrix multiplication, softmax normalization, and softmax((Qc + Qr)*(Kc + Kr)^T)*Vc into a single kernel.
The kernel provides support for page table storage and variable-length KV cache sequences. It implements KV splitting
functionality to minimize latency when processing long KV sequences.

The kernel implements key optimizations including:
- Warp specialization for different computation phases (load, MMA, softmax, correction, epilogue)
- Pipeline stages between different warps for overlapping computation and memory access
- Support for different precision data types
- Two sub-kernels (split KV kernel and reduction kernel) that enable split KV processing

To run this example:

.. code-block:: bash

    python examples/cute/blackwell/kernel/attention/dsa/hca_fp8.py                                        \
      --batch_size 4 --latent_dim 512                                    \
      --num_heads 128 --seq_len_q 1 --seq_len_k 1024                     \
      --in_dtype Float8E4M3FN --out_dtype Float8E4M3FN                   \
      --acc_dtype Float32 --lse_dtype Float32                            \
      --is_var_seq --is_var_split_kv                                     \
      --is_persistent

The above example runs Heavily Compressed Attention (HCA) with the following configuration:
- Batch size: 4
- Sequence length of Q: 1
- Sequence length of K: 1024
- Per-head depth (head_dim, last 64 lanes pre-rotated by caller): 512
- Number of heads: 128
- Data types: Float8E4M3FN (input), Float8E4M3FN (output), Float32 (accumulation and LSE)

It utilizes page table storage for the KV cache and enables both variable-length KV cache sequences
and variable split KV processing with persistent scheduling.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/cute/blackwell/kernel/attention/dsa/hca_fp8.py                                    \
      --batch_size 4 --latent_dim 512                                    \
      --num_heads 128 --seq_len_q 1 --seq_len_k 1024                     \
      --in_dtype Float8E4M3FN --out_dtype Float8E4M3FN                   \
      --acc_dtype Float32 --lse_dtype Float32                            \
      --is_var_seq --is_var_split_kv                                     \
      --is_persistent --warmup_iterations 3                              \
      --iterations 10 --skip_ref_check

Constraints for this example:
* Data type requirements:
  - Input/output: Float8E4M3FN
  - Accumulation and LSE: Float32
* Fixed architecture parameters:
  - Number of attention heads: 128
  - Per-head depth (head_dim): 512 (last `qk_rope_head_dim` lanes pre-rotated)
* Input query modes should be (NumHeads, HeadDim, SeqLenQ, BatchSize)
* Input kv modes should be (SeqLenK, HeadDim, BatchSize)
* Query sequence length must be positive. The non-persistent launch grid
  currently requires ``batch_size * seq_len_q <= 65535``.
* Only supports 2-CTA instructions
* Variable sequence length requires page table storage enabled
"""


class BlackwellHeavilyCompressedAttentionForwardFP8:
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
        page_size_win: int,
        skip_correction_threshold: float,
        is_persistent: bool,
        is_var_seq: bool,
        is_var_split_kv: bool,
        is_causal: bool = False,
        seq_len_q: int = 1,
        hca_compress_ratio: int = 128,
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
        :param page_size_win: Page size of the sliding-window page table
            (must be a power-of-two multiple of page_size_cmp, or vice versa)
        :type page_size_win: int
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
        self.page_size_win = page_size_win
        self.is_var_seq = is_var_seq
        self.is_var_split_kv = is_var_split_kv
        self.is_causal = is_causal
        self.seq_len_q = seq_len_q
        self.hca_compress_ratio = hca_compress_ratio
        self.cluster_shape_mnk = (2, 1, 1)
        self.use_2cta_instrs = True
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
        # warps 12-15 (odd k-tiles). Correction warps 4-7. MMA split: W8
        # issues QK only, W11 issues PV only.
        self.compute_warp_ids = (0, 1, 2, 3)
        self.correction_warp_ids = (4, 5, 6, 7)
        self.mma_qk_warp_id = 8
        self.load_tma_k_warp_id = 9
        self.load_tma_v_warp_id = 10
        self.mma_pv_warp_id = 11
        self.second_compute_warp_ids = (12, 13, 14, 15)
        self.num_total_compute_warps = self.num_compute_warps + len(
            self.second_compute_warp_ids
        )
        self.threads_per_cta = self.threads_per_warp * len(
            (
                self.mma_qk_warp_id,
                self.load_tma_k_warp_id,
                self.load_tma_v_warp_id,
                *self.compute_warp_ids,
                *self.second_compute_warp_ids,
                *self.correction_warp_ids,
                self.mma_pv_warp_id,
            )
        )

        # Register settings for the 16-warp launch. HCA's win/cmp TMA path
        # needs more registers in the load/MMA warps than dense MLA. Keep the
        # CTA register pool within 64K: 8*144 + 4*144 + 4*80 = 2048.
        self.softmax_reg_num = 144
        self.correction_reg_num = 144
        self.other_reg_num = 80
        # Named barriers
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
        self.init_row_max = -float("inf")
        self.tmem_corr_stage_cols = 4

    def _setup_attributes(self):
        """Set up configurations and parameters for the HCA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the multi-head latent attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        self.load_q_stage = 1
        self.load_k_stage = 3
        self.load_v_stage = 2
        self.mma_s_stage = 2
        self.p_mma_stage = 2
        self.p_cor_stage = 2
        self.mma_o_stage = 2

        self.tmem_o_offset = self.mma_s_stage * self.mma_qk_tiler[1] // self.warps_in_n
        self.correction_factor_offset = (
            self.tmem_o_offset + self.latent_dim // self.warps_in_n
        )

    @cute.jit
    def __call__(
        self,
        q_latent: cute.Tensor,
        c_latent_win: cute.Tensor,
        c_latent_cmp: cute.Tensor,
        page_table_win: cute.Tensor,
        page_table_cmp: cute.Tensor,
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
        :param c_latent_win: Sliding-window key tensor with shape
            [mma_qk_tiler[1], latent_dim, batch_size]
        :type c_latent_win: cute.Tensor
        :param c_latent_cmp: Compressed key tensor with shape
            [seq_len_k_cmp, latent_dim, batch_size]
        :type c_latent_cmp: cute.Tensor
        :param page_table_win: Page table for the sliding-window stream
        :type page_table_win: cute.Tensor
        :param page_table_cmp: Page table for the compressed stream
        :type page_table_cmp: cute.Tensor
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
        :type sparse_mla_topk_lens: cute.Tensor
        :param window_valid_lens: Per-query valid length within the sliding-window
            pool, using the same row layout as sparse_mla_topk_lens.
        :type window_valid_lens: cute.Tensor
        :param softmax_scale: The scale factor for softmax
        :type softmax_scale: cutlass.Float32
        :param output_scale: The scale factor for the output
        :type output_scale: cutlass.Float32
        :param attn_sink_unscaled: Per-head attention-sink logit divided by
            softmax_scale (unscaled-S space). Shape [num_heads]. Acts as a
            virtual extra softmax logit with V=0.
        :type attn_sink_unscaled: cute.Tensor
        :param stream: The CUDA stream to execute the kernel on
        :type stream: cuda.CUstream

        :raises TypeError: If tensor data types don't match or aren't supported
        """

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q_latent.element_type
        self.k_dtype = c_latent_cmp.element_type
        self.v_dtype = c_latent_cmp.element_type
        self.o_dtype = o.element_type

        # check type consistency
        if cutlass.const_expr(
            self.q_dtype != self.k_dtype
            or self.q_dtype != self.v_dtype
            or c_latent_win.element_type != self.k_dtype
        ):
            raise TypeError("Type mismatch among q/c_win/c_cmp")

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

        def _reinterpret_page_table(tensor):
            # [rows, page_count] -> [page_count, rows]
            return cute.make_tensor(
                tensor.iterator,
                cute.make_layout(
                    (tensor.shape[1], tensor.shape[0]),
                    stride=(tensor.stride[1], tensor.stride[0]),
                ),
            )

        q_latent = _reinterpret_4d(q_latent)
        c_latent_win = _reinterpret_3d_kv(c_latent_win)
        c_latent_cmp = _reinterpret_3d_kv(c_latent_cmp)
        page_table_win = _reinterpret_page_table(page_table_win)
        page_table_cmp = _reinterpret_page_table(page_table_cmp)
        o = _reinterpret_4d(o)
        # [B, S_q, H] -> [H, S_q, B]
        lse = cute.make_tensor(
            lse.iterator,
            cute.make_layout(
                (lse.shape[2], lse.shape[1], lse.shape[0]),
                stride=(lse.stride[2], lse.stride[1], lse.stride[0]),
            ),
        )

        # check leading dimensions of input/output
        if cutlass.const_expr(q_latent.stride[1] != 1):
            raise ValueError("q_latent must have leading dimension 1")
        if cutlass.const_expr(c_latent_cmp.stride[1] != 1):
            raise ValueError("c_latent_cmp must have leading dimension 1")
        if cutlass.const_expr(c_latent_win.stride[1] != 1):
            raise ValueError("c_latent_win must have leading dimension 1")
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

        c_latent_cmp_transpose_layout = cute.select(c_latent_cmp.layout, mode=[1, 0, 2])
        c_latent_cmp_transpose = cute.make_tensor(
            c_latent_cmp.iterator, c_latent_cmp_transpose_layout
        )
        c_latent_win_transpose_layout = cute.select(c_latent_win.layout, mode=[1, 0, 2])
        c_latent_win_transpose = cute.make_tensor(
            c_latent_win.iterator, c_latent_win_transpose_layout
        )

        self.q_major_mode = OperandMajorMode.K
        self.k_major_mode = OperandMajorMode.K
        self.v_major_mode = OperandMajorMode.MN

        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.TWO
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
        cta_kv_m = qk_tiled_mma.op.shape_mnk[0] // qk_tiled_mma.thr_id.shape
        kc_page_tile_size_cmp = min(self.page_size_cmp, cta_kv_m)
        kc_page_tile_size_win = min(self.page_size_win, cta_kv_m)
        kc_latent_smem_layout_staged = cute.logical_divide(
            kc_latent_smem_layout_staged, (None, None, None, self.iterations_qk_latent)
        )

        kc_latent_smem_layout_for_tma_base = sm100_utils.make_smem_layout(
            OperandMajorMode.K,
            (self.mma_qk_tiler[0] // qk_tiled_mma.thr_id.shape, self.mma_qk_tiler[2]),
            self.k_dtype,
            (self.iterations_qk_latent * self.load_k_stage),
        )
        kc_latent_smem_layout_for_tma_cmp = cute.tiled_divide(
            kc_latent_smem_layout_for_tma_base,
            (kc_page_tile_size_cmp, self.mma_qk_tiler[2]),
        )
        kc_latent_smem_layout_for_tma_cmp = cute.logical_divide(
            kc_latent_smem_layout_for_tma_cmp,
            (None, None, None, self.iterations_qk_latent),
        )
        kc_latent_smem_layout_for_tma_win = cute.tiled_divide(
            kc_latent_smem_layout_for_tma_base,
            (kc_page_tile_size_win, self.mma_qk_tiler[2]),
        )
        kc_latent_smem_layout_for_tma_win = cute.logical_divide(
            kc_latent_smem_layout_for_tma_win,
            (None, None, None, self.iterations_qk_latent),
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
        vc_page_tile_size_cmp = min(self.page_size_cmp, self.mma_pv_tiler[2])
        vc_page_tile_size_win = min(self.page_size_win, self.mma_pv_tiler[2])
        vc_pv_n = pv_tiled_mma.op.shape_mnk[1] // pv_tiled_mma.thr_id.shape
        vc_smem_layout_for_tma_base = sm100_utils.make_smem_layout(
            OperandMajorMode.MN,
            (self.mma_pv_tiler[1] // pv_tiled_mma.thr_id.shape, self.mma_pv_tiler[2]),
            self.v_dtype,
            (self.iterations_pv_k * self.iterations_pv_n * self.load_v_stage),
        )
        vc_smem_layout_for_tma_cmp = cute.tiled_divide(
            vc_smem_layout_for_tma_base,
            (vc_pv_n, vc_page_tile_size_cmp),
        )
        vc_smem_layout_for_tma_cmp = cute.logical_divide(
            cute.logical_divide(
                vc_smem_layout_for_tma_cmp,
                (None, None, None, self.iterations_pv_k * self.iterations_pv_n),
            ),
            (None, None, None, (self.iterations_pv_n, None)),
        )
        vc_smem_layout_for_tma_win = cute.tiled_divide(
            vc_smem_layout_for_tma_base,
            (vc_pv_n, vc_page_tile_size_win),
        )
        vc_smem_layout_for_tma_win = cute.logical_divide(
            cute.logical_divide(
                vc_smem_layout_for_tma_win,
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
        # TMA load for c latent (cmp + win)
        kc_smem_layout_cmp = cute.select(kc_latent_smem_layout_for_tma_cmp, mode=[0])
        tma_atom_c_latent_cmp, tma_tensor_c_latent_cmp = self.make_paged_tiled_tma_atom(
            tma_load_op,
            c_latent_cmp,
            kc_smem_layout_cmp,
            (self.mma_qk_tiler[1], self.mma_qk_tiler[2]),
            qk_tiled_mma,
            is_k_load=True,
            page_size=self.page_size_cmp,
        )
        kc_smem_layout_win = cute.select(kc_latent_smem_layout_for_tma_win, mode=[0])
        tma_atom_c_latent_win, tma_tensor_c_latent_win = self.make_paged_tiled_tma_atom(
            tma_load_op,
            c_latent_win,
            kc_smem_layout_win,
            (self.mma_qk_tiler[1], self.mma_qk_tiler[2]),
            qk_tiled_mma,
            is_k_load=True,
            page_size=self.page_size_win,
        )

        # TMA load for c latent transpose (cmp + win)
        vc_smem_layout_cmp = cute.select(vc_smem_layout_for_tma_cmp, mode=[0])
        tma_atom_c_latent_transpose_cmp, tma_tensor_c_latent_transpose_cmp = (
            self.make_paged_tiled_tma_atom(
                tma_load_op,
                c_latent_cmp_transpose,
                vc_smem_layout_cmp,
                (self.mma_pv_tiler[1], self.mma_pv_tiler[2]),
                pv_tiled_mma,
                is_k_load=False,
                page_size=self.page_size_cmp,
            )
        )
        vc_smem_layout_win = cute.select(vc_smem_layout_for_tma_win, mode=[0])
        tma_atom_c_latent_transpose_win, tma_tensor_c_latent_transpose_win = (
            self.make_paged_tiled_tma_atom(
                tma_load_op,
                c_latent_win_transpose,
                vc_smem_layout_win,
                (self.mma_pv_tiler[1], self.mma_pv_tiler[2]),
                pv_tiled_mma,
                is_k_load=False,
                page_size=self.page_size_win,
            )
        )

        q_latent_copy_size = (
            cute.size_in_bytes(self.q_dtype, q_smem_layout)
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_latent
        )
        kc_latent_copy_size = (
            cute.size_in_bytes(
                self.k_dtype,
                cute.select(kc_latent_smem_layout_staged, mode=[0, 1, 2]),
            )
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_latent
        )
        vc_copy_size = (
            cute.size_in_bytes(
                self.v_dtype, cute.select(vc_smem_layout_staged, mode=[0, 1, 2])
            )
            * cute.size(pv_tiled_mma.thr_id.shape)
            * self.iterations_pv_n
            * self.iterations_pv_k
        )

        self.tma_copy_q_bytes = q_latent_copy_size
        self.tma_copy_kc_bytes = kc_latent_copy_size
        self.tma_copy_vc_bytes = vc_copy_size

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
                cute.struct.MemRange[self.v_dtype, cute.cosize(vc_smem_layout_staged)],
                1024,
            ]
            softmax_smem_exchange: cute.struct.MemRange[
                self.acc_dtype, 2 * self.num_compute_warps * self.threads_per_warp
            ]
            epilogue_smem_exchange: cute.struct.MemRange[
                self.acc_dtype, self.num_compute_warps * self.threads_per_warp
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
            tma_atom_c_latent_win,
            tma_tensor_c_latent_win,
            tma_atom_c_latent_cmp,
            tma_tensor_c_latent_cmp,
            tma_atom_c_latent_transpose_win,
            tma_tensor_c_latent_transpose_win,
            tma_atom_c_latent_transpose_cmp,
            tma_tensor_c_latent_transpose_cmp,
            page_table_win,
            page_table_cmp,
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
            q_latent_smem_layout_staged,
            kc_latent_smem_layout_staged,
            p_smem_layout_staged,
            vc_smem_layout_staged,
            kc_latent_smem_layout_for_tma_win,
            kc_latent_smem_layout_for_tma_cmp,
            vc_smem_layout_for_tma_win,
            vc_smem_layout_for_tma_cmp,
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
            self.reduction_kernel(
                o,
                lse,
                acc_o,
                acc_lse,
                split_kv,
                cache_seqs,
                block_split_kvs,
            ).launch(
                grid=(q_latent.shape[0], q_latent.shape[2], q_latent.shape[3]),
                block=[self.threads_per_warp * self.num_compute_warps, 1, 1],
                stream=stream,
                min_blocks_per_mp=1,
            )

    @cute.jit
    def make_paged_tiled_tma_atom(
        self,
        tma_load_op: cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp,
        gmem: cute.Tensor,
        smem_layout: cute.Layout,
        mma_tiler,
        tiled_mma: cute.TiledMma,
        is_k_load: bool,
        page_size: int,
    ):
        ident = cute.make_identity_layout(gmem.shape)
        g_tile = cute.composition(ident, mma_tiler)
        cta_mn = mma_tiler[0] // tiled_mma.thr_id.shape
        cta_v_map = cute.flat_divide(g_tile, (cta_mn,))
        cta_v_map = cute.select(cta_v_map, mode=[0, 2])
        page_tile_size = (
            min(page_size, cta_mn) if is_k_load else min(page_size, mma_tiler[1])
        )
        cta_v_map = cute.zipped_divide(
            cta_v_map,
            (page_tile_size, mma_tiler[1]) if is_k_load else (cta_mn, page_tile_size),
        )
        cta_v_map = cute.select(cta_v_map, mode=[0])
        from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir

        res = _cute_nvgpu_ir.atom_make_non_exec_tiled_tma_load(
            gmem.value,
            smem_layout.value,
            cta_v_map,
            tma_load_op._to_ir(),
            num_multicast=1,
        )
        return cute.CopyAtom(
            tma_load_op, cpasync.CopyBulkTensorTileG2SNonExecTrait(res[0])
        ), res[1]

    @cute.kernel
    def split_kv_kernel(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tma_atom_q_latent: Optional[cute.CopyAtom],
        mQL: cute.Tensor,
        tma_atom_c_latent_win: Optional[cute.CopyAtom],
        mCL_win: cute.Tensor,
        tma_atom_c_latent_cmp: Optional[cute.CopyAtom],
        mCL_cmp: cute.Tensor,
        tma_atom_c_latent_transpose_win: Optional[cute.CopyAtom],
        mCLT_win: cute.Tensor,
        tma_atom_c_latent_transpose_cmp: Optional[cute.CopyAtom],
        mCLT_cmp: cute.Tensor,
        mPT_win: cute.Tensor,
        mPT_cmp: cute.Tensor,
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
        q_latent_smem_layout_staged: cute.ComposedLayout,
        kc_latent_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        vc_smem_layout_staged: cute.ComposedLayout,
        kc_latent_smem_layout_for_tma_win: Optional[cute.ComposedLayout],
        kc_latent_smem_layout_for_tma_cmp: Optional[cute.ComposedLayout],
        vc_smem_layout_for_tma_win: Optional[cute.ComposedLayout],
        vc_smem_layout_for_tma_cmp: Optional[cute.ComposedLayout],
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
        :param tma_atom_c_latent_win: TMA copy atom for window-stream K
        :type tma_atom_c_latent_win: cute.CopyAtom
        :param mCL_win: Window-stream key tensor
        :type mCL_win: cute.Tensor
        :param tma_atom_c_latent_cmp: TMA copy atom for compressed-stream K
        :type tma_atom_c_latent_cmp: cute.CopyAtom
        :param mCL_cmp: Compressed-stream key tensor
        :type mCL_cmp: cute.Tensor
        :param mCLT_win: Window-stream V transpose tensor
        :type mCLT_win: cute.Tensor
        :param mCLT_cmp: Compressed-stream V transpose tensor
        :type mCLT_cmp: cute.Tensor
        :param mPT_win: Window-stream page table tensor
        :type mPT_win: cute.Tensor
        :param mPT_cmp: Compressed-stream page table tensor
        :type mPT_cmp: cute.Tensor
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
            cpasync.prefetch_descriptor(tma_atom_c_latent_win)
            cpasync.prefetch_descriptor(tma_atom_c_latent_cmp)
            cpasync.prefetch_descriptor(tma_atom_c_latent_transpose_win)
            cpasync.prefetch_descriptor(tma_atom_c_latent_transpose_cmp)

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
        load_k_pipeline = self.make_and_init_load_qkv_pipeline(
            storage.load_k_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            self.load_k_stage,
            self.tma_copy_kc_bytes,
        )
        load_v_pipeline = self.make_and_init_load_qkv_pipeline(
            storage.load_v_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            self.load_v_stage,
            self.tma_copy_vc_bytes,
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
        # Two TMA-views over the same physical KC SMEM, one per page-size
        sKC_for_tma_win = storage.smem_kc_latent.get_tensor(
            kc_latent_smem_layout_for_tma_win.outer,
            swizzle=kc_latent_smem_layout_for_tma_win.inner,
        )
        sKC_for_tma_cmp = storage.smem_kc_latent.get_tensor(
            kc_latent_smem_layout_for_tma_cmp.outer,
            swizzle=kc_latent_smem_layout_for_tma_cmp.inner,
        )
        # (MMA, MMA_D, MMA_K, PIPE)
        sVC = storage.smem_vc.get_tensor(
            vc_smem_layout_staged.outer, swizzle=vc_smem_layout_staged.inner
        )
        sVC_for_tma_win = storage.smem_vc.get_tensor(
            vc_smem_layout_for_tma_win.outer, swizzle=vc_smem_layout_for_tma_win.inner
        )
        sVC_for_tma_cmp = storage.smem_vc.get_tensor(
            vc_smem_layout_for_tma_cmp.outer, swizzle=vc_smem_layout_for_tma_cmp.inner
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

        # ///////////////////////////////////////////////////////////////////////////////
        #  Load warps, including page table and data tensors
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.load_tma_k_warp_id:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            load_q_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_q_stage
            )
            load_k_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_k_stage
            )
            tile_sched = create_hca_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv,
                    cache_seqs,
                    block_split_kvs,
                    blk_coord,
                )
                if k_tile_count > 0:
                    # Construct fixed common/tma_qk/tma_pv params for load_tma
                    tma_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        local_split_kv=local_split_kv,
                        load_q_pipeline=load_q_pipeline,
                        load_k_pipeline=load_k_pipeline,
                        load_v_pipeline=load_v_pipeline,
                        mPT_win=mPT_win,
                        mPT_cmp=mPT_cmp,
                    )
                    tma_qk_params = SimpleNamespace(
                        tiled_mma_qk=tiled_mma_qk,
                        tma_atom_q_latent=tma_atom_q_latent,
                        tma_atom_c_latent_win=tma_atom_c_latent_win,
                        tma_atom_c_latent_cmp=tma_atom_c_latent_cmp,
                        mQL=mQL,
                        mCL_win=mCL_win,
                        mCL_cmp=mCL_cmp,
                        sQ=sQ,
                        sKC_win=sKC_for_tma_win,
                        sKC_cmp=sKC_for_tma_cmp,
                    )
                    # Load tma
                    load_q_producer_state, load_k_producer_state = self.load_tma_qk(
                        tma_common_params,
                        tma_qk_params,
                        k_index,
                        k_tile_count,
                        load_q_producer_state,
                        load_k_producer_state,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            load_q_pipeline.producer_tail(load_q_producer_state)
            load_k_pipeline.producer_tail(load_k_producer_state)

        if warp_idx == self.load_tma_v_warp_id:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            load_v_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_v_stage
            )
            tile_sched = create_hca_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv,
                    cache_seqs,
                    block_split_kvs,
                    blk_coord,
                )
                if k_tile_count > 0:
                    # Construct fixed common/tma_qk/tma_pv params for load_tma
                    tma_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        local_split_kv=local_split_kv,
                        load_v_pipeline=load_v_pipeline,
                        mPT_win=mPT_win,
                        mPT_cmp=mPT_cmp,
                    )
                    tma_pv_params = SimpleNamespace(
                        tiled_mma_pv=tiled_mma_pv,
                        tma_atom_c_latent_transpose_win=tma_atom_c_latent_transpose_win,
                        tma_atom_c_latent_transpose_cmp=tma_atom_c_latent_transpose_cmp,
                        mCLT_win=mCLT_win,
                        mCLT_cmp=mCLT_cmp,
                        sVC_win=sVC_for_tma_win,
                        sVC_cmp=sVC_for_tma_cmp,
                    )
                    # Load tma
                    load_v_producer_state = self.load_tma_v(
                        tma_common_params,
                        tma_pv_params,
                        k_index,
                        k_tile_count,
                        load_v_producer_state,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            load_v_pipeline.producer_tail(load_v_producer_state)

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA-QK warp (W8): issues all Q*K^T MMA and produces S.
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_qk_warp_id:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            tmem.wait_for_alloc()
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
                        is_leader_cta=is_leader_cta,
                        L=mCL_cmp.shape[1],
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
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols(self.arch_str))
            tmem.wait_for_alloc()
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
                        is_leader_cta=is_leader_cta,
                        L=mCL_cmp.shape[1],
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
            cute.arch.setmaxregister_increase(self.softmax_reg_num)
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
            tmem.wait_for_alloc()
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
                        L=mCL_cmp.shape[1],
                        tmem_ptr=tmem_ptr,
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
        #  Compute warp - second group (g1, warps 12-15, odd k-tiles).
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.second_compute_warp_ids[0]
            and warp_idx <= self.second_compute_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.softmax_reg_num)
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
            tmem.wait_for_alloc()
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
                        L=mCL_cmp.shape[1],
                        tmem_ptr=tmem_ptr,
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
            cute.arch.setmaxregister_increase(self.correction_reg_num)
            p_cor_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.p_cor_stage
            )
            mma_o_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_o_stage
            )
            # sync with mma warp before retrieving tmem ptr
            tmem.wait_for_alloc()

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
                        mAccO=mAccO,
                        mO=mO,
                        L=mCL_cmp.shape[1],
                        H=mQL.shape[0],
                        tmem_ptr=tmem_ptr,
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

        # Alloc shared memory
        smem = utils.SmemAllocator()
        storage = smem.allocate(MAX_SPLITS * self.acc_dtype.width // 8, 16)
        lse_scale_ptr = cute.recast_ptr(storage, dtype=self.acc_dtype)
        smem_lse_scale = cute.make_tensor(lse_scale_ptr, cute.make_layout(MAX_SPLITS))

        gLSE = mAccLSE[blk_coord[0], None, blk_coord[1], blk_coord[2]]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            # calculate the global lse and exp ^ (local_lse - global_lse)
            lse_per_thread = cute.ceil_div(MAX_SPLITS, self.threads_per_warp)

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
            # calculate the global_lse
            global_lse = (
                lse_max + cute.math.log2(sum_lse, fastmath=True)
                if not sum_lse == self.lse_dtype(0.0) or sum_lse != sum_lse  # noqa: SIM201
                else self.lse_dtype.inf
            )
            if tidx == 0:
                mLSE[blk_coord[0], blk_coord[1], blk_coord[2]] = global_lse
            # store the scale to shared memory
            for i in cutlass.range_constexpr(lse_per_thread):
                split_kv_idx = tidx + i * self.threads_per_warp
                if cute.elem_less(split_kv_idx, local_split_kv):
                    smem_lse_scale[split_kv_idx] = cute.math.exp2(
                        local_lse[i] - global_lse, fastmath=True
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
        for i in range(local_split_kv):
            for j in cutlass.range_constexpr(elements_per_thread):
                element_idx = tidx + j * self.threads_per_warp * self.num_compute_warps
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

    @cute.jit
    def load_tma_qk(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        load_q_producer_state: pipeline.PipelineState | None = None,
        load_k_producer_state: pipeline.PipelineState | None = None,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState]:
        """Load wrap to load Q/K tensors. Updates the load qk producer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param qk_params: The qk parameters
        :type qk_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param k_tile_count: The k tile count
        :type k_tile_count: cutlass.Int32
        :param load_q_producer_state: The load q producer state
        :type load_q_producer_state: pipeline.PipelineState
        :param load_k_producer_state: The load k producer state
        :type load_k_producer_state: pipeline.PipelineState

        :return: The load q producer state and load k producer state
        :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState]
        """
        page_table_row = self.get_page_table_row(common_params.blk_coord)
        mPT_win_b = common_params.mPT_win[None, page_table_row]
        mPT_cmp_b = common_params.mPT_cmp[None, page_table_row]

        # Flatten divide and partition global tensors for QK TMA load
        mma_qk_tiler_mk = cute.select(self.mma_qk_tiler, mode=[0, 2])
        gQL = cute.flat_divide(qk_params.mQL, mma_qk_tiler_mk)

        thr_mma_qk = qk_params.tiled_mma_qk.get_slice(
            common_params.blk_coord[0] % cute.size(qk_params.tiled_mma_qk.thr_id)
        )
        tSgQL = thr_mma_qk.partition_A(gQL)

        cta_kv_m = (
            qk_params.tiled_mma_qk.op.shape_mnk[0]
            // qk_params.tiled_mma_qk.thr_id.shape
        )

        # K partition for window stream
        cta_m_win = min(cta_kv_m, self.page_size_win)
        page_tile_size_k_win = min(self.page_size_win, cta_m_win)
        gCL_win = cute.tiled_divide(
            qk_params.mCL_win, (page_tile_size_k_win, self.mma_qk_tiler[2])
        )
        tSgCL_win = (
            gCL_win[
                None,
                common_params.blk_coord[0] % qk_params.tiled_mma_qk.thr_id.shape,
                None,
                None,
            ]
            if cta_m_win < self.page_size_win
            else gCL_win[None, 0, None, None]
        )

        # K partition for compressed stream
        cta_m_cmp = min(cta_kv_m, self.page_size_cmp)
        page_tile_size_k_cmp = min(self.page_size_cmp, cta_m_cmp)
        gCL_cmp = cute.tiled_divide(
            qk_params.mCL_cmp, (page_tile_size_k_cmp, self.mma_qk_tiler[2])
        )
        tSgCL_cmp = (
            gCL_cmp[
                None,
                common_params.blk_coord[0] % qk_params.tiled_mma_qk.thr_id.shape,
                None,
                None,
            ]
            if cta_m_cmp < self.page_size_cmp
            else gCL_cmp[None, 0, None, None]
        )

        # tma partition for q (one stream) and k (two streams sharing SMEM)
        tQsQ, tQLgQL_mkl = cpasync.tma_partition(
            qk_params.tma_atom_q_latent,
            0,
            cute.make_layout(1),
            cute.group_modes(qk_params.sQ, 0, 3),
            cute.group_modes(tSgQL, 0, 3),
        )

        tKCsKC_win, tCLgCL_win = cpasync.tma_partition(
            qk_params.tma_atom_c_latent_win,
            0,
            cute.make_layout(1),
            qk_params.sKC_win,
            tSgCL_win,
        )
        tKCsKC_cmp, tCLgCL_cmp = cpasync.tma_partition(
            qk_params.tma_atom_c_latent_cmp,
            0,
            cute.make_layout(1),
            qk_params.sKC_cmp,
            tSgCL_cmp,
        )

        tQLgQL = tQLgQL_mkl[
            None, None, None, common_params.blk_coord[1], common_params.blk_coord[2]
        ]

        # set extra params (both streams threaded through)
        common_params.mPT_win = mPT_win_b
        common_params.mPT_cmp = mPT_cmp_b
        qk_params.tQLgQL = tQLgQL
        qk_params.tCLgCL_win = tCLgCL_win
        qk_params.tCLgCL_cmp = tCLgCL_cmp
        qk_params.tQsQ = tQsQ
        qk_params.tKCsKC_win = tKCsKC_win
        qk_params.tKCsKC_cmp = tKCsKC_cmp

        k_tile_count_init = k_tile_count
        while k_tile_count > 0:
            # {$nv-internal-release begin}
            # TODO: figure out how to support SingleNamespace/struct in ast
            # {$nv-internal-release end}
            load_q_producer_state, load_k_producer_state = self.load_tma_qk_one_k_tile(
                common_params,
                qk_params,
                k_index,
                k_tile_count,
                load_q_producer_state,
                load_k_producer_state,
                load_q=k_tile_count_init == k_tile_count,
            )
            k_index += 1
            k_tile_count -= 1

        return load_q_producer_state, load_k_producer_state

    @cute.jit
    def load_tma_v(
        self,
        common_params: SimpleNamespace,
        v_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        load_v_producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Load wrap to load V tensors. Updates the load v producer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param v_params: The v parameters
        :type v_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param k_tile_count: The k tile count
        :type k_tile_count: cutlass.Int32
        :param load_v_producer_state: The load v producer state
        :type load_v_producer_state: pipeline.PipelineState

        :return: The load v producer state
        :rtype: pipeline.PipelineState
        """
        page_table_row = self.get_page_table_row(common_params.blk_coord)
        mPT_win_b = common_params.mPT_win[None, page_table_row]
        mPT_cmp_b = common_params.mPT_cmp[None, page_table_row]

        cta_n = self.mma_pv_tiler[1] // v_params.tiled_mma_pv.thr_id.shape

        # V partition for window stream
        page_tile_size_v_win = min(self.page_size_win, self.mma_pv_tiler[2])
        gCLT_win = cute.flat_divide(
            v_params.mCLT_win, (self.mma_pv_tiler[1], page_tile_size_v_win)
        )
        gCLT_win = cute.logical_divide(gCLT_win, (cta_n,))[
            (None, common_params.blk_coord[0]), None, None, None, None
        ]
        tOgCLT_win = cute.tiled_divide(gCLT_win, (cta_n, page_tile_size_v_win))
        tOgCLT_win = tOgCLT_win[None, 0, 0, None, None, None]

        # V partition for compressed stream
        page_tile_size_v_cmp = min(self.page_size_cmp, self.mma_pv_tiler[2])
        gCLT_cmp = cute.flat_divide(
            v_params.mCLT_cmp, (self.mma_pv_tiler[1], page_tile_size_v_cmp)
        )
        gCLT_cmp = cute.logical_divide(gCLT_cmp, (cta_n,))[
            (None, common_params.blk_coord[0]), None, None, None, None
        ]
        tOgCLT_cmp = cute.tiled_divide(gCLT_cmp, (cta_n, page_tile_size_v_cmp))
        tOgCLT_cmp = tOgCLT_cmp[None, 0, 0, None, None, None]

        tVCsVC_win, tCLTgCLT_win = cpasync.tma_partition(
            v_params.tma_atom_c_latent_transpose_win,
            0,
            cute.make_layout(1),
            v_params.sVC_win,
            tOgCLT_win,
        )
        tVCsVC_cmp, tCLTgCLT_cmp = cpasync.tma_partition(
            v_params.tma_atom_c_latent_transpose_cmp,
            0,
            cute.make_layout(1),
            v_params.sVC_cmp,
            tOgCLT_cmp,
        )

        # set extra params
        common_params.mPT_win = mPT_win_b
        common_params.mPT_cmp = mPT_cmp_b
        v_params.tCLTgCLT_win = tCLTgCLT_win
        v_params.tCLTgCLT_cmp = tCLTgCLT_cmp
        v_params.tVCsVC_win = tVCsVC_win
        v_params.tVCsVC_cmp = tVCsVC_cmp

        while k_tile_count > 0:
            # {$nv-internal-release begin}
            # TODO: figure out how to support SingleNamespace/struct in ast
            # {$nv-internal-release end}
            load_v_producer_state = self.load_tma_v_one_k_tile(
                common_params,
                v_params,
                k_index,
                load_v_producer_state,
            )
            k_index += 1
            k_tile_count -= 1
        return load_v_producer_state

    @cute.jit
    def load_tma_qk_one_k_tile(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        load_q_producer_state: pipeline.PipelineState,
        load_k_producer_state: pipeline.PipelineState,
        load_q: bool,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState]:
        """Load one k-tile of Q/C latent tensors. Updates the load qkv producer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param qk_params: The qk parameters
        :type qk_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param k_tile_count: The k tile count
        :type k_tile_count: cutlass.Int32
        :param load_q_producer_state: The load q producer state
        :type load_q_producer_state: pipeline.PipelineState
        :param load_k_producer_state: The load kv producer state
        :type load_k_producer_state: pipeline.PipelineState
        :param load_q: Whether to load q
        :type load_q: bool

        :return: The load q producer state and load kv producer state
        :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState]
        """
        page_per_tile_win = ceil_div(
            self.mma_qk_tiler[1] // self.page_size_win,
            qk_params.tiled_mma_qk.thr_id.shape,
        )
        page_per_tile_cmp = ceil_div(
            self.mma_qk_tiler[1] // self.page_size_cmp,
            qk_params.tiled_mma_qk.thr_id.shape,
        )
        # Either stream may have the larger per-CTA page count.
        page_per_tile_max = max(page_per_tile_win, page_per_tile_cmp)
        k_idx = cute.make_rmem_tensor(
            cute.make_layout(page_per_tile_max), cutlass.Int32
        )

        # k_index == 0 → window stream (page indices come from mPT_win[0]).
        # k_index >= 1 → compressed stream (offset by 1 into mPT_cmp).
        is_win_for_idx = k_index == 0
        if is_win_for_idx:
            if cutlass.const_expr(self.mma_qk_tiler[1] // self.page_size_win == 1):
                for i in cutlass.range_constexpr(page_per_tile_max):
                    k_idx[i] = common_params.mPT_win[0]
            else:
                for i in cutlass.range_constexpr(page_per_tile_max):
                    k_idx[i] = common_params.mPT_win[
                        common_params.blk_coord[0] * page_per_tile_max + i
                    ]
        else:
            cmp_offset = k_index - 1
            if cutlass.const_expr(self.mma_qk_tiler[1] // self.page_size_cmp == 1):
                for i in cutlass.range_constexpr(page_per_tile_max):
                    k_idx[i] = common_params.mPT_cmp[cmp_offset]
            else:
                for i in cutlass.range_constexpr(page_per_tile_max):
                    k_idx[i] = common_params.mPT_cmp[
                        (
                            cmp_offset * qk_params.tiled_mma_qk.thr_id.shape
                            + common_params.blk_coord[0]
                        )
                        * page_per_tile_max
                        + i
                    ]
        # load q once at first iteration (single Q stream)
        load_q_pipeline = common_params.load_q_pipeline
        if load_q:
            tma_bar_ptr = load_q_pipeline.producer_get_barrier(load_q_producer_state)
            load_q_pipeline.producer_acquire(load_q_producer_state)
            for i in cutlass.range_constexpr(self.iterations_qk_latent):
                cute.copy(
                    qk_params.tma_atom_q_latent,
                    qk_params.tQLgQL[None, 0, i],
                    qk_params.tQsQ[None, (i, 0)],
                    tma_bar_ptr=tma_bar_ptr,
                )
            load_q_producer_state.advance()
        # K load: branch on stream
        is_win = k_index == 0
        tma_bar_ptr = common_params.load_k_pipeline.producer_get_barrier(
            load_k_producer_state
        )
        common_params.load_k_pipeline.producer_acquire(load_k_producer_state)
        for i in range(self.iterations_qk_latent):
            if is_win:
                for k in range(page_per_tile_win):
                    cute.copy(
                        qk_params.tma_atom_c_latent_win,
                        qk_params.tCLgCL_win[None, i, k_idx[k]],
                        qk_params.tKCsKC_win[
                            None, k, 0, (i, load_k_producer_state.index)
                        ],
                        tma_bar_ptr=tma_bar_ptr,
                    )
            else:
                for k in range(page_per_tile_cmp):
                    cute.copy(
                        qk_params.tma_atom_c_latent_cmp,
                        qk_params.tCLgCL_cmp[None, i, k_idx[k]],
                        qk_params.tKCsKC_cmp[
                            None, k, 0, (i, load_k_producer_state.index)
                        ],
                        tma_bar_ptr=tma_bar_ptr,
                    )
        load_k_producer_state.advance()

        return load_q_producer_state, load_k_producer_state

    @cute.jit
    def load_tma_v_one_k_tile(
        self,
        common_params: SimpleNamespace,
        v_params: SimpleNamespace,
        k_index: cutlass.Int32,
        load_v_producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Load one k-tile of compressed latent transpose tensor(v). Updates the load qkv producer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param v_params: The load tma v parameters
        :type v_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param load_v_producer_state: The load v producer state
        :type load_v_producer_state: pipeline.PipelineState

        :return: The load qkv producer state
        :rtype: pipeline.PipelineState
        """
        page_per_tile_win = (
            self.mma_pv_tiler[2] * self.iterations_pv_k // self.page_size_win
        )
        page_per_tile_cmp = (
            self.mma_pv_tiler[2] * self.iterations_pv_k // self.page_size_cmp
        )
        page_per_subtile_win = ceil_div(page_per_tile_win, self.iterations_pv_k)
        page_per_subtile_cmp = ceil_div(page_per_tile_cmp, self.iterations_pv_k)
        page_per_tile_max = max(page_per_tile_win, page_per_tile_cmp)
        k_idx = cute.make_rmem_tensor(
            cute.make_layout(page_per_tile_max), cutlass.Int32
        )

        is_win_for_idx = k_index == 0
        if is_win_for_idx:
            if cutlass.const_expr(page_per_tile_win == 1):
                for i in cutlass.range_constexpr(page_per_tile_max):
                    k_idx[i] = common_params.mPT_win[0]
            else:
                for i in cutlass.range_constexpr(page_per_tile_max):
                    k_idx[i] = common_params.mPT_win[i]
        else:
            cmp_offset = k_index - 1
            if cutlass.const_expr(page_per_tile_cmp == 1):
                for i in cutlass.range_constexpr(page_per_tile_max):
                    k_idx[i] = common_params.mPT_cmp[cmp_offset]
            else:
                for i in cutlass.range_constexpr(page_per_tile_max):
                    k_idx[i] = common_params.mPT_cmp[cmp_offset * page_per_tile_max + i]

        # get the mbar ptr from pipeline.
        tma_bar_ptr = common_params.load_v_pipeline.producer_get_barrier(
            load_v_producer_state
        )
        common_params.load_v_pipeline.producer_acquire(load_v_producer_state)
        is_win = k_index == 0
        for j in cutlass.range_constexpr(self.iterations_pv_n):
            for i in cutlass.range_constexpr(self.iterations_pv_k):
                if is_win:
                    if cutlass.const_expr(page_per_tile_win > 1):
                        for k in cutlass.range_constexpr(page_per_subtile_win):
                            k_idx_i = k_idx[k + i * page_per_subtile_win]
                            cute.copy(
                                v_params.tma_atom_c_latent_transpose_win,
                                v_params.tCLTgCLT_win[None, j, 0, k_idx_i],
                                v_params.tVCsVC_win[
                                    None, 0, k, ((j, i), load_v_producer_state.index)
                                ],
                                tma_bar_ptr=tma_bar_ptr,
                            )
                    else:
                        cute.copy(
                            v_params.tma_atom_c_latent_transpose_win,
                            v_params.tCLTgCLT_win[None, j, i, k_idx[0]],
                            v_params.tVCsVC_win[
                                None, 0, 0, ((j, i), load_v_producer_state.index)
                            ],
                            tma_bar_ptr=tma_bar_ptr,
                        )
                else:
                    if cutlass.const_expr(page_per_tile_cmp > 1):
                        for k in cutlass.range_constexpr(page_per_subtile_cmp):
                            k_idx_i = k_idx[k + i * page_per_subtile_cmp]
                            cute.copy(
                                v_params.tma_atom_c_latent_transpose_cmp,
                                v_params.tCLTgCLT_cmp[None, j, 0, k_idx_i],
                                v_params.tVCsVC_cmp[
                                    None, 0, k, ((j, i), load_v_producer_state.index)
                                ],
                                tma_bar_ptr=tma_bar_ptr,
                            )
                    else:
                        cute.copy(
                            v_params.tma_atom_c_latent_transpose_cmp,
                            v_params.tCLTgCLT_cmp[None, j, i, k_idx[0]],
                            v_params.tVCsVC_cmp[
                                None, 0, 0, ((j, i), load_v_producer_state.index)
                            ],
                            tma_bar_ptr=tma_bar_ptr,
                        )
        load_v_producer_state.advance()
        return load_v_producer_state

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

        tStS_shape = tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tStS_staged_fake = tiled_mma_qk.make_fragment_C(
            cute.append(tStS_shape, self.mma_s_stage)
        )
        tStS_staged = cute.make_tensor(common_params.tmem_ptr, tStS_staged_fake.layout)

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
                stride=self.mma_pv_tiler[1] // self.warps_in_n,
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
        for q_stage in range(self.iterations_qk_latent):
            kc_stage = load_k_consumer_state.index
            for k_block in cutlass.range_constexpr(cute.size(qk_params.tSrQ.shape[2])):
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
        while k_tile_count > 0:
            p_cor_consumer_state, row_sum, row_max, correction_factor, no_correction = (
                self.get_correction_factor(common_params, p_cor_consumer_state)
            )
            if k_tile_count_init != k_tile_count:
                mma_o_consumer_state = self.rescale(
                    common_params,
                    mma_o_consumer_state,
                    correction_factor,
                    no_correction,
                )
            k_tile_count = k_tile_count - 1
            if k_tile_count == 0:
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
        corr_tmem_store_tiled_copy = tcgen05.make_tmem_copy(corr_tmem_store_atom, tCor)
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
        cute.arch.fence_view_async_tmem_store()
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

        cta_qk_tiler_local = (
            self.mma_qk_tiler[0] // self.cluster_shape_mnk[0],
            self.mma_qk_tiler[1],
            self.mma_qk_tiler[2],
        )
        cS_for_sink = cute.make_identity_tensor(
            cute.select(cta_qk_tiler_local, mode=[0, 1])
        )
        tStS_shape_for_sink = softmax_params.tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tStS_staged_fake_for_sink = softmax_params.tiled_mma_qk.make_fragment_C(
            cute.append(tStS_shape_for_sink, self.mma_s_stage)
        )
        tStS_staged_for_sink = cute.make_tensor(
            common_params.tmem_ptr, tStS_staged_fake_for_sink.layout
        )
        tAcc_for_sink = tStS_staged_for_sink[(None, None), 0, 0, 0]
        tidx_compute_for_sink = common_params.tidx % (
            self.num_compute_warps * self.threads_per_warp
        )
        tmem_load_atom_for_sink = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tmem_tiled_copy_for_sink = tcgen05.make_tmem_copy(
            tmem_load_atom_for_sink, tAcc_for_sink
        )
        tmem_thr_copy_for_sink = tmem_tiled_copy_for_sink.get_slice(
            tidx_compute_for_sink
        )
        tTR_tS_for_sink = tmem_thr_copy_for_sink.partition_D(cS_for_sink)
        local_row = tTR_tS_for_sink[0][0]
        head_idx = common_params.blk_coord[0] * cta_qk_tiler_local[0] + local_row
        sink_row_max = self.acc_dtype(self.init_row_max)
        sink_row_sum = self.acc_dtype(0.0)
        if common_params.blk_coord[3] == 0:
            sink_row_max = common_params.attn_sink_unscaled[head_idx]
            if sink_row_max != self.acc_dtype(self.init_row_max):
                sink_row_sum = self.acc_dtype(1.0)
            elif cutlass.const_expr(self.is_causal):
                if common_params.window_valid_len == 0:
                    # No sink and no visible window entry: use a finite max
                    # with a zero sum so the epilogue emits O=0, LSE=-inf.
                    sink_row_max = self.acc_dtype(0.0)
        elif not cute.elem_less(k_index * self.mma_qk_tiler[1], common_params.K_valid):
            # A causal query can have fewer visible HCA tiles than its batch's
            # cache length.  Keep an empty split numerically well-defined so
            # split-K reduction can discard it using an LSE of -inf.
            sink_row_max = self.acc_dtype(0.0)

        row_max = self.acc_dtype(self.init_row_max)
        row_sum = self.acc_dtype(0.0)
        correction_factor = self.acc_dtype(1)
        odd_k_tile = k_tile_count % 2 == 1
        if cutlass.const_expr(is_second_compute_warp):
            k_index = k_index + 1
            k_tile_count = k_tile_count // 2
        else:
            k_tile_count = (k_tile_count + 1) // 2
        valid_k_tile_count = k_tile_count > 0
        self.softmax_warps_initial_sync_bar.arrive_and_wait()
        common_params.p_cor_pipeline.producer_acquire(p_cor_producer_state)

        if cutlass.const_expr(is_second_compute_warp):
            self.init_p_cor_metadata(
                common_params,
                softmax_params,
                p_cor_producer_state,
                sink_row_max,
                sink_row_sum,
            )
            self.softmax_order_bar_0.arrive()

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
                self.softmax_order_bar_1.arrive_and_wait()
        else:
            if not odd_k_tile:
                self.softmax_order_bar_0.arrive_and_wait()
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
        init_tidx = common_params.tidx % (
            self.num_compute_warps * self.threads_per_warp
        )
        init_tStS_shape = softmax_params.tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        init_tStS_layout = softmax_params.tiled_mma_qk.make_fragment_C(
            cute.append(init_tStS_shape, self.mma_s_stage)
        ).layout
        init_tStS = cute.make_tensor(common_params.tmem_ptr, init_tStS_layout)
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
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def load_other_group_metadata(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        p_cor_producer_state: pipeline.PipelineState,
    ) -> tuple[cutlass.Float32, cutlass.Float32]:
        other_stage = (p_cor_producer_state.index + 1) % self.mma_s_stage
        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        tStS_shape = softmax_params.tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tStS_layout = softmax_params.tiled_mma_qk.make_fragment_C(
            cute.append(tStS_shape, self.mma_s_stage)
        ).layout
        tStS = cute.make_tensor(common_params.tmem_ptr, tStS_layout)
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
        rCor = cute.make_fragment_like(cCor_part[None, None, None, 0], self.acc_dtype)
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
        cute.copy(tiled_copy, rCor, tCor_for_copy[None, None, None, saved_stage_idx])
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def softmax(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        k_index: cutlass.Int32,
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

        tStS_shape = softmax_params.tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tStS_staged_fake = softmax_params.tiled_mma_qk.make_fragment_C(
            cute.append(tStS_shape, self.mma_s_stage)
        )
        tStS_staged = cute.make_tensor(common_params.tmem_ptr, tStS_staged_fake.layout)
        tStS = tStS_staged[None, None, None, mma_s_consumer_state.index]

        tAcc = tStS[(None, None), 0, 0]
        cta_qk_tiler = (
            self.mma_qk_tiler[0] // self.cluster_shape_mnk[0],
            self.mma_qk_tiler[1],
            self.mma_qk_tiler[2],
        )
        cS = cute.make_identity_tensor(cute.select(cta_qk_tiler, mode=[0, 1]))

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc)

        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc = tmem_thr_copy.partition_S(tAcc)
        tTR_tS = tmem_thr_copy.partition_D(cS)
        tTR_rAcc = cute.make_fragment_like(tTR_tS, self.acc_dtype)

        window_valid_len = cutlass.Int32(0)
        if cutlass.const_expr(self.is_causal):
            window_valid_len = common_params.window_valid_len

        row_max_new = row_max
        arch = BaseDSL._get_dsl().get_arch_enum()
        if cutlass.const_expr(arch >= Arch.sm_100 and arch <= Arch.sm_100f):
            cute.copy(tmem_tiled_copy, tTR_tAcc, tTR_rAcc)
            for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                if apply_mask:
                    sparse_k_idx = tTR_tS[i][1] + self.mma_qk_tiler[1] * k_index
                    if cutlass.const_expr(self.is_causal):
                        is_valid = cute.elem_less(sparse_k_idx, common_params.K_valid)
                        if cute.elem_less(sparse_k_idx, self.mma_qk_tiler[1]):
                            is_valid = cute.elem_less(sparse_k_idx, window_valid_len)
                        tTR_rAcc[i] = tTR_rAcc[i] if is_valid else -self.acc_dtype.inf
                    else:
                        tTR_rAcc[i] = (
                            tTR_rAcc[i]
                            if cute.elem_less(sparse_k_idx, common_params.K_valid)
                            else -self.acc_dtype.inf
                        )
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
                    if cutlass.const_expr(self.is_causal):
                        is_valid = cute.elem_less(sparse_k_idx, common_params.K_valid)
                        if cute.elem_less(sparse_k_idx, self.mma_qk_tiler[1]):
                            is_valid = cute.elem_less(sparse_k_idx, window_valid_len)
                        tTR_rAcc[i] = tTR_rAcc[i] if is_valid else -self.acc_dtype.inf
                    else:
                        tTR_rAcc[i] = (
                            tTR_rAcc[i]
                            if cute.elem_less(sparse_k_idx, common_params.K_valid)
                            else -self.acc_dtype.inf
                        )
                row_max_new = tTR_rAcc.load().reduce(
                    cute.ReductionOp.MAX, row_max_new, 0
                )
            else:
                row_max_new = cute.arch.fmax(row_max_new, tTR_rMax[0])

        cute.arch.fence_view_async_tmem_load()
        softmax_params.mma_s_pipeline.consumer_release(mma_s_consumer_state)

        group_offset = self.num_compute_warps * self.threads_per_warp
        if cutlass.const_expr(is_second_compute_warp):
            my_base = group_offset
        else:
            my_base = 0
        if cutlass.const_expr(self.warps_in_n == 2):
            common_params.smem_exchange[my_base + tidx] = row_max_new
            softmax_exchange_sync_bar.arrive_and_wait()
            row_max_new = cute.arch.fmax(
                row_max_new,
                common_params.smem_exchange[
                    my_base
                    + (tidx + 64) % (self.num_compute_warps * self.threads_per_warp)
                ],
            )

        if cutlass.const_expr(is_second_compute_warp):
            self.softmax_order_bar_1.arrive_and_wait()
        else:
            self.softmax_order_bar_0.arrive_and_wait()

        other_row_max, other_row_sum = self.load_other_group_metadata(
            common_params, softmax_params, p_cor_producer_state
        )
        row_max_new = cute.arch.fmax(row_max_new, other_row_max)
        row_max = other_row_max
        row_sum = other_row_sum

        correction_factor = cute.math.exp2(
            (row_max - row_max_new) * softmax_params.softmax_scale_log2, fastmath=True
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
        fma_c = (0.0 - row_max_new) * softmax_params.softmax_scale_log2
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
        cute.arch.fence_view_async_shared()
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
            self.softmax_order_bar_0.arrive()
        else:
            self.softmax_order_bar_1.arrive()

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
        cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma
    ]:
        """Tensor memory load partition for rescale and epilogue.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param tiled_mma_pv: The tiled mma pv
        :type tiled_mma_pv: cute.TiledMma
        :param iter_n: The iteration number
        :type iter_n: int

        :return: The tiled mma pv, the tiled mma pv, the tiled mma pv, the tiled mma pv, the tiled mma pv
        :rtype: tuple[cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma]
        """

        tOtO_shape = tiled_mma_pv.partition_shape_C(
            cute.select(self.mma_pv_tiler, mode=[0, 1])
        )
        tOtO = tiled_mma_pv.make_fragment_C(tOtO_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                common_params.L // self.mma_pv_tiler[1],
                stride=self.mma_pv_tiler[1] // self.warps_in_n,
            ),
        )
        tOtO = cute.make_tensor(
            common_params.tmem_ptr + self.tmem_o_offset, tOtO_layout
        )
        tOtO = tOtO[None, None, None, iter_n]

        tAcc = tOtO[(None, None), 0, 0]

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
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

    def get_correction_factor(
        self,
        common_params: SimpleNamespace,
        p_cor_consumer_state: pipeline.PipelineState,
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
        corr_tmem_load_tiled_copy = tcgen05.make_tmem_copy(corr_tmem_load_atom, tCor)
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
        return p_cor_consumer_state, row_sum, row_max, correction_factor, no_correction

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
                    tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
                )
                tmem_store_tiled_copy = tcgen05.make_tmem_copy(tmem_store_atom, tAcc)

                # load o
                cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)
                # rescale, using `mul_packed_f32x2` to reduce the number of instructions
                for i in cutlass.range(
                    cute.size(tTR_rAcc), vectorize=True, unroll_full=True
                ):
                    tTR_rAcc[i] = tTR_rAcc[i] * correction_factor

                # store o to tensor memory for next k tile
                cute.copy(tmem_store_tiled_copy, tTR_rAcc, tTR_tAcc)

            cute.arch.fence_view_async_tmem_store()
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
        """Epilogue for one k-tile. Updates the related pipeline state.

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
            self.epilogue_exchange_sync_bar.wait()
            # (64, 2)
            row_sum = (
                row_sum
                + common_params.smem_exchange[
                    (tidx + 64) % (self.num_compute_warps * self.threads_per_warp)
                ]
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
            cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)

            # Apply output scale and normalize by row_sum.  An empty causal
            # split has row_sum == 0 and contributes a zero vector with -inf
            # LSE to the split-K reduction.
            if row_sum == self.acc_dtype(0.0):
                tTR_rAcc.fill(0.0)
            else:
                for i in cutlass.range(
                    cute.size(tTR_rAcc), vectorize=True, unroll_full=True
                ):
                    tTR_rAcc[i] = (
                        tTR_rAcc[i]
                        * epilogue_params.output_scale
                        * cute.arch.rcp_approx(row_sum)
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

            # store the lse to global memory
            cta_pv_tiler = (
                self.mma_pv_tiler[0] // self.cluster_shape_mnk[0],
                self.mma_pv_tiler[1],
                self.mma_pv_tiler[2],
            )
            gLSE = None
            cLSE = None
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
            if cutlass.const_expr(self.warps_in_n == 2):
                if cute.elem_less(cLSE[tidx][0], common_params.H):
                    gLSE[tidx] = lse

            cute.arch.fence_view_async_tmem_load()
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
        page_size_win: int,
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
        :param page_size_win: Page size for window-KV stream (power-of-two
            multiple of page_size_cmp, or vice versa)
        :type page_size_win: int

        :return: Whether the HCA kernel can be implemented
        :rtype: bool
        """
        if L != 512:
            return False
        if in_dtype not in [cutlass.Float8E4M3FN]:
            return False
        if out_dtype not in [cutlass.Float8E4M3FN, cutlass.BFloat16]:
            return False
        if acc_dtype != cutlass.Float32 or lse_dtype != cutlass.Float32:
            return False
        # Both page sizes must divide mma_qk_tiler[1]; neither may equal 1
        # (TMA 128B alignment requirement).
        if mma_qk_tiler_mn[1] % page_size_cmp != 0 or page_size_cmp == 1:
            return False
        if mma_qk_tiler_mn[1] % page_size_win != 0 or page_size_win == 1:
            return False
        big, small = (
            max(page_size_win, page_size_cmp),
            min(page_size_win, page_size_cmp),
        )
        if big % small != 0:
            return False
        ratio = big // small
        if ratio & (ratio - 1) != 0:
            return False
        if mma_qk_tiler_mn[0] != mma_pv_tiler_mn[0] or mma_qk_tiler_mn[0] != 128:
            return False
        if is_var_split_kv and not is_var_seq:
            return False
        if H > 128 or (H < 128 and split_kv != 1):
            return False
        if S <= 0:
            return False
        if K <= 0:
            return False
        return True
