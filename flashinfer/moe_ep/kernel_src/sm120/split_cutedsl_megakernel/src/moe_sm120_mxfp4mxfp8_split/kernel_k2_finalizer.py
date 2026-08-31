# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""One-CTA cross-rank finalizer for split SM120 MegaMoE K2."""

from typing import Optional

import cutlass
import cutlass.cute as cute

from .fc1_fc2_fuse_sched import MoEFusedFc12SchedulerParams
from .kernel_fc2_combine import Sm120Fc2CombineKernel


class Sm120K2TailFinalizerKernel(Sm120Fc2CombineKernel):
    """Finalize peer stores after the stream-ordered K2 completion event.

    The persistent K2 grid can oversubscribe its Green partition. Keeping the
    cross-rank rendezvous out of those workers avoids coupling network progress
    to CTA admission and also handles graph-capture buckets with no valid work.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threads_per_cta = 4 * 32
        self.occupancy = 1
        self.token_comm.kernel_tail_threads = self.threads_per_cta

    @cute.kernel
    def fc2_combine_kernel_impl(self, tiled_mma: cute.TiledMma, tiled_mma_sfb: cute.TiledMma, tma_atom_fc1_weight: cute.CopyAtom, tma_tensor_fc1_weight: cute.Tensor, tma_atom_activation: cute.CopyAtom, tma_tensor_activation: cute.Tensor, tma_atom_fc1_weight_sf: cute.CopyAtom, tma_tensor_fc1_weight_sf: cute.Tensor, tma_atom_activation_sf: cute.CopyAtom, tma_tensor_activation_sf: cute.Tensor, tma_atom_fc2_weight: cute.CopyAtom, tma_tensor_fc2_weight: cute.Tensor, tma_atom_fc1_output_as_fc2_input: cute.CopyAtom, tma_tensor_fc1_output_as_fc2_input: cute.Tensor, tma_atom_fc2_weight_sf: cute.CopyAtom, tma_tensor_fc2_weight_sf: cute.Tensor, tma_atom_fc1_output_sf_as_fc2_input: cute.CopyAtom, tma_tensor_fc1_output_sf_as_fc2_input: cute.Tensor, fc1_weight_gemm: cute.Tensor, activation_gemm: cute.Tensor, fc1_output_gemm: cute.Tensor, fc1_weight_sf_gemm: cute.Tensor, activation_sf_gemm: cute.Tensor, fc1_output_sf_gemm: cute.Tensor, fc2_weight_gemm: cute.Tensor, fc2_output: cute.Tensor, fc2_weight_sf_gemm: cute.Tensor, fc1_output_sf_gemm_for_fc2_load: cute.Tensor, topk_scores: cute.Tensor, fc1_done_counter: cute.Tensor, combine_ready_flags: Optional[cute.Tensor], fc2_block_done_counter: Optional[cute.Tensor], fc1_alpha: Optional[cute.Tensor], fc2_alpha: Optional[cute.Tensor], fc1_norm_const: Optional[cute.Tensor], sched_params: MoEFusedFc12SchedulerParams, cluster_layout_vmnk: cute.Layout, cluster_layout_sfb_vmnk: cute.Layout, a_smem_layout_staged: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout, sfa_smem_layout_staged: cute.Layout, sfb_smem_layout_staged: cute.Layout, fc1_output_smem_layout_staged: cute.ComposedLayout, token_comm_args=None, green_trace: Optional[cute.Tensor]=None, k2_ready_queue_desc: Optional[cute.Tensor]=None, k2_ready_queue_ready: Optional[cute.Tensor]=None, k2_ready_queue_state: Optional[cute.Tensor]=None):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        self.token_comm.kernel_tail_after_grid_drain(
            token_comm_args,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
        )

    fc1fc2_kernel_impl = fc2_combine_kernel_impl


__all__ = ["Sm120K2TailFinalizerKernel"]
