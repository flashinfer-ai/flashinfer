# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# mypy: disable-error-code="call-overload, misc"

"""Common state for the specialized SM120 NVFP4 Conv3d kernel."""

import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm120_utils

from flashinfer.cute_dsl.sm120_blockscaled import (
    compute_sm120_blockscaled_stages,
    make_sm120_blockscaled_smem_layouts,
    make_sm120_fp4_mma_op,
)


class Sm120Nvfp4KernelBase:
    """Initialize the SM120 NVFP4 MMA, pipelines, and staged layouts."""

    def __init__(
        self,
        acc_dtype,
        sf_vec_size,
        tile_shape_mnk,
        epi_tile,
    ):
        self.acc_dtype = acc_dtype
        self.sf_vec_size = sf_vec_size
        self.cluster_shape_mnk = (1, 1, 1)
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        self.epi_tile = tuple(epi_tile)
        self.tiled_mma = None

        self.occupancy = 1
        self.num_mma_warps = 8
        self.tma_load_warp_id = self.num_mma_warps
        self.num_threads_per_warp = 32
        self.threads_per_cta = (self.num_mma_warps + 1) * self.num_threads_per_warp
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")

        self.ab_stage = None
        self.epi_stage = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.buffer_align_bytes = 1024

        self.mma_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.num_threads_per_warp // 2,
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

    def _setup_attributes(self):
        mma_op = make_sm120_fp4_mma_op(
            self.a_dtype,
            self.b_dtype,
            self.acc_dtype,
            self.sf_dtype,
            self.sf_vec_size,
        )
        self.smem_alloc_a_dtype = self.a_dtype
        self.smem_alloc_b_dtype = self.b_dtype
        atom_layout = cute.make_layout((2, 2, 1))
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk,
            self.sf_vec_size,
            False,
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op,
            atom_layout,
            permutation_mnk=permutation_mnk,
        )
        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        from flashinfer.cute_dsl.utils import (
            sm120_make_smem_layout_sfa,
            sm120_make_smem_layout_sfb,
        )

        sfa_smem_layout_per_stage = sm120_make_smem_layout_sfa(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )
        sfb_smem_layout_per_stage = sm120_make_smem_layout_sfb(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )
        self.ab_stage, self.epi_stage = compute_sm120_blockscaled_stages(
            self.tile_shape_mnk,
            self.smem_alloc_a_dtype,
            self.smem_alloc_b_dtype,
            self.sf_dtype,
            sfa_smem_layout_per_stage,
            sfb_smem_layout_per_stage,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )
        if self.ab_stage <= 0 or self.epi_stage <= 0:
            raise ValueError("insufficient SM120 shared memory for this Conv3d tile")

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = make_sm120_blockscaled_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.smem_alloc_a_dtype,
            self.a_layout,
            self.smem_alloc_b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
            self.sf_vec_size,
            self.tiled_mma,
        )

    @cute.jit
    def advance(self, state: pipeline.PipelineState, iterations):
        """Advance a pipeline state by a compile-time number of iterations."""
        if iterations < state.stages and ((state._index + iterations) >= state.stages):
            state._phase ^= 1
        if (
            iterations >= state.stages
            and (((state._index + iterations) // state.stages) % 2) == 1
        ):
            state._phase ^= 1
        state._index = (state._index + iterations) % state.stages
        state._count += iterations
        return state

    @cute.jit
    def make_and_init_order_barrier(self, order_mbar_ptr, group_id):
        """Create the two-group ping-pong ordering barrier."""
        return pipeline.PipelineOrder.create(
            barrier_storage=order_mbar_ptr,
            depth=2,
            length=2,
            group_id=group_id,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                128,
            ),
            defer_sync=True,
        )


__all__ = ["Sm120Nvfp4KernelBase"]
