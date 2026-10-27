# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""SM100 blockscaled split MoE with weights on M and activations on runtime N.

FC1 and FC2 share native NVFP4, MXFP4, MXFP8, and mixed MXFP4/MXFP8
mainloops. Both phases use regular TMA for prepared weights and routed
activation rows. Canonical activation scales use TMA for N128/N256 route
tiles and cp.async for smaller tiles that can start inside a 128-row SF atom.
FC1 preparation preserves payload and scale bytes; per-token activation
scales retain their original token indexing through the routing map.

M128 uses one CTA; M256 uses two CTAs with 128 channels each. Logical M512
reuses an activation tile across two M256 channel tiles. Allocation
uses the tactic width while each MMA descriptor rounds the live routed-token
count to 16 columns.
"""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass._mlir import ir
from cutlass._mlir.dialects import builtin, llvm, nvvm
from cutlass._mlir.dialects import cute as _cute_ir
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import dsl_user_op

from flashinfer.tllm_enums import (
    ActivationType,
    DEFAULT_SWIGLU_ALPHA,
    DEFAULT_SWIGLU_BETA,
    DEFAULT_SWIGLU_LIMIT,
)
from .custom_pipeline import PipelineCpAsyncUmma, PipelineTmaUmma, PipelineUmmaAsync
from .utils import (
    Sm100BlockScaledMoeEpilogue,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    release_gpu,
)


# Imported only for explicitly instrumented diagnostic builds.
_iket = None


@dsl_user_op
def _blockscaled_cluster_try_wait(mbarrier, phase, loc=None, ip=None):
    """Acquire the peer CTA's scale publication at cluster scope."""
    return cutlass.Boolean(
        nvvm.mbarrier_wait_parity(
            mbarrier.to_llvm_ptr(loc=loc, ip=ip),
            cutlass.Int32(phase).ir_value(loc=loc, ip=ip),
            nvvm.MBarrierWaitKind.TRY,
            scope=nvvm.MemScopeKind.CLUSTER,
            order=nvvm.MemOrderKind.ACQUIRE,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _blockscaled_tma_sfb(
    atom, destination, mbarrier, k_block, row_block, cta_group, loc=None, ip=None
):
    """Load one canonical 128-row SF block, replicated in each CTA."""
    executable = _cute_nvgpu_ir.atom_make_exec_tma(atom._trait.value, loc=loc, ip=ip)
    descriptor_type = _cute_ir.PtrType.get(
        _cute_nvgpu_ir.TmaDescriptorTiledType.get(), AddressSpace.generic, 64
    )
    descriptor = _cute_nvgpu_ir.get_tma_desc_addr(
        descriptor_type, executable, loc=loc, ip=ip
    )
    barrier = cutlass.Int32(mbarrier.toint())
    if cta_group == 2:
        # Both CTAs contribute their full local SFB tile to the leader barrier.
        barrier = barrier & cutlass.Int32(-16777217)
    elif cta_group != 1:
        raise ValueError("Unsupported TMA CTA group")
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(destination.toint()).ir_value(loc=loc, ip=ip),
            descriptor.toint().ir_value(loc=loc, ip=ip),
            barrier.ir_value(loc=loc, ip=ip),
            cutlass.Int32(k_block).ir_value(loc=loc, ip=ip),
            cutlass.Int32(row_block).ir_value(loc=loc, ip=ip),
            cutlass.Int64(0).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.tensor.3d.shared::cta.global."
        f"mbarrier::complete_tx::bytes.cta_group::{cta_group}.L2::cache_hint "
        "[$0], [$1, {0, $3, $4}], [$2], $5;",
        "r,l,r,r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _blockscaled_dynamic_mma(
    acc,
    a,
    b,
    sfa,
    sfb,
    valid_tokens,
    k_tile,
    instruction,
    descriptor,
    cta_group,
    loc=None,
    ip=None,
):
    """Issue native blockscaled MMAs with a runtime token-width descriptor."""
    i32 = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)
    n_bits = (((cutlass.Int32(valid_tokens) + 15) & -16) >> 3) << 17
    for inner_k in range(cute.size(a, mode=[2])):
        a_atom = a[(None, 0, inner_k)]
        b_atom = b[(None, 0, inner_k)]
        sfa_atom = sfa[(None, 0, inner_k)]
        sfb_atom = sfb[(None, 0, inner_k)]
        acc_atom = acc[(None, 0, 0)]
        a_value = (
            a_atom.iterator.value
            if hasattr(a_atom.iterator, "value")
            else a_atom.iterator
        )
        a_desc = builtin.unrealized_conversion_cast([i64], [a_value])
        b_value = (
            b_atom.iterator.value
            if hasattr(b_atom.iterator, "value")
            else b_atom.iterator
        )
        b_desc = builtin.unrealized_conversion_cast([i64], [b_value])
        a_sf = builtin.unrealized_conversion_cast([i32], [sfa_atom.iterator.value])
        b_sf = builtin.unrealized_conversion_cast([i32], [sfb_atom.iterator.value])
        dst = builtin.unrealized_conversion_cast([i32], [acc_atom.iterator.value])
        desc = cutlass.Int32(descriptor) | n_bits
        desc = desc | (((cutlass.Int32(a_sf) >> 30) & 3) << 29)
        desc = desc | (((cutlass.Int32(b_sf) >> 30) & 3) << 4)
        accumulate = k_tile != 0 if inner_k == 0 else True
        with cute.arch.elect_one():
            llvm.inline_asm(
                None,
                [
                    dst,
                    a_desc,
                    b_desc,
                    desc.ir_value(),
                    cutlass.Int32(cutlass.Boolean(accumulate)).ir_value(),
                    a_sf,
                    b_sf,
                ],
                "{ .reg .pred p; setp.ne.b32 p, $4, 0; "
                f"tcgen05.mma.cta_group::{cta_group}.kind::{instruction} "
                "[$0], $1, $2, $3, [$5], [$6], p; }",
                "r,l,l,r,r,r,r",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )


@cute.jit
def _blockscaled_sf_offset(
    row, channel, padded_channels, sf_vec_size: cutlass.Constexpr
):
    """Offset in the prepared 128-row, 4-scale blockscaled layout."""
    return (
        (row // 128) * (padded_channels // (4 * sf_vec_size)) * 512
        + (channel // (4 * sf_vec_size)) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + (channel % (4 * sf_vec_size)) // sf_vec_size
    )


class Sm100BlockScaledGroupedGemmKernel:
    """Persistent one/two-CTA grouped GEMM shared by the split MoE phases."""

    def __init__(
        self,
        activation_dtype,
        weight_dtype,
        sf_dtype,
        sf_vec_size,
        scatter_output,
        per_token_scale_is_routed,
        k_steps,
        num_stages,
        overlap_epilogue,
        overlap_activation_accumulators,
        pdl_signal_at,
        use_clc_scheduler,
        num_tile_info_stages,
        mma_tiler_mnk=(128, 128, 256),
        activation_type=None,
        top_k=1,
        swiglu_alpha=DEFAULT_SWIGLU_ALPHA,
        swiglu_beta=DEFAULT_SWIGLU_BETA,
        swiglu_limit=DEFAULT_SWIGLU_LIMIT,
        situ_beta=None,
        situ_linear_beta=None,
        use_per_token_scale=False,
        use_fused_finalize=False,
        enable_pdl=False,
        output_dtype=cutlass.BFloat16,
        final_scale_dtype=cutlass.Float32,
        max_active_clusters=1,
        generate_sfc=False,
        prefetch_weights_before_wait=False,
        publish_routing_before_signal=False,
        weight_scale_cache_policy=0,
        activation_cache_policy=0,
        enable_iket=False,
    ):
        if not self.is_valid_dtypes(
            activation_dtype, weight_dtype, sf_dtype, sf_vec_size
        ):
            raise ValueError("Unsupported SM100 blockscaled operand/scale types")
        if (
            k_steps is not None and (type(k_steps) is not int or k_steps not in (4, 8))
        ) or (num_stages is not None and num_stages < 1):
            raise ValueError("Invalid block-scaled pipeline depth or K steps")
        if pdl_signal_at not in ("startup", "producer_exit"):
            raise ValueError("Unsupported PDL signal boundary")
        if use_fused_finalize and not scatter_output:
            raise ValueError("Fused finalization requires output scattering")
        if num_tile_info_stages < 1:
            raise ValueError("Tile metadata needs at least one pipeline stage")
        self.use_clc_scheduler = use_clc_scheduler
        self.num_tile_info_stages = num_tile_info_stages
        self.scatter_output = scatter_output
        self.per_token_scale_is_routed = per_token_scale_is_routed
        self.pdl_signal_at = pdl_signal_at
        self.prefetch_weights_before_wait = prefetch_weights_before_wait
        self.publish_routing_before_signal = publish_routing_before_signal
        self.weight_scale_cache_policy = weight_scale_cache_policy
        self.activation_cache_policy = activation_cache_policy
        self.enable_iket = enable_iket
        if enable_iket:
            # Compile with options="iket" to collect these ranges. Keep optional
            # tracing dependencies and all device markers out of normal builds.
            global _iket
            try:
                from cutlass.cute.experimental import iket as _iket
            except (ImportError, NotImplementedError):
                from cutlass.cute import iket as _iket
        self.activation_dtype = activation_dtype
        self.weight_dtype = weight_dtype
        self.sf_dtype = sf_dtype
        self.sf_vec_size = sf_vec_size
        both_fp4 = (
            activation_dtype is cutlass.Float4E2M1FN
            and weight_dtype is cutlass.Float4E2M1FN
        )
        # Reuse the routed activation tile across physical M256 channel tiles.
        if mma_tiler_mnk[0] not in (128, 256, 512):
            raise ValueError("Unsupported logical output-channel tile")
        self.channel_tiles = max(1, mma_tiler_mnk[0] // 256)
        self.gather_k = 256 if both_fp4 else 128
        if mma_tiler_mnk[2] != self.gather_k:
            raise ValueError("Unsupported public blockscaled K tile")
        if k_steps is None:
            k_steps = self._select_k_steps(
                mma_tiler_mnk,
                activation_dtype,
                weight_dtype,
                sf_vec_size,
                num_stages,
                overlap_epilogue,
                overlap_activation_accumulators,
                output_dtype.width // 8
                if (
                    scatter_output
                    and not use_fused_finalize
                    and output_dtype in (cutlass.BFloat16, cutlass.Float16)
                    and not generate_sfc
                )
                else 0,
                use_clc_scheduler,
                num_tile_info_stages,
            )
        self.k_steps = k_steps
        self.tile_k = self.gather_k * self.k_steps // 4
        mma_tiler_mnk = (
            mma_tiler_mnk[0] // self.channel_tiles,
            mma_tiler_mnk[1],
            self.tile_k,
        )
        self.unpack_weights = weight_dtype.width < activation_dtype.width
        self.a_storage_dtype = cutlass.Int8 if self.unpack_weights else weight_dtype
        self.mma_m = mma_tiler_mnk[0]
        if self.mma_m not in (128, 256):
            raise ValueError(
                "Blockscaled split MoE requires M128 or M256 with a supported K tile"
            )
        if mma_tiler_mnk[1] not in (16, 32, 64, 128, 256):
            raise ValueError("Unsupported routed-token tile")
        self.cta_group_size = self.mma_m // 128
        self.cta_group = (
            tcgen05.CtaGroup.ONE if self.cta_group_size == 1 else tcgen05.CtaGroup.TWO
        )
        if both_fp4:
            if sf_vec_size == 16:
                self.instruction = "mxf4nvf4.block_scale.block16"
            elif sf_vec_size == 32:
                self.instruction = "mxf4.block_scale.block32"
            else:
                raise ValueError("Unsupported FP4 scale vector size")
            self.descriptor = (self.mma_m << 20) | 0x480
        else:
            formats = {
                cutlass.Float8E4M3FN: 0,
                cutlass.Float8E5M2: 1,
                cutlass.Float4E2M1FN: 5,
            }
            self.instruction = "mxf8f6f4.block_scale.block32"
            self.descriptor = (
                (self.mma_m << 20)
                | (formats[weight_dtype] << 7)
                | (formats[activation_dtype] << 10)
            )
        if sf_dtype is cutlass.Float8E8M0FNU:
            self.descriptor |= 1 << 23
        if publish_routing_before_signal and pdl_signal_at != "startup":
            raise ValueError("Routing publication requires a startup signal")
        self.route_tile = mma_tiler_mnk[1]
        # The epilogue transposes 32-column chunks even for a 16-row tile.
        self.allocated_n = max(32, self.route_tile)
        self.physical_tiler = (self.mma_m, self.allocated_n, self.tile_k)
        self.b_rows = self.allocated_n // self.cta_group_size
        self.b_copy_rows = self.route_tile // self.cta_group_size
        self.sf_block_k = 4 * sf_vec_size
        self.sf_stage_bytes = 128 * self.tile_k // sf_vec_size
        # Group-2 S2T reads each CTA's local scales: SFB remains full-N even
        # though each CTA stores only its half of the B activation tile.
        self.sfb_row_blocks = max(1, self.allocated_n // 128)
        self.sfb_stage_bytes = self.sfb_row_blocks * self.sf_stage_bytes
        # Route tiles aligned to 128 rows expose complete canonical SF atoms.
        # Smaller route tiles can start inside an atom and retain cp.async.
        self.tma_sfb = self.route_tile >= 128
        self.sf_requires_cluster_join = self.cta_group_size > 1 and not self.tma_sfb
        self.activation_type = activation_type
        self.top_k = top_k
        self.use_fused_finalize = use_fused_finalize
        self.enable_pdl = enable_pdl
        self.output_dtype = output_dtype
        self.max_active_clusters = max_active_clusters
        if max_active_clusters < self.cta_group_size:
            raise ValueError("The launch needs at least one complete CTA group")
        self.requested_num_stages = num_stages
        self.num_stages = num_stages
        self.overlap_epilogue = overlap_epilogue
        self.threads_per_cta = 288 if self.overlap_epilogue else 160
        self.mma_warp = self.threads_per_cta // 32 - 1
        self.schedule_warp = self.threads_per_cta // 32
        if self.use_clc_scheduler:
            self.threads_per_cta += 32
        self.overlap_acc = False
        self.acc_cols = self.allocated_n * self.channel_tiles
        self.sf_cols = self.tile_k // self.sf_vec_size
        self.sfb_cols = self.sf_cols * self.sfb_row_blocks
        scale_cols = self.sf_cols * self.channel_tiles + self.sfb_cols
        self.acc_stages, self.can_overlap_accumulators = self._accumulator_storage(
            self.acc_cols, scale_cols, self.overlap_epilogue
        )
        self.overlap_activation_accumulators = (
            self.can_overlap_accumulators
            if overlap_activation_accumulators is None
            else overlap_activation_accumulators
        )
        if self.overlap_activation_accumulators and not self.can_overlap_accumulators:
            raise ValueError("Unsupported overlapping activation accumulator geometry")
        self.tmem_cols = (
            1
            << (
                self.acc_stages * self.acc_cols
                + self.sf_cols * self.channel_tiles
                + self.sfb_cols
                - 1
            ).bit_length()
        )
        # Stream weights while preserving activation reuse across channel tiles.
        self.weight_cache_policy = 0x12F0000000000000  # EVICT_FIRST
        self.epilogue = Sm100BlockScaledMoeEpilogue(
            mma_tiler_mnk=(128, mma_tiler_mnk[1], mma_tiler_mnk[2]),
            activation_type=activation_type,
            scatter_output=scatter_output,
            per_token_scale_is_routed=per_token_scale_is_routed,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            use_per_token_scale=use_per_token_scale,
            use_fused_finalize=use_fused_finalize,
            top_k=top_k,
            output_dtype=output_dtype,
            final_scale_dtype=final_scale_dtype,
            sf_vec_size=sf_vec_size,
            sf_dtype=sf_dtype,
            generate_sfc=generate_sfc,
        )

    @staticmethod
    def _accumulator_storage(acc_cols, scale_cols, overlap_epilogue):
        capacity = cute.arch.get_max_tmem_alloc_cols("sm_100")
        stages = min(2, (capacity - scale_cols) // acc_cols) if overlap_epilogue else 1
        if stages < 1 or stages * acc_cols + scale_cols > capacity:
            raise ValueError("Not enough SM100 tensor memory for one accumulator tile")
        # The current epilogues release at most 64 columns before the next MMA:
        # one/two 32-column activation slabs, or the first 64-column bulk slab.
        # This ownership constraint is separate from the hardware capacity.
        can_overlap = overlap_epilogue and 2 * acc_cols == capacity and scale_cols <= 64
        return stages, can_overlap

    @staticmethod
    def _stage_storage_bytes(
        tile_k,
        allocated_n,
        cta_group_size,
        channel_tiles,
        activation_dtype,
        weight_dtype,
        sf_vec_size,
    ):
        # Each CTA owns 128 weight rows. Mixed FP4 weights occupy byte slots
        # for mxf8f6f4, and each SFB atom retains all 128 scale rows.
        return (
            128
            * tile_k
            * max(weight_dtype.width, activation_dtype.width)
            // 8
            * channel_tiles
            + allocated_n // cta_group_size * tile_k * activation_dtype.width // 8
            + 128 * tile_k // sf_vec_size * (channel_tiles + max(1, allocated_n // 128))
        )

    @staticmethod
    def _pipeline_metadata_bytes(
        num_stages,
        acc_stages,
        use_clc_scheduler,
        num_tile_info_stages,
        sf_requires_cluster_join,
    ):
        scheduler_bytes = 32 * num_tile_info_stages + 48 if use_clc_scheduler else 0
        return (
            (
                (num_stages * (3 + int(sf_requires_cluster_join)) + acc_stages * 2) * 8
                + 4
                + scheduler_bytes
                + 127
            )
            // 128
        ) * 128 + 128

    @staticmethod
    def _epilogue_storage(available_bytes, element_bytes):
        if element_bytes:
            # Supported TMEM/stmatrix slabs, in preference order.
            for tokens, stages in ((64, 2), (32, 1)):
                scratch_bytes = tokens * 136 * stages * element_bytes
                if scratch_bytes <= available_bytes:
                    return tokens, stages, scratch_bytes
        return 0, 0, 128 * 4

    @classmethod
    def _operand_stage_capacity(
        cls,
        stage_bytes,
        bulk_output_bytes,
        acc_stages,
        use_clc_scheduler,
        num_tile_info_stages,
        sf_requires_cluster_join,
    ):
        capacity = utils.get_smem_capacity_in_bytes("sm_100")
        available = (
            capacity
            - stage_bytes
            - cls._pipeline_metadata_bytes(
                1,
                acc_stages,
                use_clc_scheduler,
                num_tile_info_stages,
                sf_requires_cluster_join,
            )
        )
        _, _, scratch_bytes = cls._epilogue_storage(available, bulk_output_bytes)
        stages = (capacity - scratch_bytes) // stage_bytes
        while stages > 0 and (
            stages * stage_bytes
            + scratch_bytes
            + cls._pipeline_metadata_bytes(
                stages,
                acc_stages,
                use_clc_scheduler,
                num_tile_info_stages,
                sf_requires_cluster_join,
            )
            > capacity
        ):
            stages -= 1
        return stages

    @classmethod
    def _k_stage_capacity(
        cls,
        mma_tiler_mnk,
        activation_dtype,
        weight_dtype,
        sf_vec_size,
        overlap_epilogue,
        bulk_output_bytes,
        use_clc_scheduler,
        num_tile_info_stages,
    ):
        """Return the operand-ring capacity and accumulator overlap eligibility."""
        logical_m, route_tile, tile_k = mma_tiler_mnk
        channel_tiles = max(1, logical_m // 256)
        cta_group_size = logical_m // channel_tiles // 128
        allocated_n = max(32, route_tile)
        acc_cols = allocated_n * channel_tiles
        scale_cols = (
            tile_k // sf_vec_size * (channel_tiles + max(1, allocated_n // 128))
        )
        if acc_cols + scale_cols > cute.arch.get_max_tmem_alloc_cols("sm_100"):
            return 0, False
        acc_stages, can_overlap = cls._accumulator_storage(
            acc_cols, scale_cols, overlap_epilogue
        )
        sf_requires_cluster_join = cta_group_size > 1 and route_tile < 128
        stage_bytes = cls._stage_storage_bytes(
            tile_k,
            allocated_n,
            cta_group_size,
            channel_tiles,
            activation_dtype,
            weight_dtype,
            sf_vec_size,
        )
        return cls._operand_stage_capacity(
            stage_bytes,
            bulk_output_bytes if route_tile >= 128 else 0,
            acc_stages,
            use_clc_scheduler,
            num_tile_info_stages,
            sf_requires_cluster_join,
        ), can_overlap

    @classmethod
    def _select_k_steps(
        cls,
        mma_tiler_mnk,
        activation_dtype,
        weight_dtype,
        sf_vec_size,
        num_stages,
        overlap_epilogue,
        overlap_activation_accumulators,
        bulk_output_bytes,
        use_clc_scheduler,
        num_tile_info_stages,
    ):
        """Maximize staged reduction coverage within the actual storage budget.

        Automatic depth reserves at least two operand slots. Explicit stage
        counts remain authoritative. Prefer the larger K tile on equal coverage
        to reduce issue overhead, while preserving the epilogue's early-release
        window when two full accumulator buffers cannot coexist with scales.
        """
        logical_m, route_tile, base_k = mma_tiler_mnk
        channel_tiles = max(1, logical_m // 256)
        allocated_n = max(32, route_tile)
        acc_cols = allocated_n * channel_tiles
        tmem_capacity = cute.arch.get_max_tmem_alloc_cols("sm_100")
        bulk_output = bulk_output_bytes > 0 and route_tile >= 128
        bulk_overlap = (
            bulk_output and channel_tiles == 1 and allocated_n == tmem_capacity // 2
        )
        preserve_overlap = (
            overlap_epilogue
            and 2 * acc_cols == tmem_capacity
            and (overlap_activation_accumulators is not False or bulk_overlap)
        )
        minimum_stages = 2 if num_stages is None else num_stages
        selected_steps = None
        queued_k = 0
        for steps in (8, 4):
            tile_k = base_k * steps // 4
            stage_capacity, can_overlap = cls._k_stage_capacity(
                (logical_m, route_tile, tile_k),
                activation_dtype,
                weight_dtype,
                sf_vec_size,
                overlap_epilogue,
                bulk_output_bytes,
                use_clc_scheduler,
                num_tile_info_stages,
            )
            if preserve_overlap and not can_overlap:
                continue
            if minimum_stages <= stage_capacity:
                stages = stage_capacity if num_stages is None else num_stages
                if stages * tile_k > queued_k:
                    selected_steps = steps
                    queued_k = stages * tile_k
        if selected_steps is not None:
            return selected_steps
        raise ValueError(
            "No supported K tile fits the requested SM100 pipeline storage"
        )

    def _operand_layouts(self, tiled_mma, num_stages):
        return (
            sm100_utils.make_smem_layout_a(
                tiled_mma,
                self.physical_tiler,
                self.a_storage_dtype,
                num_stages * self.channel_tiles,
            ),
            sm100_utils.make_smem_layout_b(
                tiled_mma, self.physical_tiler, self.activation_dtype, num_stages
            ),
            blockscaled_utils.make_smem_layout_sfa(
                tiled_mma,
                self.physical_tiler,
                self.sf_vec_size,
                num_stages * self.channel_tiles,
            ),
            blockscaled_utils.make_smem_layout_sfb(
                tiled_mma, self.physical_tiler, self.sf_vec_size, num_stages
            ),
        )

    def _operand_bytes(self, layouts):
        a, b, sfa, sfb = layouts
        return (
            cute.cosize(a.outer) * self.a_storage_dtype.width // 8
            + cute.cosize(b.outer) * self.activation_dtype.width // 8
            + cute.cosize(sfa) * self.sf_dtype.width // 8
            + cute.cosize(sfb) * self.sf_dtype.width // 8
        )

    def _metadata_bytes(self, num_stages):
        # The CLC response has its own 16-byte alignment inside the slab.
        return self._pipeline_metadata_bytes(
            num_stages,
            self.acc_stages,
            self.use_clc_scheduler,
            self.num_tile_info_stages,
            self.sf_requires_cluster_join,
        )

    def _configure_pipeline_storage(self, tiled_mma, reduction_size):
        """Fit SM100 operand stages around the epilogue's working storage.

        ``num_stages=None`` selects capacity at compile time. An explicit depth
        remains available for controlled tuning. Rubin owns a separate policy
        because its operand, TMEM and epilogue layouts differ.
        """
        capacity = utils.get_smem_capacity_in_bytes("sm_100")
        stage_bytes = self._operand_bytes(self._operand_layouts(tiled_mma, 1))
        if stage_bytes != self._stage_storage_bytes(
            self.tile_k,
            self.allocated_n,
            self.cta_group_size,
            self.channel_tiles,
            self.activation_dtype,
            self.weight_dtype,
            self.sf_vec_size,
        ):
            raise ValueError(
                "SM100 operand layout no longer matches the storage planner"
            )
        bulk_output = (
            self.scatter_output
            and not self.use_fused_finalize
            and self.output_dtype in (cutlass.BFloat16, cutlass.Float16)
            and not self.epilogue.generate_sfc
            and self.route_tile >= 128
        )
        bulk_output_bytes = self.output_dtype.width // 8 if bulk_output else 0
        num_stages = self.requested_num_stages
        if num_stages is None:
            num_stages = self._operand_stage_capacity(
                stage_bytes,
                bulk_output_bytes,
                self.acc_stages,
                self.use_clc_scheduler,
                self.num_tile_info_stages,
                self.sf_requires_cluster_join,
            )
            if not self.overlap_epilogue:
                # This producer executes its own epilogue before the next tile.
                # Dedicated epilogue warps instead permit cross-tile prefetch.
                num_stages = min(num_stages, cute.ceil_div(reduction_size, self.tile_k))
            if num_stages < 1:
                raise ValueError("Not enough SM100 shared memory for one operand stage")
        self.num_stages = num_stages

        layouts = self._operand_layouts(tiled_mma, self.num_stages)
        available = (
            capacity
            - self._operand_bytes(layouts)
            - self._metadata_bytes(self.num_stages)
        )
        (self.epilogue.bulk_tokens, self.epilogue.bulk_stages, scratch_bytes) = (
            self._epilogue_storage(available, bulk_output_bytes)
        )
        self.epilogue_scratch_floats = scratch_bytes // 4
        if available < self.epilogue_scratch_floats * 4:
            raise ValueError(
                "Blockscaled pipeline exceeds SM100 shared memory capacity"
            )
        self.a_layout, self.b_layout, self.sfa_layout, self.sfb_layout = layouts

    @staticmethod
    def is_valid_dtypes(activation_dtype, weight_dtype, sf_dtype, sf_vec_size):
        supported = (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, cutlass.Float8E5M2)
        if activation_dtype not in supported or weight_dtype not in supported:
            return False
        if activation_dtype is not weight_dtype:
            return (
                activation_dtype is cutlass.Float8E4M3FN
                and weight_dtype is cutlass.Float4E2M1FN
                and sf_dtype is cutlass.Float8E8M0FNU
                and sf_vec_size == 32
            )
        if sf_vec_size == 16:
            return activation_dtype is cutlass.Float4E2M1FN and sf_dtype in (
                cutlass.Float8E4M3FN,
                cutlass.Float8E8M0FNU,
            )
        if sf_vec_size == 32:
            return sf_dtype is cutlass.Float8E8M0FNU
        return False

    @staticmethod
    def can_implement(
        m,
        n,
        k,
        output_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        activation_dtype,
        weight_dtype,
        sf_dtype,
        sf_vec_size,
        k_steps=None,
    ):
        """M is output channels; N is allocated routed-token capacity.

        Explicit K choices must fit at least one operand and accumulator slot.
        The constructor applies the requested overlap and stage-count policy.
        """
        if k_steps is not None and (type(k_steps) is not int or k_steps not in (4, 8)):
            return False
        if not Sm100BlockScaledGroupedGemmKernel.is_valid_dtypes(
            activation_dtype, weight_dtype, sf_dtype, sf_vec_size
        ):
            return False
        if output_dtype not in (
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
            cutlass.Float32,
            cutlass.BFloat16,
            cutlass.Float16,
        ):
            return False
        if len(mma_tiler_mn) != 2:
            return False
        tile_m, tile_n = mma_tiler_mn
        if tile_m not in (128, 256, 512) or tile_n not in (16, 32, 64, 128, 256):
            return False
        channel_tiles = max(1, tile_m // 256)
        cta_group_size = tile_m // channel_tiles // 128
        if cluster_shape_mn != (cta_group_size, 1):
            return False
        base_k = (
            256 if activation_dtype is weight_dtype is cutlass.Float4E2M1FN else 128
        )
        steps = 4 if k_steps is None else k_steps
        stage_capacity, _ = Sm100BlockScaledGroupedGemmKernel._k_stage_capacity(
            (tile_m, tile_n, base_k * steps // 4),
            activation_dtype,
            weight_dtype,
            sf_vec_size,
            False,
            0,
            False,
            1,
        )
        if stage_capacity < 1:
            return False
        if m <= 0 or n <= 0 or k <= 0:
            return False
        return m % 128 == 0 and k % (4 * sf_vec_size) == 0 and n % tile_n == 0

    @cute.jit
    def wrapper(
        self,
        weight_ptr: cute.Pointer,
        weight_sf_ptr: cute.Pointer,
        activation_ptr: cute.Pointer,
        activation_sf_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        output_sf_ptr: Optional[cute.Pointer],
        alpha_ptr: cute.Pointer,
        norm_const_ptr: Optional[cute.Pointer],
        token_mapping_ptr: cute.Pointer,
        expert_idx_ptr: cute.Pointer,
        mn_limit_ptr: cute.Pointer,
        num_tiles_ptr: cute.Pointer,
        token_final_scales_ptr: Optional[cute.Pointer],
        per_token_scale_ptr: Optional[cute.Pointer],
        num_channels: cutlass.Constexpr,
        num_routes: cutlass.Int64,
        reduction_size: cutlass.Constexpr,
        num_experts: cutlass.Constexpr,
        num_tokens: cutlass.Int64,
        stream: cuda.CUstream,
    ):
        num_channels = cute.assume(num_channels, divby=128)
        reduction_size = cute.assume(reduction_size, divby=self.sf_block_k)
        gated = self.activation_type in (
            ActivationType.Swiglu.value,
            ActivationType.GegluTanh.value,
        )
        out_channels = cute.assume(num_channels // (2 if gated else 1), divby=16)
        source_rows = num_routes
        # FC1 scales still belong to original tokens; the epilogue indexes them
        # through the unchanged route mapping. FC2 scales belong to route rows.
        per_token_scale_rows = (
            num_routes if self.per_token_scale_is_routed else num_tokens
        )
        output_rows = num_routes
        if cutlass.const_expr(self.scatter_output):
            output_rows = (
                num_tokens if self.use_fused_finalize else num_tokens * self.top_k
            )
        weights = cute.make_tensor(
            weight_ptr,
            cute.make_layout(
                (num_channels, reduction_size, num_experts),
                stride=(reduction_size, 1, num_channels * reduction_size),
            ),
        )
        activations = cute.make_tensor(
            activation_ptr,
            cute.make_layout((source_rows, reduction_size), stride=(reduction_size, 1)),
        )
        weights_sf = cute.make_tensor(
            weight_sf_ptr,
            cute.make_layout(
                (
                    cute.ceil_div(num_channels, 128)
                    * 128
                    * cute.ceil_div(reduction_size, self.sf_block_k)
                    * 4
                    * num_experts,
                )
            ),
        )
        activations_sf = cute.make_tensor(
            activation_sf_ptr,
            cute.make_layout(
                (
                    cute.ceil_div(source_rows, 128)
                    * 128
                    * cute.ceil_div(reduction_size, self.sf_block_k)
                    * 4,
                )
            ),
        )
        output = cute.make_tensor(
            output_ptr,
            cute.make_layout((output_rows, out_channels), stride=(out_channels, 1)),
        )
        output_sf = None
        if cutlass.const_expr(output_sf_ptr is not None):
            sf_layout = blockscaled_utils.tile_atom_to_shape_SF(
                (cute.ceil_div(num_routes, 128) * 128, out_channels, 1),
                self.sf_vec_size,
            )
            output_sf = cute.make_tensor(output_sf_ptr, sf_layout)[(None, None, 0)]
        alpha = cute.make_tensor(alpha_ptr, cute.make_layout((num_experts,)))
        norm_const = None
        if cutlass.const_expr(norm_const_ptr is not None):
            norm_const = cute.make_tensor(norm_const_ptr, cute.make_layout((1,)))
        token_mapping = cute.make_tensor(
            token_mapping_ptr, cute.make_layout((num_routes,))
        )
        expert_idx = cute.make_tensor(
            expert_idx_ptr, cute.make_layout((num_routes // self.route_tile,))
        )
        mn_limit = cute.make_tensor(mn_limit_ptr, expert_idx.layout)
        num_tiles = cute.make_tensor(num_tiles_ptr, cute.make_layout((1,)))
        token_final_scales = None
        if cutlass.const_expr(token_final_scales_ptr is not None):
            token_final_scales = cute.make_tensor(
                token_final_scales_ptr,
                cute.make_layout((num_tokens, self.top_k), stride=(self.top_k, 1)),
            )
        per_token_scale = None
        if cutlass.const_expr(per_token_scale_ptr is not None):
            per_token_scale = cute.make_tensor(
                per_token_scale_ptr, cute.make_layout((per_token_scale_rows,))
            )
        self(
            weights,
            activations,
            weights_sf,
            activations_sf,
            output,
            output_sf,
            alpha,
            norm_const,
            token_mapping,
            expert_idx,
            mn_limit,
            num_tiles,
            token_final_scales,
            per_token_scale,
            stream,
        )

    @cute.jit
    def __call__(
        self,
        weights,
        activations,
        weights_sf,
        activations_sf,
        output,
        output_sf,
        alpha,
        norm_const,
        token_mapping,
        expert_idx,
        mn_limit,
        num_tiles,
        token_final_scales,
        per_token_scale,
        stream,
    ):
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.weight_dtype,
            self.activation_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            (self.mma_m, self.allocated_n),
        )
        cluster_layout = cute.tiled_divide(
            cute.make_layout((self.cta_group_size, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        self._configure_pipeline_storage(tiled_mma, weights.shape[1])
        self.acc_layout = tiled_mma.make_fragment_C(
            tiled_mma.partition_shape_C((self.mma_m, self.allocated_n))
        ).layout

        tma_atom_a, tma_weights = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
            weights,
            cute.slice_(self.a_layout, (None, None, None, 0)),
            self.physical_tiler,
            tiled_mma,
            cluster_layout.shape,
            internal_type=self.a_storage_dtype if self.unpack_weights else None,
        )

        weight_sf_tensor = cute.make_tensor(
            weights_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(weights.shape, self.sf_vec_size),
        )
        tma_atom_sfa, tma_weights_sf = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
            weight_sf_tensor,
            cute.slice_(self.sfa_layout, (None, None, None, 0)),
            self.physical_tiler,
            tiled_mma,
            cluster_layout.shape,
            internal_type=cutlass.Int16,
        )
        # Copy the logical route extent while retaining the MMA allocation's
        # K-slab and stage strides, including any epilogue row padding.
        b_copy_layout = cute.composition(
            self.b_layout,
            cute.make_layout(
                (self.b_copy_rows, self.tile_k, self.num_stages),
                stride=(1, self.b_rows, self.b_rows * self.tile_k),
            ),
        )
        tma_atom_b, tma_activations = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
            activations,
            cute.slice_(b_copy_layout, (None, None, 0)),
            (self.b_copy_rows, self.tile_k),
        )

        tma_atom_sfb = None
        if cutlass.const_expr(self.tma_sfb):
            # The canonical SF layout consists of contiguous 512-byte blocks.
            # Keep K-block and row-block coordinates separate so TMA zero-fills
            # a short final K tile without crossing into the next row block.
            sf_k_blocks = cute.ceil_div(weights.shape[1], self.sf_block_k)
            sf_row_blocks = cute.ceil_div(activations.shape[0], 128)
            # Preserve the 3D descriptor's K axis even when it has one block.
            # This types the constant K extent without adding a runtime argument.
            sf_source = cute.make_tensor(
                cute.recast_ptr(activations_sf.iterator, dtype=cutlass.Int16),
                cute.make_layout(
                    (256, cutlass.Int32(sf_k_blocks), cutlass.Int32(sf_row_blocks)),
                    stride=(1, 256, sf_k_blocks * 256),
                ),
            )
            sf_tile = (256, self.tile_k // self.sf_block_k, 1)
            sf_destination = cute.make_layout(
                sf_tile, stride=(1, 256, self.tile_k // self.sf_block_k * 256)
            )
            tma_atom_sfb, _ = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                sf_source,
                sf_destination,
                sf_tile,
            )

        # Alternate overlapping accumulator windows while keeping scales
        # disjoint. The epilogue reads their shared interval before release.
        self.overlap_acc = (
            self.can_overlap_accumulators
            and self.allocated_n == 256
            and self.epilogue.bulk_tokens > 0
        ) or self.overlap_activation_accumulators
        if cutlass.const_expr(
            self.overlap_acc and self.sf_cols * self.channel_tiles + self.sfb_cols > 64
        ):
            raise ValueError("Overlapping accumulators require at most 64 SF columns")
        # TMEM scale copies occupy four columns per 128-bit scale atom. The
        # FC1 transpose works in 32-column slabs, so only ownership is rounded.
        sf_footprint = self.sf_cols * self.channel_tiles + self.sfb_cols
        self.protected_acc_cols = (
            ((sf_footprint + 31) // 32) * 32
            if self.overlap_activation_accumulators
            else 64
        )
        self.acc_window_stride = self.acc_cols - self.protected_acc_cols
        self.acc_reserved_cols = (
            self.acc_cols + self.acc_window_stride
            if self.overlap_acc
            else self.acc_stages * self.acc_cols
        )
        self.epilogue.overlap_acc = self.overlap_acc
        self.epilogue.overlap_activation_steps = self.protected_acc_cols // 32

        @cute.struct
        class SharedStorage:
            a_barriers: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            ab_barriers: cute.struct.MemRange[cutlass.Int64, self.num_stages]
            sf_ready_barriers: cute.struct.MemRange[
                cutlass.Int64,
                self.num_stages if self.sf_requires_cluster_join else 0,
            ]
            acc_barriers: cute.struct.MemRange[cutlass.Int64, 2 * self.acc_stages]
            tmem_holding: cutlass.Int32
            tile_info_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2 * self.num_tile_info_stages if self.use_clc_scheduler else 0,
            ]
            tile_info: cute.struct.MemRange[
                cutlass.Int32,
                4 * self.num_tile_info_stages if self.use_clc_scheduler else 0,
            ]
            clc_barriers: cute.struct.MemRange[
                cutlass.Int64, 2 if self.use_clc_scheduler else 0
            ]
            clc_response: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 4 if self.use_clc_scheduler else 0],
                16,
            ]
            epilogue_scratch: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.epilogue_scratch_floats], 128
            ]
            tmem_dealloc_mbar: cutlass.Int64
            a: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_storage_dtype, cute.cosize(self.a_layout.outer)
                ],
                128,
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[
                    self.activation_dtype, cute.cosize(self.b_layout.outer)
                ],
                128,
            ]
            sfa: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_layout)],
                128,
            ]
            sfb: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_layout)],
                128,
            ]

        self.storage_type = SharedStorage
        if cutlass.const_expr(
            SharedStorage.size_in_bytes() > utils.get_smem_capacity_in_bytes("sm_100")
        ):
            raise ValueError("SharedStorage exceeds SM100 shared memory capacity")
        sched_params = None
        launch_grid = (
            self.max_active_clusters // self.cta_group_size * self.cta_group_size,
            1,
            1,
        )
        if cutlass.const_expr(self.use_clc_scheduler):
            sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
                (
                    cute.ceil_div(weights.shape[0], self.mma_m * self.channel_tiles)
                    * self.cta_group_size,
                    expert_idx.shape[0],
                    1,
                ),
                (self.cta_group_size, 1, 1),
            )
            launch_grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(
                sched_params
            )
        self.kernel(
            tiled_mma,
            cluster_layout,
            self.a_layout,
            self.b_layout,
            b_copy_layout,
            self.sfa_layout,
            self.sfb_layout,
            self.acc_layout,
            tma_atom_a,
            tma_weights,
            tma_atom_b,
            tma_activations,
            tma_atom_sfa,
            tma_weights_sf,
            tma_atom_sfb,
            activations_sf,
            output,
            output_sf,
            alpha,
            norm_const,
            token_mapping,
            expert_idx,
            mn_limit,
            num_tiles,
            token_final_scales,
            per_token_scale,
            sched_params,
        ).launch(
            grid=launch_grid,
            cluster=(self.cta_group_size, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
        )

    def _s2t_partition(self, source, target):
        source = cute.filter_zeros(source)
        target = cute.filter_zeros(target)
        atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        copy = tcgen05.make_s2t_copy(atom, target)
        thr = copy.get_slice(0)
        source = tcgen05.get_s2t_smem_desc_tensor(copy, thr.partition_S(source))
        return copy, source, thr.partition_D(target)

    @cute.jit
    def _copy_tile(self, atom, source_ptr, destination_ptr, count, valid):
        source = cute.make_tensor(source_ptr, cute.make_layout((count,)))
        destination = cute.make_tensor(destination_ptr, cute.make_layout((count,)))
        predicate = cute.make_rmem_tensor((1,), cutlass.Boolean)
        predicate[0] = valid
        cute.copy_atom_call(atom, source, destination, pred=predicate)

    @cute.jit
    def next_work(self, info_pipe, state, info, static_tile, total_tiles):
        tile = cutlass.Int32(static_tile)
        expert = cutlass.Int32(0)
        valid = cutlass.Int32(0)
        if cutlass.const_expr(self.use_clc_scheduler):
            info_pipe.consumer_wait(state)
            tile = info[0, state.index]
            expert = info[1, state.index]
            valid = info[2, state.index]
            info_pipe.consumer_release(state)
            state.advance()
        else:
            if tile >= total_tiles:
                tile = cutlass.Int32(-1)
        return tile, expert, valid, state

    @cute.jit
    def load_weight_stage(
        self,
        a_pipe,
        producer,
        tma_atom_a,
        tAgA,
        tAsA,
        tma_atom_sfa,
        tAgSFA,
        tAsSFA,
        tile,
        total_m_tiles,
        k_tile,
        expert,
    ):
        # TMA copy lowering elects internally; keep the whole warp active.
        if cutlass.const_expr(self.enable_iket):
            _iket.range_push("producer_tma_empty")
        a_pipe.producer_acquire(producer)
        if cutlass.const_expr(self.enable_iket):
            _iket.range_pop()
        if cutlass.const_expr(self.enable_iket):
            _iket.range_push("producer_weight_tma")
        for channel_tile in cutlass.range_constexpr(self.channel_tiles):
            cute.copy(
                tma_atom_a,
                tAgA[
                    (
                        None,
                        (tile % total_m_tiles) * self.channel_tiles + channel_tile,
                        k_tile,
                        expert,
                    )
                ],
                tAsA[
                    (
                        None,
                        producer.index * self.channel_tiles + channel_tile,
                    )
                ],
                tma_bar_ptr=a_pipe.producer_get_barrier(producer),
                cache_policy=cutlass.Int64(self.weight_cache_policy),
            )
            cute.copy(
                tma_atom_sfa,
                tAgSFA[
                    (
                        None,
                        (tile % total_m_tiles) * self.channel_tiles + channel_tile,
                        k_tile,
                        expert,
                    )
                ],
                tAsSFA[
                    (
                        None,
                        producer.index * self.channel_tiles + channel_tile,
                    )
                ],
                tma_bar_ptr=a_pipe.producer_get_barrier(producer),
                cache_policy=cutlass.Int64(self.weight_scale_cache_policy),
            )
        if cutlass.const_expr(self.enable_iket):
            _iket.range_pop()

    @cute.kernel
    def kernel(
        self,
        tiled_mma,
        cluster_layout,
        a_layout,
        b_layout,
        b_copy_layout,
        sfa_layout,
        sfb_layout,
        acc_layout,
        tma_atom_a,
        weights,
        tma_atom_b,
        tma_activations,
        tma_atom_sfa,
        tma_weights_sf,
        tma_atom_sfb,
        activations_sf,
        output,
        output_sf,
        alpha,
        norm_const,
        token_mapping,
        expert_idx,
        mn_limit,
        num_tiles,
        token_final_scales,
        per_token_scale,
        sched_params,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp = tidx // 32
        block, block_y, block_z = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        # Static grids have a guaranteed physical CTA0. CLC can cancel it, so
        # each executing cluster keeps its own ancestor-publication owner.
        publishes_ancestors = (block == 0) and (block_y == 0) and (block_z == 0)
        if cutlass.const_expr(self.use_clc_scheduler):
            publishes_ancestors = (self.cta_group_size == 1) or (cta_rank == 0)
        block = block // self.cta_group_size
        grid = grid // self.cta_group_size
        storage = utils.SmemAllocator().allocate(self.storage_type)
        a_pipe = PipelineTmaUmma.create(
            barrier_storage=storage.a_barriers.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout,
            tx_count=self.cta_group_size
            * (
                (128 * self.tile_k * self.weight_dtype.width // 8 + self.sf_stage_bytes)
                * self.channel_tiles
                + self.b_copy_rows * self.tile_k * self.activation_dtype.width // 8
                + (self.sfb_stage_bytes if self.tma_sfb else 0)
            ),
        )
        ab = None
        if cutlass.const_expr(not self.tma_sfb):
            ab = PipelineCpAsyncUmma.create(
                barrier_storage=storage.ab_barriers.data_ptr(),
                num_stages=self.num_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                defer_sync=True,
                cta_layout_vmnk=cluster_layout,
                empty_owner=a_pipe,
            )
        acc_pipe = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_barriers.data_ptr(),
            num_stages=self.acc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 128 * self.cta_group_size
            ),
            cta_layout_vmnk=cluster_layout,
        )
        info_pipe = None
        clc_pipe = None
        if cutlass.const_expr(self.use_clc_scheduler):
            info_pipe = pipeline.PipelineAsync.create(
                barrier_storage=storage.tile_info_barriers.data_ptr(),
                num_stages=self.num_tile_info_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.threads_per_cta - 32
                ),
                defer_sync=True,
            )
            clc_pipe = pipeline.PipelineClcFetchAsync.create(
                barrier_storage=storage.clc_barriers.data_ptr(),
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 32 * self.cta_group_size
                ),
                tx_count=16,
                cta_layout_vmnk=cluster_layout,
                defer_sync=True,
            )
        tmem = utils.TmemAllocator(
            storage.tmem_holding.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(
                barrier_id=3, num_threads=self.threads_per_cta
            ),
            allocator_warp_id=0,
            is_two_cta=self.cta_group_size == 2,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr
            if self.cta_group_size == 2
            else None,
        )
        if cutlass.const_expr(self.sf_requires_cluster_join):
            if tidx == 0 and cta_rank == 0:
                for stage in cutlass.range_constexpr(self.num_stages):
                    cute.arch.mbarrier_init(
                        storage.sf_ready_barriers.data_ptr() + stage,
                        self.cta_group_size,
                    )
        if cutlass.const_expr(self.use_clc_scheduler or self.sf_requires_cluster_join):
            cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.cta_group_size == 2):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        tmem.allocate(self.tmem_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        sA = storage.a.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sB = storage.b.get_tensor(b_layout.outer, swizzle=b_layout.inner)
        sB_copy = storage.b.get_tensor(b_copy_layout.outer, swizzle=b_copy_layout.inner)
        sSFA = storage.sfa.get_tensor(sfa_layout)
        sSFB = storage.sfb.get_tensor(sfb_layout)
        gA = cute.local_tile(weights, (self.mma_m, self.tile_k), (None, None, None))
        tCgA = tiled_mma.get_slice(cta_rank).partition_A(gA)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        gSFA = cute.local_tile(
            tma_weights_sf, (self.mma_m, self.tile_k), (None, None, None)
        )
        tCgSFA = tiled_mma.get_slice(cta_rank).partition_A(gSFA)
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            0,
            cute.make_layout(1),
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        total_m_tiles = cute.ceil_div(weights.shape[0], self.mma_m * self.channel_tiles)
        k_tiles = cute.ceil_div(weights.shape[1], self.tile_k)
        startup_stages = min(self.num_stages, k_tiles)
        first_tile = block
        first_tile_valid = cutlass.Boolean(True)
        sched = None
        initial_work = None
        if cutlass.const_expr(self.use_clc_scheduler):
            sched = utils.ClcDynamicPersistentTileScheduler.create(
                sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                storage.clc_response.data_ptr(),
            )
            initial_work = sched.initial_work_tile_info()
            first_tile = (
                initial_work.tile_idx[1] * total_m_tiles
                + initial_work.tile_idx[0] // self.cta_group_size
            )
            first_tile_valid = initial_work.is_valid_tile
        if cutlass.const_expr(self.prefetch_weights_before_wait):
            # The caller must bind this to a prerequisite startup release+signal
            # in the same memory domain. Initial CLC work uses the same mapping
            # as the scheduler; padded routes leave the operand ring unacquired.
            # Both CTA producers use the existing group-scaled A/SFA/B barrier.
            # B/SFB remain protected by the wait.
            if warp == 0:
                if first_tile_valid and first_tile < num_tiles[0] * total_m_tiles:
                    first_expert = expert_idx[first_tile // total_m_tiles]
                    first_producer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.num_stages
                    )
                    if cutlass.const_expr(self.enable_iket):
                        _iket.range_push("startup_weight_prefetch")
                    for k_tile in cutlass.range_constexpr(startup_stages):
                        self.load_weight_stage(
                            a_pipe,
                            first_producer,
                            tma_atom_a,
                            tAgA,
                            tAsA,
                            tma_atom_sfa,
                            tAgSFA,
                            tAsSFA,
                            first_tile,
                            total_m_tiles,
                            k_tile,
                            first_expert,
                        )
                        first_producer.advance()
                    if cutlass.const_expr(self.enable_iket):
                        _iket.range_pop()
        waits_for_payload = cutlass.Boolean(True)
        if cutlass.const_expr(
            self.use_clc_scheduler and self.prefetch_weights_before_wait
        ):
            # The same startup release that permits weight prefetch publishes
            # routing metadata. Only payload readers need the dependency wait.
            waits_for_payload = warp != self.schedule_warp
        if waits_for_payload:
            if cutlass.const_expr(self.enable_iket):
                _iket.range_push("startup_dependency_wait")
            griddepcontrol_wait()
            if cutlass.const_expr(self.enable_iket):
                _iket.range_pop()
        if cutlass.const_expr(self.pdl_signal_at == "startup"):
            if cutlass.const_expr(self.publish_routing_before_signal):
                # Only warp zero triggers at startup; on publication owners,
                # its first trigger follows the completed-ancestor release.
                if warp == 0:
                    if cutlass.const_expr(self.enable_pdl):
                        if publishes_ancestors:
                            release_gpu()
                    griddepcontrol_launch_dependents()
            elif waits_for_payload:
                if cutlass.const_expr(
                    self.enable_pdl
                    and self.scatter_output
                    and not self.use_fused_finalize
                ):
                    if publishes_ancestors:
                        release_gpu()
                griddepcontrol_launch_dependents()
        num_routed_tiles = num_tiles[0]
        total_tiles = num_routed_tiles * total_m_tiles
        sInfo = None
        if cutlass.const_expr(self.use_clc_scheduler):
            sInfo = storage.tile_info.get_tensor(
                cute.make_layout((4, self.num_tile_info_stages), stride=(1, 4))
            )
        padded_k = cute.ceil_div(weights.shape[1], self.sf_block_k) * self.sf_block_k

        if warp < 4:
            copy_sf = cute.make_copy_atom(
                cpasync.CopyG2SOp(), self.sf_dtype, num_bits_per_copy=32
            )
            producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.acc_stages
            )
            info_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_info_stages
            )
            tile, scheduled_expert, scheduled_valid, info_state = self.next_work(
                info_pipe, info_state, sInfo, block, total_tiles
            )
            while tile >= 0:
                route_idx = tile // total_m_tiles
                channel_start = (
                    tile % total_m_tiles
                ) * self.mma_m * self.channel_tiles + cta_rank * 128
                route_start = route_idx * self.route_tile
                expert = (
                    scheduled_expert
                    if cutlass.const_expr(self.use_clc_scheduler)
                    else expert_idx[route_idx]
                )
                valid = (
                    scheduled_valid
                    if cutlass.const_expr(self.use_clc_scheduler)
                    else cutlass.min(self.route_tile, mn_limit[route_idx] - route_start)
                )
                gB = cute.local_tile(
                    cute.domain_offset(
                        (
                            route_start
                            + cta_rank * (((valid + 15) & -16) // self.cta_group_size),
                            0,
                        ),
                        tma_activations,
                    ),
                    (self.b_copy_rows, self.tile_k),
                    (0, None),
                )
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_b,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sB_copy, 0, 2),
                    cute.group_modes(gB, 0, 2),
                )
                # Give each lane a different SMEM bank for its four scale bytes.
                sf_row = 8 * warp + 32 * ((tidx % 32) // 8) + tidx % 8
                activation_sf_offsets = cute.make_rmem_tensor(
                    (self.sfb_row_blocks,), cutlass.Int64
                )
                for row_block in cutlass.range_constexpr(self.sfb_row_blocks):
                    row = sf_row + row_block * 128
                    sf_source_row = cutlass.Int32(0)
                    if row < valid:
                        sf_source_row = route_start + row
                    activation_sf_offsets[row_block] = _blockscaled_sf_offset(
                        sf_source_row, 0, padded_k, self.sf_vec_size
                    )
                shared_sf_offset = (sf_row % 32) * 16 + (sf_row // 32) * 4
                for k_tile in cutlass.range(k_tiles):
                    if cutlass.const_expr(not self.tma_sfb):
                        if warp != 0:
                            if cutlass.const_expr(self.enable_iket):
                                _iket.range_push("producer_sf_empty")
                            # Warp zero acquires this ring with its weight load,
                            # including any stages prefetched before the PDL wait.
                            a_pipe.sync_object_empty.wait(
                                producer.index, producer.phase
                            )
                            if cutlass.const_expr(self.enable_iket):
                                _iket.range_pop()
                    if warp == 0:
                        if cutlass.const_expr(self.prefetch_weights_before_wait):
                            # Startup acquired the first weight stages. B/SFB
                            # complete their existing expected-byte counts below.
                            if tile != first_tile or k_tile >= startup_stages:
                                self.load_weight_stage(
                                    a_pipe,
                                    producer,
                                    tma_atom_a,
                                    tAgA,
                                    tAsA,
                                    tma_atom_sfa,
                                    tAgSFA,
                                    tAsSFA,
                                    tile,
                                    total_m_tiles,
                                    k_tile,
                                    expert,
                                )
                        else:
                            self.load_weight_stage(
                                a_pipe,
                                producer,
                                tma_atom_a,
                                tAgA,
                                tAsA,
                                tma_atom_sfa,
                                tAgSFA,
                                tAsSFA,
                                tile,
                                total_m_tiles,
                                k_tile,
                                expert,
                            )
                        if cutlass.const_expr(self.enable_iket):
                            _iket.range_push("producer_activation_tma")
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, k_tile)],
                            tBsB[(None, producer.index)],
                            tma_bar_ptr=a_pipe.producer_get_barrier(producer),
                            cache_policy=cutlass.Int64(self.activation_cache_policy),
                        )
                        if cutlass.const_expr(self.enable_iket):
                            _iket.range_pop()
                    if cutlass.const_expr(self.enable_iket):
                        _iket.range_push("producer_sf_issue")
                    if cutlass.const_expr(self.tma_sfb):
                        if warp == 0:
                            for row_block in cutlass.range_constexpr(
                                self.sfb_row_blocks
                            ):
                                with cute.arch.elect_one():
                                    _blockscaled_tma_sfb(
                                        tma_atom_sfb,
                                        sSFB.iterator
                                        + producer.index * self.sfb_stage_bytes
                                        + row_block * self.sf_stage_bytes,
                                        a_pipe.producer_get_barrier(producer),
                                        k_tile * (self.tile_k // self.sf_block_k),
                                        route_start // 128 + row_block,
                                        self.cta_group_size,
                                    )
                    else:
                        # Four contiguous SF bytes form one canonical scale-factor block.
                        # Small route tiles use cp.async when they start inside
                        # a 128-row SF atom.
                        for row_block in cutlass.range_constexpr(self.sfb_row_blocks):
                            row = sf_row + row_block * 128
                            for sf_block in cutlass.range_constexpr(
                                self.tile_k // self.sf_block_k
                            ):
                                channel = (
                                    k_tile * self.tile_k + sf_block * self.sf_block_k
                                )
                                shared_offset = (
                                    producer.index * self.sfb_stage_bytes
                                    + row_block * self.sf_stage_bytes
                                    + sf_block * 512
                                    + shared_sf_offset
                                )
                                activation_offset = (
                                    activation_sf_offsets[row_block]
                                    + (channel // self.sf_block_k) * 512
                                )
                                self._copy_tile(
                                    copy_sf,
                                    activations_sf.iterator
                                    + cute.assume(activation_offset, divby=4),
                                    sSFB.iterator + cute.assume(shared_offset, divby=4),
                                    4,
                                    (row < valid) & (channel < weights.shape[1]),
                                )
                    if cutlass.const_expr(self.enable_iket):
                        _iket.range_pop()
                    if cutlass.const_expr(not self.tma_sfb):
                        if cutlass.const_expr(self.enable_iket):
                            _iket.range_push("producer_sf_commit")
                        ab.producer_commit(producer)
                        if cutlass.const_expr(self.enable_iket):
                            _iket.range_pop()
                    producer.advance()
                if cutlass.const_expr(not self.overlap_epilogue):
                    if cutlass.const_expr(self.enable_iket):
                        _iket.range_push("epi_acc_wait")
                    acc_pipe.consumer_wait(consumer)
                    if cutlass.const_expr(self.enable_iket):
                        _iket.range_pop()
                    if cutlass.const_expr(self.enable_iket):
                        _iket.range_push("epi_body")
                    norm = cutlass.Float32(1.0)
                    if cutlass.const_expr(norm_const is not None):
                        norm = norm_const[0]
                    for channel_tile in cutlass.range_constexpr(self.channel_tiles):
                        epi_accumulator = cute.make_tensor(
                            tmem_ptr
                            + cute.assume(
                                consumer.index * self.acc_cols
                                + channel_tile * self.allocated_n,
                                divby=32,
                            ),
                            acc_layout,
                        )
                        self.epilogue.run(
                            epi_accumulator,
                            output,
                            output_sf,
                            alpha[expert],
                            norm,
                            per_token_scale,
                            token_mapping,
                            token_final_scales,
                            channel_start + channel_tile * self.mma_m,
                            route_start,
                            valid,
                            tidx,
                            storage.epilogue_scratch.data_ptr(),
                        )
                    cute.arch.fence_view_async_tmem_load()
                    acc_pipe.consumer_release(consumer)
                    if cutlass.const_expr(self.enable_iket):
                        _iket.range_pop()
                    consumer.advance()
                tile, scheduled_expert, scheduled_valid, info_state = self.next_work(
                    info_pipe, info_state, sInfo, tile + grid, total_tiles
                )
            if warp == 0:
                if cutlass.const_expr(self.pdl_signal_at == "producer_exit"):
                    if cutlass.const_expr(
                        self.enable_pdl
                        and self.scatter_output
                        and not self.use_fused_finalize
                    ):
                        # Republish prior routing writes before combine can
                        # preload them. Queued MMA/epilogue work remains.
                        if publishes_ancestors:
                            release_gpu()
                    griddepcontrol_launch_dependents()
                # One empty completion protects all operand storage, including SF.
                a_pipe.producer_tail(producer.clone())
        elif warp < (8 if self.overlap_epilogue else 4):
            # Normalization is invariant across experts and scheduled tiles.
            norm = cutlass.Float32(1.0)
            if cutlass.const_expr(norm_const is not None):
                norm = norm_const[0]
            consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.acc_stages
            )
            info_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_info_stages
            )
            tile, scheduled_expert, scheduled_valid, info_state = self.next_work(
                info_pipe, info_state, sInfo, block, total_tiles
            )
            while tile >= 0:
                route_idx = tile // total_m_tiles
                channel_start = (
                    tile % total_m_tiles
                ) * self.mma_m * self.channel_tiles + cta_rank * 128
                route_start = route_idx * self.route_tile
                expert = (
                    scheduled_expert
                    if cutlass.const_expr(self.use_clc_scheduler)
                    else expert_idx[route_idx]
                )
                valid = (
                    scheduled_valid
                    if cutlass.const_expr(self.use_clc_scheduler)
                    else cutlass.min(self.route_tile, mn_limit[route_idx] - route_start)
                )
                if cutlass.const_expr(self.enable_iket):
                    _iket.range_push("epi_acc_wait")
                acc_pipe.consumer_wait(consumer)
                if cutlass.const_expr(self.enable_iket):
                    _iket.range_pop()
                if cutlass.const_expr(self.enable_iket):
                    _iket.range_push("epi_body")
                if cutlass.const_expr(self.overlap_activation_accumulators):
                    epi_accumulator = cute.make_tensor(
                        tmem_ptr
                        + cute.assume(
                            consumer.phase * self.acc_window_stride, divby=32
                        ),
                        acc_layout,
                    )
                    self.epilogue.run_overlapping_fc1(
                        epi_accumulator,
                        output,
                        output_sf,
                        alpha[expert],
                        norm,
                        per_token_scale,
                        token_mapping,
                        channel_start,
                        route_start,
                        valid,
                        tidx - 128,
                        storage.epilogue_scratch.data_ptr(),
                        acc_pipe,
                        consumer,
                        self.mma_m,
                        self.channel_tiles,
                    )
                else:
                    for channel_tile in cutlass.range_constexpr(self.channel_tiles):
                        epi_accumulator = cute.make_tensor(
                            tmem_ptr
                            + cute.assume(
                                (
                                    consumer.phase * self.acc_window_stride
                                    if self.overlap_acc
                                    else consumer.index * self.acc_cols
                                )
                                + channel_tile * self.allocated_n,
                                divby=32,
                            ),
                            acc_layout,
                        )
                        self.epilogue.run(
                            epi_accumulator,
                            output,
                            output_sf,
                            alpha[expert],
                            norm,
                            per_token_scale,
                            token_mapping,
                            token_final_scales,
                            channel_start + channel_tile * self.mma_m,
                            route_start,
                            valid,
                            tidx - 128,
                            storage.epilogue_scratch.data_ptr(),
                            acc_pipe,
                            consumer,
                        )
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(not self.overlap_acc):
                    acc_pipe.consumer_release(consumer)
                if cutlass.const_expr(self.enable_iket):
                    _iket.range_pop()
                consumer.advance()
                tile, scheduled_expert, scheduled_valid, info_state = self.next_work(
                    info_pipe, info_state, sInfo, tile + grid, total_tiles
                )
        elif warp == self.mma_warp:
            tSFA = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + self.acc_reserved_cols, dtype=self.sf_dtype),
                blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma,
                    self.physical_tiler,
                    self.sf_vec_size,
                    cute.slice_(sfa_layout, (None, None, None, 0)),
                ),
            )
            tSFB = cute.make_tensor(
                cute.recast_ptr(
                    tmem_ptr
                    + self.acc_reserved_cols
                    + self.sf_cols * self.channel_tiles,
                    dtype=self.sf_dtype,
                ),
                blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma,
                    self.physical_tiler,
                    self.sf_vec_size,
                    cute.slice_(sfb_layout, (None, None, None, 0)),
                ),
            )
            copy_a, source_a, target_a = self._s2t_partition(sSFA, tSFA)
            copy_b, source_b, target_b = self._s2t_partition(sSFB, tSFB)
            if cutlass.const_expr(self.channel_tiles == 2):
                tSFA1 = cute.make_tensor(
                    cute.recast_ptr(
                        tmem_ptr + self.acc_reserved_cols + self.sf_cols,
                        dtype=self.sf_dtype,
                    ),
                    tSFA.layout,
                )
                copy_a1, source_a1, target_a1 = self._s2t_partition(sSFA, tSFA1)
            fragment_a = tiled_mma.make_fragment_A(sA)
            fragment_b = tiled_mma.make_fragment_B(sB)
            consumer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_stages
            )
            producer = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.acc_stages
            )
            info_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_info_stages
            )
            tile, scheduled_expert, scheduled_valid, info_state = self.next_work(
                info_pipe, info_state, sInfo, block, total_tiles
            )
            while tile >= 0:
                route_idx = tile // total_m_tiles
                route_start = route_idx * self.route_tile
                valid = (
                    scheduled_valid
                    if cutlass.const_expr(self.use_clc_scheduler)
                    else cutlass.min(self.route_tile, mn_limit[route_idx] - route_start)
                )
                mma_accumulator = cute.make_tensor(
                    tmem_ptr
                    + cute.assume(
                        (producer.phase ^ 1) * self.acc_window_stride
                        if self.overlap_acc
                        else producer.index * self.acc_cols,
                        divby=32,
                    ),
                    acc_layout,
                )
                if cutlass.const_expr(self.enable_iket):
                    _iket.range_push("mma_acc_empty")
                if cta_rank == 0:
                    acc_pipe.producer_acquire(producer)
                if cutlass.const_expr(self.enable_iket):
                    _iket.range_pop()
                for k_tile in cutlass.range(k_tiles):
                    if cutlass.const_expr(not self.tma_sfb):
                        if cutlass.const_expr(self.enable_iket):
                            _iket.range_push("mma_sf_ready")
                        ab.consumer_wait(consumer)
                        # Publish each CTA's local cp.async scale bytes before
                        # group-wide S2T reads. The operand empty ring prevents
                        # reuse of this readiness slot until UMMA releases it.
                        cute.arch.fence_proxy("async.shared", space="cta")
                        if cutlass.const_expr(self.sf_requires_cluster_join):
                            ready = (
                                storage.sf_ready_barriers.data_ptr() + consumer.index
                            )
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(
                                    ready, 0, scope=nvvm.MemScopeKind.CLUSTER
                                )
                            if cta_rank == 0:
                                while not _blockscaled_cluster_try_wait(
                                    ready, consumer.phase
                                ):
                                    pass
                        if cutlass.const_expr(self.enable_iket):
                            _iket.range_pop()
                    if cta_rank == 0:
                        if cutlass.const_expr(self.enable_iket):
                            _iket.range_push("mma_tma_ready")
                        a_pipe.consumer_wait(consumer)
                        if cutlass.const_expr(self.enable_iket):
                            _iket.range_pop()
                        if cutlass.const_expr(self.enable_iket):
                            _iket.range_push("mma_sf_copy_issue")
                        sf_stage = (None, None, None, None, consumer.index)
                        sf_stage_a = (
                            None,
                            None,
                            None,
                            None,
                            consumer.index * self.channel_tiles,
                        )
                        cute.copy(copy_a, source_a[sf_stage_a], target_a)
                        cute.copy(copy_b, source_b[sf_stage], target_b)
                        if cutlass.const_expr(self.channel_tiles == 2):
                            sf_stage_a1 = (
                                None,
                                None,
                                None,
                                None,
                                consumer.index * self.channel_tiles + 1,
                            )
                            cute.copy(copy_a1, source_a1[sf_stage_a1], target_a1)
                        if cutlass.const_expr(self.enable_iket):
                            _iket.range_pop()
                        if cutlass.const_expr(self.enable_iket):
                            _iket.range_push("mma_umma0_issue")
                        _blockscaled_dynamic_mma(
                            mma_accumulator,
                            fragment_a[
                                (None, None, None, consumer.index * self.channel_tiles)
                            ],
                            fragment_b[(None, None, None, consumer.index)],
                            tSFA,
                            tSFB,
                            valid,
                            k_tile,
                            self.instruction,
                            self.descriptor,
                            self.cta_group_size,
                        )
                        if cutlass.const_expr(self.enable_iket):
                            _iket.range_pop()
                        if cutlass.const_expr(self.channel_tiles == 2):
                            if cutlass.const_expr(self.enable_iket):
                                _iket.range_push("mma_umma1_issue")
                            mma_accumulator1 = cute.make_tensor(
                                mma_accumulator.iterator
                                + cute.assume(self.allocated_n, divby=32),
                                acc_layout,
                            )
                            _blockscaled_dynamic_mma(
                                mma_accumulator1,
                                fragment_a[
                                    (
                                        None,
                                        None,
                                        None,
                                        consumer.index * self.channel_tiles + 1,
                                    )
                                ],
                                fragment_b[(None, None, None, consumer.index)],
                                tSFA1,
                                tSFB,
                                valid,
                                k_tile,
                                self.instruction,
                                self.descriptor,
                                self.cta_group_size,
                            )
                            if cutlass.const_expr(self.enable_iket):
                                _iket.range_pop()
                        if cutlass.const_expr(self.enable_iket):
                            _iket.range_push("mma_release_issue")
                        # Both channel tiles must finish reading the shared B/SFB
                        # stage before any operand pipeline can overwrite it.
                        a_pipe.consumer_release(consumer)
                        if cutlass.const_expr(self.enable_iket):
                            _iket.range_pop()
                    consumer.advance()
                if cta_rank == 0:
                    acc_pipe.producer_commit(producer)
                producer.advance()
                tile, scheduled_expert, scheduled_valid, info_state = self.next_work(
                    info_pipe, info_state, sInfo, tile + grid, total_tiles
                )
            if cta_rank == 0:
                acc_pipe.producer_tail(producer)
        elif warp == self.schedule_warp:
            if cutlass.const_expr(self.use_clc_scheduler):
                work = initial_work
                info_producer = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_tile_info_stages
                )
                clc_producer = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.ProducerConsumer, 1
                )
                clc_consumer = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, 1
                )
                active = cutlass.Boolean(True)
                while active:
                    if cutlass.const_expr(self.enable_iket):
                        _iket.range_push("scheduler_item")
                    route = work.tile_idx[1]
                    is_routed = work.is_valid_tile and (route < num_routed_tiles)
                    if is_routed or not work.is_valid_tile:
                        info_pipe.producer_acquire(info_producer)
                    if work.is_valid_tile and cta_rank == 0:
                        clc_pipe.producer_acquire(clc_producer)
                        sched.advance_to_next_work(
                            clc_pipe.producer_get_barrier(clc_producer)
                        )
                        clc_producer.advance()
                    if is_routed or not work.is_valid_tile:
                        next_tile = cutlass.Int32(-1)
                        next_expert = cutlass.Int32(0)
                        next_valid = cutlass.Int32(0)
                        if is_routed:
                            next_tile = (
                                route * total_m_tiles
                                + work.tile_idx[0] // self.cta_group_size
                            )
                            next_expert = expert_idx[route]
                            next_valid = cutlass.min(
                                self.route_tile,
                                mn_limit[route] - route * self.route_tile,
                            )
                        with cute.arch.elect_one():
                            sInfo[0, info_producer.index] = next_tile
                            sInfo[1, info_producer.index] = next_expert
                            sInfo[2, info_producer.index] = next_valid
                            # Generic shared stores publish through this owner's
                            # release arrival to the consumers' acquire waits.
                            info_pipe.producer_commit(info_producer)
                        info_producer.advance()
                    if work.is_valid_tile:
                        clc_pipe.consumer_wait(clc_consumer)
                        work = sched.get_current_work()
                        clc_pipe.consumer_release(clc_consumer)
                        clc_consumer.advance()
                    else:
                        active = cutlass.Boolean(False)
                    if cutlass.const_expr(self.enable_iket):
                        _iket.range_pop()
                info_pipe.producer_tail(info_producer)
                if cta_rank == 0:
                    clc_pipe.producer_tail(clc_producer)
                if cutlass.const_expr(self.prefetch_weights_before_wait):
                    griddepcontrol_wait()
        pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta
        ).arrive_and_wait()
        if warp < 4:
            tmem.relinquish_alloc_permit()
            pipeline.NamedBarrier(barrier_id=2, num_threads=128).arrive_and_wait()
            tmem.free(tmem_ptr, num_columns=self.tmem_cols)


class Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel(
    Sm100BlockScaledGroupedGemmKernel
):
    """FC2 capability adapter retaining the public logical activation/weight API."""

    @classmethod
    def can_implement(
        cls,
        a_dtype,
        b_dtype,
        sf_dtype,
        sf_vec_size,
        out_dtype,
        final_scale_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        m,
        n,
        k,
        l,
        a_major,
        b_major,
        out_major,
    ):
        if a_major != "k" or b_major != "k" or out_major != "n" or l <= 0:
            return False
        if len(mma_tiler_mn) != 2:
            return False
        if final_scale_dtype not in (
            cutlass.Float32,
            cutlass.BFloat16,
            cutlass.Float16,
        ):
            return False
        if out_dtype not in (cutlass.Float32, cutlass.BFloat16, cutlass.Float16):
            return False
        if a_dtype is not b_dtype and out_dtype is not cutlass.BFloat16:
            return False
        return Sm100BlockScaledGroupedGemmKernel.can_implement(
            n,
            m,
            k,
            out_dtype,
            mma_tiler_mn,
            cluster_shape_mn,
            a_dtype,
            b_dtype,
            sf_dtype,
            sf_vec_size,
        )
