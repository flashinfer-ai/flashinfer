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
mainloops. FC1 gathers activation rows with TMA Gather4 and copies their
scales with cp.async; FC2 uses regular TMA for routed inputs. Each route tile
owns complete 128-row scale-factor atoms. FC1 gathering preserves payload
and scale bytes; per-token activation scales retain their original token
indexing through the routing map.

M128 uses one CTA; M256 uses two CTAs with 128 channels each. Logical M512
reuses an activation tile across two M256 channel tiles. Allocation
uses the tactic width while each MMA descriptor rounds the live routed-token
count to 16 columns.
"""

from math import lcm
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
from cutlass._mlir.dialects import builtin, llvm
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
from flashinfer.quantization.quantization_cute_dsl_utils import (
    _nvfp4_standard_quant_from_amax,
    float_to_ue8m0_fast,
    ue8m0_to_inv_scale_fast,
)
from ..common.kernel_utils import (
    f32_reciprocal,
    fmin,
    gelu_tanh_f32,
    situ_f32,
    tanh_f32,
    vectorized_atomic_add_fp32x2,
)
from .custom_pipeline import PipelineUmmaAsync
from .utils import (
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    release_gpu,
    tma_gather4 as _blockscaled_tma_gather4,
)


# Imported only for explicitly instrumented diagnostic builds.
_iket = None


@dsl_user_op
def _copy_gather_sf(destination, source, issue, load, loc=None, ip=None):
    """Copy one SF word, zeroing MMA padding and skipping unused rows."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(destination.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip),
            cutlass.Int64(source.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip),
            cutlass.Int32(issue).ir_value(loc=loc, ip=ip),
            cutlass.Int32(load).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred issue, ignore_source; "
        "setp.ne.b32 issue, $2, 0; setp.eq.b32 ignore_source, $3, 0; "
        "@issue cp.async.ca.shared.global [$0], [$1], 4, ignore_source; }",
        "r,l,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _gather_mbarrier_arrive(mbarrier, cta_group_size, loc=None, ip=None):
    """Publish this CTA's completed shared-memory inputs to the MMA leader."""
    if cta_group_size == 1:
        cute.arch.mbarrier_arrive(mbarrier, loc=loc, ip=ip)
        return
    elif cta_group_size != 2:
        raise ValueError("Gather publication requires one or two CTAs")
    llvm.inline_asm(
        None,
        [cutlass.Int32(mbarrier.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 leader_bar; "
        "mapa.shared::cluster.u32 leader_bar, $0, 0; "
        "fence.release.sync_restrict::shared::cta.cluster; "
        "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [leader_bar]; }",
        "r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _gather_mbarrier_wait(mbarrier, phase, cta_group_size, loc=None, ip=None):
    """Acquire the shared-memory inputs after the gathered phase."""
    if cta_group_size == 1:
        cute.arch.mbarrier_wait(mbarrier, phase, loc=loc, ip=ip)
        return
    elif cta_group_size != 2:
        raise ValueError("Gather publication requires one or two CTAs")
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(mbarrier.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip),
            cutlass.Int32(phase).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred complete; "
        "wait_loop: "
        "mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 "
        "complete, [$0], $1, 10000000; "
        "@complete bra.uni wait_done; "
        "bra.uni wait_loop; "
        "wait_done: "
        "fence.acquire.sync_restrict::shared::cluster.cluster; }",
        "r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _blockscaled_tma_sfb(
    atom,
    destination,
    mbarrier,
    k_block,
    row_block,
    cta_group,
    cache_policy,
    loc=None,
    ip=None,
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
            cutlass.Int64(cache_policy).ir_value(loc=loc, ip=ip),
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
    tiled_mma,
    stage_k,
    native_mma_k,
    loc=None,
    ip=None,
):
    """Issue native blockscaled MMAs with a runtime token-width descriptor."""
    i32 = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)
    cta_group = cute.size(tiled_mma.thr_id.shape)
    atom_k = cute.size(tiled_mma.shape_mnk, mode=[2])
    padded_stage_k = cute.ceil_div(stage_k, atom_k) * atom_k
    group_k = lcm(atom_k, native_mma_k)
    # Blockscaled MMA requires eight columns per CTA in the group.
    n_alignment = 8 * cta_group
    n_bits = (
        ((cutlass.Int32(valid_tokens) + n_alignment - 1) & -n_alignment) >> 3
    ) << 17
    widths = (native_mma_k,) * (
        padded_stage_k // group_k * (group_k // native_mma_k)
    ) + (atom_k,) * ((padded_stage_k % group_k) // atom_k)
    if native_mma_k == 96:
        stage_descriptors = []
        for operand in (a, b):
            value = tcgen05.make_umma_smem_desc(
                operand.iterator, operand[(None, 0, 0)].layout, "k"
            )
            value = cutlass.Uint64(builtin.unrealized_conversion_cast([i64], [value]))
            stage_descriptors.append(
                (cutlass.Uint32(value), cutlass.Uint32(value >> 32))
            )
    start = 0
    for inner_k, width in enumerate(widths):
        if native_mma_k == 96:
            values = []
            for operand, (lo, hi) in zip((a, b), stage_descriptors, strict=True):
                offset = cute.crd2idx(((0, start % 64), 0, start // 64), operand.layout)
                # A 1024B-aligned stage and row-zero offsets preserve SW128.
                assert offset % 32 == 0 and (offset // 32) & 0x38 == 0
                if width == 96:
                    next_k = ((start + width - 1) // 256) * 256
                    next_offset = cute.crd2idx(
                        ((0, 0), 0, next_k // 64), operand.layout
                    )
                    assert next_offset % 32 == 0 and (next_offset // 32) & 0x38 == 0
                    # Absolute LBO points at the next segment; all other fields
                    # are invariant, including the high word's SW128 base.
                    address = lo & 0x3FFF
                    lo = address | ((address + next_offset // 32) << 16)
                    hi = hi | (1 << 20)
                values.append(
                    (
                        cutlass.Uint64(lo + offset // 32) | (cutlass.Uint64(hi) << 32)
                    ).ir_value()
                )
            a_value, b_value = values
            sfa_atom = sfa[(None, 0, start // 16)]
            sfb_atom = sfb[(None, 0, start // 16)]
        else:
            a_atom = a[(None, 0, inner_k)]
            b_atom = b[(None, 0, inner_k)]
            sfa_atom = sfa[(None, 0, inner_k)]
            sfb_atom = sfb[(None, 0, inner_k)]
            a_value = (
                a_atom.iterator.value
                if hasattr(a_atom.iterator, "value")
                else a_atom.iterator
            )
            b_value = (
                b_atom.iterator.value
                if hasattr(b_atom.iterator, "value")
                else b_atom.iterator
            )
        acc_atom = acc[(None, 0, 0)]
        a_desc = builtin.unrealized_conversion_cast([i64], [a_value])
        b_desc = builtin.unrealized_conversion_cast([i64], [b_value])
        a_sf = builtin.unrealized_conversion_cast([i32], [sfa_atom.iterator.value])
        b_sf = builtin.unrealized_conversion_cast([i32], [sfb_atom.iterator.value])
        dst = builtin.unrealized_conversion_cast([i32], [acc_atom.iterator.value])
        # Both scale tensors start on TMEM words; K selects the byte within one.
        sf_id = (start // tiled_mma.op.sf_vec_size) % 4
        desc = descriptor | ((1 << 31) if width == 96 else 0)
        desc = cutlass.Int32(desc | (sf_id << 29) | (sf_id << 4)) | n_bits
        start += width
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


@dsl_user_op
def _bulk_tensor_store(atom, source, channel, row, loc=None, ip=None):
    executable = _cute_nvgpu_ir.atom_make_exec_tma(atom._trait.value, loc=loc, ip=ip)
    descriptor_type = _cute_ir.PtrType.get(
        _cute_nvgpu_ir.TmaDescriptorTiledType.get(), AddressSpace.generic, 64
    )
    descriptor = _cute_nvgpu_ir.get_tma_desc_addr(
        descriptor_type, executable, loc=loc, ip=ip
    )
    llvm.inline_asm(
        None,
        [
            descriptor.toint().ir_value(loc=loc, ip=ip),
            cutlass.Int32(source.toint()).ir_value(loc=loc, ip=ip),
            cutlass.Int32(channel).ir_value(loc=loc, ip=ip),
            cutlass.Int32(row).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group "
        "[$0, {$2, $3}], [$1];",
        "l,r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_stmatrix4(destination, word0, word1, word2, word3, loc=None, ip=None):
    """Transpose four 8x8 BF16/FP16 fragments into token-major shared rows."""
    address = cutlass.Int32(destination.toint(loc=loc, ip=ip))
    address = address ^ ((address >> 3) & 0x70)
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            word0.ir_value(loc=loc, ip=ip),
            word1.ir_value(loc=loc, ip=ip),
            word2.ir_value(loc=loc, ip=ip),
            word3.ir_value(loc=loc, ip=ip),
        ],
        "stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


# The 16x32 TMEM transpose follows the NVFP4 MegaMoE epilogue. Each lane
# receives one token's 16 contiguous channels. SF32 combines two adjacent
# fragments when computing its scale; SF16 uses one fragment.
# Prepared weights interleave up16/gate16 within each warp's 32 TMEM rows.


def _blockscaled_tmem_view(ptr, rows, cols):
    return cute.make_tensor(
        ptr,
        cute.make_layout((((rows, cols), 1),), stride=((((1 << 16), 1), 0),)),
    )


def _blockscaled_rmem_view(values, count, offset=0):
    return cute.make_tensor(
        values.iterator + offset,
        cute.make_layout((((count,), 1),), stride=(((1,), 0),)),
    )


@cute.jit
def _blockscaled_load16x32(ptr):
    values = cute.make_rmem_tensor((16,), cutlass.Float32)
    cute.copy(
        cute.make_copy_atom(
            tcgen05.Ld16x64bOp(tcgen05.Repetition.x16), cutlass.Float32
        ),
        _blockscaled_tmem_view(ptr, 16, 32),
        _blockscaled_rmem_view(values, 16),
    )
    return values


@cute.jit
def _blockscaled_transpose16x32(values, scratch_ptr):
    """Transpose loaded values into lane=token, register=channel order.

    The scratch rectangle is 32 TMEM rows by 32 columns.  All source values
    must have been loaded before the caller reuses their TMEM as scratch.
    The caller must wait for final loads before releasing the scratch.
    """
    perm1 = (0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15)
    perm3 = (0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15)
    fragments = len(values)
    work = cute.make_rmem_tensor((16, fragments), cutlass.Float32)
    result = cute.make_rmem_tensor((16, fragments), cutlass.Float32)
    cute.arch.fence_view_async_tmem_load()
    for fragment in cutlass.range_constexpr(fragments):
        for i in cutlass.range_constexpr(16):
            result[i, fragment] = values[fragment][perm1[i]]
        cute.copy(
            cute.make_copy_atom(
                tcgen05.St16x128bOp(tcgen05.Repetition.x8), cutlass.Float32
            ),
            _blockscaled_rmem_view(result[None, fragment], 16),
            _blockscaled_tmem_view(scratch_ptr + fragment * (16 << 16), 16, 32),
        )
    cute.arch.fence_view_async_tmem_store()
    # Load both source halves before either transposed destination overwrites them.
    for fragment in cutlass.range_constexpr(fragments):
        cute.copy(
            cute.make_copy_atom(
                tcgen05.Ld16x64bOp(tcgen05.Repetition.x16), cutlass.Float32
            ),
            _blockscaled_tmem_view(scratch_ptr + fragment * (16 << 16), 16, 32),
            _blockscaled_rmem_view(work[None, fragment], 16),
        )
    cute.arch.fence_view_async_tmem_load()
    atom_store = cute.make_copy_atom(
        tcgen05.St32x32bOp(tcgen05.Repetition.x16), cutlass.Float32
    )
    for fragment in cutlass.range_constexpr(fragments):
        cute.copy(
            atom_store,
            _blockscaled_rmem_view(work[None, fragment], 16),
            _blockscaled_tmem_view(scratch_ptr + fragment * 16, 32, 16),
        )
    cute.arch.fence_view_async_tmem_store()
    atom_load3 = cute.make_copy_atom(
        tcgen05.Ld16x256bOp(tcgen05.Repetition.x2), cutlass.Float32
    )
    for fragment in cutlass.range_constexpr(fragments):
        for half in cutlass.range_constexpr(2):
            cute.copy(
                atom_load3,
                _blockscaled_tmem_view(
                    scratch_ptr + fragment * 16 + half * (16 << 16), 16, 16
                ),
                _blockscaled_rmem_view(work[None, fragment], 8, half * 8),
            )
    cute.arch.fence_view_async_tmem_load()
    for fragment in cutlass.range_constexpr(fragments):
        for i in cutlass.range_constexpr(16):
            result[i, fragment] = work[perm3[i], fragment]
        cute.copy(
            atom_store,
            _blockscaled_rmem_view(result[None, fragment], 16),
            _blockscaled_tmem_view(scratch_ptr + fragment * 16, 32, 16),
        )
    cute.arch.fence_view_async_tmem_store()
    atom_load4 = cute.make_copy_atom(
        tcgen05.Ld16x128bOp(tcgen05.Repetition.x4), cutlass.Float32
    )
    for fragment in cutlass.range_constexpr(fragments):
        for half in cutlass.range_constexpr(2):
            cute.copy(
                atom_load4,
                _blockscaled_tmem_view(
                    scratch_ptr + fragment * 16 + half * (16 << 16), 16, 16
                ),
                _blockscaled_rmem_view(work[None, fragment], 8, half * 8),
            )
        for i in cutlass.range_constexpr(16):
            result[i, fragment] = work[perm1[i], fragment]
    return result


@dsl_user_op
def _blockscaled_atomic_add_pair(values, output, loc=None, ip=None):
    if values.element_type == cutlass.BFloat16:
        instruction = "red.global.add.noftz.bf16x2"
    elif values.element_type == cutlass.Float16:
        instruction = "red.global.add.noftz.f16x2"
    elif values.element_type == cutlass.Float32:
        vectorized_atomic_add_fp32x2(values, output, loc=loc, ip=ip)
        return
    else:
        raise ValueError("The split MoE fused output must be BF16, FP16, or FP32")
    llvm.inline_asm(
        None,
        [
            output.iterator.llvm_ptr,
            llvm.bitcast(ir.IntegerType.get_signless(32), values.load().ir_value()),
        ],
        f"{instruction} [$0], $1;",
        "l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


class Sm100BlockScaledMoeEpilogue:
    """Process each CTA's 128-channel slice of a one/two-CTA GEMM."""

    def __init__(
        self,
        mma_tiler_mnk,
        activation_type,
        scatter_output,
        per_token_scale_is_routed,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        situ_beta,
        situ_linear_beta,
        use_per_token_scale,
        use_fused_finalize,
        top_k,
        output_dtype,
        final_scale_dtype,
        sf_vec_size,
        sf_dtype,
        generate_sfc,
    ):
        if mma_tiler_mnk[0] != 128:
            raise ValueError(
                "The split block-scaled epilogue requires 128 channels per CTA"
            )
        if activation_type not in (
            None,
            ActivationType.Swiglu.value,
            ActivationType.GegluTanh.value,
            ActivationType.Relu2.value,
        ):
            raise ValueError(
                f"Unsupported split block-scaled activation: {activation_type}"
            )
        if output_dtype not in (
            cutlass.Float4E2M1FN,
            cutlass.BFloat16,
            cutlass.Float16,
            cutlass.Float32,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        ):
            raise ValueError(f"Unsupported split block-scaled output: {output_dtype}")
        if sf_vec_size not in (16, 32):
            raise ValueError("Block-scaled MoE requires SF16 or SF32")
        if sf_dtype not in (cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU):
            raise ValueError("Block-scaled MoE requires E4M3 or E8M0 scales")
        if generate_sfc and output_dtype not in (
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
        ):
            raise ValueError("Scale generation requires FP4 or E4M3 output")
        if use_fused_finalize and output_dtype not in (
            cutlass.BFloat16,
            cutlass.Float16,
            cutlass.Float32,
        ):
            raise ValueError("Fused combine requires BF16, FP16, or FP32 output")
        self.bulk_tokens = 0
        self.overlap_acc = False
        self.overlap_activation_steps = 0
        self.bulk_stages = 0
        self.tile_n = mma_tiler_mnk[1]
        self.activation_type = activation_type
        self.scatter_output = scatter_output
        self.per_token_scale_is_routed = per_token_scale_is_routed
        self.gated = activation_type in (
            ActivationType.Swiglu.value,
            ActivationType.GegluTanh.value,
        )
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.swiglu_limit = swiglu_limit
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta
        self.use_per_token_scale = use_per_token_scale
        self.use_fused_finalize = use_fused_finalize
        self.top_k = top_k
        self.output_dtype = output_dtype
        self.final_scale_dtype = final_scale_dtype
        self.sf_vec_size = sf_vec_size
        self.sf_dtype = sf_dtype
        self.generate_sfc = generate_sfc
        self.is_mxfp8_output = (
            generate_sfc
            and output_dtype is cutlass.Float8E4M3FN
            and sf_dtype is cutlass.Float8E8M0FNU
            and sf_vec_size == 32
        )
        self.transpose_barrier = pipeline.NamedBarrier(6, 128)

    @cute.jit
    def _activate(self, up, gate, alpha):
        result = cute.make_rmem_tensor((16,), cutlass.Float32)
        for i in cutlass.range_constexpr(0, 16, 2):
            u = cute.arch.mul_packed_f32x2((up[i], up[i + 1]), (alpha, alpha))
            if cutlass.const_expr(self.activation_type == ActivationType.Relu2.value):
                relu = (
                    cute.arch.fmax(u[0], cutlass.Float32(0.0)),
                    cute.arch.fmax(u[1], cutlass.Float32(0.0)),
                )
                result[i], result[i + 1] = cute.arch.mul_packed_f32x2(relu, relu)
            elif cutlass.const_expr(self.gated):
                g = cute.arch.mul_packed_f32x2((gate[i], gate[i + 1]), (alpha, alpha))
                if cutlass.const_expr(self.situ_beta is not None):
                    if cutlass.const_expr(self.situ_linear_beta is not None):
                        linear_beta = cutlass.Float32(self.situ_linear_beta)
                        inv_beta = cutlass.Float32(
                            f32_reciprocal(self.situ_linear_beta)
                        )
                        u = (
                            linear_beta * tanh_f32(u[0] * inv_beta, fastmath=True),
                            linear_beta * tanh_f32(u[1] * inv_beta, fastmath=True),
                        )
                    g = (
                        situ_f32(g[0], self.situ_beta, fastmath=True),
                        situ_f32(g[1], self.situ_beta, fastmath=True),
                    )
                elif cutlass.const_expr(
                    self.activation_type == ActivationType.GegluTanh.value
                ):
                    g = (
                        gelu_tanh_f32(g[0], fastmath=True),
                        gelu_tanh_f32(g[1], fastmath=True),
                    )
                elif cutlass.const_expr(
                    self.activation_type == ActivationType.Swiglu.value
                ):
                    limit = cutlass.Float32(self.swiglu_limit)
                    g = (fmin(g[0], limit, nan=True), fmin(g[1], limit, nan=True))
                    u = (
                        -fmin(-fmin(u[0], limit, nan=True), limit, nan=True),
                        -fmin(-fmin(u[1], limit, nan=True), limit, nan=True),
                    )
                    neg_alpha_log2e = -(
                        cutlass.Float32(self.swiglu_alpha)
                        * cutlass.Float32(1.4426950408889634)
                    )
                    arg = cute.arch.mul_packed_f32x2(
                        g,
                        (neg_alpha_log2e, neg_alpha_log2e),
                    )
                    exp = cute.arch.add_packed_f32x2(
                        (
                            cute.math.exp2(arg[0], fastmath=True),
                            cute.math.exp2(arg[1], fastmath=True),
                        ),
                        (cutlass.Float32(1.0), cutlass.Float32(1.0)),
                    )
                    g = cute.arch.mul_packed_f32x2(
                        g, (cute.arch.rcp_approx(exp[0]), cute.arch.rcp_approx(exp[1]))
                    )
                    u = cute.arch.add_packed_f32x2(
                        u,
                        (
                            cutlass.Float32(self.swiglu_beta),
                            cutlass.Float32(self.swiglu_beta),
                        ),
                    )
                result[i], result[i + 1] = cute.arch.mul_packed_f32x2(u, g)
        return result

    @cute.jit
    def _absmax(self, values):
        result = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(16):
            result = cute.arch.fmax(result, cute.arch.fmax(values[i], -values[i]))
        return result

    @cute.jit
    def _gated_sf32_absmax(self, values, scratch_ptr, warp, lane):
        """Join adjacent warps' 16-channel fragments for each token's SF32."""
        shared = cute.make_tensor(
            scratch_ptr, cute.make_layout((4, 32), stride=(32, 1))
        )
        local = self._absmax(values)
        shared[warp, lane] = local
        self.transpose_barrier.arrive_and_wait()
        result = cute.arch.fmax(local, shared[warp ^ 1, lane])
        self.transpose_barrier.arrive_and_wait()
        return result

    @cute.jit
    def _quantization_params(self, absmax, norm_const):
        if cutlass.const_expr(self.is_mxfp8_output):
            # Match mxfp8_quantize's saturating round-up E8M0 code, including
            # code zero for an all-zero block. Keep raw bytes for its store.
            sf = float_to_ue8m0_fast(absmax * cutlass.Float32(1.0 / 448.0))
            scale = ue8m0_to_inv_scale_fast(sf)
            sf = sf.to(cutlass.Uint8)
        elif cutlass.const_expr(
            self.output_dtype == cutlass.Float4E2M1FN
            and self.sf_dtype == cutlass.Float8E4M3FN
            and self.sf_vec_size == 16
        ):
            # Share standalone standard fast quantization while keeping
            # the activation values in FP32 through the FP4 conversion.
            sf, scale = _nvfp4_standard_quant_from_amax(absmax, norm_const, False)
            sf = sf.bitcast(cutlass.Float8E4M3FN)
        elif cutlass.const_expr(self.output_dtype == cutlass.Float4E2M1FN):
            sf = (absmax * cutlass.Float32(1.0 / 6.0) * norm_const).to(self.sf_dtype)
            scale = fmin(
                norm_const * cute.arch.rcp_approx(cutlass.Float32(sf)),
                cutlass.Float32(3.40282346638528859812e38),
                nan=True,
            )
        elif cutlass.const_expr(self.output_dtype == cutlass.Float8E4M3FN):
            sf = (absmax * cutlass.Float32(1.0 / 448.0) * norm_const).to(self.sf_dtype)
            scale = fmin(
                norm_const * cute.arch.rcp_approx(cutlass.Float32(sf)),
                cutlass.Float32(3.40282346638528859812e38),
                nan=True,
            )
        return sf, scale

    @cute.jit
    def _convert(self, values, output_sf, row, channel, quantization):
        converted = cute.make_rmem_tensor((16,), self.output_dtype)
        if cutlass.const_expr(self.generate_sfc):
            sf, scale = quantization
            sf_row = row + row // self.tile_n * (
                cute.ceil_div(self.tile_n, 128) * 128 - self.tile_n
            )
            if channel % self.sf_vec_size == 0:
                output_sf[sf_row, channel] = sf
            for i in cutlass.range_constexpr(0, 16, 2):
                values[i], values[i + 1] = cute.arch.mul_packed_f32x2(
                    (values[i], values[i + 1]), (scale, scale)
                )
        converted.store(values.load().to(self.output_dtype))
        return converted

    @cute.jit
    def _store_converted(self, converted, output, row, channel):
        count = (
            cute.size(converted)
            * converted.element_type.width
            // self.output_dtype.width
        )
        target = cute.recast_tensor(
            cute.coalesce(cute.local_tile(output, (1, count), (row, channel // count))),
            converted.element_type,
        )
        if cutlass.const_expr(self.use_fused_finalize):
            converted = cute.recast_tensor(converted, self.output_dtype)
            target = cute.recast_tensor(target, self.output_dtype)
            for i in cutlass.range_constexpr(0, count, 2):
                pair = cute.make_tensor(converted.iterator + i, cute.make_layout((2,)))
                target_pair = cute.make_tensor(
                    target.iterator + i, cute.make_layout((2,))
                )
                _blockscaled_atomic_add_pair(pair, target_pair)
        else:
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    converted.element_type,
                    num_bits_per_copy=min(256, self.output_dtype.width * count),
                ),
                converted,
                cute.coalesce(target),
            )

    @cute.jit
    def _scale_pack_transpose(self, top, bottom, scratch, alpha, token_scale, lane):
        """Transpose pairs of rounded half outputs in one 32-bit TMEM pass."""
        scaled_values = cute.make_rmem_tensor((32,), cutlass.Float32)
        scale = alpha
        if cutlass.const_expr(
            self.scatter_output
            and self.final_scale_dtype in (cutlass.BFloat16, cutlass.Float16)
        ):
            scale = cutlass.Float32(self.final_scale_dtype(scale))
        for i in cutlass.range_constexpr(16):
            if cutlass.const_expr(
                self.scatter_output
                and (self.use_per_token_scale or self.use_fused_finalize)
            ):
                # Native Ld16x64 ownership: each pair shares one token and
                # channels c/c+16. Metadata already includes its scale rounding.
                native_token = 2 * i + (lane // 2) % 2
                scale = cute.arch.shuffle_sync(token_scale, native_token)
            scaled = cute.arch.mul_packed_f32x2((top[i], bottom[i]), (scale, scale))
            scaled_values[2 * i], scaled_values[2 * i + 1] = scaled
        packed = cute.make_rmem_tensor((16,), cutlass.Float32)
        packed.store(
            scaled_values.load().to(self.output_dtype).bitcast(cutlass.Float32)
        )
        words = _blockscaled_transpose16x32((packed,), scratch)[None, 0]
        halves = cute.make_rmem_tensor((32,), self.output_dtype)
        halves.store(words.load().bitcast(self.output_dtype))
        result = cute.make_rmem_tensor((32,), self.output_dtype)
        cute.autovec_copy(
            cute.composition(halves, cute.make_layout(((16, 2),), stride=((2, 1),))),
            result,
        )
        return result

    @cute.jit
    def _run_token_group(
        self,
        acc_tensor,
        output_tensor,
        output_sf_tensor,
        alpha,
        norm_const,
        per_token_scale,
        token_mapping,
        channel_start,
        route_start,
        valid_tokens,
        tidx,
        smem_scratch,
        token_start,
        fc2_output_row,
        fc2_scale,
        release_acc: cutlass.Constexpr = False,
        acc_pipeline=None,
        acc_state=None,
    ):
        warp = (tidx // 32) % 4
        lane = tidx % 32
        ptr = acc_tensor.iterator + cute.assume(token_start, divby=32)
        scratch = ptr + cute.assume(warp * (32 << 16), divby=32)
        if cutlass.const_expr(self.gated):
            # The prepared up16/gate16 groups both belong to this
            # warp's 32-row TMEM datapath. Load both before reusing
            # the accumulator as transpose scratch.
            up = _blockscaled_load16x32(scratch)
            gate = _blockscaled_load16x32(scratch + (16 << 16))
            if cutlass.const_expr(self.use_per_token_scale):
                transposed = _blockscaled_transpose16x32((up, gate), scratch)
                up, gate = transposed[None, 0], transposed[None, 1]
                if cutlass.const_expr(release_acc):
                    cute.arch.fence_view_async_tmem_load()
                    acc_pipeline.consumer_release(acc_state)
                row = route_start + token_start + lane
                token_alpha = alpha
                if token_start + lane < valid_tokens:
                    token_alpha = (
                        alpha
                        * per_token_scale[
                            row
                            if self.per_token_scale_is_routed
                            else token_mapping[row] // self.top_k
                        ]
                    )
                values = self._activate(up, gate, token_alpha)
            else:
                values = self._activate(up, gate, alpha)
                values = _blockscaled_transpose16x32((values,), scratch)[None, 0]
            if cutlass.const_expr(release_acc and not self.use_per_token_scale):
                # All overlapping accumulator and transpose-scratch reads have
                # completed before either CTA allows the next MMA to overwrite.
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_state)
            absmax = cutlass.Float32(0.0)
            if cutlass.const_expr(self.generate_sfc):
                if cutlass.const_expr(self.sf_vec_size == 32):
                    absmax = self._gated_sf32_absmax(values, smem_scratch, warp, lane)
                elif cutlass.const_expr(self.sf_vec_size == 16):
                    absmax = self._absmax(values)
            channel = channel_start // 2 + warp * 16
            if token_start + lane < valid_tokens:
                if channel < output_tensor.shape[1]:
                    quantization = None
                    if cutlass.const_expr(self.generate_sfc):
                        quantization = self._quantization_params(absmax, norm_const)
                    converted = self._convert(
                        values,
                        output_sf_tensor,
                        route_start + token_start + lane,
                        channel,
                        quantization,
                    )
                    self._store_converted(
                        converted,
                        output_tensor,
                        route_start + token_start + lane,
                        channel,
                    )
        elif cutlass.const_expr(
            self.activation_type is None
            and self.output_dtype in (cutlass.BFloat16, cutlass.Float16)
        ):
            top = _blockscaled_load16x32(scratch)
            bottom = _blockscaled_load16x32(scratch + (16 << 16))
            converted = self._scale_pack_transpose(
                top, bottom, scratch, alpha, fc2_scale, lane
            )
            if cutlass.const_expr(release_acc):
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_state)
            if token_start + lane < valid_tokens:
                output_row = route_start + token_start + lane
                if cutlass.const_expr(self.scatter_output):
                    output_row = fc2_output_row
                for half in cutlass.range_constexpr(2):
                    channel = channel_start + warp * 32 + half * 16
                    if channel < output_tensor.shape[1]:
                        values = cute.make_tensor(
                            converted.iterator + half * 16, cute.make_layout((16,))
                        )
                        self._store_converted(
                            values, output_tensor, output_row, channel
                        )
        elif cutlass.const_expr(
            self.activation_type == ActivationType.Relu2.value
            or self.activation_type is None
        ):
            top = _blockscaled_load16x32(scratch)
            bottom = _blockscaled_load16x32(scratch + (16 << 16))
            transposed = _blockscaled_transpose16x32((top, bottom), scratch)
            top, bottom = transposed[None, 0], transposed[None, 1]
            if cutlass.const_expr(release_acc):
                # All overlapping accumulator and transpose-scratch reads have
                # completed before either CTA allows the next MMA to overwrite.
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_state)
            row = route_start + token_start + lane
            scale = alpha
            output_row = row
            if token_start + lane < valid_tokens:
                if cutlass.const_expr(
                    self.use_per_token_scale
                    and self.activation_type == ActivationType.Relu2.value
                ):
                    scale = (
                        scale
                        * per_token_scale[
                            row
                            if self.per_token_scale_is_routed
                            else token_mapping[row] // self.top_k
                        ]
                    )
                if cutlass.const_expr(self.scatter_output):
                    output_row = fc2_output_row
                    scale = fc2_scale
            if cutlass.const_expr(self.activation_type == ActivationType.Relu2.value):
                top = self._activate(top, None, scale)
                bottom = self._activate(bottom, None, scale)
            elif cutlass.const_expr(self.activation_type is None):
                for i in cutlass.range_constexpr(0, 16, 2):
                    top[i], top[i + 1] = cute.arch.mul_packed_f32x2(
                        (top[i], top[i + 1]), (scale, scale)
                    )
                    bottom[i], bottom[i + 1] = cute.arch.mul_packed_f32x2(
                        (bottom[i], bottom[i + 1]), (scale, scale)
                    )
            shared_quantization = None
            if cutlass.const_expr(self.generate_sfc and self.sf_vec_size == 32):
                absmax = cute.arch.fmax(self._absmax(top), self._absmax(bottom))
                shared_quantization = self._quantization_params(absmax, norm_const)
            channel = channel_start + warp * 32
            if token_start + lane < valid_tokens:
                if channel < output_tensor.shape[1]:
                    words_per_half = 16 * self.output_dtype.width // 32
                    converted = cute.make_rmem_tensor(
                        (2 * words_per_half,), cutlass.Uint32
                    )
                    for half in cutlass.range_constexpr(2):
                        values = top if half == 0 else bottom
                        quantization = shared_quantization
                        if cutlass.const_expr(
                            self.generate_sfc and self.sf_vec_size == 16
                        ):
                            quantization = self._quantization_params(
                                self._absmax(values), norm_const
                            )
                        fragment = self._convert(
                            values,
                            output_sf_tensor,
                            output_row,
                            channel + half * 16,
                            quantization,
                        )
                        target = cute.make_tensor(
                            converted.iterator + half * words_per_half,
                            cute.make_layout((words_per_half,)),
                        )
                        target.store(fragment.load().bitcast(cutlass.Uint32))
                    self._store_converted(converted, output_tensor, output_row, channel)

    @cute.jit
    def run_overlapping_fc1(
        self,
        acc_tensor,
        output_tensor,
        output_sf_tensor,
        alpha,
        norm_const,
        per_token_scale,
        token_mapping,
        channel_start,
        route_start,
        valid_tokens,
        tidx,
        smem_scratch,
        acc_pipeline,
        acc_state,
        channel_stride: cutlass.Constexpr,
        channel_tiles: cutlass.Constexpr,
    ):
        # Treat all channel tiles as one virtual accumulator column range.
        # The actual scale footprint, rounded to a 32-column transpose slab,
        # determines the shared interval. Read it before releasing ownership.
        groups = self.tile_n // 32
        slabs = channel_tiles * groups
        for step in cutlass.range_constexpr(slabs):
            slab = (
                step + (1 - acc_state.phase) * (slabs - self.overlap_activation_steps)
            ) % slabs
            channel_tile = slab // groups
            token_start = (slab % groups) * 32
            tile = cute.make_tensor(
                acc_tensor.iterator + cute.assume(channel_tile * self.tile_n, divby=32),
                acc_tensor.layout,
            )
            if token_start < valid_tokens:
                self._run_token_group(
                    tile,
                    output_tensor,
                    output_sf_tensor,
                    alpha,
                    norm_const,
                    per_token_scale,
                    token_mapping,
                    channel_start + channel_tile * channel_stride,
                    route_start,
                    valid_tokens,
                    tidx,
                    smem_scratch,
                    token_start,
                    cutlass.Int32(0),
                    alpha,
                    step + 1 == self.overlap_activation_steps,
                    acc_pipeline,
                    acc_state,
                )
            elif cutlass.const_expr(step + 1 == self.overlap_activation_steps):
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_state)

    @cute.jit
    def _scatter_metadata(
        self,
        alpha,
        per_token_scale,
        token_mapping,
        token_final_scales,
        route_start,
        valid_tokens,
        lane,
    ):
        """Keep route indices and rounded output scales in registers for every slab."""
        groups = max(1, self.tile_n // 32)
        output_rows = None
        if cutlass.const_expr(self.use_fused_finalize):
            output_rows = cute.make_rmem_tensor((groups,), cutlass.Int32)
        output_scales = cute.make_rmem_tensor((groups,), cutlass.Float32)
        for group in cutlass.range_constexpr(groups):
            if cutlass.const_expr(self.use_fused_finalize):
                output_rows[group] = 0
            output_scales[group] = alpha
            row = route_start + group * 32 + lane
            if group * 32 + lane < valid_tokens:
                route_scale = cutlass.Float32(1.0)
                if cutlass.const_expr(self.use_fused_finalize):
                    expanded = token_mapping[row]
                    output_row = expanded // self.top_k
                    route_scale = cutlass.Float32(
                        token_final_scales[output_row, expanded % self.top_k]
                    )
                if cutlass.const_expr(self.use_per_token_scale):
                    route_scale = (
                        route_scale
                        * per_token_scale[
                            row
                            if self.per_token_scale_is_routed
                            else token_mapping[row] // self.top_k
                        ]
                    )
                scale = alpha * route_scale
                if cutlass.const_expr(
                    self.final_scale_dtype in (cutlass.BFloat16, cutlass.Float16)
                ):
                    # Match the former metadata buffer's rounding before
                    # multiplying the FP32 accumulator.
                    scale = cutlass.Float32(self.final_scale_dtype(scale))
                if cutlass.const_expr(self.use_fused_finalize):
                    output_rows[group] = output_row
                output_scales[group] = scale
        return output_rows, output_scales

    @cute.jit
    def _run_bulk(
        self,
        acc_tensor,
        output_tensor,
        tma_output,
        alpha,
        per_token_scale,
        token_mapping,
        channel_start,
        route_start,
        valid_tokens,
        tidx,
        smem_scratch,
        acc_pipeline,
        acc_state,
    ):
        warp = (tidx // 32) % 4
        lane = tidx % 32
        tokens = self.bulk_tokens
        stages = self.bulk_stages
        if cutlass.const_expr(tokens not in (32, 64)):
            raise ValueError("Unsupported FC2 bulk token slab")
        uniform_scale = alpha
        if cutlass.const_expr(
            self.final_scale_dtype in (cutlass.BFloat16, cutlass.Float16)
        ):
            uniform_scale = cutlass.Float32(self.final_scale_dtype(uniform_scale))
        scales = None
        if cutlass.const_expr(self.use_per_token_scale):
            _, scales = self._scatter_metadata(
                alpha,
                per_token_scale,
                token_mapping,
                None,
                route_start,
                valid_tokens,
                lane,
            )
        shared_all = cute.make_tensor(
            cute.recast_ptr(smem_scratch, dtype=self.output_dtype),
            cute.make_layout((tokens * 128, stages), stride=(1, tokens * 128)),
        )
        groups = self.tile_n // tokens
        for step in cutlass.range_constexpr(groups):
            group = step
            if cutlass.const_expr(self.overlap_acc):
                # Consume the shared interval first without two epilogue copies.
                rotated_group = (step + groups - 64 // tokens) % groups
                group = (
                    step + (1 - acc_state.phase) * (groups - 64 // tokens)
                ) % groups
            if (group + 1) * tokens <= valid_tokens:
                shared = shared_all[None, step % stages]
                ptr = (
                    acc_tensor.iterator
                    + cute.assume(group * tokens, divby=32)
                    + cute.assume(warp * (32 << 16), divby=32)
                )
                # Ld16x256 assigns lane//4 to one of 8 channels and lane%4
                # to a pair of tokens. Alternate register pairs select the
                # second 8 channels; subsequent quartets select 8 more tokens.
                # After conversion this is stmatrix.trans's native fragment.
                packed = cute.make_rmem_tensor((tokens,), self.output_dtype)
                for half in cutlass.range_constexpr(2):
                    values = cute.make_rmem_tensor((tokens // 2,), cutlass.Float32)
                    cute.copy(
                        cute.make_copy_atom(
                            tcgen05.Ld16x256bOp(tcgen05.Repetition(tokens // 8)),
                            cutlass.Float32,
                        ),
                        _blockscaled_tmem_view(ptr + (half * 16 << 16), 16, tokens),
                        _blockscaled_rmem_view(values, tokens // 2),
                    )
                    if cutlass.const_expr(
                        self.overlap_acc and (step + 1) * tokens == 64 and half == 1
                    ):
                        # The next MMA overlaps only these 64 columns. Complete
                        # every TMEM read before either CTA releases ownership.
                        cute.arch.fence_view_async_tmem_load()
                        acc_pipeline.consumer_release(acc_state)
                    if cutlass.const_expr(self.use_per_token_scale):
                        for i in cutlass.range_constexpr(0, tokens // 2, 2):
                            token = (i // 4) * 8 + (lane % 4) * 2
                            # Keep both register indices static when rotating.
                            scale = scales[step * (tokens // 32) + i // 16]
                            if cutlass.const_expr(self.overlap_acc):
                                scale = cutlass.select_(
                                    acc_state.phase == 0,
                                    scales[rotated_group * (tokens // 32) + i // 16],
                                    scale,
                                )
                            scale_pair = (
                                cute.arch.shuffle_sync(scale, token % 32),
                                cute.arch.shuffle_sync(scale, (token + 1) % 32),
                            )
                            values[i], values[i + 1] = cute.arch.mul_packed_f32x2(
                                (values[i], values[i + 1]), scale_pair
                            )
                    else:
                        for i in cutlass.range_constexpr(0, tokens // 2, 2):
                            values[i], values[i + 1] = cute.arch.mul_packed_f32x2(
                                (values[i], values[i + 1]),
                                (uniform_scale, uniform_scale),
                            )
                    converted = cute.make_tensor(
                        packed.iterator + half * (tokens // 2),
                        cute.make_layout((tokens // 2,)),
                    )
                    converted.store(values.load().to(self.output_dtype))
                words = cute.recast_tensor(packed, cutlass.Uint32)
                # Wait only before overwriting the shared stage.
                if cutlass.const_expr(step >= stages):
                    cute.arch.cp_async_bulk_wait_group(stages - 1, read=True)
                    self.transpose_barrier.arrive_and_wait()
                for token8 in cutlass.range_constexpr(tokens // 8):
                    address = (
                        shared.iterator
                        + (warp // 2) * tokens * 64
                        + (token8 * 8 + lane % 8) * 64
                        + (warp % 2) * 32
                        + (lane // 8) * 8
                    )
                    _bulk_stmatrix4(
                        address,
                        words[token8 * 2],
                        words[token8 * 2 + 1],
                        words[tokens // 4 + token8 * 2],
                        words[tokens // 4 + token8 * 2 + 1],
                    )
                cute.arch.fence_proxy("async.shared", space="cta")
                self.transpose_barrier.arrive_and_wait()
                if warp < 2:
                    with cute.arch.elect_one():
                        _bulk_tensor_store(
                            tma_output,
                            shared.iterator + warp * tokens * 64,
                            channel_start + warp * 64,
                            route_start + group * tokens,
                        )
                    cute.arch.cp_async_bulk_commit_group()
            elif group * tokens < valid_tokens:
                for sub in cutlass.range_constexpr(tokens // 32):
                    token_start = group * tokens + sub * 32
                    if token_start < valid_tokens:
                        scale = uniform_scale
                        if cutlass.const_expr(self.use_per_token_scale):
                            scale = scales[step * (tokens // 32) + sub]
                            if cutlass.const_expr(self.overlap_acc):
                                scale = cutlass.select_(
                                    acc_state.phase == 0,
                                    scales[rotated_group * (tokens // 32) + sub],
                                    scale,
                                )
                        self._run_token_group(
                            acc_tensor,
                            output_tensor,
                            None,
                            alpha,
                            None,
                            per_token_scale,
                            token_mapping,
                            channel_start,
                            route_start,
                            valid_tokens,
                            tidx,
                            smem_scratch,
                            token_start,
                            route_start + token_start + lane,
                            scale,
                            self.overlap_acc
                            and (step + 1) * tokens == 64
                            and sub + 1 == tokens // 32,
                            acc_pipeline,
                            acc_state,
                        )
                    elif cutlass.const_expr(
                        self.overlap_acc
                        and (step + 1) * tokens == 64
                        and sub + 1 == tokens // 32
                    ):
                        cute.arch.fence_view_async_tmem_load()
                        acc_pipeline.consumer_release(acc_state)
            elif cutlass.const_expr(self.overlap_acc and (step + 1) * tokens == 64):
                # A skipped overlap group has no live TMEM values.
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_state)
        if valid_tokens >= tokens:
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            self.transpose_barrier.arrive_and_wait()

    @cute.jit
    def run(
        self,
        acc_tensor,
        output_tensor,
        output_sf_tensor,
        alpha,
        norm_const,
        per_token_scale,
        token_mapping,
        token_final_scales,
        channel_start,
        route_start,
        valid_tokens,
        tidx,
        smem_scratch,
        acc_pipeline=None,
        acc_state=None,
        tma_output=None,
    ):
        lane = tidx % 32
        if cutlass.const_expr(self.bulk_tokens > 0):
            # A final two-CTA channel tile can have no live channels in its
            # second CTA. Avoid issuing a zero-byte global bulk copy there.
            if channel_start < output_tensor.shape[1]:
                self._run_bulk(
                    acc_tensor,
                    output_tensor,
                    tma_output,
                    alpha,
                    per_token_scale,
                    token_mapping,
                    channel_start,
                    route_start,
                    valid_tokens,
                    tidx,
                    smem_scratch,
                    acc_pipeline,
                    acc_state,
                )
            elif cutlass.const_expr(self.overlap_acc):
                # This CTA has no live output channels, hence no TMEM reads.
                acc_pipeline.consumer_release(acc_state)
        elif cutlass.const_expr(self.scatter_output):
            # Fetch the whole tile's routing metadata before its stores. Keeping
            # the slab indices static preserves these values in registers.
            groups = max(1, self.tile_n // 32)
            output_rows = None
            output_scales = None
            if cutlass.const_expr(self.use_per_token_scale or self.use_fused_finalize):
                output_rows, output_scales = self._scatter_metadata(
                    alpha,
                    per_token_scale,
                    token_mapping,
                    token_final_scales,
                    route_start,
                    valid_tokens,
                    lane,
                )
            uniform_scale = alpha
            if cutlass.const_expr(
                self.final_scale_dtype in (cutlass.BFloat16, cutlass.Float16)
            ):
                uniform_scale = cutlass.Float32(self.final_scale_dtype(uniform_scale))
            for group in cutlass.range_constexpr(groups):
                if group * 32 < valid_tokens:
                    self._run_token_group(
                        acc_tensor,
                        output_tensor,
                        output_sf_tensor,
                        alpha,
                        norm_const,
                        per_token_scale,
                        token_mapping,
                        channel_start,
                        route_start,
                        valid_tokens,
                        tidx,
                        smem_scratch,
                        group * 32,
                        (
                            output_rows[group]
                            if cutlass.const_expr(self.use_fused_finalize)
                            else route_start + group * 32 + lane
                        ),
                        (
                            output_scales[group]
                            if cutlass.const_expr(
                                self.use_per_token_scale or self.use_fused_finalize
                            )
                            else uniform_scale
                        ),
                    )
        elif cutlass.const_expr(
            self.gated or self.activation_type == ActivationType.Relu2.value
        ):
            for token_start in cutlass.range(0, self.tile_n, 32, unroll=1):
                if token_start < valid_tokens:
                    self._run_token_group(
                        acc_tensor,
                        output_tensor,
                        output_sf_tensor,
                        alpha,
                        norm_const,
                        per_token_scale,
                        token_mapping,
                        channel_start,
                        route_start,
                        valid_tokens,
                        tidx,
                        smem_scratch,
                        token_start,
                        cutlass.Int32(0),
                        alpha,
                    )


class Sm100BlockScaledGroupedGemmKernel:
    """Persistent one/two-CTA grouped GEMM shared by the split MoE phases."""

    def __init__(
        self,
        activation_dtype,
        weight_dtype,
        sf_dtype,
        sf_vec_size,
        native_mma_k,
        scatter_output,
        gather_input,
        per_token_scale_is_routed,
        k_steps,
        num_stages,
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
        self.gather_input = gather_input
        self.per_token_scale_is_routed = per_token_scale_is_routed
        self.pdl_signal_at = pdl_signal_at
        self.prefetch_weights_before_wait = prefetch_weights_before_wait
        self.publish_routing_before_signal = publish_routing_before_signal
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
        self.gather_k, atom_k = (256, 64) if both_fp4 else (128, 32)
        if type(native_mma_k) is not int or (
            native_mma_k != atom_k
            and not (native_mma_k == 96 and both_fp4 and sf_vec_size == 16)
        ):
            raise ValueError("Unsupported native blockscaled MMA K")
        self.native_mma_k = native_mma_k
        self.operand_alignment = 1024 if native_mma_k == 96 else 128
        if mma_tiler_mnk[2] != self.gather_k:
            raise ValueError("Unsupported public blockscaled K tile")
        self.bulk_output_bytes = self.bulk_output_element_bytes(
            mma_tiler_mnk[1],
            output_dtype,
            scatter_output,
            use_fused_finalize,
            generate_sfc,
        )
        max_release_cols = (
            cute.arch.get_max_tmem_alloc_cols("sm_100") // 2
            if overlap_activation_accumulators is not False
            else 64
        )
        if k_steps is None:
            k_steps = self._select_k_steps(
                mma_tiler_mnk,
                activation_dtype,
                weight_dtype,
                sf_vec_size,
                num_stages,
                overlap_activation_accumulators,
                max_release_cols,
                self.bulk_output_bytes,
                use_clc_scheduler,
                num_tile_info_stages,
                self.operand_alignment,
                self.gather_input,
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
        self.top_k = top_k
        self.use_fused_finalize = use_fused_finalize
        self.enable_pdl = enable_pdl
        self.max_active_clusters = max_active_clusters
        if max_active_clusters < self.cta_group_size:
            raise ValueError("The launch needs at least one complete CTA group")
        self.requested_num_stages = num_stages
        self.num_stages = num_stages
        self.threads_per_cta = 384 if gather_input else 256
        self.weight_warp = 0
        self.activation_warp = 1
        self.mma_warp = 2
        self.schedule_warp = 3
        self.overlap_acc = False
        self.acc_cols = self.allocated_n * self.channel_tiles
        self.sf_cols = self.tile_k // self.sf_vec_size
        self.sfb_cols = self.sf_cols * self.sfb_row_blocks
        scale_cols = self.sf_cols * self.channel_tiles + self.sfb_cols
        self.acc_stages, self.can_overlap_accumulators = self._accumulator_storage(
            self.acc_cols, scale_cols, max_release_cols
        )
        self.overlap_activation_accumulators = (
            self.can_overlap_accumulators
            if overlap_activation_accumulators is None
            else overlap_activation_accumulators
        )
        if self.overlap_activation_accumulators and not self.can_overlap_accumulators:
            raise ValueError("Unsupported overlapping activation accumulator geometry")
        self.tmem_cols = (
            1 << (self.acc_stages * self.acc_cols + scale_cols - 1).bit_length()
        )
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
    def bulk_output_element_bytes(
        route_tile, output_dtype, scatter_output, use_fused_finalize, generate_sfc
    ):
        return (
            output_dtype.width // 8
            if (
                scatter_output
                and not use_fused_finalize
                and output_dtype in (cutlass.BFloat16, cutlass.Float16)
                and not generate_sfc
                and route_tile >= 128
            )
            else 0
        )

    @staticmethod
    def _accumulator_storage(acc_cols, scale_cols, max_release_cols):
        capacity = cute.arch.get_max_tmem_alloc_cols("sm_100")
        stages = min(2, (capacity - scale_cols) // acc_cols)
        if stages < 1:
            raise ValueError("Not enough SM100 tensor memory for one accumulator tile")
        # Activation slabs can release the whole window; bulk releases 64 columns.
        # This ownership constraint is separate from the hardware capacity.
        can_overlap = 2 * acc_cols == capacity and scale_cols <= max_release_cols
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
        operand_alignment=128,
        gather_input=False,
    ):
        # The CLC response has its own 16-byte alignment inside the slab.
        scheduler_bytes = 32 * num_tile_info_stages + 48 if use_clc_scheduler else 0
        return (
            (
                (num_stages * (2 + 2 * gather_input) + acc_stages * 2) * 8
                + 4
                + scheduler_bytes
                + 127
            )
            // 128
        ) * 128 + operand_alignment

    @staticmethod
    def _epilogue_storage(available_bytes, element_bytes):
        if element_bytes:
            # Supported TMEM/stmatrix slabs, in preference order.
            for tokens, stages in ((64, 2), (32, 1)):
                scratch_bytes = tokens * 128 * stages * element_bytes
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
        operand_alignment=128,
        gather_input=False,
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
                operand_alignment,
                gather_input,
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
                operand_alignment,
                gather_input,
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
        max_release_cols,
        bulk_output_bytes,
        use_clc_scheduler,
        num_tile_info_stages,
        operand_alignment=128,
        gather_input=False,
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
            acc_cols, scale_cols, max_release_cols
        )
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
            bulk_output_bytes,
            acc_stages,
            use_clc_scheduler,
            num_tile_info_stages,
            operand_alignment,
            gather_input,
        ), can_overlap

    @classmethod
    def _select_k_steps(
        cls,
        mma_tiler_mnk,
        activation_dtype,
        weight_dtype,
        sf_vec_size,
        num_stages,
        overlap_activation_accumulators,
        max_release_cols,
        bulk_output_bytes,
        use_clc_scheduler,
        num_tile_info_stages,
        operand_alignment=128,
        gather_input=False,
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
        preserve_overlap = 2 * acc_cols == tmem_capacity and (
            overlap_activation_accumulators is not False
            or (bulk_output_bytes > 0 and channel_tiles == 1)
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
                max_release_cols,
                bulk_output_bytes,
                use_clc_scheduler,
                num_tile_info_stages,
                operand_alignment,
                gather_input,
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

    def _configure_pipeline_storage(self, tiled_mma):
        """Fit SM100 operand stages around the epilogue's working storage.

        ``num_stages=None`` selects capacity at compile time. An explicit depth
        remains available for controlled tuning.
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
        num_stages = self.requested_num_stages
        if num_stages is None:
            num_stages = self._operand_stage_capacity(
                stage_bytes,
                self.bulk_output_bytes,
                self.acc_stages,
                self.use_clc_scheduler,
                self.num_tile_info_stages,
                self.operand_alignment,
                self.gather_input,
            )
            if num_stages < 1:
                raise ValueError("Not enough SM100 shared memory for one operand stage")
        self.num_stages = num_stages

        layouts = self._operand_layouts(tiled_mma, self.num_stages)
        available = (
            capacity
            - self._operand_bytes(layouts)
            - self._pipeline_metadata_bytes(
                self.num_stages,
                self.acc_stages,
                self.use_clc_scheduler,
                self.num_tile_info_stages,
                self.operand_alignment,
                self.gather_input,
            )
        )
        (self.epilogue.bulk_tokens, self.epilogue.bulk_stages, scratch_bytes) = (
            self._epilogue_storage(available, self.bulk_output_bytes)
        )
        self.epilogue_scratch_floats = scratch_bytes // 4
        if available < scratch_bytes:
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
            64,
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
        out_channels = cute.assume(
            num_channels // (2 if self.epilogue.gated else 1), divby=16
        )
        source_rows = num_tokens if self.gather_input else num_routes
        # FC1 scales still belong to original tokens; the epilogue indexes them
        # through the unchanged route mapping. FC2 scales belong to route rows.
        per_token_scale_rows = (
            num_routes if self.per_token_scale_is_routed else num_tokens
        )
        output_rows = num_routes
        if cutlass.const_expr(self.use_fused_finalize):
            output_rows = num_tokens
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
        activation_sf_size = (
            source_rows * reduction_size // self.sf_vec_size
            if self.gather_input
            else cute.ceil_div(source_rows, self.route_tile)
            * self.sfb_row_blocks
            * 128
            * cute.ceil_div(reduction_size, self.sf_block_k)
            * 4
        )
        activations_sf = cute.make_tensor(
            activation_sf_ptr, cute.make_layout((activation_sf_size,))
        )
        output = cute.make_tensor(
            output_ptr,
            cute.make_layout((output_rows, out_channels), stride=(out_channels, 1)),
        )
        output_sf = None
        if cutlass.const_expr(output_sf_ptr is not None):
            sf_layout = blockscaled_utils.tile_atom_to_shape_SF(
                (
                    cute.ceil_div(num_routes, self.route_tile)
                    * self.sfb_row_blocks
                    * 128,
                    out_channels,
                    1,
                ),
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
        self._configure_pipeline_storage(tiled_mma)
        acc_layout = tiled_mma.make_fragment_C(
            tiled_mma.partition_shape_C((self.mma_m, self.allocated_n))
        ).layout

        tma_output = None
        if cutlass.const_expr(self.epilogue.bulk_tokens > 0):
            output_smem_layout = cute.make_composed_layout(
                cute.make_swizzle(3, 4, 3),
                0,
                cute.make_layout((self.epilogue.bulk_tokens, 64), stride=(64, 1)),
            )
            tma_output, _ = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                output,
                output_smem_layout,
                (self.epilogue.bulk_tokens, 64),
            )

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
        tma_atom_sfb = None
        if cutlass.const_expr(self.gather_input):
            # Gather4 transfers four independently indexed 128-byte rows. Keep
            # both extents dynamic so a one-token input still has a 2D descriptor.
            tma_activations = cute.make_tensor(
                activations.iterator,
                cute.make_layout(
                    (
                        cutlass.Int32(activations.shape[0]),
                        cutlass.Int32(activations.shape[1]),
                    ),
                    stride=activations.stride,
                ),
            )
            gather_layout = cute.make_composed_layout(
                cute.make_swizzle(3, 4, 3),
                0,
                cute.make_layout((1, self.gather_k), stride=(self.gather_k, 1)),
            )
            tma_atom_b, _ = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                tma_activations,
                gather_layout,
                (1, self.gather_k),
            )
        else:
            tma_atom_b, tma_activations = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
                activations,
                cute.slice_(b_copy_layout, (None, None, 0)),
                (self.b_copy_rows, self.tile_k),
            )

            # The canonical SF layout consists of contiguous 512-byte blocks.
            # Keep K-block and row-block coordinates separate so TMA zero-fills
            # a short final K tile without crossing into the next row block.
            sf_k_blocks = cute.ceil_div(weights.shape[1], self.sf_block_k)
            sf_row_blocks = (
                cute.ceil_div(activations.shape[0], self.route_tile)
                * self.sfb_row_blocks
            )
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
            and self.channel_tiles == 1
            and self.epilogue.bulk_tokens > 0
        ) or self.overlap_activation_accumulators
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
            operand_barriers: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            gather_barriers: cute.struct.MemRange[
                cutlass.Int64, self.num_stages if self.gather_input else 0
            ]
            gather_local_barriers: cute.struct.MemRange[
                cutlass.Int64, self.num_stages if self.gather_input else 0
            ]
            acc_barriers: cute.struct.MemRange[cutlass.Int64, 2 * self.acc_stages]
            tmem_holding: cutlass.Int32
            tile_info_barriers: cute.struct.MemRange[
                cutlass.Int64,
                2 * self.num_tile_info_stages if self.use_clc_scheduler else 0,
            ]
            tile_info: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int32,
                    4 * self.num_tile_info_stages if self.use_clc_scheduler else 0,
                ],
                16,
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
                self.operand_alignment,
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[
                    self.activation_dtype, cute.cosize(self.b_layout.outer)
                ],
                self.operand_alignment,
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
        work_clusters = (
            cute.ceil_div(weights.shape[0], self.mma_m * self.channel_tiles)
            * expert_idx.shape[0]
        )
        launch_grid = (
            cutlass.max(
                1,
                cutlass.min(
                    self.max_active_clusters // self.cta_group_size, work_clusters
                ),
            )
            * self.cta_group_size,
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
            acc_layout,
            tma_atom_a,
            tma_weights,
            tma_atom_b,
            tma_activations,
            tma_atom_sfa,
            tma_weights_sf,
            tma_atom_sfb,
            activations_sf if self.gather_input else None,
            output,
            output_sf,
            tma_output,
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
    def _copy_sf(self, copy, source, target, stage_k: cutlass.Constexpr):
        # Copy complete four-scale words, keeping the stage and row-bank strides.
        words = cute.ceil_div(stage_k, self.sf_block_k)
        if cutlass.const_expr(words < target.shape[3]):
            source = cute.make_tensor(
                source.iterator,
                cute.make_layout((*source.shape[:3], words), stride=source.stride),
            )
            target = cute.make_tensor(
                target.iterator,
                cute.make_layout((*target.shape[:3], words), stride=target.stride),
            )
        cute.copy(copy, source, target)

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
    def weight_cache_hint(self, route, expert, expert_idx, num_routes):
        # Preserve weights reused by other token tiles of the same expert.
        has_next = (route + 1 < num_routes) & (
            expert_idx[cutlass.min(route + 1, num_routes - 1)] == expert
        )
        has_previous = (route > 0) & (expert_idx[cutlass.max(route - 1, 0)] == expert)
        return (
            cutlass.Int64(0)
            if has_next | has_previous
            else cutlass.Int64(0x12F0000000000000)
        )

    @cute.jit
    def load_weight_stage(
        self,
        operand_pipe,
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
        weight_cache_hint,
    ):
        # TMA copy lowering elects internally; keep the whole warp active.
        if cutlass.const_expr(self.enable_iket):
            _iket.range_push("producer_tma_empty")
        operand_pipe.producer_acquire(
            producer,
            expected_tx=self.cta_group_size
            * (
                (128 * self.tile_k * self.weight_dtype.width // 8 + self.sf_stage_bytes)
                * self.channel_tiles
                + (
                    0
                    if self.gather_input
                    else self.b_copy_rows
                    * self.tile_k
                    * self.activation_dtype.width
                    // 8
                )
                + (0 if self.gather_input else self.sfb_stage_bytes)
            ),
        )
        if cutlass.const_expr(self.enable_iket):
            _iket.range_pop()
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
                tma_bar_ptr=operand_pipe.producer_get_barrier(producer),
                cache_policy=weight_cache_hint,
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
                tma_bar_ptr=operand_pipe.producer_get_barrier(producer),
                cache_policy=weight_cache_hint,
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
        gather_activations_sf,
        output,
        output_sf,
        tma_output,
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
        cta_rank = (
            cutlass.Int32(0)
            if cutlass.const_expr(self.cta_group_size == 1)
            else cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        )
        # Static grids have a guaranteed physical CTA0. CLC can cancel it, so
        # each executing cluster keeps its own ancestor-publication owner.
        publishes_ancestors = (block == 0) and (block_y == 0) and (block_z == 0)
        if cutlass.const_expr(self.use_clc_scheduler):
            publishes_ancestors = (self.cta_group_size == 1) or (cta_rank == 0)
        block = block // self.cta_group_size
        grid = grid // self.cta_group_size
        storage = utils.SmemAllocator().allocate(self.storage_type)
        # Weight announces all operand bytes; both producers wait for reuse.
        # MMA releases their shared stage only after consuming all operands.
        operand_pipe = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.operand_barriers.data_ptr(),
            defer_sync=True,
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout,
            tx_count=0,
        )
        acc_pipe = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_barriers.data_ptr(),
            defer_sync=True,
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
                    pipeline.Agent.Thread,
                    # Only the leader CTA's MMA warp consumes tile metadata.
                    32 * (6 + 4 * self.gather_input + cutlass.Int32(cta_rank == 0)),
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
        if cutlass.const_expr(self.gather_input):
            if tidx == 0:
                for stage in cutlass.range_constexpr(self.num_stages):
                    cute.arch.mbarrier_init(
                        storage.gather_barriers.data_ptr() + stage,
                        self.cta_group_size,
                    )
                    cute.arch.mbarrier_init(
                        storage.gather_local_barriers.data_ptr() + stage,
                        1 + min(self.route_tile, 128),
                    )
        # Publish all pipeline barriers once before any role can use them.
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
        total_m_tiles = cute.ceil_div(weights.shape[0], self.mma_m * self.channel_tiles)
        k_tiles = cute.ceil_div(weights.shape[1], self.tile_k)
        if cutlass.const_expr(self.enable_pdl):
            waits_for_payload = cutlass.Boolean(True)
            if cutlass.const_expr(self.prefetch_weights_before_wait):
                # The same startup release that permits weight prefetch publishes
                # routing metadata. Only payload readers need the dependency wait.
                waits_for_payload = (warp == self.activation_warp) | (warp >= 4)
            if waits_for_payload:
                if cutlass.const_expr(self.enable_iket):
                    _iket.range_push("startup_dependency_wait")
                griddepcontrol_wait()
                if cutlass.const_expr(self.enable_iket):
                    _iket.range_pop()
        if cutlass.const_expr(self.enable_pdl and self.pdl_signal_at == "startup"):
            # The activation warp has acquired ancestor payload publication.
            # The weight and scheduler warps may still be running ahead of it.
            if warp == self.activation_warp:
                if cutlass.const_expr(
                    self.publish_routing_before_signal
                    or (self.scatter_output and not self.use_fused_finalize)
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
        # Keep each warpgroup's register budget scoped to its role until the join.
        if warp < 4:
            if cutlass.const_expr(self.gather_input):
                cute.arch.setmaxregister_decrease(128)
            if warp == self.schedule_warp:
                if cutlass.const_expr(self.use_clc_scheduler):
                    sched = utils.ClcDynamicPersistentTileScheduler.create(
                        sched_params,
                        cute.arch.block_idx(),
                        cute.arch.grid_dim(),
                        storage.clc_response.data_ptr(),
                    )
                    work = sched.initial_work_tile_info()
                    info_producer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.num_tile_info_stages
                    )
                    clc_producer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.ProducerConsumer, 1
                    )
                    clc_consumer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, 1
                    )
                    while work.is_valid_tile:
                        if cutlass.const_expr(self.enable_iket):
                            _iket.range_push("scheduler_item")
                        route = work.tile_idx[1]
                        is_routed = route < num_routed_tiles
                        if is_routed:
                            info_pipe.producer_acquire(info_producer)
                        if cta_rank == 0:
                            clc_pipe.producer_acquire(clc_producer)
                            sched.advance_to_next_work(
                                clc_pipe.producer_get_barrier(clc_producer)
                            )
                            clc_producer.advance()
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
                        clc_pipe.consumer_wait(clc_consumer)
                        work = sched.get_current_work()
                        clc_pipe.consumer_release(clc_consumer)
                        clc_consumer.advance()
                        if cutlass.const_expr(self.enable_iket):
                            _iket.range_pop()
                    if cutlass.const_expr(self.enable_iket):
                        _iket.range_push("scheduler_item")
                    info_pipe.producer_acquire(info_producer)
                    with cute.arch.elect_one():
                        sInfo[0, info_producer.index] = cutlass.Int32(-1)
                        sInfo[1, info_producer.index] = cutlass.Int32(0)
                        sInfo[2, info_producer.index] = cutlass.Int32(0)
                        info_pipe.producer_commit(info_producer)
                    info_producer.advance()
                    if cutlass.const_expr(self.enable_iket):
                        _iket.range_pop()
                    # The final CTA join drains all terminal metadata consumers.
                    if cta_rank == 0:
                        clc_pipe.producer_tail(clc_producer)
            elif warp == self.weight_warp:
                gA = cute.local_tile(
                    weights, (self.mma_m, self.tile_k), (None, None, None)
                )
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
                producer = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_stages
                )
                info_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_info_stages
                )
                tile, scheduled_expert, _, info_state = self.next_work(
                    info_pipe, info_state, sInfo, block, total_tiles
                )
                while tile >= 0:
                    route_idx = tile // total_m_tiles
                    expert = (
                        scheduled_expert
                        if cutlass.const_expr(self.use_clc_scheduler)
                        else expert_idx[route_idx]
                    )
                    weight_hint = self.weight_cache_hint(
                        route_idx, expert, expert_idx, num_routed_tiles
                    )
                    for k_tile in cutlass.range(k_tiles):
                        self.load_weight_stage(
                            operand_pipe,
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
                            weight_hint,
                        )
                        producer.advance()
                    tile, scheduled_expert, _, info_state = self.next_work(
                        info_pipe, info_state, sInfo, tile + grid, total_tiles
                    )
                if cutlass.const_expr(
                    self.enable_pdl and self.prefetch_weights_before_wait
                ):
                    griddepcontrol_wait()
                if cutlass.const_expr(
                    self.enable_pdl and self.pdl_signal_at == "producer_exit"
                ):
                    if cutlass.const_expr(
                        self.scatter_output and not self.use_fused_finalize
                    ):
                        if publishes_ancestors:
                            release_gpu()
                    griddepcontrol_launch_dependents()
                # A single tail protects storage shared by both independent producers.
                operand_pipe.producer_tail(producer.clone())
            elif warp == self.activation_warp:
                if cutlass.const_expr(not self.gather_input):
                    producer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.num_stages
                    )
                    info_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.num_tile_info_stages
                    )
                    tile, _, scheduled_valid, info_state = self.next_work(
                        info_pipe, info_state, sInfo, block, total_tiles
                    )
                    while tile >= 0:
                        route_idx = tile // total_m_tiles
                        route_start = route_idx * self.route_tile
                        valid = (
                            scheduled_valid
                            if cutlass.const_expr(self.use_clc_scheduler)
                            else cutlass.min(
                                self.route_tile, mn_limit[route_idx] - route_start
                            )
                        )
                        gB = cute.local_tile(
                            cute.domain_offset(
                                (
                                    route_start
                                    + cta_rank
                                    * (((valid + 15) & -16) // self.cta_group_size),
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
                        for k_tile in cutlass.range(k_tiles):
                            if cutlass.const_expr(self.enable_iket):
                                _iket.range_push("producer_activation_empty")
                            operand_pipe.sync_object_empty.wait(
                                producer.index, producer.phase
                            )
                            if cutlass.const_expr(self.enable_iket):
                                _iket.range_pop()
                                _iket.range_push("producer_activation_tma")
                            cute.copy(
                                tma_atom_b,
                                tBgB[(None, k_tile)],
                                tBsB[(None, producer.index)],
                                tma_bar_ptr=operand_pipe.producer_get_barrier(producer),
                                cache_policy=cutlass.Int64(
                                    self.activation_cache_policy
                                ),
                            )
                            for row_block in cutlass.range_constexpr(
                                self.sfb_row_blocks
                            ):
                                with cute.arch.elect_one():
                                    _blockscaled_tma_sfb(
                                        tma_atom_sfb,
                                        sSFB.iterator
                                        + producer.index * self.sfb_stage_bytes
                                        + row_block * self.sf_stage_bytes,
                                        operand_pipe.producer_get_barrier(producer),
                                        k_tile * (self.tile_k // self.sf_block_k),
                                        route_idx * self.sfb_row_blocks + row_block,
                                        self.cta_group_size,
                                        self.activation_cache_policy,
                                    )
                            if cutlass.const_expr(self.enable_iket):
                                _iket.range_pop()
                            producer.advance()
                        tile, _, scheduled_valid, info_state = self.next_work(
                            info_pipe, info_state, sInfo, tile + grid, total_tiles
                        )
                else:
                    # The issuing threads publish local cp.async completion without
                    # blocking. This warp relays it once per CTA to the MMA leader.
                    gather_consumer = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.num_stages
                    )
                    info_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.num_tile_info_stages
                    )
                    tile, _, _, info_state = self.next_work(
                        info_pipe, info_state, sInfo, block, total_tiles
                    )
                    while tile >= 0:
                        for _k_tile in cutlass.range(k_tiles):
                            if cutlass.const_expr(self.enable_iket):
                                _iket.range_push("producer_gather_relay_wait")
                            cute.arch.mbarrier_wait(
                                storage.gather_local_barriers.data_ptr()
                                + gather_consumer.index,
                                gather_consumer.phase,
                            )
                            if cutlass.const_expr(self.enable_iket):
                                _iket.range_pop()
                            cute.arch.fence_view_async_shared()
                            with cute.arch.elect_one():
                                _gather_mbarrier_arrive(
                                    storage.gather_barriers.data_ptr()
                                    + gather_consumer.index,
                                    self.cta_group_size,
                                )
                            gather_consumer.advance()
                        tile, _, _, info_state = self.next_work(
                            info_pipe, info_state, sInfo, tile + grid, total_tiles
                        )
            elif (warp == self.mma_warp) & (cta_rank == 0):
                tSFA = cute.make_tensor(
                    cute.recast_ptr(
                        tmem_ptr + self.acc_reserved_cols, dtype=self.sf_dtype
                    ),
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
                if cutlass.const_expr(self.native_mma_k == 96):
                    # Preserve the existing SF copy layout, but address individual
                    # SF16 entries for the mixed six/four-scale MMA updates.
                    sfa_iter = cute.make_layout(
                        ((((32, 4), 1), (self.sf_vec_size, 1)), 1, self.sf_cols)
                    )
                    sfb_iter = cute.make_layout(
                        (
                            (((32, 4), self.sfb_row_blocks), (self.sf_vec_size, 1)),
                            1,
                            self.sf_cols,
                        )
                    )
                    sfa_mma_layout = blockscaled_utils.make_tmem_layout_sfa(
                        tiled_mma, self.physical_tiler, self.sf_vec_size, sfa_iter
                    )
                    sfb_mma_layout = blockscaled_utils.make_tmem_layout_sfb(
                        tiled_mma, self.physical_tiler, self.sf_vec_size, sfb_iter
                    )
                    tSFA = cute.make_tensor(tSFA.iterator, sfa_mma_layout)
                    tSFB = cute.make_tensor(tSFB.iterator, sfb_mma_layout)
                    if cutlass.const_expr(self.channel_tiles == 2):
                        tSFA1 = cute.make_tensor(tSFA1.iterator, sfa_mma_layout)
                    fragment_a = sA
                    fragment_b = sB
                else:
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
                tile, _, scheduled_valid, info_state = self.next_work(
                    info_pipe, info_state, sInfo, block, total_tiles
                )
                while tile >= 0:
                    route_idx = tile // total_m_tiles
                    route_start = route_idx * self.route_tile
                    valid = (
                        scheduled_valid
                        if cutlass.const_expr(self.use_clc_scheduler)
                        else cutlass.min(
                            self.route_tile, mn_limit[route_idx] - route_start
                        )
                    )
                    tile_target_b = target_b
                    tile_sfb = tSFB
                    if cutlass.const_expr(
                        self.native_mma_k == 96 and self.sfb_row_blocks > 1
                    ):
                        # K96 crosses SF words using the instruction's actual N.
                        # Compact the first bank for narrow tiles and place the
                        # unused bank after it in the same allocated TMEM region.
                        wide_sf = cutlass.Int32(valid > 128)
                        sf_words = self.sf_cols // 4
                        sf_word_stride = target_b.stride[1]
                        sf_k_stride = sf_word_stride * (1 + wide_sf)
                        sf_n_stride = sf_word_stride * (
                            sf_words - (sf_words - 1) * wide_sf
                        )
                        tile_target_b = cute.make_tensor(
                            target_b.iterator,
                            cute.make_layout(
                                target_b.shape,
                                stride=(
                                    target_b.stride[0],
                                    sf_n_stride,
                                    0,
                                    sf_k_stride,
                                ),
                            ),
                        )
                        tile_sfb = cute.make_tensor(
                            tSFB.iterator,
                            cute.make_layout(
                                tSFB.shape,
                                stride=(tSFB.stride[0], 0, (1, sf_k_stride)),
                            ),
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
                    acc_pipe.producer_acquire(producer)
                    if cutlass.const_expr(self.enable_iket):
                        _iket.range_pop()
                    # The reduction is static: emit a separate short final stage
                    # rather than testing the valid K extent at each MMA instruction.
                    full_k_tiles = weights.shape[1] // self.tile_k
                    tail_k = weights.shape[1] % self.tile_k
                    for section in cutlass.range_constexpr(1 + (tail_k != 0)):
                        first_stage = 0 if section == 0 else full_k_tiles
                        last_stage = full_k_tiles if section == 0 else k_tiles
                        stage_k = self.tile_k if section == 0 else tail_k
                        for k_tile in cutlass.range(
                            first_stage, last_stage, unroll_full=(section == 1)
                        ):
                            if cutlass.const_expr(self.enable_iket):
                                _iket.range_push("mma_tma_ready")
                            operand_pipe.consumer_wait(consumer)
                            if cutlass.const_expr(self.enable_iket):
                                _iket.range_pop()
                                _iket.range_push("mma_sf_copy_issue")
                            sf_stage = (None, None, None, None, consumer.index)
                            sf_stage_a = (
                                None,
                                None,
                                None,
                                None,
                                consumer.index * self.channel_tiles,
                            )
                            self._copy_sf(
                                copy_a, source_a[sf_stage_a], target_a, stage_k
                            )
                            if cutlass.const_expr(self.channel_tiles == 2):
                                sf_stage_a1 = (
                                    None,
                                    None,
                                    None,
                                    None,
                                    consumer.index * self.channel_tiles + 1,
                                )
                                self._copy_sf(
                                    copy_a1, source_a1[sf_stage_a1], target_a1, stage_k
                                )
                            if cutlass.const_expr(self.gather_input):
                                # Weight scales are ready independently of gathered B.
                                # Issue their S2T copies while the gather can finish.
                                if cutlass.const_expr(self.enable_iket):
                                    _iket.range_pop()
                                    _iket.range_push("mma_gather_ready")
                                _gather_mbarrier_wait(
                                    storage.gather_barriers.data_ptr() + consumer.index,
                                    consumer.phase,
                                    self.cta_group_size,
                                )
                                if cutlass.const_expr(self.enable_iket):
                                    _iket.range_pop()
                                    _iket.range_push("mma_sfb_copy_issue")
                            self._copy_sf(
                                copy_b, source_b[sf_stage], tile_target_b, stage_k
                            )
                            if cutlass.const_expr(self.enable_iket):
                                _iket.range_pop()
                            for channel_tile in cutlass.range_constexpr(
                                self.channel_tiles
                            ):
                                if cutlass.const_expr(self.enable_iket):
                                    _iket.range_push(
                                        "mma_umma0_issue"
                                        if channel_tile == 0
                                        else "mma_umma1_issue"
                                    )
                                tile_accumulator = mma_accumulator
                                tile_sfa = tSFA
                                if cutlass.const_expr(channel_tile > 0):
                                    tile_accumulator = cute.make_tensor(
                                        mma_accumulator.iterator
                                        + cute.assume(
                                            channel_tile * self.allocated_n, divby=32
                                        ),
                                        acc_layout,
                                    )
                                    tile_sfa = tSFA1
                                _blockscaled_dynamic_mma(
                                    tile_accumulator,
                                    fragment_a[
                                        (
                                            None,
                                            None,
                                            None,
                                            consumer.index * self.channel_tiles
                                            + channel_tile,
                                        )
                                    ],
                                    fragment_b[(None, None, None, consumer.index)],
                                    tile_sfa,
                                    tile_sfb,
                                    valid,
                                    k_tile,
                                    self.instruction,
                                    self.descriptor,
                                    tiled_mma,
                                    stage_k,
                                    self.native_mma_k,
                                )
                                if cutlass.const_expr(self.enable_iket):
                                    _iket.range_pop()
                            if cutlass.const_expr(self.enable_iket):
                                _iket.range_push("mma_release_issue")
                            # Both channel tiles must finish reading the shared B/SFB
                            # stage before any operand pipeline can overwrite it.
                            operand_pipe.consumer_release(consumer)
                            if cutlass.const_expr(self.enable_iket):
                                _iket.range_pop()
                            consumer.advance()
                    acc_pipe.producer_commit(producer)
                    producer.advance()
                    tile, _, scheduled_valid, info_state = self.next_work(
                        info_pipe, info_state, sInfo, tile + grid, total_tiles
                    )
        elif warp < 8:
            if cutlass.const_expr(self.gather_input):
                cute.arch.setmaxregister_increase(248)
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
                            tma_output,
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
        else:
            if cutlass.const_expr(self.gather_input):
                cute.arch.setmaxregister_decrease(128)
                gather_warp = warp - 8
                gather_thread = tidx - 256
                producer = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_stages
                )
                info_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_info_stages
                )
                tile, _, scheduled_valid, info_state = self.next_work(
                    info_pipe, info_state, sInfo, block, total_tiles
                )
                while tile >= 0:
                    route_idx = tile // total_m_tiles
                    route_start = route_idx * self.route_tile
                    valid = (
                        scheduled_valid
                        if cutlass.const_expr(self.use_clc_scheduler)
                        else cutlass.min(
                            self.route_tile, mn_limit[route_idx] - route_start
                        )
                    )
                    # Keep all four coordinates warp-uniform, including OOB rows.
                    # Dynamic MMA N splits payload ownership between the two CTAs.
                    groups_per_warp = cute.ceil_div(self.b_copy_rows, 16)
                    groups_per_load = min(8, 1 << (groups_per_warp - 1).bit_length())
                    row_ids = cute.make_rmem_tensor(
                        (4 * groups_per_warp,), cutlass.Int32
                    )
                    for batch in cutlass.range_constexpr(
                        cute.ceil_div(groups_per_warp, groups_per_load)
                    ):
                        group = batch * groups_per_load + (
                            (gather_thread >> 2) & (groups_per_load - 1)
                        )
                        row_base = (
                            (gather_warp + group * 4) % (self.b_copy_rows // 4)
                        ) * 4
                        source_row = (
                            row_base
                            + cta_rank * (((valid + 15) & -16) // self.cta_group_size)
                            + (gather_thread & 3)
                        )
                        row_id = cutlass.Int32(tma_activations.shape[0])
                        if source_row < valid:
                            row_id = cutlass.Int32(
                                cutlass.Uint32(token_mapping[route_start + source_row])
                                // self.top_k
                            )
                        for i in cutlass.range_constexpr(
                            min(
                                groups_per_load,
                                groups_per_warp - batch * groups_per_load,
                            )
                        ):
                            for j in cutlass.range_constexpr(4):
                                row_ids[4 * (batch * groups_per_load + i) + j] = (
                                    cute.arch.shuffle_sync(row_id, 4 * i + j)
                                )
                    # Each lane owns a complete four-byte SF word in a different
                    # shared bank. SFB remains full-N in each CTA for group-2 S2T.
                    sf_row = (
                        8 * gather_warp
                        + 32 * ((gather_thread & 31) >> 3)
                        + (gather_thread & 7)
                    )
                    sf_offsets = cute.make_rmem_tensor(
                        (self.sfb_row_blocks,), cutlass.Int64
                    )
                    for row_block in cutlass.range_constexpr(self.sfb_row_blocks):
                        row = sf_row + row_block * 128
                        source_row = cutlass.Int32(0)
                        if row < valid:
                            source_row = cutlass.Int32(
                                cutlass.Uint32(token_mapping[route_start + row])
                                // self.top_k
                            )
                        sf_offsets[row_block] = cutlass.Int64(source_row) * (
                            weights.shape[1] // self.sf_vec_size
                        )
                    shared_sf_offset = (sf_row & 31) * 16 + (sf_row >> 5) * 4
                    mma_n_alignment = 8 * self.cta_group_size
                    mma_n = (valid + mma_n_alignment - 1) & -mma_n_alignment
                    for k_tile in cutlass.range(k_tiles):
                        if cutlass.const_expr(self.enable_iket):
                            _iket.range_push("producer_gather_empty")
                        operand_pipe.sync_object_empty.wait(
                            producer.index, producer.phase
                        )
                        if cutlass.const_expr(self.enable_iket):
                            _iket.range_pop()
                            _iket.range_push("producer_gather_issue")
                        # One local arrival accounts for all Gather4 bytes;
                        # the participating threads complete their SF copies here too.
                        if gather_thread == 0:
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                storage.gather_local_barriers.data_ptr()
                                + producer.index,
                                self.b_copy_rows
                                * self.tile_k
                                * self.activation_dtype.width
                                // 8,
                            )
                        for row_block in cutlass.range_constexpr(self.sfb_row_blocks):
                            row = sf_row + row_block * 128
                            for sf_block in cutlass.range_constexpr(
                                self.tile_k // self.sf_block_k
                            ):
                                channel = (
                                    k_tile * self.tile_k + sf_block * self.sf_block_k
                                )
                                source = gather_activations_sf.iterator + cute.assume(
                                    sf_offsets[row_block] + channel // self.sf_vec_size,
                                    divby=4,
                                )
                                destination = sSFB.iterator + cute.assume(
                                    producer.index * self.sfb_stage_bytes
                                    + row_block * self.sf_stage_bytes
                                    + sf_block * 512
                                    + shared_sf_offset,
                                    divby=4,
                                )
                                _copy_gather_sf(
                                    destination,
                                    source,
                                    row < mma_n,
                                    (row < valid) & (channel < weights.shape[1]),
                                )
                        if sf_row < self.route_tile:
                            cute.arch.cp_async_mbarrier_arrive_noinc(
                                storage.gather_local_barriers.data_ptr()
                                + producer.index
                            )
                        # Distribute row/K chunks together so all four warps can
                        # issue even when the route tile has fewer than four groups.
                        row_groups = self.b_copy_rows // 4
                        tasks = row_groups * (self.k_steps // 4)
                        for task_group in cutlass.range_constexpr(
                            cute.ceil_div(tasks, 4)
                        ):
                            task = gather_warp + task_group * 4
                            row_base = (task % row_groups) * 4
                            k_chunk = task // row_groups
                            if task < tasks:
                                offset = cute.assume(
                                    producer.index * self.b_rows * self.tile_k
                                    + k_chunk * self.b_rows * self.gather_k
                                    + row_base * self.gather_k,
                                    divby=4 * self.gather_k,
                                )
                                first = (task_group % cute.ceil_div(row_groups, 4)) * 4
                                with cute.arch.elect_one():
                                    _blockscaled_tma_gather4(
                                        tma_atom_b,
                                        sB.iterator + offset,
                                        storage.gather_local_barriers.data_ptr()
                                        + producer.index,
                                        k_tile * self.tile_k + k_chunk * self.gather_k,
                                        row_ids[first],
                                        row_ids[first + 1],
                                        row_ids[first + 2],
                                        row_ids[first + 3],
                                        1,
                                        self.activation_cache_policy,
                                    )
                        if cutlass.const_expr(self.enable_iket):
                            _iket.range_pop()
                        producer.advance()
                    tile, _, scheduled_valid, info_state = self.next_work(
                        info_pipe, info_state, sInfo, tile + grid, total_tiles
                    )
        if cutlass.const_expr(self.enable_pdl and self.prefetch_weights_before_wait):
            if (warp == self.mma_warp) | (warp == self.schedule_warp):
                griddepcontrol_wait()
        pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta
        ).arrive_and_wait()
        if warp == 0:
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr, num_columns=self.tmem_cols)
