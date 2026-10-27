# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This file is copied and modified from cutlass https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/core.py

"""
Blackwell (SM100) specific kernel utilities.

Re-exports common pointer and activation helpers. Blackwell additions include
PDL publication, bulk output stores and BF16 reduction, and the channel-major
blockscaled epilogue shared by the split MoE FC1 and FC2 kernels.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import dsl_user_op

from flashinfer.tllm_enums import ActivationType
from flashinfer.quantization.quantization_cute_dsl_utils import (
    _nvfp4_standard_quant_from_amax,
    float_to_ue8m0_fast,
    ue8m0_to_inv_scale_fast,
)

# Re-export all shared utilities so existing imports continue to work
from ..common.kernel_utils import (  # noqa: F401
    _Pointer,
    _nvvm_fmin_needs_res,
    atomic_add_func,
    f32_reciprocal,
    fmin,
    gelu_tanh_f32,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    is_power_of_2,
    make_ptr,
    sigmoid_f32,
    silu_f32,
    situ_f32,
    tanh_f32,
    vectorized_atomic_add_bf16x8,
    vectorized_atomic_add_fp32x2,
)


# ============================================================================
# Blackwell-specific functions
# ============================================================================


@dsl_user_op
def release_gpu(loc=None, ip=None):
    """Publish completed writes before triggering the dependent grid."""
    llvm.fence(llvm.AtomicOrdering.release, syncscope="device", loc=loc, ip=ip)


@dsl_user_op
def _bulk_stmatrix4(destination, word0, word1, word2, word3, loc=None, ip=None):
    """Transpose four 8x8 BF16/FP16 fragments into token-major shared rows."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(destination.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip),
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


@dsl_user_op
def blk_copy(dst_gemm, src_smem, size, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            src_smem.iterator.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            size.ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def blk_reduce_bf16(dst_gemm, src_smem, size, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [$0], [$1], $2;",
        "l,l,r",
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
def _blockscaled_transpose16x32(values, scratch_ptr, bottom: cutlass.Constexpr = False):
    """Transpose loaded values into lane=token, register=channel order.

    The scratch rectangle is 32 TMEM rows by 32 columns.  All source values
    must have been loaded before the caller reuses their TMEM as scratch.
    """
    perm1 = (0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15)
    perm3 = (0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15)
    work = cute.make_rmem_tensor((16,), cutlass.Float32)
    result = cute.make_rmem_tensor((16,), cutlass.Float32)
    source_ptr = scratch_ptr + ((16 << 16) if bottom else 0)
    dest_ptr = scratch_ptr + (16 if bottom else 0)
    source = _blockscaled_tmem_view(source_ptr, 16, 32)
    dest = _blockscaled_tmem_view(dest_ptr, 32, 16)
    dest_top = _blockscaled_tmem_view(dest_ptr, 16, 16)
    dest_bottom = _blockscaled_tmem_view(dest_ptr + (16 << 16), 16, 16)
    for i in cutlass.range_constexpr(16):
        result[i] = values[perm1[i]]
    cute.copy(
        cute.make_copy_atom(
            tcgen05.St16x128bOp(tcgen05.Repetition.x8), cutlass.Float32
        ),
        _blockscaled_rmem_view(result, 16),
        source,
    )
    cute.copy(
        cute.make_copy_atom(
            tcgen05.Ld16x64bOp(tcgen05.Repetition.x16), cutlass.Float32
        ),
        source,
        _blockscaled_rmem_view(work, 16),
    )
    atom_store = cute.make_copy_atom(
        tcgen05.St32x32bOp(tcgen05.Repetition.x16), cutlass.Float32
    )
    cute.copy(atom_store, _blockscaled_rmem_view(work, 16), dest)
    atom_load3 = cute.make_copy_atom(
        tcgen05.Ld16x256bOp(tcgen05.Repetition.x2), cutlass.Float32
    )
    cute.copy(atom_load3, dest_top, _blockscaled_rmem_view(work, 8))
    cute.copy(atom_load3, dest_bottom, _blockscaled_rmem_view(work, 8, 8))
    for i in cutlass.range_constexpr(16):
        result[i] = work[perm3[i]]
    cute.copy(atom_store, _blockscaled_rmem_view(result, 16), dest)
    atom_load4 = cute.make_copy_atom(
        tcgen05.Ld16x128bOp(tcgen05.Repetition.x4), cutlass.Float32
    )
    cute.copy(atom_load4, dest_top, _blockscaled_rmem_view(work, 8))
    cute.copy(atom_load4, dest_bottom, _blockscaled_rmem_view(work, 8, 8))
    for i in cutlass.range_constexpr(16):
        result[i] = work[perm1[i]]
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
    def _store(self, values, output, output_sf, row, channel, norm_const, absmax):
        converted = cute.make_rmem_tensor((16,), self.output_dtype)
        if cutlass.const_expr(self.generate_sfc):
            if cutlass.const_expr(self.is_mxfp8_output):
                # Match mxfp8_quantize's saturating round-up E8M0 code, including
                # code zero for an all-zero block. Keep raw bytes for its store.
                sf = float_to_ue8m0_fast(absmax * cutlass.Float32(1.0 / 448.0))
                scale = ue8m0_to_inv_scale_fast(sf)
                if channel % self.sf_vec_size == 0:
                    output_sf[row, channel] = sf.to(cutlass.Uint8)
            elif cutlass.const_expr(
                self.output_dtype == cutlass.Float4E2M1FN
                and self.sf_dtype == cutlass.Float8E4M3FN
                and self.sf_vec_size == 16
            ):
                # Share standalone standard fast quantization while keeping
                # the activation values in FP32 through the FP4 conversion.
                sf, scale = _nvfp4_standard_quant_from_amax(absmax, norm_const, False)
                if channel % self.sf_vec_size == 0:
                    output_sf[row, channel] = sf.bitcast(cutlass.Float8E4M3FN)
            elif cutlass.const_expr(self.output_dtype == cutlass.Float4E2M1FN):
                sf = (absmax * cutlass.Float32(1.0 / 6.0) * norm_const).to(
                    self.sf_dtype
                )
                if channel % self.sf_vec_size == 0:
                    output_sf[row, channel] = sf
                scale = fmin(
                    norm_const * cute.arch.rcp_approx(cutlass.Float32(sf)),
                    cutlass.Float32(3.40282346638528859812e38),
                    nan=True,
                )
            elif cutlass.const_expr(self.output_dtype == cutlass.Float8E4M3FN):
                sf = (absmax * cutlass.Float32(1.0 / 448.0) * norm_const).to(
                    self.sf_dtype
                )
                if channel % self.sf_vec_size == 0:
                    output_sf[row, channel] = sf
                scale = fmin(
                    norm_const * cute.arch.rcp_approx(cutlass.Float32(sf)),
                    cutlass.Float32(3.40282346638528859812e38),
                    nan=True,
                )
            for i in cutlass.range_constexpr(0, 16, 2):
                values[i], values[i + 1] = cute.arch.mul_packed_f32x2(
                    (values[i], values[i + 1]), (scale, scale)
                )
        converted.store(values.load().to(self.output_dtype))
        self._store_converted(converted, output, row, channel)

    @cute.jit
    def _store_converted(self, converted, output, row, channel):
        target = cute.local_tile(output, (1, 16), (row, channel // 16))
        if cutlass.const_expr(self.use_fused_finalize):
            for i in cutlass.range_constexpr(0, 16, 2):
                pair = cute.make_tensor(converted.iterator + i, cute.make_layout((2,)))
                target_pair = cute.make_tensor(
                    target.iterator + i, cute.make_layout((2,))
                )
                _blockscaled_atomic_add_pair(pair, target_pair)
        else:
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.output_dtype,
                    num_bits_per_copy=min(256, self.output_dtype.width * 16),
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
        words = _blockscaled_transpose16x32(packed, scratch)
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
                up = _blockscaled_transpose16x32(up, scratch)
                gate = _blockscaled_transpose16x32(gate, scratch, True)
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
                values = _blockscaled_transpose16x32(values, scratch)
            if cutlass.const_expr(release_acc):
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
                    self._store(
                        values,
                        output_tensor,
                        output_sf_tensor,
                        route_start + token_start + lane,
                        channel,
                        norm_const,
                        absmax,
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
            top = _blockscaled_transpose16x32(top, scratch)
            bottom = _blockscaled_transpose16x32(bottom, scratch, True)
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
            absmax = cutlass.Float32(0.0)
            if cutlass.const_expr(self.generate_sfc and self.sf_vec_size == 32):
                absmax = cute.arch.fmax(self._absmax(top), self._absmax(bottom))
            for half in cutlass.range_constexpr(2):
                values = top if half == 0 else bottom
                if cutlass.const_expr(self.generate_sfc and self.sf_vec_size == 16):
                    absmax = self._absmax(values)
                channel = channel_start + warp * 32 + half * 16
                if token_start + lane < valid_tokens:
                    if channel < output_tensor.shape[1]:
                        self._store(
                            values,
                            output_tensor,
                            output_sf_tensor,
                            output_row,
                            channel,
                            norm_const,
                            absmax,
                        )

    @cute.jit
    def _run_overlapping_fc1_direction(
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
        reverse: cutlass.Constexpr,
    ):
        # Treat all channel tiles as one virtual accumulator column range.
        # The actual scale footprint, rounded to a 32-column transpose slab,
        # determines the shared interval. Read it before releasing ownership.
        groups = self.tile_n // 32
        for step in cutlass.range_constexpr(channel_tiles * groups):
            slab = channel_tiles * groups - 1 - step if reverse else step
            channel_tile = slab // groups
            token_start = (slab % groups) * 32
            tile = cute.make_tensor(
                acc_tensor.iterator + channel_tile * self.tile_n, acc_tensor.layout
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
        if acc_state.phase == 0:
            self._run_overlapping_fc1_direction(
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
                channel_stride,
                channel_tiles,
                True,
            )
        else:
            self._run_overlapping_fc1_direction(
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
                channel_stride,
                channel_tiles,
                False,
            )

    @cute.jit
    def _run_bulk_direction(
        self,
        acc_tensor,
        output_tensor,
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
        reverse: cutlass.Constexpr,
    ):
        warp = (tidx // 32) % 4
        lane = tidx % 32
        groups = max(1, self.tile_n // 32)
        tokens = self.bulk_tokens
        stages = self.bulk_stages
        if cutlass.const_expr(tokens not in (32, 64)):
            raise ValueError("Unsupported FC2 bulk token slab")
        uniform_scale = alpha
        if cutlass.const_expr(
            self.final_scale_dtype in (cutlass.BFloat16, cutlass.Float16)
        ):
            uniform_scale = cutlass.Float32(self.final_scale_dtype(uniform_scale))
        output_rows = cute.make_rmem_tensor((groups,), cutlass.Int32)
        scales = cute.make_rmem_tensor((groups,), cutlass.Float32)
        for group in cutlass.range_constexpr(groups):
            output_rows[group] = 0
            scales[group] = alpha
            row = route_start + group * 32 + lane
            if group * 32 + lane < valid_tokens:
                output_rows[group] = token_mapping[row]
                route_scale = cutlass.Float32(1.0)
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
                    scale = cutlass.Float32(self.final_scale_dtype(scale))
                scales[group] = scale
        # Eight padding elements rotate row banks for the transposed matrix
        # store while keeping each 128-channel bulk-copy row 16-byte aligned.
        shared_all = cute.make_tensor(
            cute.recast_ptr(smem_scratch, dtype=self.output_dtype),
            cute.make_layout((tokens, 128, stages), stride=(136, 1, tokens * 136)),
        )
        for step in cutlass.range_constexpr(self.tile_n // tokens):
            group = self.tile_n // tokens - 1 - step if reverse else step
            if group * tokens < valid_tokens:
                if cutlass.const_expr(step >= stages):
                    cute.arch.cp_async_bulk_wait_group(stages - 1, read=True)
                    self.transpose_barrier.arrive_and_wait()
                shared = shared_all[None, None, step % stages]
                ptr = (
                    acc_tensor.iterator
                    + group * tokens
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
                        for i in cutlass.range_constexpr(tokens // 2):
                            token = (i // 4) * 8 + (lane % 4) * 2 + i % 2
                            scale = cute.arch.shuffle_sync(
                                scales[group * (tokens // 32) + i // 16], token % 32
                            )
                            values[i] = values[i] * scale
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
                for token8 in cutlass.range_constexpr(tokens // 8):
                    address = (
                        shared.iterator
                        + (token8 * 8 + lane % 8) * 136
                        + warp * 32
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
                # Distribute row transactions across all four epilogue warps.
                copy_row = warp * (tokens // 4) + lane
                row_source = output_rows[group * (tokens // 32)]
                if cutlass.const_expr(tokens == 64):
                    if warp >= 2:
                        row_source = output_rows[group * 2 + 1]
                output_row = cute.arch.shuffle_sync(row_source, copy_row % 32)
                if lane < tokens // 4:
                    if group * tokens + copy_row < valid_tokens:
                        source = cute.local_tile(shared, (1, 128), (copy_row, 0))
                        target = cute.local_tile(
                            output_tensor, (1, 128), (output_row, channel_start // 128)
                        )
                        copy_channels = cutlass.min(
                            cutlass.Int32(128), output_tensor.shape[1] - channel_start
                        )
                        blk_copy(
                            target,
                            source,
                            cutlass.Int32(copy_channels * self.output_dtype.width // 8),
                        )
                cute.arch.cp_async_bulk_commit_group()
            if cutlass.const_expr(self.overlap_acc and (step + 1) * tokens == 64):
                if group * tokens >= valid_tokens:
                    # A skipped overlap slab has no live TMEM values to read.
                    cute.arch.fence_view_async_tmem_load()
                    acc_pipeline.consumer_release(acc_state)
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        self.transpose_barrier.arrive_and_wait()

    @cute.jit
    def _run_bulk(
        self,
        acc_tensor,
        output_tensor,
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
        if cutlass.const_expr(self.overlap_acc):
            if acc_state.phase == 0:
                self._run_bulk_direction(
                    acc_tensor,
                    output_tensor,
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
                    True,
                )
            else:
                self._run_bulk_direction(
                    acc_tensor,
                    output_tensor,
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
                    False,
                )
        else:
            self._run_bulk_direction(
                acc_tensor,
                output_tensor,
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
                False,
            )

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
    ):
        lane = tidx % 32
        if cutlass.const_expr(self.bulk_tokens > 0):
            # A final two-CTA channel tile can have no live channels in its
            # second CTA. Avoid issuing a zero-byte global bulk copy there.
            if channel_start < output_tensor.shape[1]:
                self._run_bulk(
                    acc_tensor,
                    output_tensor,
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
            output_rows = cute.make_rmem_tensor((groups,), cutlass.Int32)
            output_scales = cute.make_rmem_tensor((groups,), cutlass.Float32)
            for group in cutlass.range_constexpr(groups):
                output_rows[group] = 0
                output_scales[group] = alpha
                row = route_start + group * 32 + lane
                if group * 32 + lane < valid_tokens:
                    expanded = token_mapping[row]
                    output_row = expanded
                    route_scale = cutlass.Float32(1.0)
                    if cutlass.const_expr(self.use_fused_finalize):
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
                    output_rows[group] = output_row
                    output_scales[group] = scale
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
                        output_rows[group],
                        output_scales[group],
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
