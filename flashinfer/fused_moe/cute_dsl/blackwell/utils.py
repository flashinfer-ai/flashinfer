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

# This file is copied and modified from cutlass https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/core.py

"""
Blackwell (SM100) specific kernel utilities.

Re-exports shared utilities from common/kernel_utils.py (pointer helpers, the
f32 activation functions, fmin) and adds Blackwell-specific functions:
blk_reduce_bf16, blk_reduce_fp32, blk_reduce_fp16.
"""

from dataclasses import dataclass
from typing import Union

import cutlass
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T, dsl_user_op

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
def tcgen05_fence_before_thread_sync(*, loc=None, ip=None) -> None:
    """Order completed asynchronous tensor-memory accesses before handoff."""
    nvvm.tcgen05_fence(kind=nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC, loc=loc, ip=ip)


@dsl_user_op
def tcgen05_fence_after_thread_sync(*, loc=None, ip=None) -> None:
    """Order asynchronous tensor-memory operations after a completed wait.

    An accumulator mbarrier wait needs this ordering before a tcgen05 load.
    The load-completion wait emitted by fence_view_async_tmem_load serves a
    separate purpose and does not replace this fence.
    """
    nvvm.tcgen05_fence(kind=nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC, loc=loc, ip=ip)


@dataclass(frozen=True)
class UnalignedNamedBarrier:
    """Counted CTA barrier for participating warps at different instruction sites.

    The aligned PTX form requires every thread in the CTA to execute the same
    instruction. Warp-specialized kernels need the unaligned form when only
    selected warps participate or producer and consumer warps meet at distinct
    sites. Barrier ids and participant counts retain their usual semantics.
    """

    barrier_id: int
    num_threads: int

    @dsl_user_op
    def arrive_and_wait(self, *, loc=None, ip=None) -> None:
        nvvm.barrier_cta_sync(
            barrier_id=cutlass.Int32(self.barrier_id).ir_value(loc=loc, ip=ip),
            thread_count=cutlass.Int32(self.num_threads).ir_value(loc=loc, ip=ip),
            aligned=False,
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def native_tanh_f32(a, *, loc=None, ip=None):
    """Native FP32 tanh for the SiTU path; requires SM75 or newer."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def native_situ_f32(
    a: Union[float, cutlass.Float32],
    beta: Union[float, cutlass.Float32],
    fastmath: bool = False,
) -> Union[float, cutlass.Float32]:
    """Compute SiTU with native tanh and the existing sigmoid primitive."""
    x = cutlass.Float32(a)
    beta_f32 = cutlass.Float32(beta)
    if isinstance(beta, (float, int)):
        inv_beta = cutlass.Float32(f32_reciprocal(beta))
    else:
        inv_beta = cutlass.Float32(1.0) / beta_f32
    return beta_f32 * native_tanh_f32(x * inv_beta) * sigmoid_f32(x, fastmath=fastmath)


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
def blk_copy_raw(dst_gmem_addr, src_smem_addr, size, loc=None, ip=None):
    """``cp.async.bulk`` shared::cta -> global of ``size`` bytes (multiple of 16)
    from raw addresses: ``dst_gmem_addr`` Int64 generic, ``src_smem_addr`` Int32
    shared-window. Completion is tracked by the issuing thread's bulk group."""
    llvm.inline_asm(
        None,
        [
            dst_gmem_addr.ir_value(loc=loc, ip=ip),
            src_smem_addr.ir_value(loc=loc, ip=ip),
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
def red_add_bf16x2_pair_pred(dst_gmem, lo_f32, hi_f32, pred_i32, loc=None, ip=None):
    """Predicated ``red.global.add.bf16x2`` of two F32 values rounded to BF16.

    ``lo_f32`` lands at ``dst`` and ``hi_f32`` at ``dst + 1`` (element order);
    nothing is issued when ``pred_i32`` is zero. Predication instead of a
    branch keeps unrolled per-column epilogue loops free of control flow.
    """
    llvm.inline_asm(
        None,
        [
            dst_gmem.iterator.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            lo_f32.ir_value(loc=loc, ip=ip),
            hi_f32.ir_value(loc=loc, ip=ip),
            pred_i32.ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p_;\n\t.reg .b32 pk_;\n\tsetp.ne.b32 p_, $3, 0;\n\t"
        "cvt.rn.bf16x2.f32 pk_, $2, $1;\n\t@p_ red.global.add.noftz.bf16x2 [$0], pk_;\n}",
        "l,f,f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_bf16_pred(dst_gmem, v_f32, pred_i32, loc=None, ip=None):
    """Predicated 2-byte store of an F32 value rounded to BF16."""
    llvm.inline_asm(
        None,
        [
            dst_gmem.iterator.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            v_f32.ir_value(loc=loc, ip=ip),
            pred_i32.ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p_;\n\t.reg .b16 h_;\n\tsetp.ne.b32 p_, $2, 0;\n\t"
        "cvt.rn.bf16.f32 h_, $1;\n\t@p_ st.global.b16 [$0], h_;\n}",
        "l,f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_bf16_pred_rowaddr(
    dst_gmem, row_i32, row_bytes_i32, col_bytes_i32, v_f32, pred_i32, loc=None, ip=None
):
    """Predicated 2-byte BF16 store at byte ``row * row_bytes + col_bytes`` of
    ``dst`` where that byte offset may pass 2^31.

    One ``mad.wide.s32`` forms the 64-bit offset (the column term is loop
    invariant and hoisted by ptxas), so the deferred-row epilogue pays one
    wide IMAD per store instead of a 64-bit tensor layout. Nothing is issued
    when ``pred_i32`` is zero.
    """
    llvm.inline_asm(
        None,
        [
            dst_gmem.iterator.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            row_i32.ir_value(loc=loc, ip=ip),
            row_bytes_i32.ir_value(loc=loc, ip=ip),
            col_bytes_i32.ir_value(loc=loc, ip=ip),
            v_f32.ir_value(loc=loc, ip=ip),
            pred_i32.ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p_;\n\t.reg .b16 h_;\n\t.reg .s64 o_, a_, c_;\n\t"
        "setp.ne.b32 p_, $5, 0;\n\tcvt.rn.bf16.f32 h_, $4;\n\t"
        "cvt.s64.s32 c_, $3;\n\tmad.wide.s32 o_, $1, $2, c_;\n\tadd.s64 a_, $0, o_;\n\t"
        "@p_ st.global.b16 [a_], h_;\n}",
        "l,r,r,r,f,r",
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


@dsl_user_op
def blk_reduce_fp32(dst_gemm, src_smem, size, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def blk_reduce_fp16(dst_gemm, src_smem, size, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.noftz.f16 [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mapa_shared_cluster_u32(smem_ptr, cta_rank_i32, loc=None, ip=None):
    """Shared-memory address of ``smem_ptr``'s slot in cluster CTA ``cta_rank``."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                smem_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
                cta_rank_i32.ir_value(loc=loc, ip=ip),
            ],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def st_async_f32_cluster(remote_addr_i32, v_f32, remote_mbar_i32, loc=None, ip=None):
    """``st.async`` of one F32 into a peer CTA's shared memory; the peer's
    mbarrier at ``remote_mbar`` receives ``complete_tx`` of 4 bytes."""
    llvm.inline_asm(
        None,
        [
            remote_addr_i32.ir_value(loc=loc, ip=ip),
            v_f32.ir_value(loc=loc, ip=ip),
            remote_mbar_i32.ir_value(loc=loc, ip=ip),
        ],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];",
        "r,f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbarrier_arrive_cluster(remote_mbar_i32, loc=None, ip=None):
    """Release-arrive on a peer CTA's mbarrier (cluster scope)."""
    llvm.inline_asm(
        None,
        [remote_mbar_i32.ir_value(loc=loc, ip=ip)],
        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$0];",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
