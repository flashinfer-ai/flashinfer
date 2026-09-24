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

from typing import Tuple, Union

import cutlass
from cutlass import cute, utils
from cutlass._mlir.dialects import llvm, nvvm
from cutlass._mlir.dialects.nvvm import cp_async_bulk_global_shared_cta
from cutlass.cutlass_dsl import Int32, Integer, T, dsl_user_op

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


class DeviceBoundPersistentTileScheduler(utils.StaticPersistentTileScheduler):
    """Persistent scheduler with a device-provided logical N tile count.

    The launch grid and backing tensors can retain their graph-safe maximum
    extent while the persistent loop operates only on routed tiles produced by
    the preceding routing kernel.
    """

    @staticmethod
    @dsl_user_op
    def create_static_n(
        params: utils.PersistentTileSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        unused_actual_tiles_n: Int32,
        *,
        loc=None,
        ip=None,
    ):
        del unused_actual_tiles_n
        return utils.StaticPersistentTileScheduler.create(
            params, block_idx, grid_dim, loc=loc, ip=ip
        )

    @staticmethod
    @dsl_user_op
    def create_n(
        params: utils.PersistentTileSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        actual_tiles_n: Int32,
        *,
        loc=None,
        ip=None,
    ):
        sched = utils.StaticPersistentTileScheduler.create(
            params, block_idx, grid_dim, loc=loc, ip=ip
        )
        runtime_params = sched.params
        m_tiles = runtime_params.problem_shape_ntile_mnl[0]
        runtime_params.problem_shape_ntile_mnl = (m_tiles, actual_tiles_n, Int32(1))

        cluster_n = runtime_params.cluster_shape_mn[1]
        m_clusters = runtime_params.problem_layout_ncluster_mnl.shape[0]
        actual_clusters_n = (actual_tiles_n + Int32(cluster_n) - Int32(1)) // Int32(
            cluster_n
        )
        runtime_params.problem_layout_ncluster_mnl = cute.make_layout(
            (m_clusters, actual_clusters_n, Int32(1)), loc=loc, ip=ip
        )

        actual_clusters_n_fdd = cute.fast_divmod_create_divisor(
            cutlass.max(actual_clusters_n, Int32(1)), loc=loc, ip=ip
        )
        if hasattr(runtime_params, "raster_along_m"):
            if runtime_params.raster_along_m:
                runtime_params.cluster_shape_minor_fdd = actual_clusters_n_fdd
            else:
                runtime_params.cluster_shape_major_fdd = actual_clusters_n_fdd
        else:
            runtime_params.cluster_shape_n_fdd = actual_clusters_n_fdd

        return sched


def compact_sf_layout(shape, sf_vec_size: int) -> cute.Layout:
    """Scale layout: (M, padded K // sf_vec_size, L)."""
    m, k, l = shape
    scale_k = cute.ceil_div(k, sf_vec_size)
    padded_scale_k = cute.round_up(scale_k, 16)
    return cute.make_ordered_layout((m, padded_scale_k, l), order=(1, 0, 2))


@dsl_user_op
def fmax(
    a: Union[float, cutlass.Float32],
    b: Union[float, cutlass.Float32],
    *,
    nan=False,
    loc=None,
    ip=None,
) -> cutlass.Float32:
    a_val = cutlass.Float32(a).ir_value(loc=loc, ip=ip)
    b_val = cutlass.Float32(b).ir_value(loc=loc, ip=ip)
    if _nvvm_fmin_needs_res():
        result = nvvm.fmax(T.f32(), a_val, b_val, nan=nan, loc=loc, ip=ip)
    else:
        result = nvvm.fmax(a_val, b_val, nan=nan, loc=loc, ip=ip)
    return cutlass.Float32(result)


@dsl_user_op
def blk_copy(dst_gemm, src_smem, size, loc=None, ip=None):
    cp_async_bulk_global_shared_cta(
        dst_gemm.iterator.llvm_ptr,
        src_smem.iterator.llvm_ptr,
        size.ir_value(),
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
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
