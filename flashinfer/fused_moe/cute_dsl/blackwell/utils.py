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

Re-exports shared utilities from common/kernel_utils.py and adds
Blackwell-specific functions: fmin (with explicit return type),
blk_reduce_bf16, blk_reduce_fp32, blk_reduce_fp16.
"""

import ctypes
import functools
from typing import Union

import cutlass
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T, dsl_user_op

# Re-export all shared utilities so existing imports continue to work
from ..common.kernel_utils import (  # noqa: F401
    _Pointer,
    atomic_add_func,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    is_power_of_2,
    make_ptr,
    sigmoid_f32,
    silu_f32,
    vectorized_atomic_add_bf16x8,
    vectorized_atomic_add_fp32x2,
)


# ============================================================================
# Blackwell-specific functions
# ============================================================================


@functools.lru_cache(maxsize=None)
def _nvvm_fmin_needs_res():
    import inspect

    return "res" in inspect.signature(nvvm.fmin).parameters


@dsl_user_op
def fmin(
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
        # CUDA 12: nvvm.fmin(res, a, b, ...)
        result = nvvm.fmin(T.f32(), a_val, b_val, nan=nan, loc=loc, ip=ip)
    else:
        # CUDA 13: nvvm.fmin(a, b, ...)
        result = nvvm.fmin(a_val, b_val, nan=nan, loc=loc, ip=ip)
    return cutlass.Float32(result)


def tanh_f32(
    a: Union[float, cutlass.Float32], fastmath: bool = False
) -> Union[float, cutlass.Float32]:
    """Compute tanh from the existing sigmoid primitive."""
    return cutlass.Float32(2.0) * sigmoid_f32(
        cutlass.Float32(2.0) * a, fastmath=fastmath
    ) - cutlass.Float32(1.0)


def f32_reciprocal(value: float) -> float:
    """Return ``1 / fp32(value)``, rounded to fp32.

    The reciprocal must come from the same fp32 value the kernel multiplies back
    in.  Taking it from the unrounded Python float instead can differ by 1 ulp for
    betas that are not fp32-exact.  (f64 carries at least 2p+2 bits for p=24, so
    computing the quotient in f64 and rounding once to fp32 is correctly rounded.)
    """
    beta_f32 = ctypes.c_float(float(value)).value
    return ctypes.c_float(1.0 / beta_f32).value


def situ_f32(
    a: Union[float, cutlass.Float32],
    beta: Union[float, cutlass.Float32],
    fastmath: bool = False,
) -> Union[float, cutlass.Float32]:
    """Compute SiTU: beta * tanh(x / beta) * sigmoid(x)."""
    x = cutlass.Float32(a)
    beta_f32 = cutlass.Float32(beta)
    # Multiply by the reciprocal: `x / beta` is a per-element div.rn.f32 the backend
    # cannot strength-reduce (1/25.0 is inexact) nor hoist (varying numerator).
    if isinstance(beta, (float, int)):
        inv_beta = cutlass.Float32(f32_reciprocal(beta))
    else:
        inv_beta = cutlass.Float32(1.0) / beta_f32
    return (
        beta_f32
        * tanh_f32(x * inv_beta, fastmath=fastmath)
        * sigmoid_f32(x, fastmath=fastmath)
    )


def gelu_tanh_f32(
    a: Union[float, cutlass.Float32], fastmath: bool = False
) -> Union[float, cutlass.Float32]:
    """Compute GELU using the tanh approximation."""
    x = cutlass.Float32(a)
    inner = cutlass.Float32(0.7978845608028654) * (
        x + cutlass.Float32(0.044715) * x * x * x
    )
    return (
        cutlass.Float32(0.5)
        * x
        * (cutlass.Float32(1.0) + tanh_f32(inner, fastmath=fastmath))
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
