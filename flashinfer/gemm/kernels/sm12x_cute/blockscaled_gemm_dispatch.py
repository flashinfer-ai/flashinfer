# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""NVFP4-only instruction helpers for the SM12x ping-pong kernel."""

import cutlass
import cutlass.cute as cute

FP4_SHIFT_BITS = 2


def make_ldmatrix_atom(operand_dtype, transpose, num_matrices=4, mixed_mode=False):
    if mixed_mode or operand_dtype != cutlass.Float4E2M1FN:
        raise ValueError("The SM12x experimental kernel only accepts NVFP4 operands")
    return cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            transpose=transpose,
            num_matrices=num_matrices,
        ),
        operand_dtype,
    )


def make_sm120_blockscaled_mma_op(a_dtype, b_dtype, acc_dtype, sf_dtype, sf_vec_size):
    if (
        a_dtype != cutlass.Float4E2M1FN
        or b_dtype != cutlass.Float4E2M1FN
        or acc_dtype != cutlass.Float32
        or sf_dtype != cutlass.Float8E4M3FN
        or sf_vec_size != 16
    ):
        raise ValueError(
            "The SM12x experimental kernel requires NVFP4 and FP32 accumulation"
        )
    return cute.nvgpu.warp.MmaMXF4NVF4Op(a_dtype, acc_dtype, sf_dtype), False
