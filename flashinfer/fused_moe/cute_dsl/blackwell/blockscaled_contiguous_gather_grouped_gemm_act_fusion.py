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

"""FC1 capability adapter for the shared SM100 blockscaled MoE kernel.

The logical API remains activation × weight. The shared kernel places weights
on MMA M and gathered activations on runtime MMA N for every supported dtype.
"""

import cutlass

from .blockscaled_contiguous_grouped_gemm_finalize_fusion import (
    Sm100BlockScaledGroupedGemmKernel,
)


class BlockScaledContiguousGatherGroupedGemmKernel(Sm100BlockScaledGroupedGemmKernel):
    """Validate FC1 output types using the logical activation/weight API."""

    @classmethod
    def can_implement(
        cls,
        a_dtype,
        b_dtype,
        sf_dtype,
        sf_vec_size,
        c_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        m,
        n,
        k,
        l,
        a_major,
        b_major,
        c_major,
    ):
        if a_major != "k" or b_major != "k" or c_major != "n" or l <= 0:
            return False
        if a_dtype is not b_dtype and c_dtype is not cutlass.Float8E4M3FN:
            return False
        return Sm100BlockScaledGroupedGemmKernel.can_implement(
            n,
            m,
            k,
            c_dtype,
            mma_tiler_mn,
            cluster_shape_mn,
            a_dtype,
            b_dtype,
            sf_dtype,
            sf_vec_size,
        )
