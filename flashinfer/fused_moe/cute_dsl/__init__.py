# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CuteDSL-based Fused MoE Kernels:
- NVFP4 grouped GEMM on Blackwell GPUs.
- W4A8 MXFP4 (FP8 activation x MXFP4 weight) grouped GEMM on Hopper SM90.
"""

from ...cute_dsl.utils import is_cute_dsl_available

# Conditionally import CuTe-DSL kernels
if is_cute_dsl_available():
    from .fused_moe import (
        cute_dsl_fused_moe_nvfp4,
        CuteDslMoEWrapper,
    )
    from .b12x_moe import (
        b12x_fused_moe,
        B12xMoEWrapper,
    )
    from .w4a8_mxfp4_grouped_gemm_sm90 import (
        w4a8_mxfp4_grouped_gemm,
    )
    from .w4a8_mxfp4_moe import (
        w4a8_mxfp4_moe,
        interleave_w4a8_fc1_gate_up,
    )

# moe_reduce is a Triton kernel (the W4A8 top_k>=2 MoE finalize / un-fuse path); it does
# not depend on CuTe DSL, so it is imported unconditionally (it has its own triton guard).
from .moe_reduce_triton import (
    moe_reduce,
    build_reduce_index,
)

__all__ = [
    "is_cute_dsl_available",
    "moe_reduce",
    "build_reduce_index",
]

if is_cute_dsl_available():
    __all__ += [
        "cute_dsl_fused_moe_nvfp4",
        "CuteDslMoEWrapper",
        "b12x_fused_moe",
        "B12xMoEWrapper",
        "w4a8_mxfp4_grouped_gemm",
        "w4a8_mxfp4_moe",
        "interleave_w4a8_fc1_gate_up",
    ]
