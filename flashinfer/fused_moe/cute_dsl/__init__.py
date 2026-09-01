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
"""CuteDSL-based Fused MoE Kernels for block-scaled FP4 compute."""

from ...cute_dsl.availability import is_cute_dsl_available

# Conditionally import CuTe-DSL kernels
if is_cute_dsl_available():
    from . import fused_moe as _fused_moe
    from .fused_moe import (
        cute_dsl_fused_moe,
        CuteDslMoEWrapper,
    )
    from .fused_moe import (
        cute_dsl_fused_moe_nvfp4,
        cute_dsl_fused_moe_mxfp8_mxfp4,
        CuteDslMxfp8Mxfp4MoEWrapper,
    )
    from .b12x_moe import (
        b12x_fused_moe,
        B12xMoEWrapper,
    )


__all__ = [
    "is_cute_dsl_available",
]

if is_cute_dsl_available():
    __all__ += [
        "cute_dsl_fused_moe",
        "CuteDslMoEWrapper",
        "cute_dsl_fused_moe_nvfp4",
        "cute_dsl_fused_moe_mxfp8_mxfp4",
        "CuteDslMxfp8Mxfp4MoEWrapper",
        "b12x_fused_moe",
        "B12xMoEWrapper",
    ]
