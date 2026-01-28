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
Blackwell (SM100) CuteDSL Kernels
=================================

This module contains CuteDSL kernels optimized for NVIDIA Blackwell architecture.
These kernels are adapted from TensorRT-LLM.
"""

from .blockscaled_contiguous_grouped_gemm import (
    Sm100BlockScaledContiguousGroupedGemmKernel,
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L,
)
from .blockscaled_contiguous_grouped_gemm_swiglu_fusion import (
    Sm100BlockScaledContiguousGroupedGemmSwigluFusionKernel,
)
from .blockscaled_contiguous_grouped_gemm_finalize_fusion import (
    Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
)
from .blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion import (
    BlockScaledContiguousGatherGroupedGemmKernel,
)
from .utils import (
    TRTLLM_ENABLE_PDL,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    is_power_of_2,
)
