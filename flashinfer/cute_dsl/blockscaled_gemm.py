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
Backwards compatibility module.

This module has been moved to flashinfer.gemm.kernels.grouped_gemm_masked_blackwell.
Import from ``flashinfer.gemm`` for the public API.
All imports are re-exported here for backwards compatibility.

.. deprecated::
    ``flashinfer.cute_dsl.blockscaled_gemm`` is deprecated.
    Use ``flashinfer.gemm`` instead. This module will be removed in a future release.
"""

# Re-export everything from the new location
from flashinfer.gemm.kernels.grouped_gemm_masked_blackwell import (
    grouped_gemm_nt_masked,
    Sm100BlockScaledPersistentDenseGemmKernel,
    create_scale_factor_tensor,
    get_cute_dsl_compiled_masked_gemm_kernel,
    MaskedSchedulerParams,
    MaskedScheduler,
)

__all__ = [
    "grouped_gemm_nt_masked",
    "Sm100BlockScaledPersistentDenseGemmKernel",
    "create_scale_factor_tensor",
    "get_cute_dsl_compiled_masked_gemm_kernel",
    "MaskedSchedulerParams",
    "MaskedScheduler",
]
