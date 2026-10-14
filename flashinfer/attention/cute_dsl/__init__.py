# Copyright (c) 2026 by FlashInfer team.
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
CuTe DSL Attention Kernels (Cubin Distribution)
================================================

Pre-compiled FMHA kernels loaded via ExternalBinaryModule.
"""

from flashinfer.cute_dsl.availability import (
    is_cute_dsl_available,
    is_cute_dsl_experimental_available,
)

if is_cute_dsl_available():
    from .fmha import (
        get_cute_dsl_fmha_kernel,
        cute_dsl_fmha_ragged_prefill,
    )
    from .fmha_blockscaled import cute_dsl_fmha_blockscaled_prefill

    __all__ = [
        "is_cute_dsl_available",
        "is_cute_dsl_experimental_available",
        "get_cute_dsl_fmha_kernel",
        "cute_dsl_fmha_ragged_prefill",
        "cute_dsl_fmha_blockscaled_prefill",
    ]

    if is_cute_dsl_experimental_available():
        from .sm120_fmha import (
            sm120_fmha_fp8_paged_prefill,
            sm120_fmha_fp8_ragged_prefill,
        )

        __all__ += [
            "sm120_fmha_fp8_ragged_prefill",
            "sm120_fmha_fp8_paged_prefill",
        ]
else:
    __all__ = [
        "is_cute_dsl_available",
        "is_cute_dsl_experimental_available",
    ]
