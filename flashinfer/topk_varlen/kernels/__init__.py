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
"""GVR and radix Top-K CuTe DSL kernels for Blackwell (sm_100+).

The kernel modules import ``cutlass`` at module level, so their classes are
re-exported only when the CuTe DSL is importable. The pure-torch host module
``gvr2_topk_host`` (dispatch, warm-up, readiness, workspace bookkeeping) stays
importable either way, which lets the public helpers
``flashinfer.topk_varlen.warmup_prefill`` and ``prefill_ready`` answer
"unavailable" instead of failing to import on an installation without
nvidia-cutlass-dsl.
"""

from ...cute_dsl.availability import is_cute_dsl_available

if is_cute_dsl_available():
    from .config import GvrTopKConfig, GvrTopKLBConfig
    from .gvr_topk_decode import GvrTopKKernel
    from .gvr_topk_decode_lb import GvrTopKLBKernel, GvrTopKLBPrepareKernel
    from .radix_topk import SinglePassMultiCTARadixTopKKernel

    __all__ = [
        "GvrTopKConfig",
        "GvrTopKLBConfig",
        "GvrTopKKernel",
        "GvrTopKLBKernel",
        "GvrTopKLBPrepareKernel",
        "SinglePassMultiCTARadixTopKKernel",
    ]
else:
    __all__ = []
