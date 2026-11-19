"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm103a_nvcc_flags

# One module holds the single kernel instantiation: the token count is a runtime launch argument,
# so nothing is compiled per shape. A separate module name (not ``trtllm_comm``) so that a
# prebuilt ``trtllm_comm`` is never confused with it.
ALLREDUCE_FUSION_WS2_SPECIALIZED_MODULE_NAME = "allreduce_fusion_ar_rmsnorm_ws2_sm103"


def gen_allreduce_fusion_ws2_specialized_module() -> JitSpec:
    """Two-rank all-reduce + residual add + RMSNorm kernel (hidden size 6144, bf16) for compute
    capability 10.3 (B300 / GB300). Built with the default flags of the TRT-LLM comm kernels
    (``-use_fast_math``)."""
    return gen_jit_spec(
        ALLREDUCE_FUSION_WS2_SPECIALIZED_MODULE_NAME,
        [jit_env.FLASHINFER_CSRC_DIR / "allreduce_fusion_ws2_specialized.cu"],
        extra_cuda_cflags=sm103a_nvcc_flags,
    )
