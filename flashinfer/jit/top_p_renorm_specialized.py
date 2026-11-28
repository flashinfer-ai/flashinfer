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

# One module holds every instantiation the host plan can pick (9 launch configurations x float4 /
# scalar rows): batch size and vocabulary size are runtime launch arguments, so nothing is
# compiled per shape.
TOP_P_RENORM_SPECIALIZED_MODULE_NAME = "top_p_renorm_probs_specialized_sm103"


def gen_top_p_renorm_specialized_module() -> JitSpec:
    """Cluster top-p renorm kernel for compute capability 10.3 (B300 / GB300).

    Built without ``-use_fast_math``: the renormalising division is the IEEE division, so the
    output is a numerical peer of the stock path.
    """
    return gen_jit_spec(
        TOP_P_RENORM_SPECIALIZED_MODULE_NAME,
        [jit_env.FLASHINFER_CSRC_DIR / "top_p_renorm_specialized.cu"],
        extra_cuda_cflags=sm103a_nvcc_flags,
        use_fast_math=False,
    )
