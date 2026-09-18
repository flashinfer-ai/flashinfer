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

from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec


def gen_vibecuda_ssd_combined_module() -> JitSpec:
    """Generate the JIT module for the VibeCUDA SSD combined kernels.

    No Jinja and no dtype parameterization: the hand-written kernels dispatch
    dt/seq_idx/state dtypes at runtime.  Plain CUDA plus mma.sync m16n8k16, so
    it compiles for every SM80+ target, though the SSDCombined wrapper gates
    it to Blackwell datacenter parts like the other backends.
    """
    return gen_jit_spec(
        "mamba_vibecuda_ssd_combined",
        [
            jit_env.FLASHINFER_CSRC_DIR / "vibecuda_mamba_ssd_combined.cu",
            jit_env.FLASHINFER_CSRC_DIR / "vibecuda_mamba_ssd_combined_jit_binding.cu",
        ],
    )
