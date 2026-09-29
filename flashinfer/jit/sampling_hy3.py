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
from .core import JitSpec, gen_jit_spec


def gen_fused_sampling_hy3_module() -> JitSpec:
    """Build the HY3 fused sampler optimized for SM100/B200.

    ``gen_jit_spec`` adds ``-use_fast_math`` globally. Appending the explicit
    precision flags restores the source build's division/FTZ behavior, while
    the CUDA header calls libdevice directly for precise ``expf``. A separate
    JIT module keeps the existing FlashInfer sampling implementation untouched.
    """
    return gen_jit_spec(
        "fused_sampling_hy3",
        [
            jit_env.FLASHINFER_CSRC_DIR / "sampling_hy3.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_sampling_hy3_binding.cu",
        ],
        extra_cuda_cflags=[
            "--ftz=false",
            "--prec-div=true",
            "--prec-sqrt=true",
        ],
    )
