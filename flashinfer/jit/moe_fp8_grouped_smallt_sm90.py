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
from .core import JitSpec, gen_jit_spec, sm90a_nvcc_flags

# ``gen_jit_spec`` compiles with ``-use_fast_math``, which matches the
# measured kernel's own declared build (-O3 --use_fast_math, sm_90a).


def gen_moe_fp8_grouped_smallt_sm90_module() -> JitSpec:
    return gen_jit_spec(
        "moe_fp8_grouped_smallt_sm90",
        [jit_env.FLASHINFER_CSRC_DIR / "moe_fp8_grouped_smallt_sm90.cu"],
        extra_cuda_cflags=sm90a_nvcc_flags,
    )
