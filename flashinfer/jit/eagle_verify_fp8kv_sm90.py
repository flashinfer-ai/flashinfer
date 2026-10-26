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

# ``gen_jit_spec`` compiles every module with ``-use_fast_math``. This kernel
# is a numerical drop-in for the FA2 custom-mask path and was validated with
# IEEE division / no denormal flushing, so the three precision knobs that
# ``-use_fast_math`` implies are explicitly restored (nvcc honours the last
# occurrence); ``exp2f`` lowers to ``ex2.approx.f32`` either way. The PTX of
# this flag set is identical to a plain ``-O3`` build (checked with nvcc 13.0
# and 13.2 for sm_90a); without the three overrides it is not.
_PRECISE_MATH_FLAGS = ["--prec-div=true", "--prec-sqrt=true", "--ftz=false"]


def gen_eagle_verify_fp8kv_sm90_module() -> JitSpec:
    return gen_jit_spec(
        "eagle_verify_fp8kv_sm90",
        [jit_env.FLASHINFER_CSRC_DIR / "eagle_verify_fp8kv_sm90.cu"],
        extra_cuda_cflags=sm90a_nvcc_flags
        + _PRECISE_MATH_FLAGS
        + ["--expt-relaxed-constexpr"],
    )
