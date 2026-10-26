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
"""CUDA/CUB sample histogram and three-threshold publication policy."""

import functools

from . import env as jit_env
from .core import JitSpec, gen_jit_spec


def gen_selective_sample_ladder_module() -> JitSpec:
    return gen_jit_spec(
        "selective_sample_ladder",
        [jit_env.FLASHINFER_CSRC_DIR / "selective_sample_ladder.cu"],
        # Match the ladder's FP32 policy arithmetic; its one fused margin is
        # explicit in the kernel instead of contracting other expressions.
        extra_cuda_cflags=[
            "--fmad=false",
            "--ftz=false",
            "--prec-div=true",
            "--prec-sqrt=true",
        ],
    )


@functools.cache
def get_selective_sample_ladder_module():
    return gen_selective_sample_ladder_module().build_and_load()


def launch_selective_sample_ladder(module, *args):
    # Prepared phase calls append a CuTe stream argument. The FFI launch uses
    # the caller's current stream, like the other native FlashInfer operations.
    module.run(*args[:-1])
