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
"""Device-controlled CUDA graph repair capture support."""

import functools

from packaging.version import Version

from . import env as jit_env
from .core import JitSpec, current_compilation_context, gen_jit_spec
from .cpp_ext import get_cuda_version


def gen_selective_repair_if_module() -> JitSpec:
    if get_cuda_version() < Version("12.4"):
        raise RuntimeError(
            "selective repair requires CUDA 12.4 or newer for conditional graphs"
        )
    return gen_jit_spec(
        "selective_repair_if",
        [jit_env.FLASHINFER_CSRC_DIR / "selective_repair_if.cu"],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[10]
        ),
    )


@functools.cache
def get_selective_repair_if_module():
    return gen_selective_repair_if_module().build_and_load()
