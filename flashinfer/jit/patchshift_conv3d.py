"""
Copyright (c) 2026 by the PatchShift Conv3d contributors.

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
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags


def gen_patchshift_conv3d_module() -> JitSpec:
    source_dir = jit_env.FLASHINFER_CSRC_DIR / "patchshift_conv3d"
    return gen_jit_spec(
        "patchshift_conv3d_sm100a",
        [
            source_dir / "launcher.cu",
            source_dir / "pack_weights.cu",
            source_dir / "binding.cu",
        ],
        extra_cuda_cflags=[*sm100a_nvcc_flags, "-O3"],
        extra_include_paths=[source_dir],
    )
