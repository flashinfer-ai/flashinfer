"""
Copyright (c) 2025 by FlashInfer team.

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

import os

from . import env as jit_env
from .core import JitSpec, gen_jit_spec


def gen_sampling_module() -> JitSpec:
    compensated_mass = os.environ.get("FLASHINFER_SAMPLING_USE_COMPENSATED_MASS", "0")
    if compensated_mass not in ("0", "1"):
        raise ValueError("FLASHINFER_SAMPLING_USE_COMPENSATED_MASS must be 0 or 1")
    # Separate modules prevent the opt-in build from reusing the default AOT/JIT binary.
    enabled = compensated_mass == "1"
    return gen_jit_spec(
        "sampling_compensated_mass" if enabled else "sampling",
        [
            jit_env.FLASHINFER_CSRC_DIR / "sampling.cu",
            jit_env.FLASHINFER_CSRC_DIR / "renorm.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_sampling_binding.cu",
        ],
        extra_cuda_cflags=(
            ["-DFLASHINFER_SAMPLING_USE_COMPENSATED_MASS=1"] if enabled else None
        ),
    )
