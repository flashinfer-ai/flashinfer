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

import functools
from pathlib import Path
from typing import Any

from ...jit.core import JitSpec, current_compilation_context, gen_jit_spec


def gen_qsa_nvfp4_module() -> JitSpec:
    # SM80 instructions only (cp.async, mma.sync m16n8k16); one build serves SM80 through SM90.
    return gen_jit_spec(
        "qsa_nvfp4_attention",
        [Path(__file__).resolve().parent / "csrc" / "qsa_nvfp4_attention.cu"],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[8, 9]
        ),
    )


@functools.cache
def get_qsa_nvfp4_module() -> Any:
    return gen_qsa_nvfp4_module().build_and_load()
