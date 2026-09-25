# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from . import current_compilation_context, env as jit_env
from .core import JitSpec, gen_jit_spec


def gen_minimax_m3_module() -> JitSpec:
    return gen_jit_spec(
        "minimax_m3_sparse_decode_metadata",
        [jit_env.FLASHINFER_CSRC_DIR / "minimax_m3.cu"],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(
            supported_major_versions=[10]
        ),
    )
