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

import functools

from .. import env as jit_env
from ..core import (
    JitSpec,
    current_compilation_context,
    gen_jit_spec,
)


@functools.cache
def gen_grouped_mm_sm120_module_cute_mxfp8() -> JitSpec:
    """SM120 MXFP8 cute grouped GEMM module.

    Bundles 3 grouped MXFP8 entries (MoE contiguous w/wo psum_layout /
    MoE masked / MoE zero_padding) with a shared cute-DSL runner
    (`Mxfp8GemmCuteSm120Runner<e4m3_t, bf16_t, float, ue8m0_t>`). All source is
    in-tree under `include/flashinfer/grouped_mm/{mxfp8_gemm_cute_sm120.h,
    sm120_blockscaled/}` and `csrc/fused_moe/cutlass_backend/cute_sm120/`;
    the kernel uses flashinfer's own `3rdparty/cutlass`.
    """
    csrc_dir = (
        jit_env.FLASHINFER_CSRC_DIR / "fused_moe" / "cutlass_backend" / "cute_sm120"
    )
    source_paths = [
        csrc_dir / "mxfp8_gemm_cute_sm120.cu",
        csrc_dir / "group_gemm_mxfp8_cute_sm120.cu",
        csrc_dir / "group_gemm_mxfp8_cute_sm120_jit_binding.cu",
    ]

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[12],
    )

    return gen_jit_spec(
        "mxfp8_gemm_cute_sm120",
        source_paths,
        extra_cuda_cflags=nvcc_flags
        + [
            "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
        ],
    )
