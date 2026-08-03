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

from . import env as jit_env
from .core import (
    JitSpec,
    current_compilation_context,
    gen_jit_spec,
)


@functools.cache
def gen_gemm_sm120_module_cute_mxfp8() -> JitSpec:
    """SM120 MXFP8 / FP8 cute groupwise GEMM module.

    Bundles the cute SM120 MXFP8 and FP8 float-scale groupwise runners with their
    public MoE entries (``moe_gemm_mxfp8_nt_groupwise`` / ``moe_gemm_fp8_nt_groupwise``)
    in a single ``.so``. All source is in-tree under ``csrc/cute_sm120_mxfp8_groupwise/
    {cute_sm120_{mxfp8,fp8}_runner.{h,cu}, cute_sm120_{mxfp8,fp8}_op{,_jit_binding}.cu,
    sm120_blockscaled/, sm120_blockscaling/, sm120_common/}``;
    the kernel uses flashinfer's own ``3rdparty/cutlass``.
    """
    csrc_dir = jit_env.FLASHINFER_CSRC_DIR / "cute_sm120_mxfp8_groupwise"
    source_paths = [
        csrc_dir / "cute_sm120_mxfp8_runner.cu",
        csrc_dir / "cute_sm120_mxfp8_op.cu",
        csrc_dir / "cute_sm120_mxfp8_op_jit_binding.cu",
        csrc_dir / "cute_sm120_fp8_runner.cu",
        csrc_dir / "cute_sm120_fp8_op.cu",
        csrc_dir / "cute_sm120_fp8_op_jit_binding.cu",
    ]

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[12],
    )

    return gen_jit_spec(
        "cute_sm120_mxfp8_groupwise",
        source_paths,
        extra_cuda_cflags=[*nvcc_flags, "-DCUTLASS_ENABLE_GDC_FOR_SM100=1"],
        extra_include_paths=[jit_env.FLASHINFER_CSRC_DIR],
    )
