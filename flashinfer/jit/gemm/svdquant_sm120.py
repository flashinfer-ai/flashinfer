"""JIT spec for the SM120 SVDQuant fused NVFP4 GEMM module.

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

import os

import jinja2

from .. import env as jit_env
from ..core import JitSpec, current_compilation_context, gen_jit_spec
from ..utils import write_if_different
from .svdquant_sm120_configs import SVDQUANT_SM120_CONFIGS


def gen_gemm_sm120_module_cutlass_nvfp4_svdquant(lora_rank: int = 32) -> JitSpec:
    """Build the SM120 fused NVFP4 SVDQuant module for one LoRA rank."""
    if lora_rank <= 0 or lora_rank % 32:
        raise ValueError(
            f"lora_rank must be a positive multiple of 32; got {lora_rank!r}"
        )
    variant = f"_rank{lora_rank}" if lora_rank != 32 else ""
    gen_directory = (
        jit_env.FLASHINFER_GEN_SRC_DIR
        / f"gen_gemm_sm120_cutlass_nvfp4_svdquant{variant}"
    )
    os.makedirs(gen_directory, exist_ok=True)
    source_paths = [
        jit_env.FLASHINFER_CSRC_DIR / "nvfp4_svdquant_gemm_cutlass_sm120.cu",
        jit_env.FLASHINFER_CSRC_DIR / "nvfp4_smooth_quantize_sm100.cu",
    ]

    with open(
        jit_env.FLASHINFER_CSRC_DIR / "nvfp4_svdquant_gemm_cutlass_sm120.jinja"
    ) as f:
        kernel_inst_templ = jinja2.Template(f.read())
        for config in SVDQUANT_SM120_CONFIGS:
            dest_path = gen_directory / f"nvfp4_svdquant_gemm_cutlass_sm120_{config}.cu"
            source_paths.append(dest_path)
            source = kernel_inst_templ.render(config=config)
            write_if_different(dest_path, source)

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[12]
    )
    rank_flags = [f"-DSVDQ_SM120_LORA_RANK={lora_rank}"] if lora_rank != 32 else []
    return gen_jit_spec(
        f"nvfp4_svdquant_gemm_cutlass_sm120{variant}",
        source_paths,
        extra_cuda_cflags=nvcc_flags
        + [
            "-DENABLE_BF16",
            "-DENABLE_FP4",
            "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
            "-DFLASHINFER_ENABLE_SVDQ_SM120_K1_K2_FUSION",
        ]
        + rank_flags,
        extra_cflags=[
            "-DFAST_BUILD",
        ],
        extra_ldflags=["-lcublasLt"],
    )
