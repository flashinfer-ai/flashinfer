"""
Copyright (c) 2024 by FlashInfer team.

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
from itertools import product

import jinja2
import torch

from ...artifacts import ArtifactPath, CheckSumHash
from .. import env as jit_env
from ..core import (
    JitSpec,
    gen_jit_spec,
    sm90a_nvcc_flags,
    sm100a_nvcc_flags,
    sm100f_nvcc_flags,
    current_compilation_context,
)
from ..cubin_loader import get_cubin, get_meta_hash
from ..utils import dtype_cutlass_map, filename_safe_dtype_map, write_if_different


def gen_gemm_module() -> JitSpec:
    return gen_jit_spec(
        "gemm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "bmm_fp8.cu",
            jit_env.FLASHINFER_CSRC_DIR / "group_gemm.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_gemm_binding.cu",
        ],
        extra_ldflags=["-lcublas", "-lcublasLt"],
    )


def gen_gemm_sm100_module_cutlass_fp4() -> JitSpec:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / "gen_gemm_sm100_cutlass_fp4"
    os.makedirs(gen_directory, exist_ok=True)
    source_paths = [
        jit_env.FLASHINFER_CSRC_DIR / "fp4_gemm_cutlass.cu",
    ]

    with open(jit_env.FLASHINFER_CSRC_DIR / "fp4_gemm_cutlass.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())
        dtype_list = ["__nv_bfloat16", "half"]
        cta_m_n_k_list = [
            (128, 64, 128),
            (128, 256, 128),
            (128, 128, 256),
            (128, 256, 256),
        ]
        for cta_m, cta_n, cta_k in cta_m_n_k_list:
            for dtype in dtype_list:
                dest_path = (
                    gen_directory
                    / f"fp4_gemm_cutlass_{dtype}_{cta_m}_{cta_n}_{cta_k}.cu"
                )
                source_paths.append(dest_path)
                source = kernel_inst_templ.render(
                    type=dtype,
                    cta_m=cta_m,
                    cta_n=cta_n,
                    cta_k=cta_k,
                )
                write_if_different(dest_path, source)

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10, 11, 12]
    )
    return gen_jit_spec(
        "fp4_gemm_cutlass",
        source_paths,
        extra_cuda_cflags=nvcc_flags
        + [
            "-DENABLE_BF16",
            "-DENABLE_FP4",
        ],
        extra_cflags=[
            "-DFAST_BUILD",
        ],
    )


def gen_gemm_sm120_module_cutlass_fp4() -> JitSpec:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / "gen_gemm_sm120_cutlass_fp4"
    os.makedirs(gen_directory, exist_ok=True)
    source_paths = [
        jit_env.FLASHINFER_CSRC_DIR / "fp4_gemm_cutlass_sm120.cu",
    ]

    with open(jit_env.FLASHINFER_CSRC_DIR / "fp4_gemm_cutlass_sm120.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())
        dtype_list = ["__nv_bfloat16", "half"]
        # SM120/121 uses only 128x128x128 tile configuration with implied 1x1x1 cluster shape
        cta_m_n_k_list = [
            (128, 128, 128),
        ]
        for cta_m, cta_n, cta_k in cta_m_n_k_list:
            for dtype in dtype_list:
                dest_path = (
                    gen_directory
                    / f"fp4_gemm_cutlass_{dtype}_{cta_m}_{cta_n}_{cta_k}.cu"
                )
                source_paths.append(dest_path)
                source = kernel_inst_templ.render(
                    type=dtype,
                    cta_m=cta_m,
                    cta_n=cta_n,
                    cta_k=cta_k,
                )
                write_if_different(dest_path, source)

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[12]
    )
    return gen_jit_spec(
        "fp4_gemm_cutlass_sm120",
        source_paths,
        extra_cuda_cflags=nvcc_flags
        + [
            "-DENABLE_BF16",
            "-DENABLE_FP4",
        ],
        extra_cflags=[
            "-DFAST_BUILD",
        ],
    )


def gen_gemm_sm100_module_cutlass_fp8() -> JitSpec:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / "gen_gemm_sm100_cutlass_fp8"
    os.makedirs(gen_directory, exist_ok=True)
    source_paths = [
        jit_env.FLASHINFER_CSRC_DIR / "fp8_gemm_cutlass.cu",
    ]

    with open(jit_env.FLASHINFER_CSRC_DIR / "fp8_gemm_cutlass.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())
        dtype_list = ["__nv_bfloat16", "half"]
        cta_m_n_k_list = [
            (64, 64, 128),
            (64, 128, 128),
            (64, 256, 128),
            (128, 64, 128),
            (128, 128, 128),
            (128, 256, 128),
        ]
        for cta_m, cta_n, cta_k in cta_m_n_k_list:
            for dtype in dtype_list:
                dest_path = (
                    gen_directory
                    / f"fp8_gemm_cutlass_{dtype}_{cta_m}_{cta_n}_{cta_k}.cu"
                )
                source_paths.append(dest_path)
                source = kernel_inst_templ.render(
                    type=dtype,
                    cta_m=cta_m,
                    cta_n=cta_n,
                    cta_k=cta_k,
                )
                write_if_different(dest_path, source)

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10, 11, 12]
    )

    return gen_jit_spec(
        "fp8_gemm_cutlass",
        source_paths,
        extra_cuda_cflags=nvcc_flags
        + [
            "-DENABLE_BF16",
        ],
        extra_cflags=[
            "-DFAST_BUILD",
        ],
    )


def gen_gemm_sm100_module() -> JitSpec:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / "gen_gemm_sm100"
    os.makedirs(gen_directory, exist_ok=True)
    source_paths = []
    for prefix in ["gemm_groupwise", "group_gemm_fp8_groupwise"]:
        with open(
            jit_env.FLASHINFER_CSRC_DIR / f"{prefix}_sm100_kernel_inst.jinja"
        ) as f:
            kernel_inst_templ = jinja2.Template(f.read())
        dtype_in_list = [torch.float8_e4m3fn, torch.float8_e5m2]
        dtype_out_list = [torch.float16, torch.bfloat16]
        scale_major_k_list = ["true", "false"]
        mma_sm_list = [1, 2]
        for dtype_in, dtype_out, scale_major_k, mma_sm in product(
            dtype_in_list, dtype_out_list, scale_major_k_list, mma_sm_list
        ):
            name_dtype_in = filename_safe_dtype_map[dtype_in]
            name_dtype_out = filename_safe_dtype_map[dtype_out]
            dest_path = (
                gen_directory
                / f"{prefix}_{name_dtype_in}_{name_dtype_out}_major{scale_major_k}_mma{mma_sm}_sm100.cu"
            )
            source_paths.append(dest_path)
            source = kernel_inst_templ.render(
                dtype_in=dtype_cutlass_map[dtype_in],
                dtype_out=dtype_cutlass_map[dtype_out],
                scale_major_k=scale_major_k,
                mma_sm=mma_sm,
            )
            write_if_different(dest_path, source)
    prefix = "group_gemm_mxfp4_groupwise"
    with open(jit_env.FLASHINFER_CSRC_DIR / f"{prefix}_sm100_kernel_inst.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())
    dtype_a_list = [torch.float8_e4m3fn, torch.float8_e5m2]
    dtype_d_list = [torch.float16, torch.bfloat16]
    mma_sm_list = [1, 2]
    swap_ab_list = ["true", "false"]
    for dtype_a, dtype_d, mma_sm, swap_ab in product(
        dtype_a_list, dtype_d_list, mma_sm_list, swap_ab_list
    ):
        name_dtype_a = filename_safe_dtype_map[dtype_a]
        name_dtype_d = filename_safe_dtype_map[dtype_d]
        dest_path = (
            gen_directory
            / f"{prefix}_{name_dtype_a}_{name_dtype_d}_mma{mma_sm}_swap{swap_ab}_sm100.cu"
        )
        source_paths.append(dest_path)
        source = kernel_inst_templ.render(
            dtype_a=dtype_cutlass_map[dtype_a],
            dtype_b="cutlass::float_e2m1_t",
            dtype_d=dtype_cutlass_map[dtype_d],
            mma_sm=mma_sm,
            swap_ab=swap_ab,
        )
        write_if_different(dest_path, source)
    for filename in [
        "gemm_groupwise_sm100.cu",
        "group_gemm_fp8_groupwise_sm100.cu",
        "group_gemm_mxfp4_groupwise_sm100.cu",
        "gemm_sm100_binding.cu",
        "group_gemm_sm100_binding.cu",
    ]:
        src_path = jit_env.FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10, 11, 12]
    )
    return gen_jit_spec(
        "gemm_sm100",
        source_paths,
        extra_cuda_cflags=nvcc_flags,
    )


def gen_gemm_sm120_module() -> JitSpec:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / "gen_gemm_sm120"
    gen_directory.mkdir(parents=True, exist_ok=True)
    source_paths = []

    # Generate kernel instantiations following SM100's approach
    prefix = "gemm_groupwise"
    dtype_in_list = [torch.float8_e4m3fn, torch.float8_e5m2]
    dtype_out_list = [torch.float16, torch.bfloat16]
    scale_major_k_list = ["true", "false"]
    # SM120 uses fixed 128x128x128 tiles with Cooperative schedule

    with open(jit_env.FLASHINFER_CSRC_DIR / f"{prefix}_sm120_kernel_inst.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())

    for dtype_in, dtype_out, scale_major_k in product(
        dtype_in_list,
        dtype_out_list,
        scale_major_k_list,
    ):
        name_dtype_in = filename_safe_dtype_map[dtype_in]
        name_dtype_out = filename_safe_dtype_map[dtype_out]
        dest_path = (
            gen_directory
            / f"{prefix}_{name_dtype_in}_{name_dtype_out}_major{scale_major_k}_sm120.cu"
        )
        source_paths.append(dest_path)
        source = kernel_inst_templ.render(
            dtype_in=dtype_cutlass_map[dtype_in],
            dtype_out=dtype_cutlass_map[dtype_out],
            scale_major_k=scale_major_k,
        )
        write_if_different(dest_path, source)

    # Generate group gemm kernel instantiations
    prefix = "group_gemm_fp8_groupwise"
    with open(jit_env.FLASHINFER_CSRC_DIR / f"{prefix}_sm120_kernel_inst.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())

    for dtype_in, dtype_out, scale_major_k in product(
        dtype_in_list,
        dtype_out_list,
        scale_major_k_list,
    ):
        name_dtype_in = filename_safe_dtype_map[dtype_in]
        name_dtype_out = filename_safe_dtype_map[dtype_out]
        dest_path = (
            gen_directory
            / f"{prefix}_{name_dtype_in}_{name_dtype_out}_major{scale_major_k}_sm120.cu"
        )
        source_paths.append(dest_path)
        source = kernel_inst_templ.render(
            dtype_in=dtype_cutlass_map[dtype_in],
            dtype_out=dtype_cutlass_map[dtype_out],
            scale_major_k=scale_major_k,
        )
        write_if_different(dest_path, source)

    # Copy source files
    for filename in [
        "gemm_groupwise_sm120.cu",
        "group_gemm_fp8_groupwise_sm120.cu",
        "gemm_sm120_binding.cu",
        "group_gemm_sm120_binding.cu",
    ]:
        src_path = jit_env.FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)

    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[
            12,
        ]
    )

    return gen_jit_spec(
        "gemm_sm120",
        source_paths,
        extra_cuda_cflags=nvcc_flags,
    )


def gen_trtllm_gen_gemm_module() -> JitSpec:
    # Fetch "flashinferMetaInfo.h" from the online kernel cache. This file
    # contains the `tllmGenGemmList` as the list of available kernels online.
    # It is included when compiling `trtllm_gemm_runner.cu`.
    include_path = f"{ArtifactPath.TRTLLM_GEN_GEMM}/include"
    header_name = "flashinferMetaInfo"

    # Check if checksums.txt exists in the cubin directory
    checksum_path = f"{ArtifactPath.TRTLLM_GEN_GEMM}/checksums.txt"
    checksum = get_cubin(checksum_path, CheckSumHash.TRTLLM_GEN_GEMM)
    assert checksum, f"Failed to get checksums.txt from {checksum_path}"
    meta_hash = get_meta_hash(checksum)

    # use `get_cubin` to get "flashinferMetaInfo.h"
    metainfo = get_cubin(
        f"{include_path}/{header_name}.h",
        meta_hash,
    )
    # make sure "flashinferMetaInfo.h" is downloaded or cached
    assert metainfo, f"{header_name}.h not found"
    return gen_jit_spec(
        "trtllm_gemm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_gemm_runner.cu",
        ],
        extra_cuda_cflags=[
            "-DTLLM_GEN_EXPORT_INTERFACE",
            "-DTLLM_GEN_EXPORT_FLASHINFER",
            "-DTLLM_ENABLE_CUDA",
            f'-DTLLM_GEN_GEMM_CUBIN_PATH=\\"{ArtifactPath.TRTLLM_GEN_GEMM}\\"',
        ]
        + sm100a_nvcc_flags,
        # link "include" sub-directory in cache
        extra_include_paths=[jit_env.FLASHINFER_CUBIN_DIR / include_path],
    )


def gen_tgv_gemm_sm10x_module(
    dtype: torch.dtype = torch.bfloat16, use_sm_100f: bool = False
) -> JitSpec:
    """
    Generate TGV GEMM module for SM100 architecture.

    Args:
        dtype: Data type for the GEMM operation (torch.bfloat16 or torch.float16)
        use_sm_100f: Whether to compile with SM100f flags (default: False), which makes the compiled kernel
            compatible with both B200 and B300 GPUs. However, it's only available with CUDA 12.9+.

    Returns:
        JitSpec for the TGV GEMM module
    """
    if dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError(
            f"Unsupported dtype {dtype}. Only bfloat16 and float16 are supported."
        )

    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    module_name = f"tgv_gemm_{dtype_str}"

    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / f"gen_tgv_gemm_{dtype_str}"
    os.makedirs(gen_directory, exist_ok=True)
    source_paths = [
        jit_env.FLASHINFER_CSRC_DIR / "tgv_gemm.cu",
    ]

    # Read the Jinja template
    with open(jit_env.FLASHINFER_CSRC_DIR / "tgv_gemm.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())

    # Define tile size configurations (cta_m, cta_n, dma_stages)
    cta_m_n_dma_list = [
        (64, 8, 6),
        (64, 8, 8),
        (64, 8, 10),
        (64, 8, 12),
        (64, 16, 6),
        (64, 16, 8),
        (64, 16, 10),
        (64, 32, 6),
        (64, 32, 8),
        (64, 64, 6),
        (128, 16, 6),
    ]

    # Generate instances for the specified dtype
    for cta_m, cta_n, dma_stage in cta_m_n_dma_list:
        dest_path = (
            gen_directory / f"tgv_gemm_{dtype_str}_{cta_m}x{cta_n}_{dma_stage}.cu"
        )
        source_paths.append(dest_path)
        source = kernel_inst_templ.render(
            cta_m=cta_m, cta_n=cta_n, dma_stage=dma_stage, dtype=dtype_str
        )
        write_if_different(dest_path, source)

    return gen_jit_spec(
        module_name,
        source_paths,
        extra_cuda_cflags=[
            "--expt-relaxed-constexpr",
            "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
        ]
        + sm100f_nvcc_flags
        if use_sm_100f
        else sm100a_nvcc_flags,
        extra_include_paths=[
            jit_env.FLASHINFER_INCLUDE_DIR,
            jit_env.FLASHINFER_CSRC_DIR,
        ],
    )


def gen_gemm_sm90_module() -> JitSpec:
    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / "gen_gemm_sm90"
    os.makedirs(gen_directory, exist_ok=True)
    source_paths = []
    with open(jit_env.FLASHINFER_CSRC_DIR / "group_gemm_sm90_kernel_inst.jinja") as f:
        kernel_inst_templ = jinja2.Template(f.read())
    for dtype_in, dtype_out in [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float8_e4m3fn, torch.float16),
        (torch.float8_e5m2, torch.float16),
        (torch.float8_e4m3fn, torch.bfloat16),
        (torch.float8_e5m2, torch.bfloat16),
    ]:
        name_dtype_in = filename_safe_dtype_map[dtype_in]
        name_dtype_out = filename_safe_dtype_map[dtype_out]
        dest_path = (
            gen_directory / f"group_gemm_{name_dtype_in}_{name_dtype_out}_sm90.cu"
        )
        source_paths.append(dest_path)
        source = kernel_inst_templ.render(
            dtype_in=dtype_cutlass_map[dtype_in],
            dtype_out=dtype_cutlass_map[dtype_out],
        )
        write_if_different(dest_path, source)
    for filename in [
        "group_gemm_sm90.cu",
        "flashinfer_gemm_sm90_binding.cu",
    ]:
        src_path = jit_env.FLASHINFER_CSRC_DIR / filename
        dest_path = gen_directory / filename
        source_paths.append(dest_path)
        with open(src_path, "r") as f:
            source = f.read()
        write_if_different(dest_path, source)
    return gen_jit_spec(
        "gemm_sm90",
        source_paths,
        extra_cuda_cflags=sm90a_nvcc_flags,
    )


def gen_trtllm_low_latency_gemm_module() -> JitSpec:
    include_path = f"{ArtifactPath.TRTLLM_GEN_GEMM}/include"
    header_name = "flashinferMetaInfo"

    # Check if checksums.txt exists in the cubin directory
    checksum_path = f"{ArtifactPath.TRTLLM_GEN_GEMM}/checksums.txt"
    checksum = get_cubin(checksum_path, CheckSumHash.TRTLLM_GEN_GEMM)
    assert checksum, f"Failed to get checksums.txt from {checksum_path}"
    meta_hash = get_meta_hash(checksum)

    # use `get_cubin` to get "flashinferMetaInfo.h"
    metainfo = get_cubin(
        f"{include_path}/{header_name}.h",
        meta_hash,
    )
    # make sure "flashinferMetaInfo.h" is downloaded or cached
    assert metainfo, f"{header_name}.h not found"
    return gen_jit_spec(
        "trtllm_low_latency_gemm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_low_latency_gemm_runner.cu",
        ],
        extra_cuda_cflags=[
            "-DTLLM_GEN_EXPORT_INTERFACE",
            "-DTLLM_GEN_EXPORT_FLASHINFER",
            "-DTLLM_ENABLE_CUDA",
            f'-DTLLM_GEN_GEMM_CUBIN_PATH=\\"{ArtifactPath.TRTLLM_GEN_GEMM}\\"',
        ]
        + sm100a_nvcc_flags,
        # link "include" sub-directory in cache
        extra_include_paths=[jit_env.FLASHINFER_CUBIN_DIR / include_path],
        extra_ldflags=["-lcuda"],
    )
