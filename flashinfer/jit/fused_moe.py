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

from typing import List

from . import env as jit_env
from ..artifacts import ArtifactPath, CheckSumHash
from .core import (
    JitSpec,
    gen_jit_spec,
    current_compilation_context,
    sm90a_nvcc_flags,
    sm89_nvcc_flags,
)
from .cpp_ext import is_cuda_version_at_least
from .cubin_loader import get_cubin, get_meta_hash, get_file
from .utils import write_if_different
from .gemm.cutlass.generate_kernels import generate_gemm_operations


def gen_cutlass_fused_moe_sm120_module(use_fast_build: bool = False) -> JitSpec:
    nvcc_flags = [
        "-DCOMPILE_BLACKWELL_TMA_GEMMS",
        "-DCOMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP4",
        "-DUSING_OSS_CUTLASS_MOE_GEMM",
    ]

    nvcc_flags += current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[12]
    )

    return gen_cutlass_fused_moe_module(nvcc_flags, "120", use_fast_build)


def gen_cutlass_fused_moe_sm103_module(use_fast_build: bool = False) -> JitSpec:
    nvcc_flags = [
        "-DCOMPILE_BLACKWELL_TMA_GEMMS",
        "-DCOMPILE_BLACKWELL_TMA_GROUPED_GEMMS",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP4",
        "-DUSING_OSS_CUTLASS_MOE_GEMM",
        "-DCOMPILE_BLACKWELL_SM103_TMA_GROUPED_GEMMS",
    ]

    nvcc_flags += current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10]
    )

    return gen_cutlass_fused_moe_module(nvcc_flags, "103", use_fast_build)


def gen_cutlass_fused_moe_sm100_module(use_fast_build: bool = False) -> JitSpec:
    nvcc_flags = [
        "-DCOMPILE_BLACKWELL_TMA_GEMMS",
        "-DCOMPILE_BLACKWELL_TMA_GROUPED_GEMMS",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP4",
        "-DUSING_OSS_CUTLASS_MOE_GEMM",
    ]

    nvcc_flags += current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10, 11]
    )

    return gen_cutlass_fused_moe_module(nvcc_flags, "100", use_fast_build)


def gen_cutlass_fused_moe_sm90_module(use_fast_build: bool = False) -> JitSpec:
    nvcc_flags = sm90a_nvcc_flags + [
        "-DCOMPILE_HOPPER_TMA_GEMMS",
        "-DCOMPILE_HOPPER_TMA_GROUPED_GEMMS",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP8_BLOCK_SCALE" if is_cuda_version_at_least("12.8") else "",
        "-DENABLE_FP4" if is_cuda_version_at_least("12.8") else "",
        "-DUSING_OSS_CUTLASS_MOE_GEMM",
    ]
    return gen_cutlass_fused_moe_module(nvcc_flags, "90", use_fast_build)


def gen_cutlass_fused_moe_sm89_module(use_fast_build: bool = False) -> JitSpec:
    nvcc_flags = sm89_nvcc_flags + [
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        "-DENABLE_FP8_BLOCK_SCALE" if is_cuda_version_at_least("12.8") else "",
        "-DUSING_OSS_CUTLASS_MOE_GEMM",
    ]
    return gen_cutlass_fused_moe_module(nvcc_flags, "89", use_fast_build)


def gen_cutlass_fused_moe_module(
    nvcc_flags: List[str], device_arch: str, use_fast_build: bool = False
) -> JitSpec:
    """
    Generate a JitSpec for the cutlass fused moe module.
    """
    # Use FLASHINFER_GEN_SRC_DIR (user's writable cache) instead of FLASHINFER_CSRC_DIR
    # (package directory which may be read-only after installation)
    output_dir = (
        jit_env.FLASHINFER_GEN_SRC_DIR / f"cutlass_instantiations/{device_arch}"
    )

    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        generate_gemm_operations(
            output_dir,
            f"{device_arch};{device_arch}-real",
        )

    except Exception as e:
        raise RuntimeError(f"Failed to generate Cutlass kernels: {e}") from e

    return gen_jit_spec(
        f"fused_moe_{device_arch}",
        [
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_tma_warp_specialized_input.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp8_uint4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp8_fp8.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp8_fp4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp4_fp4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp32_fp32.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_uint8.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_uint4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_fp16.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_uint8.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_uint4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_fp8.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_bf16.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_fp4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_fp4.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "fused_moe/cutlass_backend/deepgemm_jit_setup.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "fused_moe/cutlass_backend/cutlass_fused_moe_instantiation.cu",
            # Add all generated kernels
            *(output_dir / kernel for kernel in output_dir.rglob("*.generated.cu")),
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/envUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/logger.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/stringUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/tllmException.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/memoryUtils.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/preQuantScaleKernel.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/lora/lora.cpp",
        ],
        extra_cuda_cflags=nvcc_flags,
        extra_cflags=["-DFAST_BUILD"] if use_fast_build else [],
        extra_ldflags=["-lnvrtc"],
        extra_include_paths=[
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal" / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "cutlass_extensions"
            / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "kernels"
            / "cutlass_kernels"
            / "include",
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal"
            / "tensorrt_llm"
            / "kernels"
            / "cutlass_kernels",
            # Include the generated output directory for header files
            output_dir,
        ],
    )


def gen_trtllm_gen_fused_moe_sm100_module() -> JitSpec:
    # Fetch "flashinferMetaInfo.h" from the online kernel cache. This file
    # contains the `tllmGenBatchedGemmList` as the list of available kernels
    # online. It is included when compiling `trtllm_fused_moe_runner.cu`, etc.
    include_path = f"{ArtifactPath.TRTLLM_GEN_BMM}/include"
    header_name = "flashinferMetaInfo"

    # Check if checksums.txt exists in the cubin directory
    checksum_path = f"{ArtifactPath.TRTLLM_GEN_BMM}/checksums.txt"
    checksum = get_cubin(checksum_path, CheckSumHash.TRTLLM_GEN_BMM)
    assert checksum, f"Failed to get checksums.txt from {checksum_path}"
    meta_hash = get_meta_hash(checksum)

    # use `get_cubin` to get "flashinferMetaInfo.h"
    metainfo = get_cubin(
        f"{include_path}/{header_name}.h",
        meta_hash,
    )
    # make sure "flashinferMetaInfo.h" is downloaded or cached
    assert metainfo, f"{header_name}.h not found"

    header_files = [
        "BatchedGemmEnums.h",
        "BatchedGemmInterface.h",
        "BatchedGemmOptions.h",
        "Enums.h",
        "GemmGatedActOptions.h",
        "GemmOptions.h",
        "KernelParams.h",
        "KernelParamsDecl.h",
        "KernelTraits.h",
        "TmaDescriptor.h",
        "trtllm/gen/CommonUtils.h",
        "trtllm/gen/CudaArchDecl.h",
        "trtllm/gen/CudaKernelLauncher.h",
        "trtllm/gen/DtypeDecl.h",
        "trtllm/gen/MmaDecl.h",
        "trtllm/gen/SfLayoutDecl.h",
        "trtllm/gen/SparsityDecl.h",
    ]

    header_path = f"{include_path}/trtllmGen_bmm_export"
    header_dest_dir = (
        jit_env.FLASHINFER_CUBIN_DIR
        / "flashinfer"
        / "trtllm"
        / "batched_gemm"
        / "trtllmGen_bmm_export"
    )
    artifact_hash_path = header_dest_dir / ".artifact_hash"

    # Check if cached headers are from a different artifact version (e.g. after git checkout)
    if artifact_hash_path.exists():
        cached_hash = artifact_hash_path.read_text().strip()
        if cached_hash != ArtifactPath.TRTLLM_GEN_BMM:
            raise RuntimeError(
                f"Detected inconsistent cached artifacts. "
                f"(Cached trtllm bmm headers were downloaded for artifact "
                f"'{cached_hash}', but current code expects "
                f"'{ArtifactPath.TRTLLM_GEN_BMM}'. "
                f"Please clear the cache to confirm and allow the new headers to be downloaded: "
                f"rm -rf {header_dest_dir}."
            )

    for file in header_files:
        uri_path = f"{header_path}/{file}"
        file_hash = get_meta_hash(checksum, file)
        file_path = str(header_dest_dir / file)
        result = get_file(uri_path, file_hash, file_path)
        assert result, f"{file} not found"

    # Record which artifact version these headers belong to
    write_if_different(artifact_hash_path, ArtifactPath.TRTLLM_GEN_BMM)

    # currently only support Blackwell
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10]
    )

    return gen_jit_spec(
        "fused_moe_trtllm_sm100",
        [
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/envUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/logger.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/stringUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/tllmException.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/memoryUtils.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_kernel_launcher.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_runner.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_routing_deepseek.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_routing_llama4.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_routing_renormalize.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_fused_moe_dev_kernel.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_batched_gemm_runner.cu",
        ],
        extra_cuda_cflags=[
            "-DTLLM_GEN_EXPORT_INTERFACE",
            "-DTLLM_GEN_EXPORT_FLASHINFER",
            "-DTLLM_ENABLE_CUDA",
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-DENABLE_FP4",
            f'-DTLLM_GEN_GEMM_CUBIN_PATH=\\"{ArtifactPath.TRTLLM_GEN_BMM}\\"',
        ]
        + nvcc_flags,
        extra_include_paths=[
            jit_env.FLASHINFER_CUBIN_DIR,
            jit_env.FLASHINFER_CUBIN_DIR / include_path,
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/include",
        ],
    )
