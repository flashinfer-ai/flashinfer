from .. import env as jit_env
from ..core import (
    JitSpec,
    gen_jit_spec,
    sm90a_nvcc_flags,
)
from ..cpp_ext import is_cuda_version_at_least


def gen_fp8_blockscale_gemm_sm90_module(use_fast_build: bool = False) -> JitSpec:
    """Generate JIT spec for FP8 block scale GEMM on SM90 (Hopper)."""
    nvcc_flags = sm90a_nvcc_flags + [
        "-DCOMPILE_HOPPER_TMA_GEMMS",
        "-DENABLE_BF16",
        "-DENABLE_FP8",
        *(("-DENABLE_FP8_BLOCK_SCALE",) if is_cuda_version_at_least("12.8") else ()),
    ]

    return gen_jit_spec(
        "fp8_blockscale_gemm_90",
        [
            jit_env.FLASHINFER_CSRC_DIR
            / "nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu",
            jit_env.FLASHINFER_CSRC_DIR / "fp8_blockscale_gemm_sm90_binding.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "fused_moe/cutlass_backend/deepgemm_jit_setup.cu",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/envUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/logger.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/stringUtils.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/tllmException.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "nv_internal/cpp/common/memoryUtils.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
        extra_cflags=["-DFAST_BUILD"] if use_fast_build else [],
        extra_ldflags=["-lnvrtc", "-lcuda"],
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
        ],
    )
