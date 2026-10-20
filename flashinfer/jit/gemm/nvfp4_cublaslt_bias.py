"""JIT definition for the benchmark's bias-enabled cuBLASLt NVFP4 GEMM."""

from .. import env as jit_env
from ..core import JitSpec, current_compilation_context, gen_jit_spec
from ..cpp_ext import is_cuda_version_at_least


def gen_nvfp4_cublaslt_bias_module() -> JitSpec:
    """Build the Blackwell cuBLASLt baseline with fused BF16 bias."""
    if not is_cuda_version_at_least("12.8"):
        raise RuntimeError(
            "cuBLASLt NVFP4 GEMM requires CUDA Toolkit 12.8 or newer"
        )
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[10, 12]
    )
    return gen_jit_spec(
        "nvfp4_cublaslt_bias",
        [jit_env.FLASHINFER_CSRC_DIR / "nvfp4_cublaslt_gemm_bias.cu"],
        extra_cuda_cflags=nvcc_flags + ["-DENABLE_BF16", "-DENABLE_FP4"],
        extra_ldflags=["-lcublas", "-lcublasLt"],
    )
