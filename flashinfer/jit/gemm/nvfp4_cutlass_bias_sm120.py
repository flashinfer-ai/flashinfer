"""JIT definition for the SM120 CUTLASS NVFP4 GEMM with fused bias."""

from .. import env as jit_env
from ..core import JitSpec, current_compilation_context, gen_jit_spec


def gen_nvfp4_cutlass_bias_sm120_module() -> JitSpec:
    """Build the benchmark's BF16-output CUTLASS baseline."""
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[12]
    )
    return gen_jit_spec(
        "nvfp4_cutlass_bias_sm120",
        [jit_env.FLASHINFER_CSRC_DIR / "nvfp4_cutlass_gemm_bias_sm120.cu"],
        extra_cuda_cflags=nvcc_flags
        + [
            "-DENABLE_BF16",
            "-DENABLE_FP4",
            "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
        ],
        extra_cflags=["-DFAST_BUILD"],
    )
