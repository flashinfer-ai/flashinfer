from .core import JitSpec, gen_jit_spec, current_compilation_context
from . import env as jit_env


def gen_tinygemm2_module() -> JitSpec:
    """Generate JIT spec for tinygemm2 kernel (SM90+ BF16 small GEMM with bias)."""
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[9, 10, 11, 12]
    )
    return gen_jit_spec(
        "tinygemm2",
        [jit_env.FLASHINFER_CSRC_DIR / "tinygemm2.cu"],
        extra_cuda_cflags=nvcc_flags,
    )
