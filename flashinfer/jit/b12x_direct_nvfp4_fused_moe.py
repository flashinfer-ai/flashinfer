"""JIT build specification for the experimental B12x Direct NVFP4 MoE."""

import functools

from . import env as jit_env
from .core import JitSpec, current_compilation_context, gen_jit_spec


@functools.cache
def gen_b12x_direct_nvfp4_fused_moe_module() -> JitSpec:
    """Build the SM120 low-token FP4-weight/FP4-activation MoE module."""
    nvcc_flags = current_compilation_context.get_nvcc_flags_list(
        supported_major_versions=[12]
    )
    return gen_jit_spec(
        "b12x_direct_nvfp4_fused_moe",
        [
            jit_env.FLASHINFER_CSRC_DIR
            / "fused_moe"
            / "b12x_direct_nvfp4_fused_moe.cu",
            jit_env.FLASHINFER_CSRC_DIR
            / "fused_moe"
            / "b12x_direct_nvfp4_fused_moe_jit_binding.cu",
        ],
        extra_cuda_cflags=nvcc_flags,
    )
