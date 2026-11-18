"""JIT spec for the SM103 GLM mHC PRE (hyper-connections) fused kernel."""

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm103a_nvcc_flags


def gen_mhc_pre_glm_module() -> JitSpec:
    return gen_jit_spec(
        "mhc_pre_glm_sm103",
        [jit_env.FLASHINFER_CSRC_DIR / "mhc_pre_glm" / "mhc_pre_glm_sm103.cu"],
        extra_cuda_cflags=sm103a_nvcc_flags + ["-lineinfo"],
    )
