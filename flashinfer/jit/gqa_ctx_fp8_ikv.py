"""JIT spec for the SM103 fp8 token-interleaved GQA-context fast path."""

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm103a_nvcc_flags


def gen_gqa_ctx_fp8_ikv_module() -> JitSpec:
    return gen_jit_spec(
        "gqa_ctx_fp8_ikv_sm103",
        [jit_env.FLASHINFER_CSRC_DIR / "gqa_ctx_fp8_ikv_sm103.cu"],
        extra_cuda_cflags=sm103a_nvcc_flags
        + ["-DQ2_CAUSAL_PRUNE=1", "-DTINY_KVCAP=1", "-lineinfo"],
    )
