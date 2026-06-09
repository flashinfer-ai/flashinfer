import functools

from .core import JitSpec, gen_jit_spec
from . import env as jit_env


@functools.cache
def gen_gpt_oss_reshape_cache_fp8_module() -> JitSpec:
    return gen_jit_spec(
        "gpt_oss_reshape_cache_fp8",
        [
            jit_env.FLASHINFER_CSRC_DIR / "gpt_oss_reshape_cache_fp8.cu",
        ],
    )
