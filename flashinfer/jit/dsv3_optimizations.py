from .core import JitSpec, gen_jit_spec
from . import env as jit_env


def gen_dsv3_router_gemm_module() -> JitSpec:
    return gen_jit_spec(
        "dsv3_router_gemm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "dsv3_router_gemm.cu",
        ],
    )
