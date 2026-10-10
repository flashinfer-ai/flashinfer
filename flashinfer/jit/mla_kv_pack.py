from . import env as jit_env
from .core import JitSpec, gen_jit_spec


def gen_mla_kv_pack_fp8_module() -> JitSpec:
    """JIT spec for the fused bf16 -> saturating-fp8 MLA context K/V pack."""
    return gen_jit_spec(
        "mla_kv_pack_fp8",
        [
            jit_env.FLASHINFER_CSRC_DIR / "mla_kv_pack_fp8.cu",
        ],
    )
