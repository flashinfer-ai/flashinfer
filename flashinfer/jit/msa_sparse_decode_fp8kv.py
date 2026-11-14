"""JIT spec for the SM103 block-sparse MSA decode (bf16 Q, fp8 E4M3 KV)
fast path."""

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm103a_nvcc_flags

_SOURCES = [
    "msa_sparse_decode_fp8kv_sm103_dispatch.cu",
    "msa_sparse_decode_fp8kv_sm103_c12.cu",
    "msa_sparse_decode_fp8kv_sm103_f16core.cu",
    "msa_sparse_decode_fp8kv_sm103_dkf16.cu",
    "msa_sparse_decode_fp8kv_sm103_dkf16_short.cu",
    "msa_sparse_decode_fp8kv_sm103_dkf16_short_b8.cu",
]


def gen_msa_sparse_decode_fp8kv_module() -> JitSpec:
    return gen_jit_spec(
        "msa_sparse_decode_fp8kv_sm103",
        [jit_env.FLASHINFER_CSRC_DIR / s for s in _SOURCES],
        extra_cuda_cflags=sm103a_nvcc_flags
        + ["--use_fast_math", "-lineinfo"],
    )
