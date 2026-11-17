"""JIT spec for the SM103 DeepSeek-V4.1 sparse MLA decode (bf16 Q, bf16
dense KV pools, combined 640-entry storage-offset table) fast path."""

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm103a_nvcc_flags

_SOURCES = [
    "sparse_mla_decode_dsv4_sm103.cu",
]


def gen_sparse_mla_decode_dsv4_module() -> JitSpec:
    return gen_jit_spec(
        "sparse_mla_decode_dsv4_sm103",
        [jit_env.FLASHINFER_CSRC_DIR / s for s in _SOURCES],
        extra_cuda_cflags=sm103a_nvcc_flags + ["--use_fast_math", "-lineinfo"],
    )
