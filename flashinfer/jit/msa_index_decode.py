# SPDX-License-Identifier: Apache-2.0
from . import env as jit_env
from .core import gen_jit_spec, sm90a_nvcc_flags


def gen_msa_index_decode_module():
    """Build the Hopper JIT specification for direct index-prefix selection."""
    return gen_jit_spec(
        "msa_index_decode_sm90",
        [jit_env.FLASHINFER_CSRC_DIR / "msa_index_decode.cu"],
        extra_cuda_cflags=sm90a_nvcc_flags + ["--fmad=false"],
    )
