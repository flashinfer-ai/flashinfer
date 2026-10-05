# SPDX-License-Identifier: Apache-2.0
from . import env as jit_env
from .core import gen_jit_spec, sm90a_nvcc_flags


def gen_minimax_m3_index_decode_module():
    return gen_jit_spec(
        "minimax_m3_index_decode_sm90",
        [jit_env.FLASHINFER_CSRC_DIR / "minimax_m3_index_decode.cu"],
        extra_cuda_cflags=sm90a_nvcc_flags + ["--fmad=false"],
    )
