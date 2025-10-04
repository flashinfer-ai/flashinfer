"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm90a_nvcc_flags

xqa_nvcc_flags = [
    "-DNDEBUG=1",
    "-DBEAM_WIDTH=1",
    "-DCACHE_ELEM_ENUM=0",
    "-DUSE_CUSTOM_BARRIER=1",
    "-DLOW_PREC_OUTPUT=0",
    "-DSPEC_DEC=0",
]


def gen_xqa_module(
    use_fp16: bool,
    token_per_page: int,
    head_size: int,
    head_grp_size: int,
    use_sliding_window: bool,
) -> JitSpec:
    if use_fp16:
        flag_use_fp16 = ["-DINPUT_FP16=1", "-DDTYPE=__half"]
    else:
        flag_use_fp16 = ["-DINPUT_FP16=0", "-DDTYPE=__nv_bfloat16"]

    if token_per_page not in [16, 32, 64, 128]:
        raise ValueError(
            f"Invalid token_per_page: {token_per_page}, only 16, 32, 64, 128 are supported"
        )
    flag_tokens_per_page = [f"-DTOKENS_PER_PAGE={token_per_page}"]

    if head_size % 16 != 0 or head_size > 256 or head_size < 16:
        raise ValueError(
            f"Invalid head_size: {head_size}, must be divisible by 16 and in range [16, 256]"
        )
    flag_head_size = [f"-DHEAD_ELEMS={head_size}"]

    flag_head_grp_size = [f"-DHEAD_GRP_SIZE={head_grp_size}"]

    if use_sliding_window:
        flag_sliding_window = ["-DSLIDING_WINDOW=1"]
    else:
        flag_sliding_window = ["-DSLIDING_WINDOW=0"]

    return gen_jit_spec(
        f"xqa_use_fp16_{use_fp16}_token_per_page_{token_per_page}_head_size_{head_size}_head_grp_size_{head_grp_size}_use_sliding_window_{use_sliding_window}",
        [
            jit_env.FLASHINFER_CSRC_DIR / "xqa/mha.cu",
            jit_env.FLASHINFER_CSRC_DIR / "xqa/xqa_wrapper.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_xqa_binding.cu",
        ],
        extra_cuda_cflags=xqa_nvcc_flags
        + sm90a_nvcc_flags
        + flag_tokens_per_page
        + flag_head_size
        + flag_use_fp16
        + flag_head_grp_size
        + flag_sliding_window,
    )
