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
from .core import (
    JitSpec,
    gen_jit_spec,
    sm90a_nvcc_flags,
    sm100a_nvcc_flags,
    sm120a_nvcc_flags,
)
from ..utils import get_compute_capability
import torch

xqa_nvcc_flags = [
    "-DNDEBUG=1",
    "-DBEAM_WIDTH=1",
    "-DUSE_INPUT_KV=0",
    "-DUSE_CUSTOM_BARRIER=1",
    "-DLOW_PREC_OUTPUT=0",
    "-DSPEC_DEC=0",
]


def gen_xqa_module(
    fp16_input: bool,
    fp8_kv_cache: bool,
    token_per_page: int,
    head_size: int,
    head_grp_size: int,
    use_sliding_window: bool,
) -> JitSpec:
    if fp16_input:
        flag_data_type = ["-DINPUT_FP16=1", "-DDTYPE=__half"]
    else:
        flag_data_type = ["-DINPUT_FP16=0", "-DDTYPE=__nv_bfloat16"]

    if fp8_kv_cache:
        flag_data_type.append("-DCACHE_ELEM_ENUM=2")
    else:
        flag_data_type.append("-DCACHE_ELEM_ENUM=0")

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

    if get_compute_capability(torch.device(device="cuda"))[0] == 10:
        sm_nvcc_flags = sm100a_nvcc_flags
    elif get_compute_capability(torch.device(device="cuda"))[0] == 12:
        sm_nvcc_flags = sm120a_nvcc_flags
    else:
        sm_nvcc_flags = sm90a_nvcc_flags

    return gen_jit_spec(
        f"xqa_fp16_input_{fp16_input}_fp8_kv_cache_{fp8_kv_cache}_token_per_page_{token_per_page}_head_size_{head_size}_head_grp_size_{head_grp_size}_use_sliding_window_{use_sliding_window}_sm_{get_compute_capability(torch.device(device='cuda'))[0]}0",
        [
            jit_env.FLASHINFER_CSRC_DIR / "xqa/mha.cu",
            jit_env.FLASHINFER_CSRC_DIR / "xqa/mha_sm90.cu",
            jit_env.FLASHINFER_CSRC_DIR / "xqa/tensorMap.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "xqa/xqa_wrapper.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_xqa_binding.cu",
        ],
        extra_cuda_cflags=xqa_nvcc_flags
        + sm_nvcc_flags
        + flag_tokens_per_page
        + flag_head_size
        + flag_data_type
        + flag_head_grp_size
        + flag_sliding_window,
        extra_ldflags=["-lcuda"],  # Add CUDA Driver API library
    )
