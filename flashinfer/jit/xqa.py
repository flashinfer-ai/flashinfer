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
import torch
from .utils import filename_safe_dtype_map
from ..compilation_context import CompilationContext
from .core import (
    JitSpec,
    gen_jit_spec,
)

xqa_nvcc_flags = [
    "-DNDEBUG=1",
    "-DBEAM_WIDTH=1",
    "-DUSE_INPUT_KV=0",
    "-DUSE_CUSTOM_BARRIER=1",
]


def gen_xqa_module(
    input_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    page_size: int,
    head_dim: int,
    head_group_ratio: int,
    use_sliding_window: bool,
    output_dtype: torch.dtype,
    q_seq_len: int = 1,
) -> JitSpec:
    if input_dtype == torch.float16:
        flag_input_dtype = ["-DINPUT_FP16=1", "-DDTYPE=__half"]
    elif input_dtype == torch.bfloat16:
        flag_input_dtype = ["-DINPUT_FP16=0", "-DDTYPE=__nv_bfloat16"]
    else:
        raise ValueError(
            f"Invalid dtype: {input_dtype} for XQA, only float16 and bfloat16 input are supported"
        )

    if kv_cache_dtype == torch.float8_e4m3fn:
        flag_kv_cache_dtype = ["-DCACHE_ELEM_ENUM=2"]
    elif kv_cache_dtype == torch.int8:
        flag_kv_cache_dtype = ["-DCACHE_ELEM_ENUM=1"]
    else:
        flag_kv_cache_dtype = ["-DCACHE_ELEM_ENUM=0"]

    if page_size not in [16, 32, 64, 128]:
        raise ValueError(
            f"Invalid page_size: {page_size}, only 16, 32, 64, 128 are supported"
        )
    flag_tokens_per_page = [f"-DTOKENS_PER_PAGE={page_size}"]

    if head_dim % 16 != 0 or head_dim > 256 or head_dim < 16:
        raise ValueError(
            f"Invalid head_dim: {head_dim}, must be divisible by 16 and in range [16, 256]"
        )
    flag_head_dim = [f"-DHEAD_ELEMS={head_dim}"]

    flag_head_group_ratio = [f"-DHEAD_GRP_SIZE={head_group_ratio}"]

    if use_sliding_window:
        flag_sliding_window = ["-DSLIDING_WINDOW=1"]
    else:
        flag_sliding_window = ["-DSLIDING_WINDOW=0"]

    if output_dtype == torch.float8_e4m3fn:
        flag_low_prec_output = ["-DLOW_PREC_OUTPUT=1"]
    else:
        flag_low_prec_output = ["-DLOW_PREC_OUTPUT=0"]

    if q_seq_len > 1:
        use_spec_dec = True
        if q_seq_len * head_group_ratio <= 32:
            flag_spec_dec = ["-DSPEC_DEC=1", f"-DSPEC_Q_SEQ_LEN={q_seq_len}"]
        else:
            flag_spec_dec = ["-DSPEC_DEC=1"]
    else:
        flag_spec_dec = ["-DSPEC_DEC=0"]
        use_spec_dec = False

    compilation_context = CompilationContext()
    nvcc_flags = compilation_context.get_nvcc_flags_list(
        supported_major_versions=[9, 10, 11, 12]
    )
    sm_nvcc_flags = nvcc_flags

    flag_mla_wrapper = ["-DMLA_WRAPPER=0"]

    sources = [
        jit_env.FLASHINFER_CSRC_DIR / "xqa/mha.cu",
        jit_env.FLASHINFER_CSRC_DIR / "xqa/xqa_wrapper.cu",
        jit_env.FLASHINFER_CSRC_DIR / "flashinfer_xqa_binding.cu",
    ]

    target_archs = compilation_context.TARGET_CUDA_ARCHS

    has_sm90 = any(major == 9 for major, minor in target_archs)
    if has_sm90:
        sources.append(jit_env.FLASHINFER_CSRC_DIR / "xqa/mha_sm90.cu")
        sources.append(jit_env.FLASHINFER_CSRC_DIR / "xqa/tensorMap.cpp")
        flag_sm90_mha = ["-DUSE_SM90_MHA=1"]
    else:
        flag_sm90_mha = ["-DUSE_SM90_MHA=0"]

    return gen_jit_spec(
        f"xqa_input_{filename_safe_dtype_map[input_dtype]}_kv_cache_{filename_safe_dtype_map[kv_cache_dtype]}_output_{filename_safe_dtype_map[output_dtype]}_page_size_{page_size}_head_dim_{head_dim}_head_group_ratio_{head_group_ratio}_use_sliding_window_{use_sliding_window}_use_spec_dec_{use_spec_dec}_spec_q_seq_len_{q_seq_len}",
        sources,
        extra_cuda_cflags=xqa_nvcc_flags
        + sm_nvcc_flags
        + flag_tokens_per_page
        + flag_head_dim
        + flag_input_dtype
        + flag_kv_cache_dtype
        + flag_head_group_ratio
        + flag_sliding_window
        + flag_low_prec_output
        + flag_spec_dec
        + flag_mla_wrapper
        + flag_sm90_mha,
        extra_ldflags=["-lcuda"],  # Add CUDA Driver API library
    )


def gen_xqa_module_mla(
    input_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    page_size: int,
    head_dim: int,
    head_group_ratio: int,
    use_sliding_window: bool = False,
) -> JitSpec:
    assert head_group_ratio == 128, "Only head group ratio 128 is supported for xqa MLA"
    assert head_dim == 576, "Only head dim 576 is supported for xqa_module_mla"
    assert input_dtype == torch.float8_e4m3fn, (
        "Only fp8 input is supported for xqa_module_mla"
    )
    assert kv_cache_dtype == torch.float8_e4m3fn, (
        "Only fp8 kv cache is supported for xqa_module_mla"
    )
    assert not use_sliding_window, "Sliding window is not supported for xqa_module_mla"

    flag_kv_cache_dtype = ["-DCACHE_ELEM_ENUM=2"]

    if page_size not in [16, 32, 64, 128]:
        raise ValueError(
            f"Invalid page_size: {page_size}, only 16, 32, 64, 128 are supported"
        )
    flag_tokens_per_page = [f"-DTOKENS_PER_PAGE={page_size}"]

    flag_head_dim = [f"-DHEAD_ELEMS={head_dim}"]

    flag_head_group_ratio = [f"-DHEAD_GRP_SIZE={head_group_ratio}"]

    flag_sliding_window = ["-DSLIDING_WINDOW=0"]

    compilation_context = CompilationContext()
    nvcc_flags = compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
    sm_nvcc_flags = nvcc_flags

    flag_mla_wrapper = ["-DMLA_WRAPPER=1"]

    return gen_jit_spec(
        f"xqa_mla_input_{filename_safe_dtype_map[input_dtype]}_kv_cache_{filename_safe_dtype_map[kv_cache_dtype]}_page_size_{page_size}_head_dim_{head_dim}_head_group_ratio_{head_group_ratio}_use_sliding_window_{use_sliding_window}",
        [
            jit_env.FLASHINFER_CSRC_DIR / "xqa/mla_sm120.cu",
            jit_env.FLASHINFER_CSRC_DIR / "xqa/tensorMap.cpp",
            jit_env.FLASHINFER_CSRC_DIR / "xqa/xqa_wrapper.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_xqa_binding.cu",
        ],
        extra_cuda_cflags=xqa_nvcc_flags
        + sm_nvcc_flags
        + flag_tokens_per_page
        + flag_head_dim
        + flag_kv_cache_dtype
        + flag_head_group_ratio
        + flag_sliding_window
        + flag_mla_wrapper,
        extra_ldflags=["-lcuda"],  # Add CUDA Driver API library
    )
