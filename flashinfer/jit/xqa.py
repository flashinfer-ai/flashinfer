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

import functools

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


# Row budget of the SM90 kernel's SPEC_Q_SEQ_LEN (SWAP_AB) specialization; see
# the ``specDecQLen * headGrpSize <= 32`` static_assert in csrc/xqa/mha_sm90.cu.
SWAP_AB_MAX_Q_ROWS = 32


def swap_ab_eligible(q_seq_len: int, head_group_ratio: int) -> bool:
    """True when the shape fits the SM90 kernel's SPEC_Q_SEQ_LEN (SWAP_AB)
    specialization."""
    return q_seq_len * head_group_ratio <= SWAP_AB_MAX_Q_ROWS


@functools.cache
def _has_sm90_target() -> bool:
    return any(major == 9 for major, _ in CompilationContext().TARGET_CUDA_ARCHS)


def ragged_q_changes_build(q_seq_len: int, head_group_ratio: int) -> bool:
    """True when ragged Q changes the build: it suppresses SPEC_Q_SEQ_LEN
    (SWAP_AB), which only the SM90 kernel uses."""
    return swap_ab_eligible(q_seq_len, head_group_ratio) and _has_sm90_target()


# Name component for a spec-dec module that carries no SPEC_Q_SEQ_LEN
# specialization. The draft length is a runtime argument there, so one module
# serves every length and the name must not claim a specialization.
SPEC_Q_SEQ_LEN_GENERIC = 0


def spec_dec_build_q_seq_len(
    q_seq_len: int, head_group_ratio: int, use_ragged_q: bool = False
) -> int:
    """The part of ``q_seq_len`` that changes the build.

    ``SPEC_Q_SEQ_LEN`` (SWAP_AB) is the only compile-time use of the draft
    length, and ``csrc/xqa/mha_sm90.cu`` is its only consumer. Where that
    specialization is not compiled in, the draft length reaches the kernel as a
    runtime argument and one module serves every length, so modules must not be
    keyed by it: those builds all report ``SPEC_Q_SEQ_LEN_GENERIC``.
    """
    if q_seq_len <= 1:
        return q_seq_len
    if use_ragged_q and ragged_q_changes_build(q_seq_len, head_group_ratio):
        return SPEC_Q_SEQ_LEN_GENERIC
    if swap_ab_eligible(q_seq_len, head_group_ratio):
        return q_seq_len
    return SPEC_Q_SEQ_LEN_GENERIC


def generic_spec_dec_q_seq_len(head_group_ratio: int) -> int:
    """A draft length that always takes the generic (non-SWAP_AB) spec-dec build
    for ``head_group_ratio``: the first length past ``swap_ab_eligible``.

    Clamped to 2 so the result still selects a spec-dec build when the group
    ratio alone exceeds the SWAP_AB row budget and no length is eligible.
    """
    return max(2, SWAP_AB_MAX_Q_ROWS // head_group_ratio + 1)


def xqa_module_key(
    q_seq_len: int, head_group_ratio: int, use_ragged_q: bool
) -> tuple[int, bool]:
    """Canonical ``(q_seq_len, use_ragged_q)`` for module identity.

    Both only change the build through the SPEC_Q_SEQ_LEN (SWAP_AB)
    specialization, so every request that does not compile it in shares one
    module and must map to one key: a second key would build an identical
    module under a second name and re-register the same torch op.
    """
    use_ragged_q = use_ragged_q and ragged_q_changes_build(q_seq_len, head_group_ratio)
    if (
        q_seq_len > 1
        and spec_dec_build_q_seq_len(q_seq_len, head_group_ratio, use_ragged_q)
        == SPEC_Q_SEQ_LEN_GENERIC
    ):
        return generic_spec_dec_q_seq_len(head_group_ratio), False
    return q_seq_len, use_ragged_q


def gen_xqa_module(
    input_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    page_size: int,
    head_dim: int,
    head_group_ratio: int,
    use_sliding_window: bool,
    output_dtype: torch.dtype,
    q_seq_len: int = 1,
    use_ragged_q: bool = False,
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
    elif kv_cache_dtype == torch.uint8:
        flag_kv_cache_dtype = ["-DCACHE_ELEM_ENUM=3"]
    else:
        flag_kv_cache_dtype = ["-DCACHE_ELEM_ENUM=0"]

    if page_size not in [16, 32, 64, 128]:
        raise ValueError(
            f"Invalid page_size: {page_size}, only 16, 32, 64, 128 are supported"
        )
    flag_tokens_per_page = [f"-DTOKENS_PER_PAGE={page_size}"]

    if head_dim % 16 != 0 or head_dim > 512 or head_dim < 16:
        raise ValueError(
            f"Invalid head_dim: {head_dim}, must be divisible by 16 and in range [16, 512]"
        )
    if head_dim > 256:
        # headElems > 256 uses per-warp head-dim splits in mha.cu (see nbHeadSplits);
        # the specialized SPEC_DEC/SM90-GMMA paths do not support it, and the
        # persistent-Q smem budget requires head_group_ratio <= 16.
        if q_seq_len > 1:
            raise ValueError(
                f"head_dim {head_dim} > 256 does not support speculative decoding (q_seq_len > 1)"
            )
        if head_group_ratio > 16:
            raise ValueError(
                f"head_dim {head_dim} > 256 requires head_group_ratio <= 16, got {head_group_ratio}"
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
        # The SPEC_Q_SEQ_LEN (SWAP_AB) specialization requires a uniform q
        # length across the batch, so it must be skipped for ragged Q.
        ragged_changes_flags = use_ragged_q and ragged_q_changes_build(
            q_seq_len, head_group_ratio
        )
        if swap_ab_eligible(q_seq_len, head_group_ratio) and not ragged_changes_flags:
            flag_spec_dec = ["-DSPEC_DEC=1", f"-DSPEC_Q_SEQ_LEN={q_seq_len}"]
        else:
            flag_spec_dec = ["-DSPEC_DEC=1"]
        # The xqa() mask API indexes draft tokens by row position (linear
        # chains only); non-tree mode enables per-row sliding-window masking.
        flag_spec_dec.append("-DIS_SPEC_DEC_TREE=0")
    else:
        if use_ragged_q:
            raise ValueError("use_ragged_q requires q_seq_len > 1 (speculative decode)")
        flag_spec_dec = ["-DSPEC_DEC=0"]
        use_spec_dec = False
        ragged_changes_flags = False

    compilation_context = CompilationContext()
    nvcc_flags = compilation_context.get_nvcc_flags_list(
        supported_major_versions=[9, 10, 12], map_sm107_to_100f=True
    )
    sm_nvcc_flags = nvcc_flags

    flag_mla_wrapper = ["-DMLA_WRAPPER=0"]

    sources = [
        jit_env.FLASHINFER_CSRC_DIR / "xqa/mha.cu",
        jit_env.FLASHINFER_CSRC_DIR / "xqa/xqa_wrapper.cu",
        jit_env.FLASHINFER_CSRC_DIR / "flashinfer_xqa_binding.cu",
    ]

    if _has_sm90_target() and head_dim <= 256:
        # The SM90 GMMA kernel does not support head_dim > 256.
        sources.append(jit_env.FLASHINFER_CSRC_DIR / "xqa/mha_sm90.cu")
        sources.append(jit_env.FLASHINFER_CSRC_DIR / "xqa/tensorMap.cpp")
        flag_sm90_mha = ["-DUSE_SM90_MHA=1"]
    else:
        flag_sm90_mha = ["-DUSE_SM90_MHA=0"]

    # Name by the draft length that reaches nvcc, not the requested one. Where
    # the SPEC_Q_SEQ_LEN specialization is not compiled in -- because the shape
    # is not SWAP_AB-eligible, or because ragged Q suppressed it -- every length
    # compiles the same module, so they must all answer to one name. This also
    # subsumes the old "_ragged_q" suffix: ragged Q only ever differed from a
    # uniform request by suppressing that specialization, which the name now
    # states directly.
    name_q_seq_len = spec_dec_build_q_seq_len(q_seq_len, head_group_ratio, use_ragged_q)
    return gen_jit_spec(
        f"xqa_input_{filename_safe_dtype_map[input_dtype]}_kv_cache_{filename_safe_dtype_map[kv_cache_dtype]}_output_{filename_safe_dtype_map[output_dtype]}_page_size_{page_size}_head_dim_{head_dim}_head_group_ratio_{head_group_ratio}_use_sliding_window_{use_sliding_window}_use_spec_dec_{use_spec_dec}_spec_q_seq_len_{name_q_seq_len}",
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
    assert input_dtype in (torch.float8_e4m3fn, torch.bfloat16), (
        f"Only fp8 and bf16 input are supported for xqa_module_mla, got {input_dtype}"
    )
    assert kv_cache_dtype in (torch.float8_e4m3fn, torch.bfloat16), (
        f"Only fp8 and bf16 kv cache are supported for xqa_module_mla, got {kv_cache_dtype}"
    )
    assert input_dtype == kv_cache_dtype, (
        f"input_dtype ({input_dtype}) and kv_cache_dtype ({kv_cache_dtype}) must match for xqa MLA"
    )
    assert not use_sliding_window, "Sliding window is not supported for xqa_module_mla"

    if input_dtype == torch.float8_e4m3fn:
        flag_kv_cache_dtype = ["-DCACHE_ELEM_ENUM=2"]
    else:
        flag_kv_cache_dtype = ["-DCACHE_ELEM_ENUM=0", "-DMLA_BF16=1"]

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
