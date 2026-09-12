# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""JIT compilers for packed-contiguous SM120 FP8 FMHA kernels.

Ragged and paged compilers use symbolic token, batch, page-pool, and
block-table dimensions. Sequence lengths are runtime metadata and never part
of the compiler cache key.

Requires a compatible CuTe DSL package providing ``cutlass.experimental``.
"""

import functools
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Int32

_CUTLASS_DTYPE = {
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}


def _validate_balanced_scheduler(is_causal: bool, balanced_scheduler: bool) -> None:
    if balanced_scheduler and not is_causal:
        raise ValueError("balanced_scheduler=True requires is_causal=True")


def _compile_sm120_fmha_fp8_ragged_kernel(
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_causal: bool,
    kv_tile: int,
    q_tile: int,
    device: torch.device,
    with_lse: bool = False,
    balanced_scheduler: bool = False,
):
    """Compile one sequence-length-independent packed ragged kernel."""

    _validate_balanced_scheduler(is_causal, balanced_scheduler)

    from .fmha_prefill_fp8_tma import SM120FusedMultiHeadAttentionFP8ForwardTMA

    in_ct = _CUTLASS_DTYPE[in_dtype]
    out_ct = _CUTLASS_DTYPE[out_dtype]

    fmha = SM120FusedMultiHeadAttentionFP8ForwardTMA(
        in_dtype=in_ct,
        out_dtype=out_ct,
        is_causal=is_causal,
        head_tile=head_dim,
        kv_tile=kv_tile,
        q_tile=q_tile,
        use_paged_kv=False,
        balanced_scheduler=balanced_scheduler,
    )

    sym_total_q = cute.sym_int()
    sym_total_k = cute.sym_int()
    sym_cu = cute.sym_int()
    fake_q = make_fake_compact_tensor(
        in_ct,
        (sym_total_q, num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_o = make_fake_compact_tensor(
        out_ct,
        (sym_total_q, num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_lse = (
        make_fake_compact_tensor(
            cutlass.Float32,
            (sym_total_q, num_qo_heads),
            stride_order=(1, 0),
            assumed_align=16,
        )
        if with_lse
        else None
    )
    fake_k = make_fake_compact_tensor(
        in_ct,
        (sym_total_k, num_kv_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_v = make_fake_compact_tensor(
        in_ct,
        (sym_total_k, num_kv_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_cu_seqlens_q = make_fake_compact_tensor(Int32, (sym_cu,), assumed_align=4)
    fake_cu_seqlens_k = make_fake_compact_tensor(Int32, (sym_cu,), assumed_align=4)
    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        fmha,
        fake_q,
        fake_k,
        fake_v,
        fake_o,
        fake_lse,
        cutlass.Float32(1.0),  # softmax_scale_log2 placeholder
        cutlass.Float32(1.0),  # output_scale placeholder
        stream_fake,
        None,  # seqlens_kv
        fake_cu_seqlens_q,
        None,  # block_tables (non-paged)
        fake_cu_seqlens_k,
        cutlass.Int32(1),  # runtime max_seqlen_q placeholder
        True,  # use_pdl placeholder (runtime-dynamic)
        options="--enable-tvm-ffi",
    )


def _compile_sm120_fmha_fp8_paged_kernel(
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_causal: bool,
    kv_tile: int,
    q_tile: int,
    num_tokens_per_page: int,
    device: torch.device,
    with_lse: bool = False,
    balanced_scheduler: bool = False,
):
    """Compile one sequence-length-independent packed-Q paged kernel."""
    _validate_balanced_scheduler(is_causal, balanced_scheduler)

    from .fmha_prefill_fp8_tma import SM120FusedMultiHeadAttentionFP8ForwardTMA

    in_ct = _CUTLASS_DTYPE[in_dtype]
    out_ct = _CUTLASS_DTYPE[out_dtype]

    fmha = SM120FusedMultiHeadAttentionFP8ForwardTMA(
        in_dtype=in_ct,
        out_dtype=out_ct,
        is_causal=is_causal,
        head_tile=head_dim,
        kv_tile=kv_tile,
        q_tile=q_tile,
        use_paged_kv=True,
        num_tokens_per_page=num_tokens_per_page,
        balanced_scheduler=balanced_scheduler,
    )

    sym_b = cute.sym_int()
    sym_m = cute.sym_int()
    sym_num_pages = cute.sym_int()
    sym_total_q = cute.sym_int()
    sym_seqlens = cute.sym_int()
    sym_cu_q = cute.sym_int()
    fake_q = make_fake_compact_tensor(
        in_ct,
        (sym_total_q, num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_o = make_fake_compact_tensor(
        out_ct,
        (sym_total_q, num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_lse = (
        make_fake_compact_tensor(
            cutlass.Float32,
            (sym_total_q, num_qo_heads),
            stride_order=(1, 0),
            assumed_align=16,
        )
        if with_lse
        else None
    )

    # HND paged KV pool: (num_pages, Hkv, num_tokens_per_page, D). Keep the
    # outer page stride symbolic so a plane view from a combined
    # [num_pages, 2, Hkv, page_size, D] allocation has the same compiled ABI
    # as a standalone compact HND pool.
    fake_k = cute.runtime.make_fake_tensor(
        in_ct,
        (sym_num_pages, num_kv_heads, num_tokens_per_page, head_dim),
        stride=(
            cute.sym_int(),
            num_tokens_per_page * head_dim,
            head_dim,
            1,
        ),
        assumed_align=16,
    )
    fake_v = cute.runtime.make_fake_tensor(
        in_ct,
        (sym_num_pages, num_kv_heads, num_tokens_per_page, head_dim),
        stride=(
            cute.sym_int(),
            num_tokens_per_page * head_dim,
            head_dim,
            1,
        ),
        assumed_align=16,
    )

    fake_seqlens_kv = make_fake_compact_tensor(Int32, (sym_seqlens,), assumed_align=4)
    fake_cu_seqlens_q = make_fake_compact_tensor(Int32, (sym_cu_q,), assumed_align=4)
    # Both dimensions are dynamic: M is a runtime row stride, not a
    # sequence-capacity specialization.
    fake_block_tables = make_fake_compact_tensor(
        Int32,
        (sym_b, sym_m),
        stride_order=(1, 0),
        assumed_align=4,
    )

    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        fmha,
        fake_q,
        fake_k,
        fake_v,
        fake_o,
        fake_lse,
        cutlass.Float32(1.0),  # softmax_scale_log2 placeholder
        cutlass.Float32(1.0),  # output_scale placeholder
        stream_fake,
        fake_seqlens_kv,
        fake_cu_seqlens_q,
        fake_block_tables,
        None,  # cu_seqlens_k
        cutlass.Int32(1),  # runtime max_seqlen_q placeholder
        True,  # use_pdl placeholder (runtime-dynamic)
        options="--enable-tvm-ffi",
    )


def _cache_key_files():
    root = Path(__file__).resolve().parent
    return (
        str(Path(__file__).resolve()),
        str(root / "fmha_prefill_fp8_tma.py"),
    )


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.").replace(".", "_")


@functools.cache
def compile_sm120_fmha_fp8_ragged_kernel(
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_causal: bool,
    kv_tile: int,
    q_tile: int,
    device: torch.device,
    with_lse: bool = False,
    balanced_scheduler: bool = False,
):
    _validate_balanced_scheduler(is_causal, balanced_scheduler)

    from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    device = torch.device(device)
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("SM120 PRIMS FMHA compilation requires SM120")
    kernel_name = (
        f"ragged_{_dtype_name(in_dtype)}_{_dtype_name(out_dtype)}"
        f"_hq{num_qo_heads}_hkv{num_kv_heads}_d{head_dim}"
        f"_causal{int(is_causal)}_kt{kv_tile}_qt{q_tile}"
        f"_lse{int(with_lse)}_balanced{int(balanced_scheduler)}"
    )
    return build_and_load_cute_dsl_kernel(
        "sm120_prims_fmha_fp8",
        kernel_name,
        lambda: _compile_sm120_fmha_fp8_ragged_kernel(
            in_dtype,
            out_dtype,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            is_causal,
            kv_tile,
            q_tile,
            device,
            with_lse,
            balanced_scheduler,
        ),
        extra_key_files=_cache_key_files(),
    )


@functools.cache
def compile_sm120_fmha_fp8_paged_kernel(
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_causal: bool,
    kv_tile: int,
    q_tile: int,
    num_tokens_per_page: int,
    device: torch.device,
    with_lse: bool = False,
    balanced_scheduler: bool = False,
):
    _validate_balanced_scheduler(is_causal, balanced_scheduler)

    from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    device = torch.device(device)
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("SM120 PRIMS FMHA compilation requires SM120")
    kernel_name = (
        f"paged_hnd_{_dtype_name(in_dtype)}_{_dtype_name(out_dtype)}"
        f"_hq{num_qo_heads}_hkv{num_kv_heads}_d{head_dim}"
        f"_causal{int(is_causal)}_kt{kv_tile}_qt{q_tile}"
        f"_page{num_tokens_per_page}_lse{int(with_lse)}"
        f"_balanced{int(balanced_scheduler)}"
    )
    return build_and_load_cute_dsl_kernel(
        "sm120_prims_fmha_fp8",
        kernel_name,
        lambda: _compile_sm120_fmha_fp8_paged_kernel(
            in_dtype,
            out_dtype,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            is_causal,
            kv_tile,
            q_tile,
            num_tokens_per_page,
            device,
            with_lse,
            balanced_scheduler,
        ),
        extra_key_files=_cache_key_files(),
    )
