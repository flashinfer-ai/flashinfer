# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Compile and launch paged NVFP4 decode attention with BF16 output.

Arguments are validated by ``flashinfer.resserve.paged_decode_residue``.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Literal

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32

from .kernel_paged_decode import GroupedQueryAttentionDecodePagedFp4

PAGE_SIZE = 128
SF_VEC = 16
# Q scales are stored per KV head in a 128-row scale-factor tile.
Q_SF_ROWS = 128


def _padded_group_size(group_size: int) -> int:
    """Base+residue Q rows per KV head for a query/KV head ratio."""
    if group_size == 4:
        return 8
    if group_size in (6, 8):
        return 16
    raise NotImplementedError(f"ResAttn supports GQA4/6/8, got {group_size}")


@functools.cache
def _compiled(
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    reduction: Literal["none", "kernel"],
):
    """Compile or load the decode kernel for one head layout and reducer."""
    group_size = num_qo_heads // num_kv_heads
    padded_group_size = _padded_group_size(group_size)
    padded_qo_heads = num_kv_heads * padded_group_size
    fmha = GroupedQueryAttentionDecodePagedFp4(
        head_dim, padded_group_size, group_size, reduction
    )

    fp4 = cutlass.Float4E2M1FN
    e4m3 = cutlass.Float8E4M3FN
    sym_batch = cute.sym_int()
    sym_pages = cute.sym_int()
    sf_blocks = head_dim // SF_VEC

    def even():
        return cute.sym_int(divisibility=2)

    def compact(dtype, shape, align=16):
        order = tuple(reversed(range(len(shape))))
        return cute.runtime.make_fake_compact_tensor(
            dtype, shape, stride_order=order, assumed_align=align
        )

    def paged_scales():
        return cute.runtime.make_fake_tensor(
            e4m3,
            (sym_pages, num_kv_heads, PAGE_SIZE, sf_blocks),
            stride=(cute.sym_int(divisibility=16), PAGE_SIZE * sf_blocks, sf_blocks, 1),
            assumed_align=16,
        )

    q_fake = compact(fp4, (sym_batch, 1, padded_qo_heads, head_dim))
    k_fake = cute.runtime.make_fake_tensor(
        fp4,
        (sym_pages, PAGE_SIZE, num_kv_heads, head_dim),
        stride=(even(), even(), even(), 1),
        assumed_align=16,
    )
    v_fake = cute.runtime.make_fake_tensor(
        fp4,
        (sym_pages, PAGE_SIZE, num_kv_heads, head_dim),
        stride=(even(), head_dim, even(), 1),
        assumed_align=16,
    )
    q_sf_fake = compact(e4m3, (cute.sym_int(),))
    o_fake = compact(cutlass.BFloat16, (sym_batch, 1, num_qo_heads, head_dim))

    o_partial_fake = l_partial_fake = m_partial_fake = None
    if reduction == "kernel":
        partial_shape = (cute.sym_int(), sym_batch, 1, padded_qo_heads)
        o_partial_fake = compact(cutlass.Float32, (*partial_shape, head_dim))
        l_partial_fake = compact(cutlass.Float32, partial_shape)
        m_partial_fake = compact(cutlass.Float32, partial_shape)

    seq_lens_fake = compact(Int32, (sym_batch,), align=4)
    block_tables_fake = cute.runtime.make_fake_tensor(
        Int32,
        (sym_batch, cute.sym_int()),
        stride=(cute.sym_int(), 1),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    def compile_kernel():
        return cute.compile(
            fmha,
            Int32(1),
            q_fake,
            k_fake,
            v_fake,
            q_sf_fake,
            paged_scales(),
            paged_scales(),
            o_fake,
            o_partial_fake,
            l_partial_fake,
            m_partial_fake,
            Float32(1.0),
            Float32(1.0),
            seq_lens_fake,
            block_tables_fake,
            stream_fake,
            True,
            options="--enable-tvm-ffi --opt-level 3",
        )

    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    specialization = (num_qo_heads, num_kv_heads, head_dim, reduction)
    kernel_name = "kernel_" + hashlib.sha256(repr(specialization).encode()).hexdigest()
    return build_and_load_cute_dsl_kernel(
        "resserve_resattn",
        kernel_name,
        compile_kernel,
        extra_key_files=[
            __file__,
            *(
                str(Path(__file__).with_name(name))
                for name in ("kernel_paged_decode.py", "qmul4.py")
            ),
        ],
    )


def _quantize_q_residue_cuda(q: torch.Tensor, num_kv_heads: int):
    """Quantize BF16 Q into packed base+residue rows and their scales."""
    from .quantization import quantize_grouped_q

    batch, num_qo_heads, head_dim = q.shape
    group_size = num_qo_heads // num_kv_heads
    padded_group_size = _padded_group_size(group_size)
    # Per-call storage prevents cross-stream and captured-graph scratch aliasing.
    packed = torch.zeros(
        batch,
        num_kv_heads,
        padded_group_size,
        head_dim // 2,
        dtype=torch.uint8,
        device=q.device,
    )
    scales = torch.zeros(
        batch * num_kv_heads * Q_SF_ROWS * (head_dim // SF_VEC),
        dtype=torch.uint8,
        device=q.device,
    )
    quantize_grouped_q(
        q.reshape(batch, num_kv_heads, group_size, head_dim).contiguous(),
        packed,
        scales,
    )
    return (
        packed.reshape(batch, 1, num_kv_heads * padded_group_size, head_dim // 2),
        scales.view(torch.float8_e4m3fn),
    )


def decode(
    q,
    k,
    v,
    k_scale,
    v_scale,
    block_tables,
    seq_lens,
    sm_scale,
    key_scale,
    value_scale,
    num_splits,
):
    """Run paged NVFP4 attention and return BF16 ``[batch, query_heads, D]``.

    One split uses no reducer; more splits use the FP32-partial kernel reducer.
    """
    batch, num_qo_heads, head_dim = q.shape
    pages, num_kv_heads = k.shape[:2]
    padded_qo_heads = num_kv_heads * _padded_group_size(num_qo_heads // num_kv_heads)
    reduction: Literal["none", "kernel"] = "none" if num_splits == 1 else "kernel"
    kernel = _compiled(num_qo_heads, num_kv_heads, head_dim, reduction)

    q_packed, q_scale = _quantize_q_residue_cuda(q, num_kv_heads)
    out = torch.empty_like(q)
    o_partial = l_partial = m_partial = None
    if reduction == "kernel":
        stats_shape = (num_splits, batch, 1, padded_qo_heads)
        o_partial = torch.empty(
            *stats_shape, head_dim, dtype=torch.float32, device=q.device
        )
        l_partial = torch.empty(stats_shape, dtype=torch.float32, device=q.device)
        m_partial = torch.empty_like(l_partial)

    fp4 = torch.float4_e2m1fn_x2
    e4m3 = torch.float8_e4m3fn
    kernel(
        Int32(num_splits),
        q_packed.view(fp4),
        k.view(fp4).permute(0, 2, 1, 3),
        v.view(fp4).permute(0, 2, 1, 3),
        q_scale,
        k_scale.view(e4m3),
        # The kernel reads the four-token-interleaved V scale bytes in place.
        v_scale.view(e4m3).view(pages, num_kv_heads, PAGE_SIZE, head_dim // SF_VEC),
        out.view(batch, 1, num_qo_heads, head_dim),
        o_partial,
        l_partial,
        m_partial,
        Float32(sm_scale * key_scale),
        Float32(value_scale),
        seq_lens,
        block_tables,
        False,
    )
    return out
