# mypy: disable-error-code="attr-defined"
# Upstream CuTe launchers attach compile caches and mixin methods dynamically.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Compile-cached launchers for the KPage forward (M128 x N128 and M64 x N256
# routes) and the KV-owned paged backward, plus the forward/backward entry
# points called from api.py.

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

import cutlass.cute as cute

from flashinfer.experimental.vsa_kpage.csrc.fwd.sm100_blk128.bsa_fwd_sm100_kpage import (
    BlockSparseAttnForwardSm100KPage,
)
from flashinfer.experimental.vsa_kpage.csrc.fwd.sm100_blk128.bsa_fwd_sm100_kpage_q64 import (
    BlockSparseAttnForwardSm100KPageQ64,
)
from flashinfer.experimental.vsa_kpage.csrc.utils.kpage_utils import (
    HEAD_DIM,
    PAGE,
)
from ._build import zero_pad_forward_state
from ._compile import compile_key, current_stream, fake_stream, to_cute_tensor
from ._layout import KPageLayout
import cutlass

torch2cute_dtype_map = {torch.bfloat16: cutlass.BFloat16}


def _fwd_m128(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    lse: Tensor,
    page_table: Tensor,
    mask_table: Tensor,
    tile_offsets: Tensor,
    tile_counts: Tensor,
    full_tile_counts: Tensor,
    work_indices: Tensor,
    softmax_scale: float,
) -> None:
    head_dim = q.shape[-1]
    qhead_per_kvhead = q.shape[1] // k.shape[1]
    tail_page = k.shape[2] % PAGE != 0  # K/V end inside a page: row-extent descriptors
    nonpositive_scale = (
        not softmax_scale > 0.0
    )  # scale the scores before the masks (NaN/inf otherwise)
    tensors = (
        q,
        k,
        v,
        out,
        lse,
        page_table,
        mask_table,
        tile_offsets,
        tile_counts,
        full_tile_counts,
        work_indices,
    )
    key = compile_key(
        "kpage_fwd_m128",
        (
            torch2cute_dtype_map[q.dtype],
            head_dim,
            qhead_per_kvhead,
            tail_page,
            nonpositive_scale,
            (
                torch.cuda.get_device_capability(q.device)[0] * 10
                + torch.cuda.get_device_capability(q.device)[1]
            ),
        ),
        tensors,
    )
    cache = _fwd_m128.compile_cache
    if key not in cache:
        q_t, k_t, v_t, o_t = [to_cute_tensor(t, (0, 1, 2)) for t in (q, k, v, out)]
        lse_t = to_cute_tensor(lse, (0, 1, 2), assumed_align=4)
        pt_t = to_cute_tensor(page_table, 0, assumed_align=32)
        mt_t = to_cute_tensor(mask_table, 0, assumed_align=16)
        to_t, tc_t, ft_t = [
            to_cute_tensor(t, (0, 1, 2), assumed_align=4)
            for t in (tile_offsets, tile_counts, full_tile_counts)
        ]
        wi_t = to_cute_tensor(work_indices, 0, assumed_align=4)
        cache[key] = cute.compile(
            BlockSparseAttnForwardSm100KPage(
                head_dim,
                head_dim,
                qhead_per_kvhead=qhead_per_kvhead,
                tail_page=tail_page,
                nonpositive_scale=nonpositive_scale,
            ),
            q_t,
            k_t,
            v_t,
            o_t,
            lse_t,
            softmax_scale,
            pt_t,
            mt_t,
            to_t,
            ft_t,
            tc_t,
            wi_t,
            fake_stream(),
            options="--enable-tvm-ffi",
        )
    cache[key](
        q.detach(),
        k.detach(),
        v.detach(),
        out.detach(),
        lse,
        softmax_scale,
        page_table,
        mask_table,
        tile_offsets,
        full_tile_counts,
        tile_counts,
        work_indices,
        current_stream(),
    )


_fwd_m128.compile_cache = {}


def _fwd_m64(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    lse: Tensor,
    page_table: Tensor,
    mask_table: Tensor,
    tile_offsets: Tensor,
    iter_counts: Tensor,
    full_iter_counts: Tensor,
    work_indices: Tensor,
    softmax_scale: float,
) -> None:
    head_dim = q.shape[-1]
    qhead_per_kvhead = q.shape[1] // k.shape[1]
    tail_page = k.shape[2] % PAGE != 0
    nonpositive_scale = (
        not softmax_scale > 0.0
    )  # scale the scores before the masks (NaN/inf otherwise)
    tensors = (
        q,
        k,
        v,
        out,
        lse,
        page_table,
        mask_table,
        tile_offsets,
        iter_counts,
        full_iter_counts,
        work_indices,
    )
    key = compile_key(
        "kpage_fwd_m64",
        (
            torch2cute_dtype_map[q.dtype],
            head_dim,
            qhead_per_kvhead,
            tail_page,
            nonpositive_scale,
            (
                torch.cuda.get_device_capability(q.device)[0] * 10
                + torch.cuda.get_device_capability(q.device)[1]
            ),
        ),
        tensors,
    )
    cache = _fwd_m64.compile_cache
    if key not in cache:
        q_t, k_t, v_t, o_t = [to_cute_tensor(t, (0, 1, 2)) for t in (q, k, v, out)]
        lse_t = to_cute_tensor(lse, (0, 1, 2), assumed_align=4)
        pt_t = to_cute_tensor(page_table, 0, assumed_align=32)
        mt_t = to_cute_tensor(mask_table, 0, assumed_align=16)
        to_t, ic_t, fi_t = [
            to_cute_tensor(t, (0, 1, 2), assumed_align=4)
            for t in (tile_offsets, iter_counts, full_iter_counts)
        ]
        wi_t = to_cute_tensor(work_indices, 0, assumed_align=4)
        cache[key] = cute.compile(
            BlockSparseAttnForwardSm100KPageQ64(
                head_dim,
                head_dim,
                qhead_per_kvhead=qhead_per_kvhead,
                tail_page=tail_page,
                nonpositive_scale=nonpositive_scale,
            ),
            q_t,
            k_t,
            v_t,
            o_t,
            lse_t,
            softmax_scale,
            pt_t,
            mt_t,
            to_t,
            fi_t,
            ic_t,
            wi_t,
            fake_stream(),
            options="--enable-tvm-ffi",
        )
    cache[key](
        q.detach(),
        k.detach(),
        v.detach(),
        out.detach(),
        lse,
        softmax_scale,
        page_table,
        mask_table,
        tile_offsets,
        full_iter_counts,
        iter_counts,
        work_indices,
        current_stream(),
    )


_fwd_m64.compile_cache = {}


def forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    layout: KPageLayout,
    softmax_scale: Optional[float],
    out: Optional[Tensor] = None,
    lse: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Forward over a prepared layout; returns (o, lse) with zeroed padding rows (blk128; the blk64 path has none)."""
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    softmax_scale = float(softmax_scale)
    out = torch.empty_like(q) if out is None else out  # keeps BHSD or BSHD strides
    lse = (
        torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
        if lse is None
        else lse
    )
    t64 = layout._blk64_tables
    if t64.work_indices.numel() > 0:
        _fwd_m64(
            q,
            k,
            v,
            out,
            lse,
            t64.page_table,
            t64.mask_table,
            t64.tile_offsets,
            t64.iter_counts,
            t64.full_iter_counts,
            t64.work_indices,
            softmax_scale,
        )
    t128 = layout._tables  # None on the blk64 path: every Q block is one M64 tile
    if t128 is not None and t128.work_indices.numel() > 0:
        _fwd_m128(
            q,
            k,
            v,
            out,
            lse,
            t128.page_table,
            t128.mask_table,
            t128.tile_offsets,
            t128.tile_counts,
            t128.full_tile_counts,
            t128.work_indices,
            softmax_scale,
        )
    if layout._has_partial_q is not False:
        zero_pad_forward_state(out, lse, layout._q_sizes)
    return out, lse
