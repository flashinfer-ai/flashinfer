# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KPage layout: the reusable, content-independent page-walk metadata.

``kv_sizes[j]`` is the number of valid rows of KV block ``j`` (the BSA
``block_sizes``); ``q_sizes[i]`` the number of valid rows of Q block ``i``
(128 everywhere for the dense Q of the base path).  Forward output rows and
backward gradient rows that fall on padding are zero.

``sparse_block_size`` is 128 or 64.  The blk64 layout is forward only: it holds
just the M64 x N256 page walks, one Q block per tile, and its Q rows are all
valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from flashinfer.experimental.vsa_kpage.csrc.utils.kpage_utils import BLOCK
from ._build import (
    Blk64ForwardTables,
    WalkTables,
    build_walk_tables,
    build_walk_tables_blk64,
)

__all__ = ["KPageLayout", "prepare_layout"]


@dataclass
class KPageLayout:
    """Page walks for one sparse pattern; reuse while the metadata is unchanged.

    Built from ``q2k_block_index`` / ``q2k_block_nums`` / ``block_sizes`` /
    ``q_block_sizes`` only, never from Q/K/V.  Backward tables are built on
    first backward use.
    """

    seqlen_q: int  # rows of q (the walks address ceil(seqlen_q / 128) blocks)
    seqlen_k: int
    num_heads: int
    num_q_blocks: int
    num_kv_blocks: int
    kmax: int
    device: torch.device
    _q_sizes: Tensor  # [nQ] int32
    _kv_sizes: Tensor  # [nKV] int32
    _tables: Optional[WalkTables]  # None for sparse_block_size == 64
    _blk64_tables: Blk64ForwardTables
    _has_partial_q: Optional[bool]  # None when validate=False
    _has_partial_kv: Optional[bool]
    _source: tuple = ()  # identity of the metadata arguments the layout was built from
    sparse_block_size: int = 128

    def __repr__(self) -> str:
        return (
            f"KPageLayout(seqlen_q={self.seqlen_q}, seqlen_k={self.seqlen_k}, num_heads={self.num_heads}, "
            f"num_q_blocks={self.num_q_blocks}, num_kv_blocks={self.num_kv_blocks}, kmax={self.kmax}, device={self.device})"
        )

    def build_backward(self) -> None:
        """Build the backward tables now instead of on the first backward call."""
        if self.sparse_block_size != 128:
            raise NotImplementedError(
                "the KPage backward is available for sparse_block_size=128 only"
            )
        self._tables.resolve_bwd_tables()


def _validate_metadata(
    q2k_block_index: Tensor,
    q2k_block_nums: Tensor,
    q_sizes: Tensor,
    kv_sizes: Tensor,
    allow_empty: bool,
    block: int = BLOCK,
) -> Tuple[bool, bool]:
    """Check the metadata contract on device with one readback; returns whether the Q / KV partition has partial blocks.

    ``allow_empty`` admits query blocks with no KV pages to attend to (count 0,
    or only zero-size blocks); they get O = 0, LSE = -inf and dQ = 0.
    """
    dev = q2k_block_index.device
    cap = q2k_block_index.shape[-1]
    num_kv_blocks = kv_sizes.numel()
    active = torch.arange(
        cap, device=dev, dtype=q2k_block_nums.dtype
    ) < q2k_block_nums.unsqueeze(-1)
    idx_oob = active & ((q2k_block_index < 0) | (q2k_block_index >= num_kv_blocks))
    # Inactive slots get unique sentinels so only active collisions survive the sort.
    sentinels = num_kv_blocks + torch.arange(cap, device=dev, dtype=torch.int32)
    sorted_idx = (
        torch.where(active, q2k_block_index.clamp(0, num_kv_blocks - 1), sentinels)
        .sort(dim=-1)
        .values
    )
    dup = sorted_idx[..., 1:] == sorted_idx[..., :-1]
    gathered = kv_sizes.clamp(0, block)[
        q2k_block_index.clamp(0, num_kv_blocks - 1).long()
    ]
    row_valid = (gathered * active).sum(dim=-1, dtype=torch.int64)

    flags = torch.stack(
        [
            q2k_block_nums.min().to(torch.int64),
            q2k_block_nums.max().to(torch.int64),
            idx_oob.any().to(torch.int64),
            dup.any().to(torch.int64),
            torch.minimum(q_sizes.min(), kv_sizes.min()).to(torch.int64),
            torch.maximum(q_sizes.max(), kv_sizes.max()).to(torch.int64),
            row_valid.min(),
            (q_sizes < block).any().to(torch.int64),
            (kv_sizes < block).any().to(torch.int64),
        ]
    ).cpu()
    (
        counts_min,
        counts_max,
        has_oob,
        has_dup,
        sizes_min,
        sizes_max,
        row_valid_min,
        partial_q,
        partial_kv,
    ) = (int(x) for x in flags)

    def first(mask: Tensor) -> tuple:
        return tuple(int(x) for x in mask.nonzero()[0])

    if counts_min < 0 or (counts_min < 1 and not allow_empty):
        b, h, i = first(q2k_block_nums < 1)
        hint = (
            ""
            if allow_empty
            else " (pass allow_empty_block_nums=True to admit empty query blocks)"
        )
        raise ValueError(
            f"q2k_block_nums[{b}, {h}, {i}] = {int(q2k_block_nums[b, h, i])}: every query block must list at least one KV block{hint}"
        )
    if counts_max > cap:
        b, h, i = first(q2k_block_nums > cap)
        raise ValueError(
            f"q2k_block_nums[{b}, {h}, {i}] = {int(q2k_block_nums[b, h, i])} exceeds q2k_block_index.shape[-1] = {cap}"
        )
    if has_oob:
        b, h, i, e = first(idx_oob)
        raise ValueError(
            f"q2k_block_index[{b}, {h}, {i}, {e}] = {int(q2k_block_index[b, h, i, e])} is outside [0, {num_kv_blocks})"
        )
    if has_dup:
        b, h, i, _ = first(dup)
        raise ValueError(
            f"q2k_block_index[{b}, {h}, {i}, :q2k_block_nums[{b}, {h}, {i}]] contains duplicate KV block ids"
        )
    if sizes_min < 0 or sizes_max > block:
        for name, sizes in (("block_sizes", kv_sizes), ("q_block_sizes", q_sizes)):
            bad = (sizes < 0) | (sizes > block)
            if bool(bad.any()):
                j = first(bad)[0]
                raise ValueError(
                    f"{name}[{j}] = {int(sizes[j])} is outside [0, {block}]"
                )
    if row_valid_min < 1 and not allow_empty:
        b, h, i = first(row_valid < 1)
        raise ValueError(
            f"query block ({b}, {h}, {i}) attends to zero valid tokens; every listed KV block has block_sizes == 0"
        )
    return bool(partial_q), bool(partial_kv)


def _check_metadata_types(
    q2k_block_index: Tensor, q2k_block_nums: Tensor, kv_sizes: Tensor, q_sizes: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor, int, int, int]:
    if q2k_block_index.dim() != 4 or q2k_block_index.shape[0] != 1:
        raise ValueError(
            f"q2k_block_index must have shape [1, num_heads, num_q_blocks, kmax], got {tuple(q2k_block_index.shape)}"
        )
    B, H, nQ, cap = q2k_block_index.shape
    if tuple(q2k_block_nums.shape) != (B, H, nQ):
        raise ValueError(
            f"q2k_block_nums must have shape [1, {H}, {nQ}], got {tuple(q2k_block_nums.shape)}"
        )
    if q2k_block_index.dtype != torch.int32 or q2k_block_nums.dtype != torch.int32:
        raise ValueError(
            f"q2k_block_index and q2k_block_nums must be int32, got {q2k_block_index.dtype} / {q2k_block_nums.dtype}"
        )
    if kv_sizes.dim() != 1 or kv_sizes.dtype != torch.int32 or kv_sizes.numel() < 1:
        raise ValueError(
            f"block_sizes must be a non-empty 1-D int32 tensor, got {tuple(kv_sizes.shape)} {kv_sizes.dtype}"
        )
    if tuple(q_sizes.shape) != (nQ,) or q_sizes.dtype != torch.int32:
        raise ValueError(
            f"q_block_sizes must be a 1-D int32 tensor with one entry per query block ({nQ}), got {tuple(q_sizes.shape)} {q_sizes.dtype}"
        )
    if cap < 1:
        raise ValueError("q2k_block_index.shape[-1] must be >= 1")
    dev = q2k_block_index.device
    if dev.type != "cuda" or any(
        t.device != dev for t in (q2k_block_nums, kv_sizes, q_sizes)
    ):
        raise ValueError(
            "q2k_block_index, q2k_block_nums, block_sizes and q_block_sizes must be on the same CUDA device"
        )
    return (
        q2k_block_index.contiguous(),
        q2k_block_nums.contiguous(),
        kv_sizes.contiguous(),
        q_sizes.contiguous(),
        H,
        nQ,
        cap,
    )


def prepare_layout(
    q2k_block_index: Tensor,
    q2k_block_nums: Tensor,
    block_sizes: Tensor,
    q_block_sizes: Tensor,
    *,
    seqlen_q: Optional[int] = None,
    seqlen_k: Optional[int] = None,
    validate: bool = True,
    allow_empty_block_nums: bool = False,
    bwd_kv_groups: Optional[int] = None,
    source: tuple = (),
    sparse_block_size: int = 128,
) -> KPageLayout:
    """Build the forward page walks; backward tables are built on first backward use.

    ``seqlen_q`` / ``seqlen_k`` default to the padded block lengths; a shorter
    value puts the rows past it into the padding of the final block.
    """
    idx, cnt, kv_sizes, q_sizes, H, nQ, cap = _check_metadata_types(
        q2k_block_index, q2k_block_nums, block_sizes, q_block_sizes
    )
    block = sparse_block_size
    if block not in (64, 128):
        raise ValueError(f"sparse_block_size must be 64 or 128, got {block}")
    seqlen_q = nQ * block if seqlen_q is None else seqlen_q
    seqlen_k = kv_sizes.numel() * block if seqlen_k is None else seqlen_k
    for name, seqlen, n in (
        ("seqlen_q", seqlen_q, nQ),
        ("seqlen_k", seqlen_k, kv_sizes.numel()),
    ):
        if not (n - 1) * block < seqlen <= n * block:
            raise ValueError(
                f"{name} = {seqlen} does not fit the {n} blocks of the metadata"
            )
    if block == 64 and (seqlen_q % block or seqlen_k % block):
        raise NotImplementedError(
            "the blk64 KPage path requires seqlen_q and seqlen_k to be multiples of 64"
        )
    if (
        seqlen_q % block
    ):  # device-side clamp of the final block: no host-device copy, no sync
        q_sizes = q_sizes.clone()
        q_sizes[-1:].clamp_(max=seqlen_q % block)
    if seqlen_k % block:
        kv_sizes = kv_sizes.clone()
        kv_sizes[-1:].clamp_(max=seqlen_k % block)
    partial_q, partial_kv = (
        _validate_metadata(idx, cnt, q_sizes, kv_sizes, allow_empty_block_nums, block)
        if validate
        else (None, None)
    )
    if block == 64:
        # Forward only: the M64 x N256 walks, one Q block per tile, every Q row valid.
        tables, blk64_tables = None, build_walk_tables_blk64(idx, cnt, kv_sizes)
        partial_q = partial_kv = False
    else:
        tables, blk64_tables = build_walk_tables(
            idx, cnt, q_sizes, kv_sizes, H, bwd_kv_groups=bwd_kv_groups
        )
    return KPageLayout(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        num_heads=H,
        num_q_blocks=nQ,
        num_kv_blocks=kv_sizes.numel(),
        kmax=cap,
        device=idx.device,
        _q_sizes=q_sizes,
        _kv_sizes=kv_sizes,
        _tables=tables,
        _blk64_tables=blk64_tables,
        _has_partial_q=partial_q if seqlen_q % block == 0 else True,
        _has_partial_kv=partial_kv if seqlen_k % block == 0 else True,
        _source=source,
        sparse_block_size=block,
    )
