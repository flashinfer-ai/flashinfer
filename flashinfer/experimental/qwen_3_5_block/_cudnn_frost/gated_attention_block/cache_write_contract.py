# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata validation for paged cache writes, and the fused prepare launch's declaration-time record, independent of CuTeDSL kernels.

Declarations and execute preflight must remain usable on the package's supported
DSL floor, before a stage checks the newer version its kernel requires.  That is
why :class:`QsaPrepareIndexArms` -- which indexer arms the prepare launch traces;
the block builds the record at DECLARATION under ``fuse_prepare`` -- lives here
and not in ``kernels/qsa_prepare.py``, which imports the DSL at module scope and
re-exports the name for its own callers (Rule 7).
"""

from typing import NamedTuple, Optional

import torch

_SLOT_DTYPES = (torch.int32, torch.int64)

DEFAULT_POOL = 4
"""Tokens per compressed block -- the QSA compress ratio (``indexer_compress_ratio``; ``QsaSpec.block_size``'s pinned value).  The ONE
definition: ``kernels/qsa_compress.py`` and ``kernels/qsa_prepare.py`` import it (DSL-free here, so a declaration can read it)."""


def check_pool(pool: torch.Tensor, name: str, *, h: int, page_size: int, d: int, dtype, device=None) -> None:
    """One pool against the kernel's contract, typed and naming the operand (no device read): a 4-D ``[P, H, page_size, D]``
    tensor of ``dtype`` with ``D`` contiguous, ``P >= 1``, every other stride a whole number of 16-byte vectors and a 16-byte
    aligned base (a ``st.global.v4`` target), on ``device`` when given.  HND compact and NHD strides both pass."""
    if not isinstance(pool, torch.Tensor):
        raise ValueError(f"{name} must be a [num_pages, {h}, {page_size}, {d}] {dtype} tensor, got {type(pool).__name__}")
    if pool.dtype != dtype:
        raise ValueError(f"{name} must be {dtype} (the activation dtype the K / V rows are written in), got {pool.dtype}")
    if pool.dim() != 4 or tuple(int(x) for x in pool.shape[1:]) != (h, page_size, d) or int(pool.shape[0]) < 1:
        raise ValueError(f"{name} must be [num_pages >= 1, H={h}, page_size={page_size}, D={d}] (HND compact, or NHD by strides), got {tuple(pool.shape)}")
    strides = tuple(int(s) for s in pool.stride())
    if strides[3] != 1:
        raise ValueError(f"{name} must be contiguous along D (stride 1 on the last axis), got strides {strides}")
    es = pool.element_size()
    for ax, s in enumerate(strides[:3]):
        if (s * es) % 16:
            raise ValueError(f"{name} stride[{ax}] = {s} elements ({s * es} bytes) must be a multiple of 16 bytes (each row is written in 16-byte vectors)")
    if device is not None and pool.device != torch.device(device):
        raise ValueError(f"{name} must live on {device}, got {pool.device}")
    if pool.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned (a 16-byte vector-store target), got data_ptr={pool.data_ptr():#x}")


def check_slot_mapping(slot: torch.Tensor, name: str, *, t: int, device=None, t_live: Optional[int] = None) -> None:
    """``slot_mapping`` against the kernel's contract, typed (no device read): a contiguous 1-D int32 / int64 tensor of
    exactly ``T`` entries (one flat slot per token; the VALUES are device data the kernel range-checks), on ``device``.
    ``t_live`` (appended; the block's ``num_tokens_live``): a second admitted count -- exactly ``T_live`` entries, the live tokens'
    slots alone (a serving stack's ``slot_mapping[rows]`` with no padding copy); the launch then reads the first ``T_live`` of a
    ``T``-entry mapping or all of a ``T_live``-entry one."""
    admitted = (t,) if t_live is None else (t, t_live)
    if not isinstance(slot, torch.Tensor):
        raise ValueError(f"{name} must be a [{t}] int32 / int64 tensor (one flat slot per token), got {type(slot).__name__}")
    if slot.dtype not in _SLOT_DTYPES:
        raise ValueError(f"{name} must be int32 or int64, got {slot.dtype}")
    if slot.dim() != 1 or int(slot.numel()) not in admitted:
        want = f"T={t}" if t_live is None else f"T={t} (the declared capacity) or T_live={t_live} (num_tokens_live)"
        raise ValueError(f"{name} must be 1-D with exactly {want} entries (one flat slot per token of this launch), got shape {tuple(slot.shape)}")
    if not slot.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got strides {tuple(slot.stride())}")
    if device is not None and slot.device != torch.device(device):
        raise ValueError(f"{name} must live on {device}, got {slot.device}")


class QsaPrepareIndexArms(NamedTuple):
    """Which indexer arms an artifact traces (every one a compile key), and the band geometry they address.

    ``raw_pool``: the raw indexer key's write-through pool (the cache-write kernel's second launch); ``query``: the
    indexer queries' norm + RoPE into a compact ``[T, H_i, D_i]`` buffer (the compress kernel at pool 1); ``compress``:
    the ring + the compressed-key cache (the compress kernel's step form; ``n_commit`` traces its accepted-prefix load).
    ``heads`` / ``head_dim`` are the band's ``index_heads`` / ``index_head_dim``; ``pool`` the compress ratio.

    Appended (the serving step's forms, each a compile key): ``paged`` -- the compress arm over the stack's PAGED ring and
    compressed-key cache through their block tables with the positions from ``kv_lens`` (needs ``compress``);
    ``ring_slots`` -- the ring WRITE rows from the stack's own slot mapping (needs ``paged``; the slots share the
    artifact's slot dtype); ``rope_table`` -- the Q / K and indexer-query rotations read from the position-indexed
    ``cos_pos`` / ``sin_pos`` at ``pos0 + j`` instead of per-token rows (needs ``compress`` and a rotation)."""

    heads: int
    head_dim: int
    raw_pool: bool = True
    query: bool = False
    compress: bool = False
    n_commit: bool = False
    pool: int = DEFAULT_POOL
    paged: bool = False
    ring_slots: bool = False
    rope_table: bool = False

    @property
    def raw(self) -> bool:
        """Whether the raw-key arm exists at all (any of its three jobs)."""
        return bool(self.raw_pool or self.compress)

    @property
    def any(self) -> bool:
        return bool(self.raw or self.query)
