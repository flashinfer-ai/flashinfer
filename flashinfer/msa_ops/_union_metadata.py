"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

Reference metadata builder for the union-tile sparse-attention prefill
(:mod:`sparse_fwd_union_sm12x`). Groups each sequence's queries into tiles of
``tokens_per_tile`` and, per (batch, q-tile, kv-head), forms the **union** of the
KV blocks those tokens selected together with a per-(union-block)
``tokens_per_tile``-bit membership mask (bit ``i`` set iff tile-token ``i``
selected that block). The union kernel walks this list with online softmax, so a
token attends only the blocks it picked.

This is a correctness-first host builder (the q2k_indices it consumes are tiny --
``total_q x topk`` ints). An on-device CuTe-DSL builder mirroring
``build_k2q_csr_sm12x`` is the production follow-up.
"""

from typing import Tuple

import torch


def build_msa_union_metadata(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    tokens_per_tile: int,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build the union-tile work list from per-query top-k block selections.

    Parameters
    ----------
    q2k_indices : torch.Tensor
        ``(num_kv_heads, total_q, topk)`` int32, batch-local KV-block ids per query
        (ascending, ``-1`` padded), shared across the kv-head's GQA query heads.
    cu_seqlens_q : torch.Tensor
        ``(batch_size + 1,)`` int32 cumulative query lengths.
    tokens_per_tile : int
        Query tokens per CTA tile (``m_block // group_size``).
    topk : int
        Per-query selection width (bounds ``max_union = tokens_per_tile * topk``).

    Returns
    -------
    union_blocks, union_masks : torch.Tensor
        ``(work_items, max_union)`` int32: the union's KV-block ids and the
        ``tokens_per_tile``-bit per-block membership masks (zero-padded past
        ``union_count``).
    union_count : torch.Tensor
        ``(work_items,)`` int32, blocks in each union.
    work_meta : torch.Tensor
        ``(work_items, 3)`` int32 ``{batch_idx, q_tile_idx, kv_head}``.
    work_items : int
        Number of emitted work items.
    """
    num_kv_heads, total_q, topk_in = q2k_indices.shape
    if topk_in != topk:
        raise ValueError(f"q2k_indices topk {topk_in} != {topk}")
    max_union = tokens_per_tile * topk
    q2k = q2k_indices.cpu().tolist()
    cuq = cu_seqlens_q.cpu().tolist()
    batch_size = len(cuq) - 1

    blocks, masks, counts, meta = [], [], [], []
    for b in range(batch_size):
        qs, qe = cuq[b], cuq[b + 1]
        seqlen_q = qe - qs
        n_tiles = (seqlen_q + tokens_per_tile - 1) // tokens_per_tile
        for h in range(num_kv_heads):
            for t in range(n_tiles):
                union: dict = {}
                for i in range(tokens_per_tile):
                    ql = t * tokens_per_tile + i
                    if ql >= seqlen_q:
                        break
                    for bid in q2k[h][qs + ql]:
                        if bid < 0:
                            continue
                        union[bid] = union.get(bid, 0) | (1 << i)
                if not union:
                    continue
                items = sorted(union.items())
                bl = [bid for bid, _ in items]
                mk = [m for _, m in items]
                counts.append(len(bl))
                blocks.append(bl + [0] * (max_union - len(bl)))
                masks.append(mk + [0] * (max_union - len(mk)))
                meta.append([b, t, h])

    dev = q2k_indices.device
    n = len(counts)
    return (
        torch.tensor(blocks, dtype=torch.int32, device=dev).reshape(n, max_union),
        torch.tensor(masks, dtype=torch.int32, device=dev).reshape(n, max_union),
        torch.tensor(counts, dtype=torch.int32, device=dev),
        torch.tensor(meta, dtype=torch.int32, device=dev).reshape(n, 3),
        n,
    )


# Compiled CuTe-DSL union builders, keyed by (topk, tokens_per_tile).
_union_compile_cache: dict = {}


def _get_compiled_union(topk: int, tokens_per_tile: int):
    import cutlass
    import cutlass.cute as cute

    from .cute_dsl.build_union_meta_sm12x import BuildUnionMetaSm12x
    from .sparse_index_utils import _fake_i32

    key = (topk, tokens_per_tile)
    compiled = _union_compile_cache.get(key)
    if compiled is not None:
        return compiled

    kernel_obj = BuildUnionMetaSm12x(topk=topk, tokens_per_tile=tokens_per_tile)
    compiled = cute.compile(
        kernel_obj,
        _fake_i32(3),  # q2k
        _fake_i32(1),  # tile_batch
        _fake_i32(1),  # tile_t
        _fake_i32(1),  # tile_qbase
        _fake_i32(1),  # tile_ntok
        _fake_i32(2),  # union_blocks (out)
        _fake_i32(2),  # union_masks (out)
        _fake_i32(1),  # union_count (out)
        _fake_i32(2),  # work_meta (out)
        cutlass.Int32(1),  # H
        cutlass.Int32(1),  # total_tiles
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _union_compile_cache[key] = compiled
    return compiled


def build_msa_union_metadata_device(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    tokens_per_tile: int,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """On-device equivalent of :func:`build_msa_union_metadata`.

    Produces the same five outputs, but keeps the ``(num_kv_heads, total_q,
    topk)`` selection on the GPU -- only ``cu_seqlens_q`` (a tiny ``(batch +
    1,)`` tensor) is read host-side to lay out the work items, so there is no
    host copy of the large q2k tensor (the production win over the reference
    builder). A CuTe-DSL warp-per-work-item k-way merge (:class:`...cute_dsl.
    build_union_meta_sm12x.BuildUnionMetaSm12x`) forms each union on device.

    The emitted blocks are ascending (k-way merge of ascending lists), matching
    the reference. Work items are enumerated over **all** (batch, q-tile,
    kv-head) -- no empty-union compaction (a tile spanning valid queries always
    selects at least one block), so the count is statically ``total_tiles *
    num_kv_heads``; the order differs from the reference but the forward scatters
    by ``work_meta`` so order is immaterial.
    """
    num_kv_heads, total_q, topk_in = q2k_indices.shape
    if topk_in != topk:
        raise ValueError(f"q2k_indices topk {topk_in} != {topk}")
    dev = q2k_indices.device
    max_union = tokens_per_tile * topk

    # Work-item geometry from cu_seqlens alone: one tile per ``tokens_per_tile``
    # queries per batch, for every kv-head.
    cu = cu_seqlens_q.detach().to(device="cpu", dtype=torch.int64)
    seqlens = cu[1:] - cu[:-1]
    batch_size = seqlens.numel()
    tpt = tokens_per_tile
    ntiles_per_batch = (seqlens + tpt - 1) // tpt
    total_tiles = int(ntiles_per_batch.sum())
    work_items = total_tiles * num_kv_heads

    if work_items == 0:
        zeros2 = torch.empty((0, max_union), dtype=torch.int32, device=dev)
        return (
            zeros2,
            zeros2.clone(),
            torch.empty((0,), dtype=torch.int32, device=dev),
            torch.empty((0, 3), dtype=torch.int32, device=dev),
            0,
        )

    # Per-global-tile (batch, within-batch tile idx, query base, valid tokens).
    tile_batch = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.int64), ntiles_per_batch
    )
    tile_base = torch.zeros(batch_size, dtype=torch.int64)
    tile_base[1:] = ntiles_per_batch.cumsum(0)[:-1]
    tile_t = torch.arange(total_tiles, dtype=torch.int64) - tile_base[tile_batch]
    tile_qbase = cu[:-1][tile_batch] + tile_t * tpt
    tile_ntok = torch.clamp(seqlens[tile_batch] - tile_t * tpt, max=tpt)

    def _to_dev(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype=torch.int32, device=dev)

    union_blocks = torch.empty((work_items, max_union), dtype=torch.int32, device=dev)
    union_masks = torch.empty((work_items, max_union), dtype=torch.int32, device=dev)
    union_count = torch.empty((work_items,), dtype=torch.int32, device=dev)
    work_meta = torch.empty((work_items, 3), dtype=torch.int32, device=dev)

    compiled = _get_compiled_union(topk, tpt)
    compiled(
        q2k_indices,
        _to_dev(tile_batch),
        _to_dev(tile_t),
        _to_dev(tile_qbase),
        _to_dev(tile_ntok),
        union_blocks,
        union_masks,
        union_count,
        work_meta,
        int(num_kv_heads),
        int(total_tiles),
    )
    return union_blocks, union_masks, union_count, work_meta, work_items
