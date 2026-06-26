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
