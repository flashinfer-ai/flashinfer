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

This host builder is the correctness-first test oracle;
``build_msa_union_metadata_device`` is the on-device CuTe-DSL builder the forward
path calls.
"""

from typing import Tuple

import torch


def _fake_i32(ndim):
    """Fake compact int32 tensor with ``ndim`` symbolic dims, for TVM-FFI compile."""
    import cutlass
    import cutlass.cute as cute

    from .sparse_attention import _fake

    return _fake(cutlass.Int32, tuple(cute.sym_int() for _ in range(ndim)), align=4)


def build_msa_union_metadata(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    tokens_per_tile: int,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Host reference union builder (test oracle). From per-query top-k selections
    ``q2k_indices`` ``(num_kv_heads, total_q, topk)`` and ``cu_seqlens_q``, returns
    ``(union_blocks, union_masks, union_count, work_meta, work_items)``: per
    (batch, q-tile, kv-head) work item, the union's KV-block ids, the
    ``tokens_per_tile``-bit membership masks, the block count, and
    ``{batch_idx, q_tile_idx, kv_head}``."""
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
    # masks built as int64 (bit 31 -> 2**31 overflows int32 at construction), then
    # cast to int32 to keep the kernel's 32-bit membership bit pattern.
    return (
        torch.tensor(blocks, dtype=torch.int32, device=dev).reshape(n, max_union),
        torch.tensor(masks, dtype=torch.int64, device=dev)
        .reshape(n, max_union)
        .to(torch.int32),
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
    """On-device equivalent of :func:`build_msa_union_metadata`, called by the
    forward path (the host reference is the test oracle).

    Same five outputs (work-item order differs but the forward scatters by
    ``work_meta``). Keeps the q2k selection on the GPU and is CUDA-graph
    capturable: tensors are sized to the static tile bound ``ceil(total_q / tpt)
    + batch_size`` and the per-tile geometry is derived on device from
    ``cu_seqlens`` (a searchsorted), so no host copy/sync. Padding slots get
    ``tile_ntok == 0`` -> ``union_count == 0`` and are skipped by the forward.
    """
    num_kv_heads, total_q, topk_in = q2k_indices.shape
    if topk_in != topk:
        raise ValueError(f"q2k_indices topk {topk_in} != {topk}")
    dev = q2k_indices.device
    max_union = tokens_per_tile * topk
    tpt = tokens_per_tile
    batch_size = cu_seqlens_q.shape[0] - 1

    # Static upper bound on the (batch, q-tile) count (shapes only, so capturable):
    # sum_b ceil(sq_b / tpt) <= ceil(total_q / tpt) + batch_size. Extra slots empty.
    total_tiles = (total_q + tpt - 1) // tpt + batch_size
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

    # On-device per-tile geometry (no cu_seqlens host copy -> capture-safe): map each
    # flat tile index to its batch via searchsorted over the per-batch tile cumsum.
    # Padding indices clamp to the last batch with tile_ntok -> 0.
    cu = cu_seqlens_q.to(dtype=torch.int64)
    seqlens = cu[1:] - cu[:-1]
    ntiles = (seqlens + tpt - 1) // tpt
    offsets = torch.cumsum(ntiles, 0)  # exclusive-end tile index per batch
    t = torch.arange(total_tiles, dtype=torch.int64, device=dev)
    tile_batch = torch.clamp(
        torch.searchsorted(offsets, t, right=True), max=batch_size - 1
    )
    tile_t = t - (offsets - ntiles)[tile_batch]  # within-batch tile index
    tile_qbase = cu[:-1][tile_batch] + tile_t * tpt
    tile_ntok = torch.clamp(seqlens[tile_batch] - tile_t * tpt, min=0, max=tpt)

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
