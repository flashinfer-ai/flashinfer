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

Cake backend of the MiniMax-M3 index-context-parallel (ICP) indexer.

The index keys of a sequence are sharded across the ``world`` ranks of a
tensor-parallel group; every rank holds every index head's query.  Each entry
point here validates tensor metadata on the host (never tensor contents), picks
the program of its role for the device's exact architecture and launches it on
the current stream.  Nothing allocates, synchronizes or reads a device value,
except the eager host plan :func:`index_score_sharded_plan`.

Layouts (``P`` page size, ``W`` world size, ``R = P // W`` local rows per page,
``r`` rank, ``c`` local page column):

* P128 interleaved: local row ``i`` of column ``c`` is global key
  ``128 c + i W + r``; every rank holds part of every B128 block.
* P256, ``W = 2``: rank ``r`` holds the 128 contiguous keys from
  ``256 c + 128 r``, the complete block ``2 c + r``.
* P256, ``W = 4``: rank ``r`` holds the 64 contiguous keys from
  ``256 c + 64 r``; block ``2 c + r // 2`` spans ranks ``{2 (r // 2), 2 (r // 2) + 1}``.

A score is the raw FP32 maximum over the block's visible local keys of
``q . k`` (no scale, NaN ignored, ``-0.0`` written as ``+0.0``, ``-inf`` when
no local key is visible).  Selection ranks scores descending (NaN as
``-inf``, forced blocks first) and breaks exact ties toward the lower global
B128 id.  A Top16 row is 16 raw score words then 16 global ids ascending,
padded with ``(-inf, -1)``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional

import torch

from . import cake_jit

HEAD_DIM = 128
TOPK = 16
# Words of a legacy (int32) row and of a tagged (int64) row.
ROW_WORDS = 2 * TOPK + 1
TAGGED_WORDS = 2 * TOPK
NUM_SLOT_SETS = 3
# A prefill merge runs the prefill program (``merge_prefill``: one exchange,
# no spin).  A legacy decode merge of at least this many tokens runs one token
# per CTA (``merge_legacy``); every other decode merge (tagged rows, fewer
# tokens) runs 4 per CTA (``merge``).
LEGACY_MERGE_MIN_TOKENS = 8
# The merge ranks at most this many B128 blocks per token.
MAX_BLOCKS = 8192
# Upper bound (exclusive) of the score-row width the selectors accept.
MAX_SELECT_COLUMNS = 12288
PLAN_WORKSPACE_BYTES = 32 * 1024 * 1024
SUPPORTED_WORLD_SIZES = (2, 4)
SUPPORTED_PAGE_SIZES = (128, 256)
# v1 scope: the MiniMax-M3 indexer's four index heads.
NUM_INDEX_HEADS = 4
# The decode scorer packs ``H * max_decode_query_len`` query rows per tile.
MAX_DECODE_ROWS = 32
DEFAULT_SPIN_LIMIT = 1 << 26
_INT32_MAX = (1 << 31) - 1
# The prefill plan's item size (its build constant tile_tokens) and the 16-bit
# fields it packs: segment and item indices, columns per item (work_info and
# work_range); the column total is an int32.
PLAN_ITEM_TOKENS = 64
_PLAN_FIELD_MAX = (1 << 16) - 1
_FP8 = torch.float8_e4m3fn


@dataclasses.dataclass(frozen=True)
class MSAIcpScorePlan:
    """A prefill scorer plan: the segment metadata it was built from and the
    caller-owned storage it lives in (``packed_work_range`` /
    ``packed_work_info``; ``workspace`` is accepted for MSA compatibility and
    unused).  ``columns`` is the B128 column bound ``ceil(max_seq_len / 128)``.
    """

    qo_segment_lens: torch.Tensor
    kv_segment_lens: torch.Tensor
    qo_segment_offsets: torch.Tensor
    kv_segment_offsets: torch.Tensor
    qo_offset: torch.Tensor
    packed_work_range: torch.Tensor
    packed_work_info: torch.Tensor
    workspace: torch.Tensor
    total_q: int
    num_qo_heads: int
    batch: int
    max_query_len: int
    columns: int
    uniform: bool = False


# --------------------------------------------------------------------------- checks


def _device_index(t: torch.Tensor) -> int:
    if not t.is_cuda:
        raise ValueError("Cake MSA ICP indexer tensors must be CUDA tensors")
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def _same_device(device: torch.device, **tensors: Optional[torch.Tensor]) -> None:
    for name, t in tensors.items():
        if t is not None and t.device != device:
            raise ValueError(f"{name} is on {t.device}, expected {device}")


def _vector(name: str, t: torch.Tensor, dtype: torch.dtype, size: int) -> None:
    if t.dtype != dtype or t.ndim != 1 or t.numel() != size or not t.is_contiguous():
        raise ValueError(
            f"{name} must be a contiguous 1-D {dtype} tensor of {size} elements"
        )


def _heads(h: int) -> None:
    if h != NUM_INDEX_HEADS:
        raise ValueError(
            f"the Cake MSA ICP indexer supports {NUM_INDEX_HEADS} index heads "
            f"(MiniMax-M3); got {h}"
        )


def _world(world_size: int, rank: int) -> None:
    if world_size not in SUPPORTED_WORLD_SIZES or not 0 <= rank < world_size:
        raise ValueError(
            f"world_size must be 2 or 4 and 0 <= rank < world_size; got {world_size}, {rank}"
        )


def _query(name: str, q: torch.Tensor) -> None:
    if (
        q.dtype != _FP8
        or q.ndim != 3
        or q.shape[2] != HEAD_DIM
        or not q.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be a contiguous float8_e4m3fn [T, H, 128] tensor"
        )
    _heads(q.shape[1])


def _index_k(name: str, k: torch.Tensor, world_size: int, page_size: int) -> int:
    """The rank's K shard ``[pages, P / W, 128]``; returns the local rows per page.

    Rows are dense (``stride(2) == 1``, ``stride(1) == 128``); the page pitch
    ``stride(0)`` may exceed the page, in multiples of 16 bytes.
    """
    rows = page_size // world_size
    if (
        k.dtype != _FP8
        or k.ndim != 3
        or k.shape[1] != rows
        or k.shape[2] != HEAD_DIM
        or k.stride(2) != 1
        or k.stride(1) != HEAD_DIM
        or k.stride(0) < rows * HEAD_DIM
        or k.stride(0) % 16
        or k.data_ptr() % 16
    ):
        raise ValueError(
            f"{name} must be a float8_e4m3fn [pages, {rows}, 128] view with dense rows, "
            "a 16-byte aligned base and a page pitch that is a multiple of 16 bytes"
        )
    return rows


def _score_rows(
    name: str,
    s: torch.Tensor,
    tokens: int,
    heads: int,
    min_cols: int,
    max_cols: Optional[int] = None,
) -> int:
    """FP32 ``[T, H, K]`` rows with ``stride(2) == 1`` and non-overlapping
    rows; returns ``K``."""
    if (
        s.dtype != torch.float32
        or s.ndim != 3
        or s.shape[0] != tokens
        or s.shape[1] != heads
        or s.stride(2) != 1
        or s.stride(1) < s.shape[2]
        or s.stride(0) < heads * s.stride(1)
    ):
        raise ValueError(
            f"{name} must be float32 [{tokens}, {heads}, K] with unit last stride and "
            "non-overlapping rows"
        )
    cols = s.shape[2]
    if cols < max(1, min_cols) or (max_cols is not None and cols >= max_cols):
        bound = f" and below {max_cols}" if max_cols is not None else ""
        raise ValueError(
            f"{name} has {cols} columns; it needs at least {max(1, min_cols)}{bound}"
        )
    return cols


def _block_table(bt: torch.Tensor, batch: int, min_cols: int) -> None:
    if (
        bt.dtype != torch.int32
        or bt.ndim != 2
        or bt.shape[0] != batch
        or bt.shape[1] < min_cols
        or bt.stride(1) != 1
    ):
        raise ValueError(
            f"block_table must be int32 [{batch}, >= {min_cols}] with unit last stride"
        )


def _forced(force_begin_blocks: int, force_end_blocks: int) -> None:
    if (
        force_begin_blocks < 0
        or force_end_blocks < 0
        or force_begin_blocks + force_end_blocks > TOPK
    ):
        raise ValueError(
            "forced begin/end blocks must be non-negative with a sum of at most 16"
        )


def _index_map(index_scale: int, index_offset: int) -> None:
    if index_scale not in (1, 2) or not 0 <= index_offset < index_scale:
        raise ValueError(
            "index_scale must be 1 or 2 and 0 <= index_offset < index_scale"
        )


def _spin(spin_limit: int) -> None:
    if not 0 < spin_limit <= _INT32_MAX:
        raise ValueError("spin_limit must be in (0, 2**31 - 1]")


def _errors(errors: torch.Tensor) -> None:
    if (
        errors.dtype != torch.int32
        or errors.ndim != 1
        or errors.numel() < 1
        or not errors.is_contiguous()
    ):
        raise ValueError(
            "errors must be a contiguous int32 tensor of at least one element"
        )


def index_layout(page_size: int, world_size: int, rank: int) -> tuple[int, int]:
    """``(index_scale, index_offset)`` mapping a local score column to its
    global B128 id: ``(1, 0)`` for P128, ``(2, rank * 2 // world_size)`` for P256."""
    if page_size == 256:
        return 2, rank * 2 // world_size
    return 1, 0


# --------------------------------------------------------------------------- sizing (host only)


def _plan_limits(batch: int, total_q: int, max_seq_len: int) -> None:
    """Reject a window whose plan fields would not fit their packing: the
    segment index and the item count below 65536, at most 65535 B128 columns
    per item and an int32 column total.  Host bounds only (no device read),
    from the caller's upper bounds total_q and max_seq_len."""
    items = min(
        (total_q + PLAN_ITEM_TOKENS - 1) // PLAN_ITEM_TOKENS + batch - 1, total_q
    )
    columns = (max_seq_len + 127) // 128
    if batch > _PLAN_FIELD_MAX + 1 or items > _PLAN_FIELD_MAX:
        raise ValueError(
            f"a prefill window holds at most 65536 segments and 65535 query items of "
            f"{PLAN_ITEM_TOKENS} tokens (got batch {batch}, up to {items} items)"
        )
    if columns > _PLAN_FIELD_MAX or items * columns > _INT32_MAX:
        raise ValueError(
            f"max_seq_len {max_seq_len} gives {columns} B128 columns per item; the plan "
            f"admits at most 65535 per item and an int32 total over {items} items"
        )


def index_score_plan_storage(
    *, num_qo_heads: int, total_q: int, batch: int, device: torch.device
) -> dict[str, int]:
    """Element counts of a prefill plan's caller-owned storage: int64
    ``packed_work_range`` (one per SM), int64 ``packed_work_info`` and the uint8
    ``workspace`` bytes."""
    if num_qo_heads < 1 or total_q < 1 or batch < 1:
        raise ValueError("num_qo_heads, total_q and batch must be positive")
    index = torch.device(device).index
    return {
        "work_range_numel": cake_jit.sm_count(
            torch.cuda.current_device() if index is None else index
        ),
        # The MSA plan formula.  The plan writes one record per item of
        # PLAN_ITEM_TOKENS queries of one segment, all heads folded in: at most
        # (total_q + 63) // 64 + batch - 1 records, which this formula covers
        # at the four heads _heads() admits.
        "work_info_numel": ((total_q + 127) // 128 + batch - 1) * num_qo_heads,
        "workspace_bytes": PLAN_WORKSPACE_BYTES,
    }


def index_score_columns(max_seq_len: int, page_size: int) -> int:
    """Score columns of a prefill row: ``ceil(max_seq_len / 128)`` B128 blocks
    for P128, half of that (rounded up, one column per local page) for the
    compact P256 rows."""
    if page_size not in SUPPORTED_PAGE_SIZES or max_seq_len < 1:
        raise ValueError("page_size must be 128 or 256 and max_seq_len positive")
    blocks = (max_seq_len + 127) // 128
    return (blocks + 1) // 2 if page_size == 256 else blocks


def decode_score_columns(max_model_len: int, page_size: int) -> int:
    """Score columns of a decode row: ``min(8192, ceil(L / (4 P)) * 4)``."""
    if page_size not in SUPPORTED_PAGE_SIZES or max_model_len < 1:
        raise ValueError("page_size must be 128 or 256 and max_model_len positive")
    return min(MAX_BLOCKS, (max_model_len + 4 * page_size - 1) // (4 * page_size) * 4)


def decode_pool_bytes(
    *, max_decode_tokens: int, num_heads: int = NUM_INDEX_HEADS
) -> int:
    """Bytes of the fused decode path's candidate pool: one sorted 16-entry
    list of 64-bit keys per (token, head, split) for up to 128 splits."""
    if max_decode_tokens < 1:
        raise ValueError("max_decode_tokens must be positive")
    _heads(num_heads)
    return max_decode_tokens * num_heads * cake_jit.MAX_DECODE_SPLITS * TOPK * 8


def comm_layout(world_size: int, heads_per_rank: int, capacity: int) -> dict[str, Any]:
    """Layout of the caller's decode receive storage (one symmetric int32
    allocation per rank, zeroed once and never reset).

    ``storage_words`` int32 words hold the legacy rows (int32
    ``legacy_shape = [3, W, hpr, capacity, 33]`` at byte offset 0) followed by
    the tagged rows (int64 ``tagged_shape = [3, W, hpr, capacity, 32]`` at
    ``tagged_offset_bytes``).  A rank's ``recv_bases`` entry ``p`` is peer
    ``p``'s storage address (plus ``tagged_offset_bytes`` for tagged mode).
    ``generations_shape`` / ``expected_shape`` are the int32 publish and merge
    counters (zeroed together with the storage on every rank).  Tagged mode
    requires consecutive uses of one slot set to be at least two sparse layers
    apart, including across steps; legacy mode has no distance requirement.
    """
    _world(world_size, 0)
    if heads_per_rank < 1 or capacity < 1:
        raise ValueError("heads_per_rank and capacity must be positive")
    rows = NUM_SLOT_SETS * world_size * heads_per_rank * capacity
    legacy_words = rows * ROW_WORDS
    tagged_offset_bytes = legacy_words * 4
    if tagged_offset_bytes % 8:
        raise ValueError("the tagged rows would not be 8-byte aligned")
    return {
        "storage_words": legacy_words + rows * 2 * TAGGED_WORDS,
        "legacy_offset_bytes": 0,
        "tagged_offset_bytes": tagged_offset_bytes,
        "legacy_shape": (
            NUM_SLOT_SETS,
            world_size,
            heads_per_rank,
            capacity,
            ROW_WORDS,
        ),
        "tagged_shape": (
            NUM_SLOT_SETS,
            world_size,
            heads_per_rank,
            capacity,
            TAGGED_WORDS,
        ),
        "generations_shape": (NUM_SLOT_SETS, world_size * heads_per_rank, capacity),
        "expected_shape": (NUM_SLOT_SETS, heads_per_rank, capacity),
    }


def decode_splits(
    name: str, batch: int, device_index: int, cap: int = cake_jit.MAX_DECODE_SPLITS
) -> int:
    """Splits per decode request: one wave of the scorer
    (``num_sms * ctas_per_sm // batch``), at least 1 and at most ``cap``
    (128 for the score plane, :data:`cake_jit.TOPK16_SPLIT_CAP` of the
    architecture for the fused path's lists).  Host metadata only."""
    resident = cake_jit.sm_count(device_index) * cake_jit.ctas_per_sm(name)
    return max(1, min(cap, cake_jit.MAX_DECODE_SPLITS, resident // max(batch, 1)))


# The split rule of the score plane on sm_103a at 32 local K rows per page
# (P128 at world 4), whose record runs four CTAs per SM: from
# ``PLANE_R32_LARGE_BATCH`` requests on, a request takes at most
# ``PLANE_R32_LARGE_BATCH_SPLITS`` splits.  One wave there is 38-43 splits at
# batch 14-16; the measured end-to-end decode step (world 4, P128, GB300) is
# faster at 32 splits than at one wave.
# Below that batch, and on every other layout and architecture, the plane
# takes one wave (:func:`decode_splits`).  The plane's scores do not depend on
# the split count.
PLANE_R32_ARCH = "sm_103a"
PLANE_R32_LARGE_BATCH = 14
PLANE_R32_LARGE_BATCH_SPLITS = 32


def plane_splits(name: str, rows: int, arch: str, batch: int, device_index: int) -> int:
    """Splits per request of the score plane record ``name`` (built for
    ``rows`` local K rows per page on ``arch``): one wave
    (:func:`decode_splits`), at most ``PLANE_R32_LARGE_BATCH_SPLITS`` for the
    rows 32 plane on ``PLANE_R32_ARCH`` at ``PLANE_R32_LARGE_BATCH`` or more
    requests.  A function of the record, the architecture, the batch and the
    device's SM count only (host metadata: no device read, no environment)."""
    splits = decode_splits(name, batch, device_index)
    if arch == PLANE_R32_ARCH and rows == 32 and batch >= PLANE_R32_LARGE_BATCH:
        splits = min(splits, PLANE_R32_LARGE_BATCH_SPLITS)
    return splits


# The decode chain of the facade (``MSAIcpIndexerPlan.decode`` without a
# caller ``pool``): (architecture, page size, world size) -> the number of
# requests from which the fused chain (list scorer and list publisher, at most
# ``TOPK16_SPLIT_CAP`` splits) replaces the score plane and the select
# publish, in a pool the facade allocates once.  On sm_107a at P128 (TP4: 32
# and TP2: 64 local K rows per page) the measured end-to-end decode step is
# faster with the fused chain from 8 requests on; below that and at P256 TP4
# the score plane is faster.  Every other layout and architecture keeps the
# score plane.  Both chains publish the same receive rows.
FUSED_DECODE_MIN_BATCH: dict[tuple[str, int, int], int] = {
    ("sm_107a", 128, 4): 8,
    ("sm_107a", 128, 2): 8,
}


def fused_decode_min_batch(arch: str, page_size: int, world_size: int) -> Optional[int]:
    """The number of decode requests from which the facade runs the fused
    chain without a caller ``pool`` on ``arch`` at ``page_size`` /
    ``world_size``, or ``None`` (always the score plane; also when the
    architecture has no fused programs).  Host metadata only (no device read,
    no environment)."""
    batch = FUSED_DECODE_MIN_BATCH.get((arch, page_size, world_size))
    if batch is None or not (
        cake_jit.has_role("decode_topk", arch)
        and cake_jit.has_role("topk16_publish", arch)
    ):
        return None
    return batch


# --------------------------------------------------------------------------- prefill


def index_score_sharded_plan_device(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    qo_segment_offsets: torch.Tensor,
    kv_segment_offsets: torch.Tensor,
    qo_offset: torch.Tensor,
    num_qo_heads: int,
    *,
    total_q: int,
    max_query_len: int,
    max_seq_len: int,
    packed_work_range: torch.Tensor,
    packed_work_info: torch.Tensor,
    workspace: torch.Tensor,
) -> MSAIcpScorePlan:
    batch = qo_segment_lens.numel()
    dev = _device_index(qo_segment_lens)
    device = qo_segment_lens.device
    for name, t, size in (
        ("qo_segment_lens", qo_segment_lens, batch),
        ("kv_segment_lens", kv_segment_lens, batch),
        ("qo_segment_offsets", qo_segment_offsets, batch + 1),
        ("kv_segment_offsets", kv_segment_offsets, batch + 1),
        ("qo_offset", qo_offset, batch),
    ):
        _vector(name, t, torch.int32, size)
    if batch < 1 or total_q < 1 or not 0 < max_query_len <= total_q or max_seq_len < 1:
        raise ValueError("invalid query/sequence bounds")
    _heads(num_qo_heads)
    _plan_limits(batch, total_q, max_seq_len)
    storage = index_score_plan_storage(
        num_qo_heads=num_qo_heads, total_q=total_q, batch=batch, device=device
    )
    _vector(
        "packed_work_range", packed_work_range, torch.int64, storage["work_range_numel"]
    )
    if (
        packed_work_info.dtype != torch.int64
        or packed_work_info.ndim != 1
        or not packed_work_info.is_contiguous()
        or packed_work_info.numel() < storage["work_info_numel"]
    ):
        raise ValueError(
            f"packed_work_info must be contiguous int64 with at least "
            f"{storage['work_info_numel']} elements"
        )
    if (
        workspace.dtype != torch.uint8
        or not workspace.is_contiguous()
        or workspace.numel() < PLAN_WORKSPACE_BYTES
    ):
        raise ValueError(
            "workspace must be a contiguous uint8 tensor of at least 32 MiB"
        )
    _same_device(
        device,
        kv_segment_lens=kv_segment_lens,
        qo_segment_offsets=qo_segment_offsets,
        kv_segment_offsets=kv_segment_offsets,
        qo_offset=qo_offset,
        packed_work_range=packed_work_range,
        packed_work_info=packed_work_info,
        workspace=workspace,
    )
    arch = cake_jit.device_arch(dev)
    plan = MSAIcpScorePlan(
        qo_segment_lens=qo_segment_lens,
        kv_segment_lens=kv_segment_lens,
        qo_segment_offsets=qo_segment_offsets,
        kv_segment_offsets=kv_segment_offsets,
        qo_offset=qo_offset,
        packed_work_range=packed_work_range,
        packed_work_info=packed_work_info,
        workspace=workspace,
        total_q=int(total_q),
        num_qo_heads=int(num_qo_heads),
        batch=batch,
        max_query_len=int(max_query_len),
        columns=(int(max_seq_len) + 127) // 128,
    )
    name = cake_jit.route("plan", None, arch)
    cake_jit.run_module(
        name,
        {
            "qo_lens": qo_segment_lens,
            "kv_lens": kv_segment_lens,
            "qo_indptr": qo_segment_offsets,
            "kv_indptr": kv_segment_offsets,
            "qo_offset": qo_offset,
            "work_range": packed_work_range,
            "work_info": packed_work_info,
            "batch": batch,
            "num_heads": plan.num_qo_heads,
            "num_ctas": packed_work_range.numel(),
            "total_q": plan.total_q,
            "columns": plan.columns,
        },
        grid=cake_jit.launch_grid(name),
        arch=arch,
        device_index=dev,
    )
    return plan


def index_score_sharded_plan(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    num_qo_heads: int,
    *,
    qo_offset: torch.Tensor,
    device: Optional[torch.device] = None,
) -> MSAIcpScorePlan:
    """Eager host plan: reads the CPU segment lengths, allocates the plan
    storage on ``device`` and runs the device plan (not for CUDA Graphs)."""
    lens = [qo_segment_lens, kv_segment_lens, qo_offset]
    if any(
        t.device.type != "cpu" or t.dtype != torch.int32 or t.ndim != 1 for t in lens
    ):
        raise ValueError("host plan segment metadata must be 1-D int32 CPU tensors")
    batch = qo_segment_lens.numel()
    if batch < 1 or kv_segment_lens.numel() != batch or qo_offset.numel() != batch:
        raise ValueError("segment metadata must have one entry per segment")
    if any(bool((t < 0).any()) for t in lens):
        raise ValueError("segment lengths and query offsets must be non-negative")
    if any(int(t.sum(dtype=torch.int64)) > _INT32_MAX for t in lens[:2]):
        raise ValueError(
            "the segment lengths of a window must sum to at most 2**31 - 1"
        )
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None
        else torch.device(device)
    )
    zero = torch.zeros(1, dtype=torch.int32)
    qo_indptr = torch.cat([zero, torch.cumsum(qo_segment_lens, 0, dtype=torch.int32)])
    kv_indptr = torch.cat([zero, torch.cumsum(kv_segment_lens, 0, dtype=torch.int32)])
    total_q = int(qo_indptr[-1])
    storage = index_score_plan_storage(
        num_qo_heads=num_qo_heads, total_q=max(total_q, 1), batch=batch, device=device
    )
    to = dict(device=device)
    return index_score_sharded_plan_device(
        qo_segment_lens.to(**to),
        kv_segment_lens.to(**to),
        qo_indptr.to(**to),
        kv_indptr.to(**to),
        qo_offset.to(**to),
        num_qo_heads,
        total_q=max(total_q, 1),
        max_query_len=max(int(qo_segment_lens.max()), 1),
        max_seq_len=max(int(kv_segment_lens.max()), 1),
        packed_work_range=torch.empty(
            storage["work_range_numel"], dtype=torch.int64, **to
        ),
        packed_work_info=torch.empty(
            storage["work_info_numel"], dtype=torch.int64, **to
        ),
        workspace=torch.empty(storage["workspace_bytes"], dtype=torch.uint8, **to),
    )


def index_score_sharded(
    q: torch.Tensor,
    k: torch.Tensor,
    block_table: torch.Tensor,
    plan: MSAIcpScorePlan,
    *,
    rank: int,
    max_score: torch.Tensor,
    sm_scale: float = 1.0,
    initialize: bool = True,
    tp: int = 2,
    compact: bool = False,
    enable_pdl: bool = False,
) -> None:
    _world(tp, rank)
    _query("q", q)
    if not isinstance(plan, MSAIcpScorePlan):
        raise ValueError(
            "plan must come from msa_icp_index_score_sharded_plan[_device]"
        )
    if plan.total_q != q.shape[0] or plan.num_qo_heads != q.shape[1]:
        raise ValueError("q does not match the plan's total_q / num_qo_heads")
    if not math.isfinite(sm_scale):
        raise ValueError("sm_scale must be finite (scores are raw maxima)")
    page_size = k.shape[1] * tp if k.ndim == 3 else 0
    if page_size not in SUPPORTED_PAGE_SIZES:
        raise ValueError("k must hold 128 / tp or 256 / tp rows per page")
    rows = _index_k("k", k, tp, page_size)
    blocks_per_page = page_size // 128
    if compact and blocks_per_page != 2:
        raise ValueError("compact scores require P256 shards")
    cols = (plan.columns + 1) // 2 if compact else plan.columns
    t, h = q.shape[0], q.shape[1]
    k_cols = _score_rows("max_score", max_score, t, h, cols)
    _block_table(
        block_table, plan.batch, (plan.columns + blocks_per_page - 1) // blocks_per_page
    )
    dev = _device_index(q)
    _same_device(
        q.device,
        k=k,
        block_table=block_table,
        max_score=max_score,
        work_range=plan.packed_work_range,
    )
    arch = cake_jit.device_arch(dev)
    if initialize:
        max_score.fill_(float("-inf"))
    index_scale, index_offset = index_layout(page_size, tp, rank) if compact else (1, 0)
    name = cake_jit.route("score", rows, arch, page=page_size)
    cake_jit.run_module(
        name,
        {
            # The TMA descriptors read FP8 bytes.
            "q": q.view(torch.uint8),
            "k": k.view(torch.uint8),
            "block_table": block_table,
            "qo_lens": plan.qo_segment_lens,
            "kv_lens": plan.kv_segment_lens,
            "qo_indptr": plan.qo_segment_offsets,
            "kv_indptr": plan.kv_segment_offsets,
            "qo_offset": plan.qo_offset,
            "work_range": plan.packed_work_range,
            "work_info": plan.packed_work_info,
            "max_score": max_score,
            "batch": plan.batch,
            "total_q": t,
            "num_tokens": t,
            "num_heads": h,
            "num_ctas": plan.packed_work_range.numel(),
            "columns": plan.columns,
            "score_cols": k_cols,
            "rank": rank,
            "world": tp,
            "page_size": page_size,
            "compact": int(compact),
            "index_scale": index_scale,
            "index_offset": index_offset,
            "bt_stride": block_table.stride(0),
            "score_stride_t": max_score.stride(0),
            "score_stride_h": max_score.stride(1),
        },
        grid=cake_jit.launch_grid(name, sm_count=plan.packed_work_range.numel()),
        arch=arch,
        device_index=dev,
        enable_pdl=enable_pdl,
    )


def topk_select_prefill(
    scores: torch.Tensor,
    num_valid_pages: torch.Tensor,
    carrier: torch.Tensor,
    *,
    force_begin_blocks: int = 0,
    force_end_blocks: int = 0,
    index_scale: int = 1,
    index_offset: int = 0,
) -> None:
    if scores.ndim != 3:
        raise ValueError("scores must be float32 [T, H, K]")
    t, h = scores.shape[0], scores.shape[1]
    _heads(h)
    k_cols = _score_rows("scores", scores, t, h, 1, MAX_SELECT_COLUMNS)
    _vector("num_valid_pages", num_valid_pages, torch.int32, t)
    if (
        carrier.dtype != torch.int32
        or carrier.ndim != 3
        or tuple(carrier.shape) != (h, t, ROW_WORDS)
        or carrier.stride(2) != 1
        or carrier.stride(1) < ROW_WORDS
        or carrier.stride(0) < t * carrier.stride(1)
    ):
        raise ValueError(
            f"carrier must be int32 [{h}, {t}, 33] with unit last stride and non-overlapping rows"
        )
    _forced(force_begin_blocks, force_end_blocks)
    _index_map(index_scale, index_offset)
    dev = _device_index(scores)
    _same_device(scores.device, num_valid_pages=num_valid_pages, carrier=carrier)
    if t == 0:
        return
    arch = cake_jit.device_arch(dev)
    name = cake_jit.route("select_prefill", None, arch)
    cake_jit.run_module(
        name,
        {
            "scores": scores,
            "num_valid_pages": num_valid_pages,
            "carrier": carrier,
            "num_tokens": t,
            "total_q": t,
            "num_heads": h,
            "score_cols": k_cols,
            "score_stride_t": scores.stride(0),
            "score_stride_h": scores.stride(1),
            "carrier_stride_h": carrier.stride(0),
            "carrier_stride_t": carrier.stride(1),
            "force_begin": force_begin_blocks,
            "force_end": force_end_blocks,
            "index_scale": index_scale,
            "index_offset": index_offset,
        },
        grid=cake_jit.launch_grid(name, tokens=t, heads=h),
        arch=arch,
        device_index=dev,
    )


# --------------------------------------------------------------------------- decode


def decode_index_score_sharded(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    score_out: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    page_size: int,
    decode_query_len: int,
    max_decode_query_len: int,
    enable_pdl: bool = False,
) -> None:
    _world(world_size, rank)
    if page_size not in SUPPORTED_PAGE_SIZES:
        raise ValueError("page_size must be 128 or 256")
    _query("index_q", index_q)
    rows = _index_k("index_k", index_k, world_size, page_size)
    t, h = index_q.shape[0], index_q.shape[1]
    batch = block_table.shape[0] if block_table.ndim == 2 else -1
    _block_table(block_table, batch, 1)
    _vector("seq_lens", seq_lens, torch.int32, batch)
    if (
        not 1 <= decode_query_len <= max_decode_query_len
        or h * max_decode_query_len > MAX_DECODE_ROWS
    ):
        raise ValueError(
            "need 1 <= decode_query_len <= max_decode_query_len and "
            f"num_heads * max_decode_query_len <= {MAX_DECODE_ROWS}"
        )
    if t != batch * decode_query_len:
        raise ValueError("index_q must hold batch * decode_query_len tokens")
    cols = _score_rows("score_out", score_out, t, h, 1, MAX_BLOCKS + 1)
    if cols % 4:
        raise ValueError("score_out columns must be a multiple of 4")
    dev = _device_index(index_q)
    _same_device(
        index_q.device,
        index_k=index_k,
        block_table=block_table,
        seq_lens=seq_lens,
        score_out=score_out,
    )
    if batch == 0:
        return
    arch = cake_jit.device_arch(dev)
    name = cake_jit.route("decode_plane", rows, arch)
    splits = plane_splits(name, rows, arch, batch, dev)
    cake_jit.run_module(
        name,
        {
            "q": index_q.view(torch.uint8),
            "k": index_k.view(torch.uint8),
            "block_table": block_table,
            "seq_lens": seq_lens,
            "scores": score_out,
            # Read only by the list form; any int32 vector serves the plane.
            "num_valid_pages": seq_lens,
            "columns": cols,
            "init_blocks": 0,
            "local_blocks": 0,
            "batch": batch,
            "num_tokens": t,
            "num_heads": h,
            "query_len": decode_query_len,
            "splits": splits,
            "score_cols": cols,
            "rank": rank,
            "world": world_size,
            "page_size": page_size,
            "bt_stride": block_table.stride(0),
            "score_stride_t": score_out.stride(0),
            "score_stride_h": score_out.stride(1),
        },
        grid=cake_jit.launch_grid(name, batch=batch, splits=splits),
        arch=arch,
        device_index=dev,
        enable_pdl=enable_pdl,
    )


def _publish_checks(
    num_tokens: int,
    num_heads: int,
    num_valid_pages: torch.Tensor,
    recv_bases: torch.Tensor,
    errors: torch.Tensor,
    generations: Optional[torch.Tensor],
    *,
    slot: int,
    rank: int,
    world_size: int,
    heads_per_rank: int,
    capacity: int,
    force_begin_blocks: int,
    force_end_blocks: int,
    spin_limit: int,
    index_scale: int,
    index_offset: int,
) -> None:
    _world(world_size, rank)
    if world_size * heads_per_rank != num_heads:
        raise ValueError("scores must hold world_size * heads_per_rank heads")
    _heads(num_heads)
    if slot not in range(NUM_SLOT_SETS):
        raise ValueError("slot must be 0, 1 or 2")
    if not 0 <= num_tokens <= capacity:
        raise ValueError("the token count exceeds the receive capacity")
    _vector("num_valid_pages", num_valid_pages, torch.int32, num_tokens)
    _vector("recv_bases", recv_bases, torch.int64, world_size)
    _errors(errors)
    if generations is not None and (
        generations.dtype != torch.int32
        or tuple(generations.shape) != (NUM_SLOT_SETS, num_heads, capacity)
        or not generations.is_contiguous()
    ):
        raise ValueError(
            f"generations must be contiguous int32 [3, {num_heads}, {capacity}]"
        )
    _forced(force_begin_blocks, force_end_blocks)
    _spin(spin_limit)
    _index_map(index_scale, index_offset)


def topk_select_publish(
    scores: torch.Tensor,
    num_valid_pages: torch.Tensor,
    recv_bases: torch.Tensor,
    errors: torch.Tensor,
    *,
    slot: int,
    rank: int,
    world_size: int,
    heads_per_rank: int,
    capacity: int,
    force_begin_blocks: int,
    force_end_blocks: int,
    spin_limit: int = DEFAULT_SPIN_LIMIT,
    enable_pdl: bool = False,
    generations: Optional[torch.Tensor] = None,
    index_scale: int = 1,
    index_offset: int = 0,
    _trusted: bool = False,
) -> None:
    # ``_trusted``: receive bases, errors and generations come from a comm the
    # caller's ``check_comm`` validated, the scalars from its validated plan
    # and ``slot`` is in range; the per-call tensors are still checked.
    if scores.ndim != 3:
        raise ValueError("scores must be float32 [T, H, K]")
    t, h = scores.shape[0], scores.shape[1]
    k_cols = _score_rows("scores", scores, t, h, 1, MAX_SELECT_COLUMNS)
    dev = _device_index(scores)
    if _trusted:
        _heads(h)
        if t > capacity:
            raise ValueError("the token count exceeds the receive capacity")
        _vector("num_valid_pages", num_valid_pages, torch.int32, t)
        _same_device(
            scores.device, num_valid_pages=num_valid_pages, recv_bases=recv_bases
        )
    else:
        _publish_checks(
            t,
            h,
            num_valid_pages,
            recv_bases,
            errors,
            generations,
            slot=slot,
            rank=rank,
            world_size=world_size,
            heads_per_rank=heads_per_rank,
            capacity=capacity,
            force_begin_blocks=force_begin_blocks,
            force_end_blocks=force_end_blocks,
            spin_limit=spin_limit,
            index_scale=index_scale,
            index_offset=index_offset,
        )
        _same_device(
            scores.device,
            num_valid_pages=num_valid_pages,
            recv_bases=recv_bases,
            errors=errors,
            generations=generations,
        )
    if t == 0:
        return
    arch = cake_jit.device_arch(dev)
    name = cake_jit.route("select_publish", None, arch)
    cake_jit.run_module(
        name,
        {
            "scores": scores,
            "num_valid_pages": num_valid_pages,
            "recv_bases": recv_bases,
            "errors": errors,
            # The legacy kernel never touches it; any int32 pointer serves.
            "generations": errors if generations is None else generations,
            "num_tokens": t,
            "total_q": t,
            "num_heads": h,
            "heads_per_rank": heads_per_rank,
            "score_cols": k_cols,
            "score_stride_t": scores.stride(0),
            "score_stride_h": scores.stride(1),
            "slot": slot,
            "rank": rank,
            "world": world_size,
            "capacity": capacity,
            "force_begin": force_begin_blocks,
            "force_end": force_end_blocks,
            "spin_limit": spin_limit,
            "tagged": int(generations is not None),
            "index_scale": index_scale,
            "index_offset": index_offset,
        },
        grid=cake_jit.launch_grid(name, tokens=t, heads=h),
        arch=arch,
        device_index=dev,
        enable_pdl=enable_pdl,
    )


def decode_score_publish(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_valid_pages: torch.Tensor,
    pool: torch.Tensor,
    recv_bases: torch.Tensor,
    errors: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    page_size: int,
    decode_query_len: int,
    slot: int,
    capacity: int,
    force_begin_blocks: int,
    force_end_blocks: int,
    spin_limit: int = DEFAULT_SPIN_LIMIT,
    generations: Optional[torch.Tensor] = None,
    enable_pdl: bool = True,
) -> None:
    _world(world_size, rank)
    if page_size not in SUPPORTED_PAGE_SIZES:
        raise ValueError("page_size must be 128 or 256")
    _query("index_q", index_q)
    rows = _index_k("index_k", index_k, world_size, page_size)
    t, h = index_q.shape[0], index_q.shape[1]
    batch = block_table.shape[0] if block_table.ndim == 2 else -1
    _block_table(block_table, batch, 1)
    _vector("seq_lens", seq_lens, torch.int32, batch)
    if decode_query_len < 1 or h * decode_query_len > MAX_DECODE_ROWS:
        raise ValueError(
            f"need decode_query_len >= 1 and num_heads * decode_query_len <= {MAX_DECODE_ROWS}"
        )
    if t != batch * decode_query_len:
        raise ValueError("index_q must hold batch * decode_query_len tokens")
    if (
        pool.dtype != torch.uint8
        or not pool.is_contiguous()
        or pool.numel() < decode_pool_bytes(max_decode_tokens=max(t, 1), num_heads=h)
        or pool.numel() % 8
        or pool.data_ptr() % 16
    ):
        raise ValueError(
            "pool must be contiguous 16-byte aligned uint8 of a multiple of 8 and at least "
            "msa_icp_decode_pool_bytes bytes"
        )
    index_scale, index_offset = index_layout(page_size, world_size, rank)
    _publish_checks(
        t,
        h,
        num_valid_pages,
        recv_bases,
        errors,
        generations,
        slot=slot,
        rank=rank,
        world_size=world_size,
        heads_per_rank=h // world_size,
        capacity=capacity,
        force_begin_blocks=force_begin_blocks,
        force_end_blocks=force_end_blocks,
        spin_limit=spin_limit,
        index_scale=index_scale,
        index_offset=index_offset,
    )
    dev = _device_index(index_q)
    _same_device(
        index_q.device,
        index_k=index_k,
        block_table=block_table,
        seq_lens=seq_lens,
        num_valid_pages=num_valid_pages,
        pool=pool,
        recv_bases=recv_bases,
        errors=errors,
        generations=generations,
    )
    if batch == 0:
        return
    arch = cake_jit.device_arch(dev)
    scorer = cake_jit.route("decode_topk", rows, arch)
    publisher = cake_jit.route("topk16_publish", None, arch)
    splits = decode_splits(scorer, batch, dev, cap=cake_jit.TOPK16_SPLIT_CAP[arch])
    values = {
        "q": index_q.view(torch.uint8),
        "k": index_k.view(torch.uint8),
        "block_table": block_table,
        "seq_lens": seq_lens,
        "num_valid_pages": num_valid_pages,
        # uint64 keys: one 16-entry list per (token, head, split), rows of
        # MAX_DECODE_SPLITS lists.
        "pool": pool.view(torch.uint64),
        "pool_row_stride": cake_jit.MAX_DECODE_SPLITS * TOPK,
        "recv_bases": recv_bases,
        "errors": errors,
        "generations": errors if generations is None else generations,
        "batch": batch,
        "num_tokens": t,
        "total_q": t,
        "num_heads": h,
        "heads_per_rank": h // world_size,
        "query_len": decode_query_len,
        "splits": splits,
        "rank": rank,
        "world": world_size,
        "page_size": page_size,
        "bt_stride": block_table.stride(0),
        "slot": slot,
        "capacity": capacity,
        "force_begin": force_begin_blocks,
        "force_end": force_end_blocks,
        # The scorer's names of the forced blocks and of the candidate column
        # bound (one local column per block-table page).
        "init_blocks": force_begin_blocks,
        "local_blocks": force_end_blocks,
        "columns": block_table.shape[1],
        "spin_limit": spin_limit,
        "tagged": int(generations is not None),
        "index_scale": index_scale,
        "index_offset": index_offset,
    }
    cake_jit.run_module(
        scorer,
        values,
        grid=cake_jit.launch_grid(scorer, batch=batch, splits=splits),
        arch=arch,
        device_index=dev,
        enable_pdl=enable_pdl,
    )
    cake_jit.run_module(
        publisher,
        values,
        grid=cake_jit.launch_grid(publisher, tokens=t, heads=h),
        arch=arch,
        device_index=dev,
        enable_pdl=enable_pdl,
    )


def merge_role(*, prefill: bool, tagged: bool, num_tokens: int) -> str:
    """The merge role of one call (launch metadata only, every arch):
    ``merge_prefill`` for a prefill merge, ``merge_legacy`` for a legacy decode
    merge of at least ``LEGACY_MERGE_MIN_TOKENS`` tokens, else ``merge``."""
    if prefill:
        return "merge_prefill"
    if not tagged and num_tokens >= LEGACY_MERGE_MIN_TOKENS:
        return "merge_legacy"
    return "merge"


def merge_candidates(
    recv_slot: torch.Tensor,
    num_valid_pages: torch.Tensor,
    output: torch.Tensor,
    errors: torch.Tensor,
    *,
    init_blocks: int,
    local_blocks: int,
    spin_limit: int = DEFAULT_SPIN_LIMIT,
    enable_pdl: bool = False,
    expected_generations: Optional[torch.Tensor] = None,
    prefill: bool = False,
    _trusted: bool = False,
) -> None:
    # ``_trusted``: the caller (MSAIcpIndexerPlan.decode) passes one slot of
    # receive rows and counters its ``check_comm`` validated, and its own
    # validated scalars; only the per-call tensors are checked here.
    tagged = expected_generations is not None
    if not _trusted:
        if prefill and (tagged or enable_pdl):
            raise ValueError(
                "the prefill merge takes neither expected_generations nor enable_pdl"
            )
        if recv_slot.ndim != 4:
            raise ValueError("recv_slot must be [world, heads_per_rank, capacity, row]")
    world, hpr, capacity, row = recv_slot.shape
    if not _trusted:
        if world not in SUPPORTED_WORLD_SIZES:
            raise ValueError("recv_slot must hold 2 or 4 sources")
        want_dtype, want_row = (
            (torch.int64, TAGGED_WORDS) if tagged else (torch.int32, ROW_WORDS)
        )
        if (
            recv_slot.dtype != want_dtype
            or row != want_row
            or recv_slot.stride(-1) != 1
        ):
            raise ValueError(
                f"recv_slot must be {want_dtype} [..., {want_row}] with unit last stride"
                + (" (tagged rows)" if tagged else " (legacy rows)")
            )
        _heads(world * hpr)
    if (
        output.dtype != torch.int32
        or output.ndim != 3
        or output.shape[1] != hpr
        or output.shape[2] != TOPK
        or output.stride(-1) != 1
    ):
        raise ValueError(f"output must be int32 [T, {hpr}, 16] with unit last stride")
    t = output.shape[0]
    if t > capacity:
        raise ValueError("output holds more tokens than the receive capacity")
    _vector("num_valid_pages", num_valid_pages, torch.int32, t)
    dev = _device_index(recv_slot)
    if _trusted:
        _same_device(recv_slot.device, num_valid_pages=num_valid_pages, output=output)
    else:
        _errors(errors)
        if tagged and (
            expected_generations.dtype != torch.int32
            or tuple(expected_generations.shape) != (hpr, capacity)
            or not expected_generations.is_contiguous()
        ):
            raise ValueError(
                f"expected_generations must be contiguous int32 [{hpr}, {capacity}]"
            )
        if init_blocks < 0 or local_blocks < 0:
            raise ValueError("init_blocks and local_blocks must be non-negative")
        _spin(spin_limit)
        _same_device(
            recv_slot.device,
            num_valid_pages=num_valid_pages,
            output=output,
            errors=errors,
            expected_generations=expected_generations,
        )
    if t == 0:
        return
    arch = cake_jit.device_arch(dev)
    role = merge_role(prefill=prefill, tagged=tagged, num_tokens=t)
    name = cake_jit.route(role, None, arch)
    # One program reads both row forms through int32 words: a tagged row is
    # its 32 int64 words viewed as 64 int32 words (strides in int32 elements).
    words = recv_slot.view(torch.int32) if tagged else recv_slot
    cake_jit.run_module(
        name,
        {
            "recv": words,
            "num_valid_pages": num_valid_pages,
            "output": output,
            "errors": errors,
            # Read only when tagged; any int32 pointer serves otherwise.
            "expected": num_valid_pages
            if expected_generations is None
            else expected_generations,
            "num_tokens": t,
            "heads_per_rank": hpr,
            "world": world,
            "capacity": capacity,
            "recv_stride_src": words.stride(0),
            "recv_stride_h": words.stride(1),
            "recv_stride_t": words.stride(2),
            "out_stride_t": output.stride(0),
            "out_stride_h": output.stride(1),
            "init_blocks": init_blocks,
            "local_blocks": local_blocks,
            "spin_limit": spin_limit,
            "tagged": int(tagged),
            "prefill": int(prefill),
        },
        grid=cake_jit.launch_grid(name, tokens=t, heads=hpr),
        arch=arch,
        device_index=dev,
        enable_pdl=enable_pdl,
    )


def warmup(device: torch.device) -> None:
    """Load (build on first use) every program of the device's architecture,
    each kernel resident in the device's context (every role a call may
    launch: plan, scorers, select / publish, merges).  Run on every rank
    before the first exchange and before CUDA Graph capture."""
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    arch = cake_jit.device_arch(index)
    with torch.cuda.device(index):
        for name, record in cake_jit.MODULES.items():
            if arch in record["arches"]:
                cake_jit.load(name, arch, index)


def is_supported(device: torch.device | str | int | None = None) -> bool:
    """Whether ``device`` is an exported architecture of this backend."""
    if not torch.cuda.is_available():
        return False
    index = (
        torch.device("cuda", device).index
        if isinstance(device, int)
        else (
            torch.cuda.current_device()
            if device is None or torch.device(device).index is None
            else torch.device(device).index
        )
    )
    return (
        cake_jit.arch_for_capability(torch.cuda.get_device_capability(index))
        is not None
    )


__all__ = [
    "MSAIcpScorePlan",
    "comm_layout",
    "decode_index_score_sharded",
    "decode_pool_bytes",
    "decode_score_columns",
    "decode_score_publish",
    "decode_splits",
    "fused_decode_min_batch",
    "index_layout",
    "index_score_columns",
    "index_score_plan_storage",
    "index_score_sharded",
    "index_score_sharded_plan",
    "index_score_sharded_plan_device",
    "is_supported",
    "merge_candidates",
    "merge_role",
    "plane_splits",
    "topk_select_prefill",
    "topk_select_publish",
    "warmup",
]
