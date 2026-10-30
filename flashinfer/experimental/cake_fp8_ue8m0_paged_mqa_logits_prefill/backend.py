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
"""

# Launch adapter for the Cake-generated CuTe DSL programs under csrc/: DSA
# indexer prefill logits over an FP8 paged key cache whose per-token scales
# are UE8M0 exponents, and the host-side work planner. One program per
# architecture, compiled for the device's exact architecture through
# FlashInfer's CuTe DSL JIT cache. The programs import only CUTLASS's CuTe DSL
# and cuda-python.

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tvm_ffi

from ...jit import cute_dsl_core
from ...utils import get_compute_capability, get_device_sm_count
from ..cake_fp8_ue8m0_paged_mqa_logits_common import (
    SUPPORTED_COMPUTE_CAPABILITIES,
    check_launcher,
    import_program,
    on_device,
    program_arch,
    program_files,
)

HEADS = 32
HEAD_DIM = 128
PAGE = 64  # tokens per KV block
TOKEN_BYTES = HEAD_DIM + 4  # a token's FP8 values plus its 4-byte scale slot
BLOCK_BYTES = PAGE * TOKEN_BYTES
ROWS = 8  # rows per work item: 4 per math warpgroup
TILE_KEYS = 128  # keys per tile, scored by both math warpgroups
ITEM_FIELDS = 8
# A work item: first row, block-table row, the first row's visible key count,
# rows (1..8), first tile, end tile, padding.
_MODULE = "cake_fp8_ue8m0_paged_mqa_logits_prefill"
_CSRC = Path(__file__).resolve().parent / "csrc"
# Items per CTA the planner aims for, and the fewest tiles per item.
_ITEMS_PER_CTA = 32
_MIN_SEGMENT_TILES = 16
# The generated launcher's parameters, in the order `_launch` binds them; its
# stream is the TVM-FFI environment stream.
_LAUNCH_PARAMETERS = (
    "q_tma",
    "_loom_tma_q_tma_dim_0",
    "_loom_tma_q_tma_dim_1",
    "_loom_tma_q_tma_stride16_0",
    "w_tma",
    "_loom_tma_w_tma_dim_0",
    "_loom_tma_w_tma_dim_1",
    "_loom_tma_w_tma_stride16_0",
    "kv_tma",
    "_loom_tma_kv_tma_dim_0",
    "_loom_tma_kv_tma_dim_1",
    "_loom_tma_kv_tma_dim_2",
    "_loom_tma_kv_tma_stride16_0",
    "_loom_tma_kv_tma_stride16_1",
    "sf_tma",
    "_loom_tma_sf_tma_dim_0",
    "_loom_tma_sf_tma_dim_1",
    "_loom_tma_sf_tma_stride16_0",
    "block_table",
    "items",
    "out",
    "block_table_stride",
    "num_items",
    "out_stride",
    "write",
    "grid_x",
    "grid_y",
    "grid_z",
    "stream",
)


def _source(arch: str) -> Path:
    return _CSRC / f"dsa_indexer_prefill_logits_{arch}.py"


@functools.cache
def _kernel(device_index: int) -> Any:
    path = _source(program_arch(device_index))
    module = import_program(path, __name__)
    check_launcher(path, module.launch_dsa_indexer_prefill_logits, _LAUNCH_PARAMETERS)
    with torch.cuda.device(device_index):
        return cute_dsl_core.build_and_load_cute_dsl_kernel(
            _MODULE,
            "dsa_indexer_prefill_logits",
            module.compile_program,
            extra_key_files=program_files(_CSRC, "dsa_indexer_prefill_logits_*.py"),
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"fp8_ue8m0_paged_mqa_logits_prefill: {message}")


def _host_ints(values: Sequence[int] | np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        _require(not values.is_cuda, "request metadata must be on the host")
        values = values.numpy()
    return np.asarray(values, dtype=np.int64).reshape(-1)


def plan_work(
    row_offsets: Sequence[int] | np.ndarray | torch.Tensor,
    row_counts: Sequence[int] | np.ndarray | torch.Tensor,
    prefixes: Sequence[int] | np.ndarray | torch.Tensor,
    sms: int,
) -> np.ndarray:
    """The work list, int32 [num_items, ITEM_FIELDS], for requests i whose
    ``row_counts[i]`` rows start at row ``row_offsets[i]`` and see
    ``prefixes[i] + j + 1`` keys of block-table row i (row j). Items are 8 rows
    of one request and a segment of its 128-key tiles, about 32 per CTA, ordered
    request by request and, within one, segment by segment, so that CTAs
    running together read the same keys."""
    offsets, counts, prefix = (
        _host_ints(row_offsets),
        _host_ints(row_counts),
        _host_ints(prefixes),
    )
    _require(
        offsets.shape == counts.shape == prefix.shape,
        "row_offsets, row_counts and prefixes must have one entry per request",
    )
    _require(
        bool((counts >= 0).all() and (prefix >= 0).all() and (offsets >= 0).all()),
        "row offsets, row counts and prefixes must be >= 0",
    )
    blocks = (counts + ROWS - 1) // ROWS
    request = np.repeat(np.arange(counts.size), blocks)
    first = np.arange(blocks.sum()) - np.repeat(np.cumsum(blocks) - blocks, blocks)
    j0 = first * ROWS
    rows = np.minimum(ROWS, counts[request] - j0)
    first_len = prefix[request] + j0 + 1
    tiles = (first_len + rows - 1 + TILE_KEYS - 1) // TILE_KEYS
    if tiles.size == 0:
        return np.zeros((0, ITEM_FIELDS), dtype=np.int32)
    segment = max(
        _MIN_SEGMENT_TILES, -(-int(tiles.sum()) // (_ITEMS_PER_CTA * max(1, sms)))
    )
    segments = (tiles + segment - 1) // segment
    # Item k of row block b is segment k; order by (request, segment, block).
    block = np.repeat(np.arange(tiles.size), segments)
    k = np.arange(segments.sum()) - np.repeat(np.cumsum(segments) - segments, segments)
    order = np.lexsort((block, k, request[block]))
    block, k = block[order], k[order]
    work = np.zeros((block.size, ITEM_FIELDS), dtype=np.int64)
    work[:, 0] = offsets[request[block]] + j0[block]
    work[:, 1] = request[block]
    work[:, 2] = first_len[block]
    work[:, 3] = rows[block]
    work[:, 4] = k * segment
    work[:, 5] = np.minimum((k + 1) * segment, tiles[block])
    _require(
        work.size == 0 or int(work.max()) < 2**31,
        "positions or rows reach 2**31; the programs index them with int32",
    )
    return work.astype(np.int32)


def check_inputs(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    block_tables: torch.Tensor,
    work: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Raise ``ValueError`` unless the generated program serves this call.

    Host metadata only (shapes, dtypes, strides, devices, alignment): nothing
    here reads device memory."""
    _require(
        q.dim() == 3 and tuple(q.shape[1:]) == (HEADS, HEAD_DIM),
        f"q must be [rows, {HEADS}, {HEAD_DIM}]; got {tuple(q.shape)}",
    )
    _require(
        q.dtype == torch.float8_e4m3fn and q.is_contiguous(),
        f"q must be contiguous float8_e4m3fn; got {q.dtype}",
    )
    _require(
        q.shape[0] >= ROWS,
        f"q must hold at least {ROWS} rows (a work item's TMA box); pad it",
    )
    _require(q.is_cuda, "q must be on a CUDA device")
    capability = get_compute_capability(q.device)
    _require(
        capability in SUPPORTED_COMPUTE_CAPABILITIES,
        f"needs compute capability in {SUPPORTED_COMPUTE_CAPABILITIES}; "
        f"{q.device} has {capability}",
    )
    _require(
        kv_fused.dtype == torch.uint8
        and kv_fused.dim() == 4
        and tuple(kv_fused.shape[1:]) == (PAGE, 1, TOKEN_BYTES)
        and kv_fused.is_contiguous()
        and kv_fused.shape[0] > 0,
        f"kv_fused must be contiguous uint8 [num_blocks > 0, {PAGE}, 1, "
        f"{TOKEN_BYTES}]; got {kv_fused.dtype} {tuple(kv_fused.shape)}",
    )
    _require(
        weights.dtype == torch.float32
        and tuple(weights.shape) == (q.shape[0], HEADS)
        and weights.is_contiguous(),
        f"weights must be contiguous float32 [{q.shape[0]}, {HEADS}]; got "
        f"{weights.dtype} {tuple(weights.shape)}",
    )
    _require(
        block_tables.dtype == torch.int32
        and block_tables.dim() == 2
        and block_tables.stride(1) == 1
        and block_tables.shape[0] > 0,
        f"block_tables must be int32 [requests, max_blocks_per_seq] with unit "
        f"column stride; got {block_tables.dtype} {tuple(block_tables.shape)} "
        f"strides {block_tables.stride()}",
    )
    _require(
        (block_tables.shape[0] - 1) * block_tables.stride(0) + block_tables.shape[1]
        < 2**31,
        "block_tables spans 2**31 or more entries; the programs index it with int32",
    )
    _require(
        work.dtype == torch.int32
        and work.dim() == 2
        and work.shape[1] == ITEM_FIELDS
        and work.is_contiguous(),
        f"work must be contiguous int32 [num_items, {ITEM_FIELDS}] from plan_work",
    )
    _require(
        out.dtype == torch.float32
        and out.dim() == 2
        and out.is_contiguous()
        # The programs take the row pitch as an int32.
        and out.shape[1] < 2**31,
        f"out must be contiguous float32 [rows, pitch < 2**31]; got {out.dtype} "
        f"{tuple(out.shape)} strides {out.stride()}",
    )
    tensors = (q, kv_fused, weights, block_tables, work, out)
    _require(
        all(tensor.device == q.device for tensor in tensors),
        f"every tensor must be on {q.device}",
    )
    _require(
        all(tensor.data_ptr() % 16 == 0 for tensor in tensors),
        "every tensor's data must be 16-byte aligned",
    )


@dataclasses.dataclass(frozen=True)
class PrefillWork:
    """A step's work list on the device and what it needs of the call's
    tensors, known on the host so that the launch checks them without
    reading device memory."""

    items: torch.Tensor  # [num_items, ITEM_FIELDS] int32, on the device
    rows: int  # q, weights and out rows read or written
    max_len: int  # the longest row's visible keys: out's minimum pitch
    requests: int  # block-table rows read


def work_to_device(work: np.ndarray, device: torch.device) -> torch.Tensor:
    """The work list on ``device``, copied from pinned memory on the current
    stream without synchronizing the host. Launches on another stream must
    wait for this one."""
    host = torch.empty(work.shape, dtype=torch.int32, pin_memory=True)
    host.numpy()[...] = work
    return host.to(device, non_blocking=True)


def plan(
    row_offsets: Sequence[int] | np.ndarray | torch.Tensor,
    row_counts: Sequence[int] | np.ndarray | torch.Tensor,
    prefixes: Sequence[int] | np.ndarray | torch.Tensor,
    device: torch.device,
) -> PrefillWork:
    """See :func:`flashinfer.attn_scores.plan_fp8_ue8m0_paged_mqa_logits_prefill`."""
    device = torch.device(device)
    _require(
        device.type == "cuda", f"the work list goes to a CUDA device; got {device}"
    )
    index = device.index if device.index is not None else torch.cuda.current_device()
    device = torch.device("cuda", index)
    offsets, counts, prefix = (
        _host_ints(row_offsets),
        _host_ints(row_counts),
        _host_ints(prefixes),
    )
    work = plan_work(offsets, counts, prefix, get_device_sm_count(device))
    busy = counts > 0
    return PrefillWork(
        items=work_to_device(work, device),
        rows=int((offsets + counts)[busy].max()) if busy.any() else 0,
        max_len=int((prefix + counts)[busy].max()) if busy.any() else 0,
        requests=counts.size,
    )


def paged_mqa_logits_prefill(
    q: torch.Tensor,
    kv_fused: torch.Tensor,
    weights: torch.Tensor,
    block_tables: torch.Tensor,
    work: PrefillWork,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """See :func:`flashinfer.attn_scores.fp8_ue8m0_paged_mqa_logits_prefill`."""
    if out is None:
        pitch = -(-max(work.max_len, 1) // 4) * 4
        out = torch.empty(
            max(work.rows, 1), pitch, dtype=torch.float32, device=q.device
        )
    check_inputs(q, kv_fused, weights, block_tables, work.items, out)
    _require(
        q.shape[0] >= work.rows,
        f"the work reads {work.rows} rows of q and weights; q has {q.shape[0]}",
    )
    _require(
        out.shape[0] >= work.rows and out.shape[1] >= work.max_len,
        f"out must hold at least [{work.rows}, {work.max_len}]; got {tuple(out.shape)}",
    )
    _require(
        block_tables.shape[0] >= work.requests,
        f"block_tables must hold a row per request ({work.requests}); got "
        f"{block_tables.shape[0]}",
    )
    # A request reads the first ceil((prefix + rows) / PAGE) entries of its
    # block-table row: a narrower table would read the next row's entries.
    _require(
        block_tables.shape[1] * PAGE >= work.max_len,
        f"block_tables must hold {-(-work.max_len // PAGE)} blocks per request "
        f"(the longest row's keys); got {block_tables.shape[1]}",
    )
    work = work.items
    if work.shape[0] == 0:
        return out
    device_index = q.device.index
    num_blocks = kv_fused.shape[0]
    kv_bytes = kv_fused.view(-1)
    rows = q.shape[0]
    table = block_tables.as_strided(
        ((block_tables.shape[0] - 1) * block_tables.stride(0) + block_tables.shape[1],),
        (1,),
    )
    with on_device(device_index):
        kernel = _kernel(device_index)
        with tvm_ffi.use_torch_stream():
            kernel(
                # Q as bytes [rows * HEADS, HEAD_DIM]; a box is an item's 8 rows.
                q.view(torch.uint8).view(-1),
                HEAD_DIM,
                rows * HEADS,
                HEAD_DIM // 16,
                # Weights as [rows, HEADS] fp32; a box is an item's 8 rows.
                weights.view(-1),
                HEADS,
                rows,
                HEADS * 4 // 16,
                # A block's values as [num_blocks, PAGE, HEAD_DIM] bytes.
                kv_bytes,
                HEAD_DIM,
                PAGE,
                num_blocks,
                HEAD_DIM // 16,
                BLOCK_BYTES // 16,
                # Blocks as byte rows: the scale slots are a box inside each row.
                kv_bytes,
                BLOCK_BYTES,
                num_blocks,
                BLOCK_BYTES // 16,
                table,
                work.view(-1),
                out.view(-1),
                block_tables.stride(0),
                work.shape[0],
                out.stride(0),
                1,
                # Persistent CTAs: CTA c takes items c, c + grid, ...
                min(work.shape[0], get_device_sm_count(q.device)),
                1,
                1,
            )
    return out
