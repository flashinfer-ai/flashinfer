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

Cake backend: on-device load-balanced BF16 paged GQA decode (SM100/SM103).

One persistent launch (one CTA per SM) serves a whole ragged decode batch.  A
scheduler warp reads the device ``seq_lens`` buffer, derives a chunk length and
four length buckets, and hands ``(query row tile, kv head, KV block range)``
work items to the attention warps through a self-resetting global ticket
counter; long requests are split into chunks whose FP32 partials are merged by
the last CTA to finish the tile.  Nothing about the plan is decided on the
host, so a runner captured once into a CUDA Graph replays correctly for any
KV-length distribution written into ``seq_lens`` later.  See
``README.md`` in this package and flashinfer-ai/flashinfer#4832.

``q_len_per_req`` 3..8 (speculative / MTP verify) is served by the packed-row
program: one ``8 * q_len``-row tile per ``(request, kv head)`` item so each KV
chunk is streamed once per request, with the same on-device scheduler and a
distributed merge (one to ``2 * q_len`` merge tickets per split tile, fewer
and fatter for tiles with few chunks).  Other ``q_len_per_req`` values use
the row-tile program.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import tvm_ffi

from .cake_jit import (
    MODULES,
    load_cake_balanced_gqa_decode_module,
    select_module,
)
from .cake_bounds import (
    BLOCK_N,
    MAX_REQUESTS,
    MTP_COUNTERS_PER_TILE,
    MTP_PARTIAL_O_PER_SLOT,
    MTP_STATS_PER_SLOT,
    max_items_bound,
    mtp_max_items_bound,
    mtp_n_rows,
    uses_packed_mtp,
    workspace_bounds,
)

HEAD_DIM = 128
PAGE_SIZE = 16
GROUP_RATIO = 8  # query heads per KV head served by one 8-row MMA tile
PAGES_PER_BLOCK = BLOCK_N // PAGE_SIZE  # the loader fetches page ids 8 at a time
PARTIAL_O_PER_SLOT = GROUP_RATIO * HEAD_DIM  # row-tile program
STATS_PER_SLOT = 16  # max[8] then sum[8]
LOG2E = 1.4426950408889634
QUEUE_COUNTERS = 4  # ticket, done CTAs (both reset in-kernel), L, total items
WORKSPACE_ALIGN = 256
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}

# Exact keyword set of the generated program's ``run`` entry (bound by the
# export's argument plan); ``grid`` is expanded to ``grid_x/y/z``.
MTP_MAIN_KWARGS = (
    "Q",
    "K",
    "V",
    "O_ptr",
    "page_table",
    "seq_lens_kv",
    "partial_o",
    "partial_stats",
    "tile_counters",
    "queue_counters",
    "max_pages_per_seq",
    "softmax_scale_log2",
    "num_q_heads",
    "num_kv_heads",
    "batch_size",
    "q_len",
    "max_items",
    "grid",
)
MAIN_KWARGS = (
    "Qt",
    "K",
    "V",
    "O_ptr",
    "page_table",
    "seq_lens_kv",
    "partial_o",
    "partial_stats",
    "tile_counters",
    "queue_counters",
    "max_pages_per_seq",
    "softmax_scale",
    "num_q_heads",
    "num_kv_heads",
    "group_ratio",
    "batch_size",
    "q_len",
    "max_items",
    "grid",
)


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


def num_persistent_ctas(device: Optional[torch.device] = None) -> int:
    """Grid size of the persistent launch: one CTA per SM."""
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _align(nbytes: int) -> int:
    return (nbytes + WORKSPACE_ALIGN - 1) // WORKSPACE_ALIGN * WORKSPACE_ALIGN


def workspace_layout(num_ctas: int, *, padded_table_ints: int = 0) -> dict:
    """Byte ``(offset, size)`` of every workspace region plus ``"total"``.

    The partial slots and counters depend only on the CTA count (they bound
    every batch); ``padded_table_ints`` is non-zero only when the caller's
    block table needs padding to a multiple of eight pages per request.
    """
    # Both programs carve the same regions; the packed-row MTP program has the
    # larger slots (64 x 128 FP32 O, 128 statistics words, two counters per
    # split tile), so the layout is sized for it and bounds either kernel.
    max_split_items, max_split_tiles = workspace_bounds(num_ctas)
    sizes = [
        (
            "partial_o",
            max_split_items * max(PARTIAL_O_PER_SLOT, MTP_PARTIAL_O_PER_SLOT) * 4,
        ),
        (
            "partial_stats",
            max_split_items * max(STATS_PER_SLOT, MTP_STATS_PER_SLOT) * 4,
        ),
        ("tile_counters", max_split_tiles * MTP_COUNTERS_PER_TILE * 4),
        ("queue_counters", QUEUE_COUNTERS * 4),
        ("page_table", padded_table_ints * 4),
    ]
    layout: dict = {}
    offset = 0
    for name, nbytes in sizes:
        layout[name] = (offset, nbytes)
        offset += _align(nbytes)
    layout["total"] = offset
    return layout


def balanced_gqa_decode_workspace_size(
    device: Optional[torch.device] = None,
    *,
    num_sms: Optional[int] = None,
    batch: int = MAX_REQUESTS,
    max_pages: int = 0,
) -> int:
    """Workspace bytes for any batch on ``device`` (or ``num_sms`` CTAs).

    The default covers every batch whose block table already has a multiple
    of eight page columns (no padding copy).  Pass ``batch`` and ``max_pages``
    to also reserve the padded table copy used for other block-table widths.
    """
    if num_sms is None:
        num_sms = num_persistent_ctas(device)
    padded = 0
    if max_pages % PAGES_PER_BLOCK:
        padded = batch * _round_up_pages(max_pages)
    return int(workspace_layout(num_sms, padded_table_ints=padded)["total"])


def _round_up_pages(max_pages: int) -> int:
    return (max_pages + PAGES_PER_BLOCK - 1) // PAGES_PER_BLOCK * PAGES_PER_BLOCK


def _carve(flat: torch.Tensor, layout: dict, name: str, dtype, shape):
    """View ``shape`` elements of ``dtype`` at the start of workspace region ``name``.

    The regions are sized for the larger packed-row MTP slots; the row-tile
    program uses a prefix of each region.
    """
    offset, nbytes = layout[name]
    needed = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if needed > nbytes:
        raise ValueError(
            f"workspace region {name!r} holds {nbytes} bytes, {needed} needed"
        )
    return flat[offset : offset + needed].view(dtype).view(shape)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BalancedGQADecodeRunner:
    """Launch the prepared balanced decode.

    Calling the runner or ``launch()`` writes the caller-owned ``out`` with no
    CUDA allocation and no host synchronization and returns ``out``.  The
    kernel reads ``seq_lens`` and ``block_tables`` on device at every launch,
    so the same runner (or a CUDA Graph capturing it) stays valid when the
    caller writes new lengths or page ids into those buffers.  Prepare a new
    runner when shapes, dtypes or tensor bindings change.
    """

    module_name: str
    main_kwargs: dict
    out: torch.Tensor
    entry: Callable[..., Any]
    arguments: tuple
    block_tables_padded: bool

    def launch(self) -> torch.Tensor:
        # Tensor maps are encoded by the host binding and passed by value.
        with tvm_ffi.use_torch_stream():
            self.entry(*self.arguments)
        return self.out

    __call__ = launch

    @property
    def num_ctas(self) -> int:
        return int(self.main_kwargs["grid"][0])

    def device_plan(self) -> tuple[int, int]:
        """``(chunk_pairs, num_items)`` published by the last launch (syncs)."""
        counters = self.main_kwargs["queue_counters"].tolist()
        return int(counters[2]), int(counters[3])


def program_kind(q_len_per_req: int) -> str:
    """``"row"`` or the packed MTP instance (``"mtp32"`` / ``"mtp64"``) for ``q_len_per_req``."""
    if uses_packed_mtp(q_len_per_req):
        return f"mtp{mtp_n_rows(q_len_per_req)}"
    return "row"


def generated_program_available(device: torch.device, q_len_per_req: int = 1) -> bool:
    """True when this checkout registers the program serving ``q_len_per_req`` on ``device``."""
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))
    kind = program_kind(q_len_per_req)
    return arch is not None and any(
        r["arch"] == arch and r.get("kind", "row") == kind for r in MODULES.values()
    )


def bind_decode_payload(
    arch: str,
    main_kwargs: dict,
    out: torch.Tensor,
    *,
    block_tables_padded: bool,
    kind: str = "row",
) -> BalancedGQADecodeRunner:
    """Bind the prepared buffers to the generated physical argument order."""
    module_name = select_module(arch, kind)
    record = MODULES[module_name]
    physical = record["main"]
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), main_kwargs["grid"], strict=True))
    arguments = tuple(
        grid[name] if kind == "grid" else main_kwargs[name]
        for kind, name in physical["arg_plan"]
    )
    module = load_cake_balanced_gqa_decode_module(module_name, "main")
    entry = getattr(module, physical["ffi_entry"])
    return BalancedGQADecodeRunner(
        module_name, main_kwargs, out, entry, arguments, block_tables_padded
    )


# ---------------------------------------------------------------------------
# Validation and preparation
# ---------------------------------------------------------------------------


def _split_kv_cache(
    kv_cache: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]], kv_layout: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if kv_layout != "HND":
        raise ValueError(
            "balanced GQA decode reads pages as [num_kv_heads, page_size, head_dim] "
            "(kv_layout='HND')"
        )
    if isinstance(kv_cache, torch.Tensor):
        if kv_cache.ndim == 5 and kv_cache.shape[1] == 2:
            k_cache, v_cache = kv_cache[:, 0], kv_cache[:, 1]
            if not (k_cache.is_contiguous() and v_cache.is_contiguous()):
                raise ValueError(
                    "balanced GQA decode needs K and V pages in separate contiguous "
                    "tensors: pass kv_cache=(k_cache, v_cache), each "
                    f"[num_pages, num_kv_heads, {PAGE_SIZE}, {HEAD_DIM}]"
                )
            return k_cache, v_cache
        raise ValueError(
            "kv_cache must be a (k_cache, v_cache) tuple of "
            f"[num_pages, num_kv_heads, {PAGE_SIZE}, {HEAD_DIM}] tensors"
        )
    if len(kv_cache) != 2:
        raise ValueError("kv_cache tuple must hold exactly (k_cache, v_cache)")
    return kv_cache[0], kv_cache[1]


def validate_balanced_gqa_decode_inputs(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    q_len_per_req: int,
    out: Optional[torch.Tensor] = None,
) -> tuple[int, int, int, int]:
    """Shape / dtype validation shared by ``prepare``.

    Returns ``(batch, num_q_heads, num_kv_heads, num_pages)``.  Device placement
    and compute capability are checked separately so this runs on host tensors.
    """
    if query.ndim != 3 or query.shape[-1] != HEAD_DIM or query.dtype != torch.bfloat16:
        raise ValueError(
            f"query must be a bfloat16 [batch * q_len, num_q_heads, {HEAD_DIM}] tensor"
        )
    if block_tables.ndim != 2 or block_tables.dtype != torch.int32:
        raise ValueError("block_tables must be an int32 [batch, max_pages] tensor")
    batch = int(block_tables.shape[0])
    total_q, num_q_heads = int(query.shape[0]), int(query.shape[1])
    if not isinstance(q_len_per_req, int) or q_len_per_req <= 0:
        raise ValueError("q_len_per_req must be a positive integer")
    if batch <= 0 or total_q != batch * q_len_per_req:
        raise ValueError(
            "query rows must equal batch * q_len_per_req "
            f"(got {total_q} rows for batch {batch}, q_len_per_req {q_len_per_req})"
        )
    if batch > MAX_REQUESTS:
        raise ValueError(
            f"balanced GQA decode plans at most {MAX_REQUESTS} requests per launch"
        )
    for name, cache in (("k_cache", k_cache), ("v_cache", v_cache)):
        if (
            cache.ndim != 4
            or cache.dtype != torch.bfloat16
            or tuple(cache.shape[2:]) != (PAGE_SIZE, HEAD_DIM)
        ):
            raise ValueError(
                f"{name} must be a bfloat16 [num_pages, num_kv_heads, {PAGE_SIZE}, "
                f"{HEAD_DIM}] tensor"
            )
    if tuple(k_cache.shape) != tuple(v_cache.shape):
        raise ValueError("k_cache and v_cache must have the same shape")
    num_pages, num_kv_heads = int(k_cache.shape[0]), int(k_cache.shape[1])
    if num_kv_heads <= 0 or num_q_heads != GROUP_RATIO * num_kv_heads:
        raise ValueError(
            f"balanced GQA decode serves exactly {GROUP_RATIO} query heads per KV head "
            f"(got {num_q_heads} query heads, {num_kv_heads} KV heads)"
        )
    if seq_lens.shape != (batch,) or seq_lens.dtype != torch.int32:
        raise ValueError("seq_lens must be an int32 [batch] tensor")
    if out is not None and (
        tuple(out.shape) != (total_q, num_q_heads, HEAD_DIM)
        or out.dtype != torch.bfloat16
    ):
        raise ValueError(
            f"out must be a bfloat16 [batch * q_len, num_q_heads, {HEAD_DIM}] tensor"
        )
    return batch, num_q_heads, num_kv_heads, num_pages


def prepare_balanced_batch_decode_with_kv_cache(
    query: torch.Tensor,
    kv_cache: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    sm_scale: Optional[float] = None,
    q_len_per_req: int = 1,
    out: Optional[torch.Tensor] = None,
    kv_layout: str = "HND",
    backend: str = "cake",
) -> BalancedGQADecodeRunner:
    """Validate and bind one balanced BF16 paged GQA decode batch.

    Every allocation happens here (only the optional output and, for block
    tables whose width is not a multiple of eight pages, a padded copy inside
    ``workspace_buffer``); the returned runner launches with none.  No host
    copy of ``seq_lens`` is made: the work plan is derived on device.
    """
    if backend != "cake":
        raise ValueError("balanced GQA decode supports backend='cake'")
    k_cache, v_cache = _split_kv_cache(kv_cache, kv_layout)
    batch, num_q_heads, num_kv_heads, num_pages = validate_balanced_gqa_decode_inputs(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        q_len_per_req=q_len_per_req,
        out=out,
    )
    tensors = [query, k_cache, v_cache, block_tables, seq_lens, workspace_buffer]
    if out is not None:
        tensors.append(out)
    device = query.device
    if not all(t.is_cuda and t.device == device for t in tensors):
        raise ValueError("Expected all tensors on one CUDA device")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("Expected contiguous tensors")
    capability = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            "balanced GQA decode requires compute capability 10.0 or 10.3 "
            f"(got {capability[0]}.{capability[1]})"
        )
    if sm_scale is None:
        sm_scale = HEAD_DIM**-0.5
    total_q = batch * q_len_per_req
    num_ctas = num_persistent_ctas(device)
    max_pages = int(block_tables.shape[1])
    padded_pages = _round_up_pages(max_pages)
    needs_padding = padded_pages != max_pages
    layout = workspace_layout(
        num_ctas, padded_table_ints=batch * padded_pages if needs_padding else 0
    )
    flat = workspace_buffer.view(-1).view(torch.uint8)
    if flat.numel() < layout["total"]:
        raise ValueError(
            f"workspace_buffer needs {layout['total']} bytes on this device "
            f"({num_ctas} CTAs"
            + (", padded block table" if needs_padding else "")
            + f"), got {flat.numel()}"
        )
    if out is None:
        out = torch.empty(
            (total_q, num_q_heads, HEAD_DIM), dtype=torch.bfloat16, device=device
        )

    max_split_items, max_split_tiles = workspace_bounds(num_ctas)
    kind = program_kind(q_len_per_req)
    packed = kind != "row"
    partial_o = _carve(
        flat,
        layout,
        "partial_o",
        torch.float32,
        (max_split_items * (MTP_PARTIAL_O_PER_SLOT if packed else PARTIAL_O_PER_SLOT),),
    )
    partial_stats = _carve(
        flat,
        layout,
        "partial_stats",
        torch.float32,
        (max_split_items * (MTP_STATS_PER_SLOT if packed else STATS_PER_SLOT),),
    )
    tile_counters = _carve(
        flat,
        layout,
        "tile_counters",
        torch.uint32,
        (max_split_tiles * (MTP_COUNTERS_PER_TILE if packed else 1),),
    )
    queue_counters = _carve(
        flat, layout, "queue_counters", torch.uint32, (QUEUE_COUNTERS,)
    )
    # The kernel resets its ticket, done and tile counters at the end of every
    # launch; they must start at zero once.  The partial slots need no
    # initial value (each is written before it is read) but start clean.
    partial_o.zero_()
    partial_stats.zero_()
    tile_counters.zero_()
    queue_counters.zero_()
    page_table = block_tables
    if needs_padding:
        # The loader fetches page ids in groups of eight per KV block; widen the
        # table so every group stays inside its own request row.  Padding
        # pages are never used for valid tokens (they are masked by seq_lens).
        page_table = _carve(
            flat, layout, "page_table", torch.int32, (batch, padded_pages)
        )
        page_table.zero_()
        page_table[:, :max_pages].copy_(block_tables)

    if packed:
        # Packed-row MTP program: Q is read in its natural [batch * q_len,
        # num_q_heads, 128] layout by a 4-D TMA box (no host packing).
        main_kwargs = dict(
            Q=query,
            K=k_cache.view(num_pages * num_kv_heads, PAGE_SIZE, HEAD_DIM),
            V=v_cache.view(num_pages * num_kv_heads, PAGE_SIZE, HEAD_DIM),
            O_ptr=out,
            page_table=page_table,
            seq_lens_kv=seq_lens,
            partial_o=partial_o,
            partial_stats=partial_stats,
            tile_counters=tile_counters,
            queue_counters=queue_counters,
            max_pages_per_seq=padded_pages,
            softmax_scale_log2=float(sm_scale) * LOG2E,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            batch_size=batch,
            q_len=q_len_per_req,
            max_items=mtp_max_items_bound(batch, q_len_per_req, num_kv_heads, num_ctas),
            grid=(num_ctas, 1, 1),
        )
        assert tuple(main_kwargs) == MTP_MAIN_KWARGS
    else:
        main_kwargs = dict(
            Qt=query.view(total_q * num_q_heads, HEAD_DIM),
            K=k_cache.view(num_pages * num_kv_heads, PAGE_SIZE, HEAD_DIM),
            V=v_cache.view(num_pages * num_kv_heads, PAGE_SIZE, HEAD_DIM),
            O_ptr=out,
            page_table=page_table,
            seq_lens_kv=seq_lens,
            partial_o=partial_o,
            partial_stats=partial_stats,
            tile_counters=tile_counters,
            queue_counters=queue_counters,
            max_pages_per_seq=padded_pages,
            softmax_scale=float(sm_scale),
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            group_ratio=GROUP_RATIO,
            batch_size=batch,
            q_len=q_len_per_req,
            max_items=max_items_bound(batch, q_len_per_req, num_kv_heads, num_ctas),
            grid=(num_ctas, 1, 1),
        )
        assert tuple(main_kwargs) == MAIN_KWARGS
    return bind_decode_payload(
        arch, main_kwargs, out, block_tables_padded=needs_padding, kind=kind
    )
