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

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import torch
import tvm_ffi

from .cake_jit import MODULES, select_module

# DeepSeek-V4 main-attention decode geometry served by the generated program:
# MQA over one 512-wide NVFP4 latent row that is both K and V, 64-token pages.
HEAD_DIM = 512
PAGE_SIZE = 64  # tokens per page; one page is one KV tile of the decode kernel
SF_VEC = 16  # E2M1 values per UE4M3 block scale
ROW_BYTES = HEAD_DIM // 2  # 256 packed E2M1 bytes per Q / KV row
SF_ROW_BYTES = HEAD_DIM // SF_VEC  # 32 UE4M3 bytes per Q / KV row
DSV4_Q_LEN = (
    6  # DeepSeek-V4 default: query tokens per request (derived from the query shape)
)
CLUSTER_PAIR = (
    2  # the two v_half CTAs of a row tile form one cluster (multicast page stream)
)
ROWS_PER_TILE = 128  # packed query rows (token * H + head) per work item
V_HALVES = 2  # each work item accumulates 256 of the 512 output dims
LOG2E = 1.4426950408889634
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}

# Host work plan ABI shared with the generated program (part of the ABI freeze).
ITEM_FIELDS = 8  # b, m_tile, v_half, tile_start, tile_end, split, num_splits, flags
FLAG_SEED_SINK = 1  # split 0 folds the attention sink into the online softmax
FLAG_DIRECT_OUT = 2  # single-split request: write O / LSE directly
MAX_SPLITS = 64  # split-KV combine kernel capacity per row
MIN_PAGES_PER_UNIT = 8  # balanced schedule: minimum pages per CTA
BALANCED_MIN_KV = 8192  # auto schedule: balanced partition from this KV length on
WORKSPACE_ALIGNMENT = 256

# Keyword names the two stages are bound with (arg plans refer to these names).
MAIN_KWARGS = (
    "Q",
    "QS",
    "KV",
    "KVS",
    "O",
    "LSE",
    "partial_o",
    "partial_lse",
    "work_table",
    "unit_first",
    "page_table",
    "seq_lens",
    "q_indptr",
    "sinks",
    "num_heads",
    "q_len",
    "max_pages",
    "max_splits",
    "scale_log2",
    "grid",
)
REDUCE_KWARGS = (
    "o",
    "lse",
    "partial_o",
    "partial_lse",
    "row_splits",
    "num_heads",
    "max_splits",
    "grid",
)


# ---------------------------------------------------------------------------
# NVFP4 (E2M1 + UE4M3 block-16 scales) preparation helper
# ---------------------------------------------------------------------------


def _nearest_e2m1_codes(values):
    """Encode E2M1 with ``cvt.rn.satfinite`` tie-to-even semantics."""
    magnitude = values.abs()
    codes = torch.zeros_like(magnitude, dtype=torch.uint8)
    codes[(magnitude > 0.25) & (magnitude < 0.75)] = 1
    codes[(magnitude >= 0.75) & (magnitude <= 1.25)] = 2
    codes[(magnitude > 1.25) & (magnitude < 1.75)] = 3
    codes[(magnitude >= 1.75) & (magnitude <= 2.5)] = 4
    codes[(magnitude > 2.5) & (magnitude < 3.5)] = 5
    codes[(magnitude >= 3.5) & (magnitude <= 5.0)] = 6
    codes[magnitude > 5.0] = 7
    return codes | (torch.signbit(values).to(torch.uint8) << 3)


def quantize_nvfp4(x):
    """Quantize the last dimension of ``x`` to packed E2M1 codes + UE4M3 scales.

    Returns ``(packed_u8[..., D/2], scale_u8[..., D/16])``. The scale of each
    16-element block is ``amax / 6`` rounded to E4M3 and clamped to the
    positive finite range; scales are returned as the raw UE4M3 bytes.
    """
    if x.shape[-1] % (2 * SF_VEC):
        raise ValueError("the last dimension must be a multiple of 32")
    x32 = x.float()
    blocks = x32.reshape(*x32.shape[:-1], x32.shape[-1] // SF_VEC, SF_VEC)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    raw_scale = (blocks.abs().amax(dim=-1) / 6.0).clamp(min=2.0**-9, max=fp8_max)
    scale = raw_scale.to(torch.float8_e4m3fn)
    normalized = blocks / scale.float().unsqueeze(-1)
    codes = _nearest_e2m1_codes(normalized).reshape(*x32.shape)
    pairs = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2)
    packed = (pairs[..., 0] & 0x0F) | ((pairs[..., 1] & 0x0F) << 4)
    return packed.contiguous(), scale.view(torch.uint8).contiguous()


# ---------------------------------------------------------------------------
# Host work plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkPlan:
    """Flat work table of the persistent decode kernel.

    One item is ``(request, m_tile, v_half, tile_start, tile_end, split,
    num_splits, flags)``; cluster ``u`` (``CLUSTER_PAIR`` CTAs) runs items
    ``unit_first[u] .. unit_first[u + 1]``, always whole (v_half 0, v_half 1)
    pairs of one piece. Requests with more than one split go through the FP32
    partials and the split-KV combine kernel.
    """

    schedule: str
    items: tuple
    unit_first: tuple
    max_splits: int
    tiles_per_split: Optional[int]

    @property
    def num_items(self) -> int:
        return len(self.items)

    @property
    def num_units(self) -> int:
        return len(self.unit_first) - 1

    @property
    def grid(self) -> tuple:
        """Launch grid: one cluster of ``CLUSTER_PAIR`` CTAs per unit."""
        return (CLUSTER_PAIR * self.num_units, 1, 1)


def kv_tiles(kv_len: int) -> int:
    return (kv_len + PAGE_SIZE - 1) // PAGE_SIZE


def _m_tiles(q_len: int, num_heads: int) -> int:
    return math.ceil(q_len * num_heads / ROWS_PER_TILE)


def choose_tiles_per_split(
    kv_lens: Sequence[int],
    *,
    q_len: int,
    num_heads: int,
    num_sms: int,
    min_tiles_per_split: int = 8,
    target_waves: int = 2,
    max_partial_fraction: float = 0.35,
) -> int:
    """Pick the uniform split granularity so the item count fills whole waves.

    Base items (no split) = requests * m_tiles * 2. The split granularity is
    halved until ``items >= target_waves * num_sms``, every split would fall
    below ``min_tiles_per_split`` tiles, or the FP32 partial traffic of the
    next split count would exceed ``max_partial_fraction`` of the KV bytes
    streamed. For bs32 / q_len 6 / 64 heads this keeps one split at 8K KV and
    two splits from 16K on.
    """
    m_tiles = _m_tiles(q_len, num_heads)
    total_tiles = [kv_tiles(kv) for kv in kv_lens]
    max_tiles = max(total_tiles)
    kv_bytes = sum(kv_lens) * (ROW_BYTES + SF_ROW_BYTES)
    partial_bytes_per_split = len(kv_lens) * q_len * num_heads * HEAD_DIM * 4
    tiles_per_split = max_tiles
    while True:
        items = (
            sum(math.ceil(t / tiles_per_split) for t in total_tiles)
            * m_tiles
            * V_HALVES
        )
        if items >= target_waves * num_sms:
            break
        candidate = max(min_tiles_per_split, math.ceil(tiles_per_split / 2))
        if candidate >= tiles_per_split:
            break
        next_splits = max(math.ceil(t / candidate) for t in total_tiles)
        if next_splits * partial_bytes_per_split > max_partial_fraction * kv_bytes:
            break
        tiles_per_split = candidate
    return tiles_per_split


def _uniform_plan(kv_lens, *, q_len, num_heads, num_sms, enable_sink, tiles_per_split):
    """Request-major table, one v_half pair per cluster (all CTAs of a split are adjacent)."""
    if tiles_per_split is None:
        tiles_per_split = choose_tiles_per_split(
            kv_lens, q_len=q_len, num_heads=num_heads, num_sms=num_sms
        )
    m_tiles = _m_tiles(q_len, num_heads)
    items = []
    max_splits = 1
    for b, kv_len in enumerate(kv_lens):
        tiles = kv_tiles(kv_len)
        num_splits = math.ceil(tiles / tiles_per_split)
        max_splits = max(max_splits, num_splits)
        for split in range(num_splits):
            start = split * tiles_per_split
            end = min(tiles, start + tiles_per_split)
            flags = 0
            if enable_sink and split == 0:
                flags |= FLAG_SEED_SINK
            if num_splits == 1:
                flags |= FLAG_DIRECT_OUT
            for m_tile in range(m_tiles):
                for v_half in range(V_HALVES):
                    items.append(
                        (b, m_tile, v_half, start, end, split, num_splits, flags)
                    )
    return WorkPlan(
        schedule="uniform",
        items=tuple(items),
        unit_first=tuple(range(0, len(items) + 1, CLUSTER_PAIR)),
        max_splits=max_splits,
        tiles_per_split=tiles_per_split,
    )


def _balanced_plan(kv_lens, *, q_len, num_heads, num_units, enable_sink):
    """Balanced contiguous partition of the concatenated (row tile, page) work.

    Unit ``u`` of ``U`` receives positions ``[floor(u*W/U), floor((u+1)*W/U))``
    of the request-major page sequence. Wherever a unit boundary cuts a
    sequence the sequence becomes two splits; both ``v_half`` items of a piece
    share ``[start, end)`` and the split index, so the LSE written by the
    ``v_half == 0`` item applies to both halves of ``O`` in the combine
    kernel.
    """
    m_tiles = _m_tiles(q_len, num_heads)
    sequences = []  # (request, m_tile, tiles)
    for b, kv_len in enumerate(kv_lens):
        tiles = kv_tiles(kv_len)
        for m_tile in range(m_tiles):
            sequences.append((b, m_tile, tiles))
    total = sum(seq[2] for seq in sequences)
    if total == 0 or num_units <= 0:
        raise ValueError("balanced plan needs work and at least one unit")
    num_units = max(1, min(num_units, total // MIN_PAGES_PER_UNIT))
    longest = max(seq[2] for seq in sequences)
    while num_units > 1 and math.ceil(longest / (total / num_units)) + 1 > MAX_SPLITS:
        num_units //= 2
    pieces: list = [[] for _ in sequences]  # (unit, start, end)
    seq_idx = 0
    seq_pos = 0
    for u in range(num_units):
        lo = (u * total) // num_units
        hi = ((u + 1) * total) // num_units
        pos = lo
        while pos < hi:
            while seq_pos + sequences[seq_idx][2] <= pos:
                seq_pos += sequences[seq_idx][2]
                seq_idx += 1
            start = pos - seq_pos
            end = min(sequences[seq_idx][2], hi - seq_pos)
            pieces[seq_idx].append((u, start, end))
            pos = seq_pos + end
    per_unit: list = [[] for _ in range(num_units)]
    max_splits = 1
    for (b, m_tile, _tiles), seq_pieces in zip(sequences, pieces, strict=False):
        num_splits = len(seq_pieces)
        max_splits = max(max_splits, num_splits)
        for split, (u, start, end) in enumerate(seq_pieces):
            flags = 0
            if enable_sink and split == 0:
                flags |= FLAG_SEED_SINK
            if num_splits == 1:
                flags |= FLAG_DIRECT_OUT
            for v_half in range(V_HALVES):
                per_unit[u].append(
                    (b, m_tile, v_half, start, end, split, num_splits, flags)
                )
    items = []
    unit_first = [0]
    for u in range(num_units):
        items.extend(per_unit[u])
        unit_first.append(len(items))
    return WorkPlan(
        schedule="balanced",
        items=tuple(items),
        unit_first=tuple(unit_first),
        max_splits=max_splits,
        tiles_per_split=None,
    )


def check_pairs(plan: WorkPlan) -> None:
    """Every unit holds consecutive (v_half 0, v_half 1) items of the same piece."""
    items, unit_first = plan.items, plan.unit_first
    for u in range(len(unit_first) - 1):
        lo, hi = unit_first[u], unit_first[u + 1]
        if (hi - lo) % CLUSTER_PAIR:
            raise ValueError(
                f"unit {u} holds {hi - lo} items; the cluster pairing needs an even count"
            )
        for i in range(lo, hi, CLUSTER_PAIR):
            a, b = items[i], items[i + 1]
            if (a[2], b[2]) != (0, 1) or a[:2] != b[:2] or a[3:7] != b[3:7]:
                raise ValueError(
                    f"items {i},{i + 1} are not a v_half pair of one piece: {a} / {b}"
                )


def build_work_plan(
    kv_lens: Sequence[int],
    *,
    num_heads: int,
    num_sms: int,
    q_len: int,
    enable_sink: bool = False,
    schedule: str = "auto",
    tiles_per_split: Optional[int] = None,
) -> WorkPlan:
    """Build the host work plan for one batch.

    ``schedule="auto"`` uses the balanced partition over ``num_sms //
    CLUSTER_PAIR`` clusters when the longest request reaches
    ``BALANCED_MIN_KV`` tokens and the uniform split policy otherwise;
    ``"balanced"`` / ``"uniform"`` force one of them. ``tiles_per_split``
    fixes the uniform split granularity (pages per split).
    """
    kv_lens = [int(v) for v in kv_lens]
    if q_len < 1 or not kv_lens or min(kv_lens) < q_len:
        raise ValueError("every request needs kv_len >= q_len >= 1")
    if schedule == "auto":
        schedule = "balanced" if max(kv_lens) >= BALANCED_MIN_KV else "uniform"
    if schedule == "balanced":
        plan = _balanced_plan(
            kv_lens,
            q_len=q_len,
            num_heads=num_heads,
            num_units=num_sms // CLUSTER_PAIR,
            enable_sink=enable_sink,
        )
    elif schedule == "uniform":
        plan = _uniform_plan(
            kv_lens,
            q_len=q_len,
            num_heads=num_heads,
            num_sms=num_sms,
            enable_sink=enable_sink,
            tiles_per_split=tiles_per_split,
        )
    else:
        raise ValueError(f"unknown schedule {schedule!r} (auto | balanced | uniform)")
    if plan.max_splits > MAX_SPLITS:
        raise ValueError(
            f"work plan needs {plan.max_splits} KV splits; the combine kernel supports {MAX_SPLITS}"
        )
    check_pairs(plan)
    return plan


def work_table_rows(plan: WorkPlan) -> torch.Tensor:
    """Host int32 ``[num_items, ITEM_FIELDS]`` table for ``plan``.

    When any request has more than one split every row goes through the
    combine kernel, so ``FLAG_DIRECT_OUT`` is cleared for all items.
    """
    table = torch.tensor(plan.items, dtype=torch.int32).reshape(-1, ITEM_FIELDS)
    if plan.max_splits > 1:
        table[:, 7] &= ~FLAG_DIRECT_OUT
    return table


# ---------------------------------------------------------------------------
# Workspace layout
# ---------------------------------------------------------------------------


def _align(nbytes: int) -> int:
    return (
        (nbytes + WORKSPACE_ALIGNMENT - 1) // WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT
    )


def partial_shapes(plan: WorkPlan, *, total_q: int, num_heads: int) -> tuple:
    """Shapes of ``partial_o`` / ``partial_lse``; one dummy element each for single-split plans."""
    if plan.max_splits == 1:
        return (1,), (1,)
    return (total_q, num_heads, plan.max_splits, HEAD_DIM), (
        total_q,
        num_heads,
        plan.max_splits,
    )


def workspace_layout(plan: WorkPlan, *, batch: int, num_heads: int, q_len: int) -> dict:
    """Byte offsets and sizes of every workspace region plus ``"total"``.

    Regions: FP32 ``partial_o [total_q, H, max_splits, 512]`` and
    ``partial_lse [total_q, H, max_splits]`` (one dummy element each when the
    plan has a single split and the kernel writes ``O`` / ``LSE`` directly),
    int32 ``work_table``, ``unit_first``, ``row_splits``, ``q_indptr`` and
    FP32 ``sinks [H]``.
    """
    total_q = batch * q_len
    o_shape, lse_shape = partial_shapes(plan, total_q=total_q, num_heads=num_heads)
    sizes = (
        ("partial_o", math.prod(o_shape) * 4),
        ("partial_lse", math.prod(lse_shape) * 4),
        ("work_table", plan.num_items * ITEM_FIELDS * 4),
        ("unit_first", (plan.num_units + 1) * 4),
        ("row_splits", total_q * 4),
        ("q_indptr", (batch + 1) * 4),
        ("sinks", num_heads * 4),
    )
    layout: dict = {}
    offset = 0
    for name, nbytes in sizes:
        layout[name] = (offset, nbytes)
        offset += _align(nbytes)
    layout["total"] = offset
    return layout


def nvfp4_mla_decode_workspace_size(
    seq_lens: Sequence[int],
    num_heads: int,
    *,
    num_sms: int,
    q_len: int = DSV4_Q_LEN,
    enable_sink: bool = False,
    schedule: str = "auto",
) -> int:
    """Workspace bytes ``prepare`` needs for the given host sequence lengths."""
    plan = build_work_plan(
        seq_lens,
        num_heads=num_heads,
        num_sms=num_sms,
        q_len=q_len,
        enable_sink=enable_sink,
        schedule=schedule,
    )
    return workspace_layout(
        plan, batch=len(seq_lens), num_heads=num_heads, q_len=q_len
    )["total"]


def max_nvfp4_mla_decode_workspace_size(
    batch: int, num_heads: int, *, q_len: int = DSV4_Q_LEN, max_splits: int = MAX_SPLITS
) -> int:
    """Upper bound of the workspace for any plan with at most ``max_splits`` splits.

    Every request contributes at most ``max_splits`` items per (row tile,
    v_half) pair; the partials dominate (``total_q * H * max_splits * 2 KiB``).
    """
    total_q = batch * q_len
    max_items = batch * _m_tiles(q_len, num_heads) * V_HALVES * max_splits
    return (
        _align(total_q * num_heads * max_splits * HEAD_DIM * 4)
        + _align(total_q * num_heads * max_splits * 4)
        + _align(max_items * ITEM_FIELDS * 4)
        + _align((max_items + 1) * 4)
        + _align(total_q * 4)
        + _align((batch + 1) * 4)
        + _align(num_heads * 4)
    )


def _carve(flat: torch.Tensor, layout: dict, name: str, dtype, shape):
    offset, nbytes = layout[name]
    return flat[offset : offset + nbytes].view(dtype).view(shape)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NVFP4MLADecodeRunner:
    """Launch the prepared decode (and, when planned, the split-KV combine).

    Calling the runner or ``launch()`` writes the caller-owned output and LSE
    with no CUDA allocation or host synchronization and returns ``out`` (or
    ``(out, lse)`` when prepared with ``return_lse=True``). Prepare a new
    runner when sequence lengths, bindings or input values change.
    """

    module_name: str
    plan: WorkPlan
    main_kwargs: dict
    reduce_kwargs: Optional[dict]
    out: torch.Tensor
    lse: torch.Tensor
    return_lse: bool
    main_entry: object
    main_arguments: tuple
    reduce_entry: object
    reduce_arguments: tuple

    def launch(self):
        # Tensor maps are encoded by the host binding and passed by value; the
        # split-KV combine runs only when the plan has more than one split.
        with tvm_ffi.use_torch_stream():
            self.main_entry(*self.main_arguments)
            if self.reduce_entry is not None:
                self.reduce_entry(*self.reduce_arguments)
        return (self.out, self.lse) if self.return_lse else self.out

    __call__ = launch


def generated_program_available(device: torch.device) -> bool:
    """True when this checkout registers a generated program for ``device``."""
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))
    return arch is not None and any(r["arch"] == arch for r in MODULES.values())


def bind_decode_payload(
    arch: str,
    plan: WorkPlan,
    main_kwargs: dict,
    reduce_kwargs: Optional[dict],
    out: torch.Tensor,
    lse: torch.Tensor,
    return_lse: bool,
) -> NVFP4MLADecodeRunner:
    """Bind the prepared buffers to the generated physical argument order."""
    module_name = select_module(arch)
    # TODO(cake_nvfp4_mla_decode ABI freeze, flashinfer-ai/flashinfer#5403):
    # once ``cake_jit.MODULES`` is populated, bind as nvfp4_attention does:
    #   record = MODULES[module_name]
    #   main_arguments = tuple(<grid_x/grid_y/grid_z from main_kwargs["grid"]>
    #       if kind == "grid" else main_kwargs[name]
    #       for kind, name in record["arg_plan"])
    #   reduce_arguments = same over record["reduce_arg_plan"] / reduce_kwargs
    #       (empty when reduce_kwargs is None, i.e. plan.max_splits == 1)
    #   module = load_cake_nvfp4_mla_decode_module(module_name)
    #   main_entry = getattr(module, record["ffi_entry"])
    #   reduce_entry = getattr(module, record["reduce_ffi_entry"]) or None
    # The kwargs assembled by ``prepare`` already follow MAIN_KWARGS /
    # REDUCE_KWARGS; only the export's arg plans, FFI entry names and
    # closure digest are missing.
    raise NotImplementedError(
        f"launch binding for {module_name} is pending the generated-program "
        "export (flashinfer-ai/flashinfer#5403)"
    )


# ---------------------------------------------------------------------------
# Validation and preparation
# ---------------------------------------------------------------------------


def _check_u8(name: str, tensor: torch.Tensor, shape: tuple) -> None:
    if tensor.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise TypeError(f"{name} must be packed uint8 (or float8_e4m3fn scales)")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def validate_nvfp4_mla_decode_inputs(
    query,
    query_scale,
    kv_cache,
    kv_scale,
    block_tables,
    seq_lens,
    *,
    sinks=None,
    out=None,
    lse=None,
) -> tuple:
    """Shape / dtype validation shared by ``prepare``; returns ``(batch, H, num_pages, q_len)``.

    Device placement and compute capability are checked separately so this
    part runs on host tensors too.
    """
    if query.ndim != 3 or query.shape[-1] != ROW_BYTES:
        raise ValueError(
            f"query must be [total_q, num_heads, {ROW_BYTES}] packed E2M1 bytes"
        )
    total_q, num_heads = int(query.shape[0]), int(query.shape[1])
    if block_tables.ndim != 2 or block_tables.dtype != torch.int32:
        raise ValueError("block_tables must be an int32 [batch, max_pages] tensor")
    batch = int(block_tables.shape[0])
    if batch <= 0 or num_heads <= 0 or total_q < batch or total_q % batch:
        raise ValueError(
            "query rows must be batch * q_len with the same number of query tokens per request"
        )
    q_len = total_q // batch
    if seq_lens.shape != (batch,) or seq_lens.dtype != torch.int32:
        raise ValueError("seq_lens must be an int32 [batch] tensor")
    if kv_cache.ndim != 3 or tuple(kv_cache.shape[1:]) != (PAGE_SIZE, ROW_BYTES):
        raise ValueError(
            f"kv_cache must be [num_pages, {PAGE_SIZE}, {ROW_BYTES}] packed E2M1 bytes"
        )
    num_pages = int(kv_cache.shape[0])
    _check_u8("query", query, (total_q, num_heads, ROW_BYTES))
    _check_u8("query_scale", query_scale, (total_q, num_heads, SF_ROW_BYTES))
    _check_u8("kv_cache", kv_cache, (num_pages, PAGE_SIZE, ROW_BYTES))
    _check_u8("kv_scale", kv_scale, (num_pages, PAGE_SIZE, SF_ROW_BYTES))
    if sinks is not None and (
        sinks.shape != (num_heads,) or sinks.dtype != torch.float32
    ):
        raise ValueError("sinks must be a float32 [num_heads] tensor")
    if out is not None and (
        out.shape != (total_q, num_heads, HEAD_DIM) or out.dtype != torch.bfloat16
    ):
        raise ValueError(
            f"out must be a bfloat16 [total_q, num_heads, {HEAD_DIM}] tensor"
        )
    if lse is not None and (
        lse.shape != (total_q, num_heads) or lse.dtype != torch.float32
    ):
        raise ValueError("lse must be a float32 [total_q, num_heads] tensor")
    return batch, num_heads, num_pages, q_len


def prepare_nvfp4_batch_decode_with_kv_cache_mla(
    query,
    query_scale,
    kv_cache,
    kv_scale,
    block_tables,
    seq_lens,
    workspace_buffer,
    *,
    sm_scale,
    sinks=None,
    out=None,
    lse=None,
    return_lse=False,
    seq_lens_cpu=None,
    backend="cake",
    schedule="auto",
    tiles_per_split=None,
):
    """Plan and bind one NVFP4 DeepSeek-V4 paged MQA decode batch.

    Host planning (one device-to-host copy of ``seq_lens`` unless
    ``seq_lens_cpu`` is given) and every allocation happen here; the returned
    runner launches with neither. ``schedule`` / ``tiles_per_split`` expose the
    host work-plan policy for tests and benchmarks.
    """
    if backend != "cake":
        raise ValueError("NVFP4 MLA decode supports backend='cake'")
    batch, num_heads, num_pages, q_len = validate_nvfp4_mla_decode_inputs(
        query,
        query_scale,
        kv_cache,
        kv_scale,
        block_tables,
        seq_lens,
        sinks=sinks,
        out=out,
        lse=lse,
    )
    tensors = [query, query_scale, kv_cache, kv_scale, block_tables, seq_lens]
    tensors += [t for t in (sinks, out, lse, workspace_buffer) if t is not None]
    device = query.device
    if not all(t.is_cuda and t.device == device for t in tensors):
        raise ValueError("Expected all tensors on one CUDA device")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("Expected contiguous tensors")
    capability = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            "NVFP4 MLA decode requires compute capability 10.0 or 10.3 "
            f"(got {capability[0]}.{capability[1]})"
        )
    if seq_lens_cpu is None:
        seq_lens_cpu = seq_lens.cpu()
    kv_lens = [int(v) for v in seq_lens_cpu.tolist()]
    if len(kv_lens) != batch:
        raise ValueError("seq_lens_cpu must hold one length per request")
    max_pages = int(block_tables.shape[1])
    if min(kv_lens) < q_len or max(kv_lens) > max_pages * PAGE_SIZE:
        raise ValueError(
            f"every request needs q_len ({q_len}) <= seq_len <= max_pages * {PAGE_SIZE}"
        )
    total_q = batch * q_len
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    plan = build_work_plan(
        kv_lens,
        num_heads=num_heads,
        num_sms=num_sms,
        q_len=q_len,
        enable_sink=sinks is not None,
        schedule=schedule,
        tiles_per_split=tiles_per_split,
    )
    layout = workspace_layout(plan, batch=batch, num_heads=num_heads, q_len=q_len)
    flat = workspace_buffer.view(-1).view(torch.uint8)
    if flat.numel() < layout["total"]:
        raise ValueError(
            f"workspace_buffer needs {layout['total']} bytes for this batch "
            f"({plan.max_splits} KV splits), got {flat.numel()}"
        )
    if out is None:
        out = torch.empty(
            (total_q, num_heads, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
    if lse is None:
        lse = torch.empty((total_q, num_heads), dtype=torch.float32, device=device)

    max_splits = plan.max_splits
    o_shape, lse_shape = partial_shapes(plan, total_q=total_q, num_heads=num_heads)
    partial_o = _carve(flat, layout, "partial_o", torch.float32, o_shape)
    partial_lse = _carve(flat, layout, "partial_lse", torch.float32, lse_shape)
    work_table = _carve(
        flat, layout, "work_table", torch.int32, (plan.num_items, ITEM_FIELDS)
    )
    unit_first = _carve(flat, layout, "unit_first", torch.int32, (plan.num_units + 1,))
    row_splits = _carve(flat, layout, "row_splits", torch.int32, (total_q,))
    q_indptr = _carve(flat, layout, "q_indptr", torch.int32, (batch + 1,))
    sink_buffer = _carve(flat, layout, "sinks", torch.float32, (num_heads,))
    # Unused partial slots keep finite zeros and -inf LSEs so the combine
    # kernel weights them by zero; every row reads max_splits slots.
    partial_o.zero_()
    partial_lse.fill_(float("-inf"))
    work_table.copy_(work_table_rows(plan))
    unit_first.copy_(torch.tensor(plan.unit_first, dtype=torch.int32))
    row_splits.fill_(max_splits)
    q_indptr.copy_(torch.arange(0, total_q + 1, q_len, dtype=torch.int32))
    if sinks is None:
        sink_buffer.zero_()
    else:
        sink_buffer.copy_(sinks)

    main_kwargs = dict(
        Q=query.view(total_q * num_heads, ROW_BYTES),
        QS=query_scale.view(torch.uint8).view(total_q * num_heads, SF_ROW_BYTES),
        KV=kv_cache.view(num_pages * PAGE_SIZE, ROW_BYTES),
        KVS=kv_scale.view(torch.uint8).view(num_pages * PAGE_SIZE, SF_ROW_BYTES),
        O=out,
        LSE=lse,
        partial_o=partial_o,
        partial_lse=partial_lse,
        work_table=work_table,
        unit_first=unit_first,
        page_table=block_tables,
        seq_lens=seq_lens,
        q_indptr=q_indptr,
        sinks=sink_buffer,
        num_heads=num_heads,
        q_len=q_len,
        max_pages=max_pages,
        max_splits=max_splits,
        scale_log2=float(sm_scale) * LOG2E,
        grid=plan.grid,
    )
    reduce_kwargs = None
    if max_splits > 1:
        reduce_kwargs = dict(
            o=out,
            lse=lse,
            partial_o=partial_o,
            partial_lse=partial_lse,
            row_splits=row_splits,
            num_heads=num_heads,
            max_splits=max_splits,
            grid=(num_heads, total_q, 1),
        )
    assert tuple(main_kwargs) == MAIN_KWARGS
    assert reduce_kwargs is None or tuple(reduce_kwargs) == REDUCE_KWARGS
    return bind_decode_payload(
        arch, plan, main_kwargs, reduce_kwargs, out, lse, return_lse
    )
