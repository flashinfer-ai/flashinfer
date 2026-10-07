"""CAKE backend for Kimi-K3 MLA attention over an FP8 (E4M3) paged latent cache (SM100 / SM103).

Cake-generated programs behind ``trtllm_batch_decode_with_kv_cache_mla(backend="cake")`` for
``kv_lora_rank=512`` / ``qk_rope_head_dim=64`` with an FP8 query and an FP8 paged cache (page
size 64): dense decode (``q_len = 1``), packed variable-Q / MTP (``cum_seq_lens_q``) and
incremental prefill on the paged cache.  BF16 output into a caller-owned buffer, bottom-right
causal mask, current stream, CUDA-Graph replayable (``launch`` allocates nothing; the split-KV
partials live in the caller's ``workspace_buffer``).

Routing uses host-known scalars only.  Requests with more than ``WIDE_MIN_ROWS`` packed
(token, head) rows whose longest KV is at least ``WIDE_MIN_KV`` tokens run the two-CTA wide
kernel (``main_wide``: 128 rows per cluster of two CTAs); every other shape runs a swapped-AB
row tile (``main_rt16`` .. ``main_rt96``: the smallest tile holding the request's rows).  Both
share the split-KV merge kernels (warp-per-row reducers ``reduce_w4`` / ``reduce_w2`` /
``reduce_w1`` for ``num_split <= 32``, the CTA reducer otherwise).  The plan (route, row tile,
split count, grids, reducer) depends only on the batch shape, the longest KV and the device's
SM count, so it is computed once per distinct shape (:func:`plan_attention`) and reused;
device facts are read once per device.

Query and KV views may be row-strided on every route: every row of ``query`` and every token
row of ``kv_cache`` must be 576 contiguous FP8 elements, and rows must be equally spaced (a
16-byte multiple); the TMA descriptors of the row-tile programs and of the two-CTA wide kernel
carry that row stride.  ``out`` rows are dense (512 BF16 elements apart); ``block_tables``,
``seq_lens`` and ``cum_seq_lens_q`` are contiguous int32.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, Optional

import torch

LATENT = 512
ROPE = 64
QK_DIM = LATENT + ROPE
V_DIM = LATENT
PAGE_SIZE = 64
TILE_TOK = 128  # tokens per KV tile of both attention schedules (two pages)
MAX_SPLITS = 256
MIN_TILES_PER_SPLIT = 2
REDUCE_WARPS = 8  # warp reducer CTA: 256 threads
REDUCE_WARP_MAX_SPLITS = 32  # lane s owns split s
REDUCE_DIM_CHUNKS = 4  # CTA reducer: 128 latent dims per CTA
ROW_TILES = (16, 32, 48, 64, 96)
# Two-CTA wide route: 128 packed rows per cluster of two CTAs, K tokens split across the pair.
# Taken for requests with more than WIDE_MIN_ROWS packed rows whose longest KV is at least
# WIDE_MIN_KV tokens: the lazy-E4M3 probability path of that schedule meets the uniform-profile
# tolerance (atol 0.01 / rtol 0.02 against an FP32 reference) from a longest KV of 8192 tokens
# on; shorter-KV prefill stays on the row tiles.
WIDE_MIN_ROWS = 64
WIDE_MIN_KV = 8192
WIDE_TILE_Q = 128  # packed rows per two-CTA cluster
WIDE_CLUSTER = 2  # CTAs per cluster: one SM pair per work item
WIDE_MIN_TILES_PER_SPLIT = 2
# Per-wave fixed cost (prologue + drain of a work item) in 128-token tile periods.  Calibrated
# on B200 (148 SMs: one wave of 305 tiles beat three waves of 99) and GB300 (152 SMs: one wave
# of 9 splits over 298 tiles beat two full waves of 19 splits by 3 % on the 8-request H96 decode
# row; forced-split sweeps fit the per-wave term at 20-25 tiles); 16 is the smallest value that
# reproduces every measured optimum on 148 / 152 / 160 SMs.
WIDE_WAVE_COST_TILES = 16
# Tail splitting: a partially filled last wave of SM pairs runs faster per tile than a full one
# at the board power cap (fewer active SMs, higher clock; B200 fits: 44 of 74 pairs ~1.19x,
# 56 of 74 ~1.10x, i.e. occupancy ** -0.33), so the full waves keep unsplit KV streams and only
# the tail items are split, and only when the model predicts at least WIDE_TAIL_MIN_GAIN.
WIDE_UNDERFILL_EXP = 0.33
WIDE_TAIL_MIN_GAIN = 0.02
TMA_ROW_ALIGN = 16  # bytes: TMA global strides are 16-byte multiples


# ---------------------------------------------------------------------------
# Planning (pure functions of host scalars)
# ---------------------------------------------------------------------------


def swapped_rt(rows_per_request: int) -> int:
    """Row tile for a request: the smallest template holding all rows, else 96-row tiles."""
    for rt in ROW_TILES:
        if rows_per_request <= rt:
            return rt
    return 96


def plan_num_split(items: int, max_seq_len: int, sm_count: int) -> int:
    """One CTA per SM per wave over (work item, split) pairs without a ragged tail."""
    target = max(1, sm_count // max(1, items))
    max_by_len = max(1, (max_seq_len + TILE_TOK - 1) // TILE_TOK // MIN_TILES_PER_SPLIT)
    return max(1, min(target, MAX_SPLITS, max_by_len))


def use_wide_route(rows_per_request: int, max_seq_len: int) -> bool:
    """Whether a request shape runs the two-CTA wide route (else a swapped-AB row tile)."""
    return rows_per_request > WIDE_MIN_ROWS and int(max_seq_len) >= WIDE_MIN_KV


def plan_num_split_wide(
    clusters: int,
    max_seq_len: int,
    sm_count: int,
    min_tiles_per_split: int = WIDE_MIN_TILES_PER_SPLIT,
    max_splits: int = MAX_SPLITS,
    wave_cost_tiles: int = WIDE_WAVE_COST_TILES,
) -> int:
    """KV splits per cluster of the wide route.

    Work items (clusters x splits) run one per SM pair; the cost of ``s`` splits in 128-token
    tile periods is ``ceil(items / pairs) * (ceil(tiles / s) + wave_cost_tiles)`` plus ~0.05 tile
    periods per item for the split merge.  The per-wave term is the prologue + drain every work
    item pays.  A split is taken only when that model predicts at least 15 %.
    """
    pairs = max(1, sm_count // WIDE_CLUSTER)
    tiles = max(1, (max_seq_len + TILE_TOK - 1) // TILE_TOK)
    max_s = max(1, min(max_splits, tiles // max(1, min_tiles_per_split)))
    best_s, best_cost, cost_one = 1, None, None
    for s in range(1, max_s + 1):
        items = clusters * s
        cost = -(-items // pairs) * (-(-tiles // s) + wave_cost_tiles) + 0.05 * items
        if s == 1:
            cost_one = cost
        if best_cost is None or cost < best_cost:
            best_s, best_cost = s, cost
    assert cost_one is not None and best_cost is not None
    if best_s > 1 and cost_one / best_cost < 1.15:
        return 1
    return best_s


def plan_wide_work(
    clusters: int,
    max_seq_len: int,
    sm_count: int,
    forced_split: Optional[int] = None,
    min_tiles_per_split: int = WIDE_MIN_TILES_PER_SPLIT,
    max_splits: int = MAX_SPLITS,
    wave_cost_tiles: int = WIDE_WAVE_COST_TILES,
) -> tuple[int, int]:
    """``(n_full_items, num_split)`` of the wide route's flat grid.

    Items before ``n_full_items`` stream their whole KV on one cluster; the remaining items take
    ``num_split`` clusters each.  With fewer items than SM pairs, or a forced / uniform split, this
    is ``(0, plan_num_split_wide)``.  Otherwise the full waves stay unsplit and the last, partially
    filled wave is split into ``s`` chunks when ``full_waves * (tiles + w) + waves(tail * s) *
    (tiles / s + w)`` (last wave discounted by ``occupancy ** -WIDE_UNDERFILL_EXP``, 0.05 tile
    periods per chunk for the merge) beats the unsplit plan by WIDE_TAIL_MIN_GAIN.
    """
    if forced_split:
        return 0, max(1, int(forced_split))
    uniform = plan_num_split_wide(
        clusters,
        max_seq_len,
        sm_count,
        min_tiles_per_split=min_tiles_per_split,
        max_splits=max_splits,
        wave_cost_tiles=wave_cost_tiles,
    )
    if uniform != 1:
        return 0, uniform
    pairs = max(1, sm_count // WIDE_CLUSTER)
    tiles = max(1, (max_seq_len + TILE_TOK - 1) // TILE_TOK)
    full_waves, tail = divmod(clusters, pairs)
    if full_waves == 0 or tail == 0:
        return 0, 1

    def wave(chunk_tiles: int, active: int) -> float:
        return (chunk_tiles + wave_cost_tiles) * (active / pairs) ** WIDE_UNDERFILL_EXP

    base = full_waves * (tiles + wave_cost_tiles)
    cost_one = base + wave(tiles, tail)
    best_s, best_cost = 1, cost_one
    for s in range(
        2, max(2, min(max_splits, tiles // max(1, min_tiles_per_split))) + 1
    ):
        chunk = -(-tiles // s)
        if chunk < min_tiles_per_split:
            break
        chunks = tail * s
        waves = -(-chunks // pairs)
        last = chunks - (waves - 1) * pairs
        cost = (
            base
            + (waves - 1) * (chunk + wave_cost_tiles)
            + wave(chunk, last)
            + 0.05 * chunks
        )
        if cost < best_cost:
            best_s, best_cost = s, cost
    if best_s > 1 and best_cost <= cost_one * (1.0 - WIDE_TAIL_MIN_GAIN):
        return full_waves * pairs, best_s
    return 0, 1


def reduce_warps_per_row(rows: int) -> int:
    if rows <= 256:
        return 4
    if rows <= 1024:
        return 2
    return 1


@dataclass(frozen=True)
class AttentionPlan:
    """Launch plan of one batch shape: route, row tile, split plan, grids and reducer."""

    wide: bool
    rt: Optional[int]
    m_tiles: int
    num_split: int
    n_full_items: int
    tile_rows: int
    rows_max: int
    reduce_rows: int
    reduce_warps: int  # 0 for the CTA reducer
    grid_main: tuple[int, int, int]
    grid_reduce: tuple[int, int, int]
    main_kind: str
    reduce_kind: str


@functools.lru_cache(maxsize=1024)
def plan_attention(
    *,
    batch: int,
    max_q_len: int,
    num_heads: int,
    max_seq_len: int,
    sm_count: int,
    num_split: Optional[int] = None,
) -> AttentionPlan:
    """The plan of a batch shape; cached, so a decode loop plans each shape once."""
    rows_max = batch * max_q_len * num_heads
    rows_per_request = max_q_len * num_heads
    wide = use_wide_route(rows_per_request, max_seq_len)
    if wide:
        rt = None
        tile_rows = WIDE_TILE_Q
        m_tiles = (rows_per_request + WIDE_TILE_Q - 1) // WIDE_TILE_Q
        # Full waves of SM pairs run unsplit items; only the tail items (if any) take num_split clusters.
        n_full_items, splits = plan_wide_work(
            batch * m_tiles, max_seq_len, sm_count, forced_split=num_split
        )
    else:
        rt = swapped_rt(rows_per_request)
        tile_rows = rt
        m_tiles = (rows_per_request + rt - 1) // rt
        splits = (
            int(num_split)
            if num_split
            else plan_num_split(batch * m_tiles, max_seq_len, sm_count)
        )
        n_full_items = 0
    num_items = m_tiles * batch
    tail_items = num_items - n_full_items
    if wide:
        # Flat grid: one two-CTA cluster per unsplit item, num_split clusters per tail item (split fastest).
        grid_main = (WIDE_CLUSTER * (n_full_items + tail_items * splits), 1, 1)
    else:
        grid_main = (splits, m_tiles, batch)
    # The merge covers the packed rows (uniform plan) or the rows of the split items only (tail plan).
    reduce_rows = rows_max if n_full_items == 0 else tail_items * tile_rows
    if splits <= REDUCE_WARP_MAX_SPLITS:
        reduce_warps = reduce_warps_per_row(reduce_rows)
        rows_per_cta = REDUCE_WARPS // reduce_warps
        grid_reduce = ((reduce_rows + rows_per_cta - 1) // rows_per_cta, 1, 1)
        reduce_kind = f"reduce_w{reduce_warps}"
    else:
        reduce_warps = 0
        grid_reduce = (reduce_rows, REDUCE_DIM_CHUNKS, 1)
        reduce_kind = "reduce_cta"
    return AttentionPlan(
        wide=wide,
        rt=rt,
        m_tiles=m_tiles,
        num_split=splits,
        n_full_items=n_full_items,
        tile_rows=tile_rows,
        rows_max=rows_max,
        reduce_rows=reduce_rows,
        reduce_warps=reduce_warps,
        grid_main=grid_main,
        grid_reduce=grid_reduce,
        main_kind="main_wide" if wide else f"main_rt{rt}",
        reduce_kind=reduce_kind,
    )


# ---------------------------------------------------------------------------
# Workspace, device facts and tensor views
# ---------------------------------------------------------------------------


def _align16(n: int) -> int:
    return (n + 15) & ~15


def workspace_bytes(rows_max: int, num_split: int) -> int:
    """Bytes of ``workspace_buffer`` needed: BF16 partial O + FP32 partial max / sum."""
    partial_o = rows_max * num_split * V_DIM * 2 if num_split > 1 else 0
    stats = rows_max * num_split * 4
    return _align16(partial_o) + 2 * _align16(stats)


def _carve_workspace(workspace: torch.Tensor, rows_max: int, num_split: int):
    if workspace.device.type != "cuda" or not workspace.is_contiguous():
        raise ValueError("workspace_buffer must be a contiguous CUDA tensor")
    raw = workspace.view(torch.uint8).reshape(-1)
    need = workspace_bytes(rows_max, num_split)
    if raw.numel() < need:
        raise ValueError(
            f"workspace_buffer needs at least {need} bytes for this CAKE Kimi-K3 MLA plan, "
            f"got {raw.numel()}"
        )
    off = 0
    partial_o = None
    if num_split > 1:
        n = rows_max * num_split * V_DIM * 2
        partial_o = (
            raw[off : off + n].view(torch.bfloat16).view(rows_max, num_split, V_DIM)
        )
        off += _align16(n)
    n = rows_max * num_split * 4
    partial_max = raw[off : off + n].view(torch.float32).view(rows_max, num_split)
    off += _align16(n)
    partial_sum = raw[off : off + n].view(torch.float32).view(rows_max, num_split)
    return partial_o, partial_max, partial_sum


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _device_facts(device_index: int) -> tuple[str, int]:
    """``(arch, sm_count)`` of a device, read once."""
    props = torch.cuda.get_device_properties(device_index)
    return f"sm_{props.major}{props.minor}a", int(props.multi_processor_count)


_DENSE_Q_INDPTR: dict[tuple[int, int, int], torch.Tensor] = {}


def _dense_q_indptr(batch: int, q_len: int, device: torch.device) -> torch.Tensor:
    """``[0, q_len, 2 q_len, ...]`` for a dense ``[B, q_len, H, 576]`` query, kept per shape and device."""
    key = (batch, q_len, _device_index(device))
    cached = _DENSE_Q_INDPTR.get(key)
    if cached is None:
        cached = torch.arange(
            0, (batch + 1) * q_len, q_len, dtype=torch.int32, device=device
        )
        # A tensor allocated during CUDA-Graph capture belongs to the graph's private pool: it
        # serves the captured launch but must not be reused by later eager calls.
        if not torch.cuda.is_current_stream_capturing():
            _DENSE_Q_INDPTR[key] = cached
    return cached


def _rows_view(t: torch.Tensor, inner: int, row_bytes: int, name: str) -> torch.Tensor:
    """``[rows, inner]`` view of the row-major leading dims of ``t`` with equally spaced rows.

    Every row keeps ``inner`` contiguous elements; the row stride (in elements) must be the same
    between any two consecutive rows of the flattened leading dimensions and a 16-byte multiple,
    so one 2-D TMA descriptor (or one dense pointer) addresses the whole tensor.
    """
    if t.ndim < 2 or t.shape[-1] != inner:
        raise ValueError(
            f"{name} must have a last dimension of {inner}, got shape {tuple(t.shape)}"
        )
    if t.shape[-1] > 1 and t.stride(-1) != 1:
        raise ValueError(
            f"{name} rows must be contiguous (last-dimension stride 1), got {t.stride(-1)}"
        )
    rows = 1
    row_stride = None
    expected = None  # stride the next leading dim must have for equal row spacing
    for axis in range(t.ndim - 2, -1, -1):
        size = int(t.shape[axis])
        if size > 1:
            stride = int(t.stride(axis))
            if row_stride is None:
                row_stride = stride
                expected = stride * size
            elif stride != expected:
                raise ValueError(
                    f"{name} rows are not equally spaced: dim {axis} stride {stride} != {expected}; "
                    "pass a view whose leading dimensions fold into one row-major run"
                )
            else:
                expected = stride * size
        rows *= size
    if row_stride is None:
        row_stride = inner
    if row_stride < inner or (row_stride * row_bytes) % TMA_ROW_ALIGN != 0:
        raise ValueError(
            f"{name} row stride must be at least {inner} elements and a {TMA_ROW_ALIGN}-byte multiple, "
            f"got {row_stride} elements ({row_stride * row_bytes} bytes)"
        )
    return torch.as_strided(t, (rows, inner), (row_stride, 1), t.storage_offset())


def _fp8_rows(t: torch.Tensor, name: str) -> torch.Tensor:
    """Byte view ``[rows, 576]`` of an FP8 query or paged cache; rows may be strided."""
    return _rows_view(t, QK_DIM, 1, name).view(torch.uint8)


def _bind_args(
    record: dict[str, Any], values: dict[str, Any], grid: tuple[int, int, int]
):
    """Positional arguments of a program in the order of its generated argument plan."""
    grid_args = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    args = []
    for kind, name in record["arg_plan"]:
        if kind == "grid":
            args.append(int(grid_args[name]))
            continue
        try:
            args.append(values[name])
        except KeyError as exc:
            raise ValueError(
                f"CAKE Kimi-K3 MLA program {record['name']} needs argument {name!r}"
            ) from exc
    return args


@functools.cache
def _kernel_entry(name: str, arch: str, ffi_entry: str):
    from ..jit.cake_kimi_k3_mla import get_cake_kimi_k3_mla_module

    return getattr(get_cake_kimi_k3_mla_module(name, arch), ffi_entry)


# ---------------------------------------------------------------------------
# Prepared launcher
# ---------------------------------------------------------------------------


class KimiK3MlaFp8PagedAttention:
    """Prepared launcher: validation and binding at construction, no allocation at ``launch``.

    Args mirror ``trtllm_batch_decode_with_kv_cache_mla``: ``query`` FP8 ``[B, q_len, H, 576]``
    or ``[total_q, H, 576]`` with ``cum_seq_lens_q`` (rows may be strided, see the module
    docstring); ``kv_cache`` FP8 ``[pages, 64, 576]`` or ``[pages, 1, 64, 576]`` (token rows may
    be strided); ``block_tables`` int32 ``[B, width]``; ``seq_lens`` int32 ``[B]``; ``out`` BF16
    ``query.shape[:-1] + (512,)`` with dense rows; ``workspace_buffer`` a CUDA byte buffer of at
    least ``workspace_bytes(rows_max, num_split)`` bytes.
    """

    def __init__(
        self,
        *,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        out: torch.Tensor,
        workspace_buffer: torch.Tensor,
        bmm1_scale: float,
        bmm2_scale: float = 1.0,
        cum_seq_lens_q: Optional[torch.Tensor] = None,
        max_q_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        num_split: Optional[int] = None,
    ):
        from ..jit.cake_kimi_k3_mla import get_cake_kimi_k3_mla_kernel

        device = query.device
        if device.type != "cuda":
            raise ValueError("query must be a CUDA tensor")
        if query.dtype != torch.float8_e4m3fn or kv_cache.dtype != torch.float8_e4m3fn:
            raise ValueError("query and kv_cache must be float8_e4m3fn")
        if query.shape[-1] != QK_DIM or kv_cache.shape[-1] != QK_DIM:
            raise ValueError(f"query / kv_cache last dim must be {QK_DIM}")
        if kv_cache.shape[-2] != PAGE_SIZE:
            raise ValueError(f"page_size must be {PAGE_SIZE}")
        if query.ndim == 4:
            batch, q_len, num_heads, _ = query.shape
            if cum_seq_lens_q is None:
                cum_seq_lens_q = _dense_q_indptr(int(batch), int(q_len), device)
            max_q_len = int(q_len)
        elif query.ndim == 3:
            if cum_seq_lens_q is None or max_q_len is None:
                raise ValueError("ragged query needs cum_seq_lens_q and max_q_len")
            batch = int(cum_seq_lens_q.shape[0]) - 1
            _, num_heads, _ = query.shape
        else:
            raise ValueError("query must be 3D or 4D")
        for name, t in (
            ("seq_lens", seq_lens),
            ("block_tables", block_tables),
            ("cum_seq_lens_q", cum_seq_lens_q),
        ):
            if t.dtype != torch.int32:
                raise ValueError(f"{name} must be int32")
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if out.dtype != torch.bfloat16 or out.shape[-1] != V_DIM:
            raise ValueError("out must be BF16 with shape query.shape[:-1] + (512,)")
        q_rows = _fp8_rows(query, "query")
        kv_rows = _fp8_rows(kv_cache, "kv_cache")
        o_rows = _rows_view(out, V_DIM, 2, "out")
        if o_rows.shape[0] != q_rows.shape[0]:
            raise ValueError(
                "out must hold one 512-element row per (token, head) row of query"
            )
        if o_rows.stride(0) != V_DIM:
            raise ValueError(
                f"out rows must be dense ({V_DIM} elements apart); got a row stride of {o_rows.stride(0)}"
            )
        if max_seq_len is None:
            max_seq_len = int(block_tables.shape[-1]) * PAGE_SIZE
        self.arch, sm_count = _device_facts(_device_index(device))
        self.batch = int(batch)
        self.num_heads = int(num_heads)
        self.max_q_len = int(max_q_len)
        self.plan = plan_attention(
            batch=self.batch,
            max_q_len=self.max_q_len,
            num_heads=self.num_heads,
            max_seq_len=int(max_seq_len),
            sm_count=sm_count,
            num_split=None if num_split is None else int(num_split),
        )
        plan = self.plan
        self.rt = plan.rt
        self.num_split = plan.num_split
        self.rows_max = plan.rows_max
        self.max_pages_per_seq = int(block_tables.shape[-1])
        self.softmax_scale_log2 = float(bmm1_scale) * math.log2(math.e)
        self.bmm2_scale = float(bmm2_scale)
        self.q_rows = q_rows
        self.kv_rows = kv_rows
        self.o_rows = o_rows
        self.seq_lens = seq_lens
        self.cum_seq_lens_q = cum_seq_lens_q
        self.block_tables = block_tables.reshape(-1)
        self.partial_O, self.partial_max, self.partial_sum = _carve_workspace(
            workspace_buffer, plan.rows_max, plan.num_split
        )
        main = get_cake_kimi_k3_mla_kernel(plan.main_kind, arch=self.arch)
        reduce = get_cake_kimi_k3_mla_kernel(plan.reduce_kind, arch=self.arch)
        self.route_metadata = dict(
            backend="cake",
            arch=self.arch,
            route="wide" if plan.wide else "swapped",
            rt=plan.rt,
            reducer=plan.reduce_kind,
            main_module=main["name"],
            reduce_module=reduce["name"],
        )
        write_target = self.o_rows if plan.num_split == 1 else self.partial_O
        main_values = dict(
            tmap_q=self.q_rows,
            tmap_qr=self.q_rows,
            tmap_k=self.kv_rows,
            tmap_kr=self.kv_rows,
            tmap_v=self.kv_rows,
            partial_O=write_target,
            partial_max=self.partial_max,
            partial_sum=self.partial_sum,
            seq_lens=self.seq_lens,
            cum_seq_lens_q=self.cum_seq_lens_q,
            page_table=self.block_tables,
            softmax_scale_log2=self.softmax_scale_log2,
            bmm2_scale=self.bmm2_scale,
            num_heads=self.num_heads,
            num_split=plan.num_split,
            max_pages_per_seq=self.max_pages_per_seq,
        )
        if plan.wide:
            # Unsplit items store the caller's O directly; the flat grid decodes items from these two scalars.
            main_values.update(
                O=self.o_rows, m_tiles=plan.m_tiles, n_full_items=plan.n_full_items
            )
        self._main_fn = _kernel_entry(main["name"], self.arch, main["ffi_entry"])
        self._main_args = _bind_args(main, main_values, plan.grid_main)
        self._reduce_fn = None
        self._reduce_args = None
        if plan.num_split > 1:
            self._reduce_fn = _kernel_entry(
                reduce["name"], self.arch, reduce["ffi_entry"]
            )
            self._reduce_args = _bind_args(
                reduce,
                dict(
                    partial_O=self.partial_O,
                    partial_max=self.partial_max,
                    partial_sum=self.partial_sum,
                    O=self.o_rows,
                    cum_seq_lens_q=self.cum_seq_lens_q,
                    batch=self.batch,
                    num_heads=self.num_heads,
                    num_split=plan.num_split,
                    bmm2_scale=self.bmm2_scale,
                    m_tiles=plan.m_tiles,
                    n_full_items=plan.n_full_items,
                    tile_rows=plan.tile_rows,
                ),
                plan.grid_reduce,
            )

    def launch(self) -> None:
        """Enqueue the attention (and the split merge) on the current stream; no allocation."""
        self._main_fn(*self._main_args)
        if self._reduce_fn is not None:
            self._reduce_fn(*self._reduce_args)


def run_cake_kimi_k3_mla_fp8_paged_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    bmm1_scale: float,
    bmm2_scale: float = 1.0,
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    max_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """One-shot entry used by ``trtllm_batch_decode_with_kv_cache_mla(backend="cake")``.

    The plan is cached per batch shape (:func:`plan_attention`) and the device facts per device,
    so a decode loop pays only argument validation and binding per step.
    """
    KimiK3MlaFp8PagedAttention(
        query=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        out=out,
        workspace_buffer=workspace_buffer,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        max_seq_len=max_seq_len,
    ).launch()
    return out


__all__ = [
    "AttentionPlan",
    "KimiK3MlaFp8PagedAttention",
    "plan_attention",
    "plan_num_split",
    "plan_num_split_wide",
    "plan_wide_work",
    "reduce_warps_per_row",
    "run_cake_kimi_k3_mla_fp8_paged_attention",
    "swapped_rt",
    "use_wide_route",
    "workspace_bytes",
]
