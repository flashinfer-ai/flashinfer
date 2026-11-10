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
row tile (``main_rt16`` .. ``main_rt96``: the smallest tile holding the request's rows).  A
16- to 48-row tile plan that merges splits over at least ``LATE_TRIGGER_MIN_MAIN_CTAS`` main CTAs
runs its tile's late-trigger program (``main_rt16_late`` .. ``main_rt48_late``).  Both routes
share the split-KV merge kernels (warp-per-row reducers ``reduce_w4`` / ``reduce_w2`` /
``reduce_w1`` for ``num_split <= 32``, a CTA reducer otherwise: ``reduce_cta_rt`` after a row
tile whose reducer grid fits one wave, ``reduce_cta`` otherwise).  The plan (route, row tile,
split count, grids, reducer) depends only on the batch shape, the longest KV and the device's SM
count, so it is computed once per distinct shape (:func:`plan_attention`) and reused; device
facts are read once per device.

A wide plan whose split merge can overlap later attention work (:func:`use_persistent_route`) runs
the persistent wide kernel instead (``main_wide_persistent``) with the flag-waiting warp reducer
(``reduce_w<N>_tile_flags``): at most the co-resident SM pairs walk the plan's work clusters, the
split chunks first, and each split item publishes a per-tile flag that its reducer rows wait on, so
the merge runs on the pairs that finish early instead of after the whole attention grid.  Same
split plan, same per-item code and fold order: the output is bitwise the shipped kernels'.  The
flag words live in a buffer of the launcher (zeroed once when it is built); one launcher must not
run on two streams at once (as its workspace).

Three options select the extended programs of the same plan (``main_<kind>_ext`` /
``reduce_<kind>_ext``; the default call launches the programs above with the arguments above):

* ``lse``: a caller-owned FP32 buffer of one value per (token, head) row, ``[total_q, H]`` or
  ``[B, q_len, H]`` (contiguous), receives ``log(sum_k exp(bmm1_scale * q . k_k))`` over the row's
  visible keys, natural log (``lse_base="e"``) or base 2 (``lse_base="2"``); a row that sees no
  key gets ``-inf`` and a zero ``O`` row.
* ``causal=False``: every query token of a request sees all ``seq_lens[b]`` keys (uniform
  non-causal multi-token verify); ``causal_seqs`` is not read.
* ``cp_world`` / ``cp_rank`` / ``causal_seqs``: the cache holds the rank's shard of a
  decode-context-parallel KV sequence.  Global position ``p`` lives on rank ``p % cp_world`` at
  local index ``p // cp_world``; ``causal_seqs[b]`` is request ``b``'s global KV length including its
  query tokens and ``seq_lens[b]`` its local length.  Query token ``q`` of a request sees the
  local keys whose global position is below ``causal_seqs[b] - q_len + q + 1``.  Merge the ranks'
  ``(O, lse)`` with a log-sum-exp weighted combine in the matching base.  Block interleave
  (``cp_interleave > 1``) is refused.

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
REDUCE_CTA_FOLD_SPLITS = (
    16  # CTA reducer: splits merged per fold iteration (8 warps x 2)
)
EMPTY_SPLIT_TRIM_MIN = (
    2  # single-item CTA-reducer plans drop this many or more trailing empty splits
)
ROW_TILES = (16, 32, 48, 64, 96)
# The late-trigger programs signal the reducer's launch (griddepcontrol.launch_dependents) after the
# last PV commit; the other row-tile programs signal nothing, so the reducer launches when the main
# grid exits.  The early launch pays off on grids of LATE_TRIGGER_MIN_MAIN_CTAS or more main CTAs
# (32 to 152 measured on 152 SMs) and costs time on smaller ones (4 and 16 measured); a plan
# without splits has no reducer.
LATE_TRIGGER_TILES = (16, 32, 48)
LATE_TRIGGER_MIN_MAIN_CTAS = 32
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
# Longest work item (KV tokens) that takes the persistent wide route, per device: (compute
# capability, SM count).  B200 ((10, 0), 148 SMs): 16384.  Every other device keeps the shipped
# kernels until its own measurement, among them GB300 ((10, 3), 152 SMs) and B300 ((10, 3), 148
# SMs): the SM count alone does not identify the measured device.
PERSISTENT_MAX_ITEM_KV = {((10, 0), 148): 16384}
# Merge reducer classes (``reduce_warps_per_row`` of the plan's merge rows) that keep the shipped
# kernels instead of the persistent wide route, per device (as above).  B200: single-warp reducer
# plans (more than 1024 merge rows) keep the shipped kernels.  On one such shape the measurement
# did not show the persistent route faster than the shipped kernels, so the class keeps them
# until a separate measurement confirms a gain.
PERSISTENT_SHIPPED_REDUCE_WARPS = {((10, 0), 148): (1,)}
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
# Persistent wide route: one CTA of the wide kernels per SM (WIDE_CTA_SMEM_BYTES of dynamic shared
# memory), WIDE_CTA_THREADS threads, clusters of WIDE_CLUSTER CTAs.
WIDE_CTA_THREADS = 384
WIDE_CTA_SMEM_BYTES = 232448
TMA_ROW_ALIGN = 16  # bytes: TMA global strides are 16-byte multiples
# log2 of the probability scale of each attention schedule (row tiles: 224 = 2^7.807355, wide: 112):
# the split statistics carry it, so the reducer's LSE subtracts it (``lse_bias``).  The literals are
# the kernels' own f32 constants.
ROW_TILE_P_SCALE_LOG2 = 7.807355
WIDE_P_SCALE_LOG2 = 6.807355
# lse_scale: log2-sum-exp -> natural log or base 2.
LSE_BASES = {"e": math.log(2.0), "2": 1.0}


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
    """One CTA per SM per wave over (work item, split) pairs without a ragged tail.

    The row-tile kernel deals each split ``ceil(tiles / num_split)`` of a row tile's own tiles. With one
    work item of ``max_seq_len`` keys only ``ceil(tiles / ceil(tiles / num_split))`` splits hold tiles.
    The plan drops the empty splits where that removes a CTA-reducer fold iteration of
    ``REDUCE_CTA_FOLD_SPLITS`` splits (the same tile runs; the reducer merges fewer partials), or when at
    least ``EMPTY_SPLIT_TRIM_MIN`` splits are empty: the reducer loads the partial rows of every split, an
    empty split never writes its rows, so those loads miss L2 and the last fold iteration waits for DRAM.
    With more items the count stays: a shorter row tile's runs depend on it.
    """
    target = max(1, sm_count // max(1, items))
    tiles = max(1, (max_seq_len + TILE_TOK - 1) // TILE_TOK)
    max_by_len = max(1, tiles // MIN_TILES_PER_SPLIT)
    splits = max(1, min(target, MAX_SPLITS, max_by_len))
    if items == 1 and splits > REDUCE_WARP_MAX_SPLITS:
        trimmed = -(-tiles // -(-tiles // splits))
        fewer_folds = -(-trimmed // REDUCE_CTA_FOLD_SPLITS) < -(
            -splits // REDUCE_CTA_FOLD_SPLITS
        )
        if fewer_folds or splits - trimmed >= EMPTY_SPLIT_TRIM_MIN:
            splits = trimmed
    return splits


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
    if wide:
        main_kind = "main_wide"
    elif (
        rt in LATE_TRIGGER_TILES
        and splits > 1
        and splits * m_tiles * batch >= LATE_TRIGGER_MIN_MAIN_CTAS
    ):
        main_kind = f"main_rt{rt}_late"
    else:
        main_kind = f"main_rt{rt}"
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
        # The row tiles' CTA reducer prefetches every split's partials; its registers cost occupancy,
        # so it serves only a row-tile plan whose reducer grid fits one wave.
        one_wave = reduce_rows * REDUCE_DIM_CHUNKS <= sm_count
        reduce_kind = "reduce_cta_rt" if not wide and one_wave else "reduce_cta"
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
        main_kind=main_kind,
        reduce_kind=reduce_kind,
    )


def wide_work_clusters(plan: AttentionPlan) -> int:
    """Work clusters of a wide plan: one per unsplit item, ``num_split`` per tail item."""
    return plan.grid_main[0] // WIDE_CLUSTER


def use_persistent_route(
    plan: AttentionPlan, capability: tuple[int, int], sm_count: int, max_seq_len: int
) -> bool:
    """Whether a plan runs the persistent wide kernel with the flag-waiting reducer.

    With ``s`` splits, ``nf`` unsplit items, ``W`` work clusters, ``P = sm_count // 2`` SM pairs
    and ``C`` split chunks (``tail x s`` on a tail plan, ``W`` on a uniform one): a warp-reducer
    merge exists (``2 <= s <= 32``), a second round of work clusters exists (``W > P``: an item's
    epilogue overlaps the next item, and a merge can overlap later work), every split chunk runs in
    the first round (``nf == 0`` or ``C <= P``: split chunks go first), and some pairs run out of
    work before the grid ends (``ceil(W / P) x P > W``: only they can host the flag-waiting merge;
    a reducer CTA never shares an SM with a running attention CTA).

    The gains (an item's epilogue hidden under the next item, the tail merge hidden under later
    work) are fixed per item boundary, so the route also needs a tail plan (``nf > 0``: unsplit
    items whose boundaries overlap) and items no longer than ``PERSISTENT_MAX_ITEM_KV`` for the
    device (``max_seq_len``, the keys one unsplit item streams).  Both tables are keyed by the
    device's compute capability and SM count (``(capability, sm_count)``): a device without a value
    keeps the shipped kernels, as does a plan whose merge reducer class is listed in
    ``PERSISTENT_SHIPPED_REDUCE_WARPS`` for the device (``reduce_warps``).
    """
    if not plan.wide or not 2 <= plan.num_split <= REDUCE_WARP_MAX_SPLITS:
        return False
    device = ((int(capability[0]), int(capability[1])), int(sm_count))
    max_item_kv = PERSISTENT_MAX_ITEM_KV.get(device)
    if plan.n_full_items == 0 or max_item_kv is None or int(max_seq_len) > max_item_kv:
        return False
    if plan.reduce_warps in PERSISTENT_SHIPPED_REDUCE_WARPS.get(device, ()):
        return False
    work = wide_work_clusters(plan)
    pairs = max(1, sm_count // WIDE_CLUSTER)
    chunks = work - plan.n_full_items
    rounds = -(-work // pairs)
    return (
        work > pairs
        and (plan.n_full_items == 0 or chunks <= pairs)
        and rounds * pairs > work
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
def _device_facts(device_index: int) -> tuple[str, tuple[int, int], int]:
    """``(arch, capability, sm_count)`` of a device, read once."""
    props = torch.cuda.get_device_properties(device_index)
    return (
        f"sm_{props.major}{props.minor}a",
        (int(props.major), int(props.minor)),
        int(props.multi_processor_count),
    )


# A minimal kernel the driver JIT-compiles for the pair capacity probe: the capacity depends on the
# launch shape (block, shared memory, cluster), not on the body.
_PAIR_PROBE_PTX = b"""
.version 7.8
.target sm_90
.address_size 64
.visible .entry cake_kimi_k3_mla_pair_probe()
{
    ret;
}
\x00"""

_WIDE_PAIR_CAPACITY: dict[int, int] = {}


def _wide_pair_capacity(device_index: int) -> int:
    """How many two-CTA clusters of the wide kernels the device holds at once.

    ``cuOccupancyMaxActiveClusters`` for a probe kernel launched with the wide CTA's threads and
    dynamic shared memory in clusters of two; the driver's answer follows the part's floorsweeping,
    which can leave it below ``sm_count // 2``.  Only an answer is cached (per device): when the
    driver cannot answer, the call returns 0 (the plan keeps the shipped wide kernels) and probes
    again on the next call.  No probe runs inside a CUDA-Graph capture (0 there too): a shape
    captured before its first eager call keeps the shipped wide kernels in that graph.
    """
    capacity = _WIDE_PAIR_CAPACITY.get(device_index)
    if capacity is not None:
        return capacity
    if torch.cuda.is_current_stream_capturing():
        return 0
    try:
        from cuda.bindings import driver

        def check(result):
            err, *values = result if isinstance(result, tuple) else (result,)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"CUDA driver error {err}")
            return values[0] if values else None

        with torch.cuda.device(device_index):
            torch.empty(
                0, device=f"cuda:{device_index}"
            )  # the primary context is current
            module = check(driver.cuModuleLoadData(_PAIR_PROBE_PTX))
            try:
                function = check(
                    driver.cuModuleGetFunction(module, b"cake_kimi_k3_mla_pair_probe")
                )
                check(
                    driver.cuFuncSetAttribute(
                        function,
                        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        WIDE_CTA_SMEM_BYTES,
                    )
                )
                config = driver.CUlaunchConfig()
                config.gridDimX, config.gridDimY, config.gridDimZ = WIDE_CLUSTER, 1, 1
                config.blockDimX, config.blockDimY, config.blockDimZ = (
                    WIDE_CTA_THREADS,
                    1,
                    1,
                )
                config.sharedMemBytes = WIDE_CTA_SMEM_BYTES
                cluster = driver.CUlaunchAttribute()
                cluster.id = (
                    driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
                )
                cluster.value.clusterDim.x = WIDE_CLUSTER
                cluster.value.clusterDim.y = 1
                cluster.value.clusterDim.z = 1
                config.attrs = [cluster]
                config.numAttrs = 1
                capacity = int(
                    check(driver.cuOccupancyMaxActiveClusters(function, config))
                )
            finally:
                driver.cuModuleUnload(module)
    except Exception:  # noqa: BLE001
        return 0
    _WIDE_PAIR_CAPACITY[device_index] = capacity
    return capacity


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


def _is_pow2(n: int) -> bool:
    return (
        isinstance(n, int) and not isinstance(n, bool) and n >= 1 and n & (n - 1) == 0
    )


def _ext_values(
    *,
    lse: Optional[torch.Tensor],
    lse_base: str,
    causal: bool,
    causal_seqs: Optional[torch.Tensor],
    cp_world: int,
    cp_rank: int,
    cp_interleave: int,
    rows: int,
    batch: int,
    seq_lens: torch.Tensor,
    device: torch.device,
) -> Optional[dict[str, Any]]:
    """Construction-time check of the LSE / DCP / mask options; ``None`` keeps the default programs.

    Otherwise the arguments the ``_ext`` programs take besides the default ones: ``lse`` (the caller's
    buffer, or a launcher-owned one when only the mask changes), ``lse_scale``, ``causal_seqs``
    (``seq_lens`` without DCP: the bound is then the local bottom-right one), ``causal`` and the shard
    geometry ``cp_shift = log2(W * I)``, ``cp_ishift = log2(I)``, ``cp_rank_off = rank * I``,
    ``cp_span = I``.
    """
    if lse_base not in LSE_BASES:
        raise ValueError(f"lse_base must be 'e' or '2', got {lse_base!r}")
    if not isinstance(causal, bool):
        raise TypeError(f"causal must be a bool, got {type(causal).__name__}")
    if not _is_pow2(cp_world):
        raise ValueError(f"cp_world must be a power of two >= 1, got {cp_world!r}")
    if (
        not isinstance(cp_rank, int)
        or isinstance(cp_rank, bool)
        or not 0 <= cp_rank < cp_world
    ):
        raise ValueError(
            f"cp_rank must be an integer with 0 <= cp_rank < cp_world={cp_world}, got {cp_rank!r}"
        )
    if not _is_pow2(cp_interleave):
        raise ValueError(
            f"cp_interleave must be a power of two >= 1, got {cp_interleave!r}"
        )
    if cp_interleave != 1:
        raise NotImplementedError(
            "CAKE Kimi-K3 MLA supports cyclic DCP sharding only (cp_interleave=1)"
        )
    if cp_world > 1:
        if causal_seqs is None:
            raise ValueError("causal_seqs is required when cp_world > 1")
        if not isinstance(causal_seqs, torch.Tensor):
            raise TypeError("causal_seqs must be a torch.Tensor")
        if causal_seqs.dtype != torch.int32:
            raise ValueError(f"causal_seqs must be int32, got {causal_seqs.dtype}")
        if causal_seqs.device != device:
            raise ValueError(
                f"causal_seqs must be on the query device {device}, got {causal_seqs.device}"
            )
        if tuple(causal_seqs.shape) != (batch,):
            raise ValueError(
                f"causal_seqs must have shape ({batch},), got {tuple(causal_seqs.shape)}"
            )
        if not causal_seqs.is_contiguous():
            raise ValueError("causal_seqs must be contiguous")
    elif causal_seqs is not None:
        raise ValueError(
            "causal_seqs is only read with cp_world > 1 (the global length of a sharded request)"
        )
    if lse is not None:
        if not isinstance(lse, torch.Tensor):
            raise TypeError("lse must be a torch.Tensor")
        if lse.dtype != torch.float32:
            raise ValueError(f"lse must be float32, got {lse.dtype}")
        if lse.device != device:
            raise ValueError(
                f"lse must be on the query device {device}, got {lse.device}"
            )
        if not lse.is_contiguous():
            raise ValueError("lse must be contiguous")
        if lse.numel() != rows:
            raise ValueError(
                f"lse must hold one value per (token, head) row of query ({rows}), got "
                f"{lse.numel()} ([total_q, H] or [B, q_len, H])"
            )
    if lse is None and causal and cp_world == 1:
        return None
    lse_rows = (
        lse.reshape(-1)
        if lse is not None
        else torch.empty(max(rows, 1), dtype=torch.float32, device=device)
    )
    shift_i = cp_interleave.bit_length() - 1
    return dict(
        lse=lse_rows,
        lse_scale=LSE_BASES[lse_base],
        causal_seqs=causal_seqs if cp_world > 1 else seq_lens,
        causal=int(causal),
        cp_shift=(cp_world * cp_interleave).bit_length() - 1,
        cp_ishift=shift_i,
        cp_rank_off=cp_rank * cp_interleave,
        cp_span=cp_interleave,
    )


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
    least ``workspace_bytes(rows_max, num_split)`` bytes (the options below do not change it).

    ``lse`` (FP32, one value per query row, caller-owned), ``lse_base`` (``"e"`` or ``"2"``),
    ``causal`` (``False``: every query token sees all ``seq_lens[b]`` keys), ``cp_world`` /
    ``cp_rank`` / ``causal_seqs`` (int32 ``[B]`` global lengths, required iff ``cp_world > 1``) and
    ``cp_interleave`` (only 1) follow the module docstring; they are bound here, so ``launch`` stays
    allocation-free and CUDA-Graph replayable with new ``seq_lens`` / ``causal_seqs`` contents.
    Plan the shard with its local ``max_seq_len``.
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
        lse: Optional[torch.Tensor] = None,
        lse_base: str = "e",
        causal: bool = True,
        causal_seqs: Optional[torch.Tensor] = None,
        cp_world: int = 1,
        cp_rank: int = 0,
        cp_interleave: int = 1,
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
        ext = _ext_values(
            lse=lse,
            lse_base=lse_base,
            causal=causal,
            causal_seqs=causal_seqs,
            cp_world=cp_world,
            cp_rank=cp_rank,
            cp_interleave=cp_interleave,
            rows=int(q_rows.shape[0]),
            batch=int(batch),
            seq_lens=seq_lens,
            device=device,
        )
        self.arch, capability, sm_count = _device_facts(_device_index(device))
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
        grid_main = plan.grid_main
        main_kind, reduce_kind = plan.main_kind, plan.reduce_kind
        self.tile_flags = None
        pairs = 0
        if use_persistent_route(plan, capability, sm_count, int(max_seq_len)):
            pairs = min(
                wide_work_clusters(plan), _wide_pair_capacity(_device_index(device))
            )
        if pairs > 0:
            # The persistent wide kernel walks the plan's work clusters on at most the co-resident
            # SM pairs; its reducer waits per tile on the flag words of this launcher: [launch epoch,
            # CTA ticket, publishes of each tail item], zeroed once here and kept consistent by the
            # kernels from launch to launch (CUDA-Graph replays included).
            main_kind = "main_wide_persistent"
            reduce_kind = f"{plan.reduce_kind}_tile_flags"
            grid_main = (WIDE_CLUSTER * pairs, 1, 1)
            tail_items = plan.m_tiles * self.batch - plan.n_full_items
            self.tile_flags = torch.zeros(
                2 + tail_items, dtype=torch.int32, device=device
            ).view(torch.uint32)
        self.lse = None if ext is None else ext["lse"]
        if ext is not None:
            # Same plan, grids and workspace; the _ext programs add the LSE store and the
            # visible-key bound.
            main_kind, reduce_kind = f"{main_kind}_ext", f"{reduce_kind}_ext"
        main = get_cake_kimi_k3_mla_kernel(main_kind, arch=self.arch)
        reduce = get_cake_kimi_k3_mla_kernel(reduce_kind, arch=self.arch)
        self.route_metadata = dict(
            backend="cake",
            arch=self.arch,
            route="wide" if plan.wide else "swapped",
            rt=plan.rt,
            reducer=reduce_kind,
            persistent=self.tile_flags is not None,
            ext=ext is not None,
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
        if self.tile_flags is not None:
            main_values.update(
                n_work_clusters=wide_work_clusters(plan), tile_flags=self.tile_flags
            )
        if ext is not None:
            main_values.update(ext)
        self._main_fn = _kernel_entry(main["name"], self.arch, main["ffi_entry"])
        self._main_args = _bind_args(main, main_values, grid_main)
        self._reduce_fn = None
        self._reduce_args = None
        if plan.num_split > 1:
            self._reduce_fn = _kernel_entry(
                reduce["name"], self.arch, reduce["ffi_entry"]
            )
            reduce_values = dict(
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
            )
            if self.tile_flags is not None:
                reduce_values.update(tile_flags=self.tile_flags)
            if ext is not None:
                p_scale_log2 = WIDE_P_SCALE_LOG2 if plan.wide else ROW_TILE_P_SCALE_LOG2
                reduce_values.update(
                    lse=ext["lse"], lse_scale=ext["lse_scale"], lse_bias=-p_scale_log2
                )
            self._reduce_args = _bind_args(reduce, reduce_values, plan.grid_reduce)

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
    lse: Optional[torch.Tensor] = None,
    lse_base: str = "e",
    causal: bool = True,
    causal_seqs: Optional[torch.Tensor] = None,
    cp_world: int = 1,
    cp_rank: int = 0,
    cp_interleave: int = 1,
) -> torch.Tensor:
    """One-shot entry used by ``trtllm_batch_decode_with_kv_cache_mla(backend="cake")``.

    The plan is cached per batch shape (:func:`plan_attention`) and the device facts per device,
    so a decode loop pays only argument validation and binding per step.  ``lse`` and the mask /
    DCP options are the :class:`KimiK3MlaFp8PagedAttention` ones.
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
        lse=lse,
        lse_base=lse_base,
        causal=causal,
        causal_seqs=causal_seqs,
        cp_world=cp_world,
        cp_rank=cp_rank,
        cp_interleave=cp_interleave,
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
    "use_persistent_route",
    "use_wide_route",
    "wide_work_clusters",
    "workspace_bytes",
]
