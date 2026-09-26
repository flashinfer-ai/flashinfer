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

# Cake MLA decode with compact variable-length queries and decode context
# parallelism (DCP).
#
# Host side of the generated Cake program for DeepSeek MLA decode (576 = 512
# latent + 64 rope per key, 512-wide value) over a rank-local paged BF16 or
# FP8 (e4m3) KV cache with
#
# * compact variable-length queries Q[total_q, H, 576]: request b owns rows
#   cum_seq_lens_q[b] .. cum_seq_lens_q[b + 1] (empty requests are legal),
#   max_q_len is the static per-request capacity;
# * static cyclic DCP: rank r of W holds the global positions W * k + r; local
#   key k of request b is visible to query token q iff k < seq_lens[b] and
#   W * k + r <= G[b] - q_len_b + q with G = causal_seqlens_kv_global;
# * compact outputs O[total_q, H, 512] (BF16) and natural-log LSE[total_q, H]
#   (FP32); rows without a visible key write O = 0 and LSE = -inf so the
#   LSE-weighted cross-rank merge works unchanged.
#
# The device kernel balances work itself: a scheduler warp per two-CTA
# cluster claims units of KV tiles from a self-resetting ticket counter (the
# *ticket* variant) or derives one balanced static unit per cluster (the
# *partition* variant, chosen by the host plan for few long items).  Items
# that split across units write BF16 partials to the caller-owned workspace;
# the ticket variant merges them with a second kernel (mla_varq_dcp_merge)
# launched programmatically behind the main kernel, the partition variant
# merges in-kernel.  The host only fixes the launch geometry, the scheduler
# knobs and replay-safe buffer bounds (plan_varq_dcp_decode); nothing here
# reads a CUDA tensor's contents.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from .cake_jit import (
    MODULES,
    load_cake_mla_varq_dcp_decode_module,
    main_route_available,
    route_name,
    select_main_module,
    select_merge_module,
)

# DeepSeek MLA geometry served by the generated program.
HEAD_DIM_QK = 576  # 512 latent (nope) + 64 rope
HEAD_DIM_V = 512
TILE_Q = 128  # flattened (token, head) rows per cluster (MMA M)
TILE_KV = 128  # KV tokens per pipeline tile
CLUSTER_SIZE = 2  # CTAs per cluster (cta_group::2)
SUPPORTED_PAGE_SIZES = (32, 64, 128)
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
KV_DTYPES = {torch.bfloat16: "bf16", torch.float8_e4m3fn: "fp8"}

# Scheduler ABI shared with the generated program (part of the ABI freeze).
MAX_ITEM_GROUPS = 16  # 32 (request, m_tile) items per scheduler lane group
MAX_ITEMS = 32 * MAX_ITEM_GROUPS
ITEM_GROUP_VARIANTS = (4, MAX_ITEM_GROUPS)  # traced main-kernel capacities
M1_GROUPS = 4  # partition variant: scheduler pass over <= 128 items
MERGE_JOBS = 8  # merge kernel jobs per split item (16 rows each)
META_WORDS = 2  # split table entry: units | slot_base << 9, item index
SCHED_COUNTER_WORDS = 8
MERGE_CTL_WORDS = 4
DBG_WORDS = 1  # %globaltimer stamp buffer of instrumented builds (one word otherwise)
DEFAULT_UNIT_MIN = 4
DEFAULT_UNIT_RATIO = (1, 1)  # U = max(unit_min, ceil(tiles_per_cluster * num / den))
LONG_ITEM_TILES = 32  # items at least this long take U = share / 2
NO_SPLIT_MAX_TILES = 8  # items this short never split
WORKSPACE_ALIGNMENT = 256
LOG2E = 1.4426950408889634

# Keyword names the two stages are bound with (the registry's arg plans refer
# to these names; ``_bind_stage`` orders them by the generated argument plan).
MAIN_KWARGS = (
    "tmap_q",
    "tmap_k",
    "tmap_v",
    "tmap_o",
    "tmap_po",
    "O",
    "LSE",
    "partial_O",
    "partial_lse",
    "page_table",
    "seq_lens",
    "cum_seq_lens_q",
    "causal_global",
    "sched_counters",
    "unit_flags",
    "split_meta",
    "merge_ctl",
    "softmax_scale_log2",
    "tiles_max",
    "num_heads",
    "max_pages",
    "cp_world",
    "cp_rank",
    "num_items",
    "unit_min",
    "static_tiles",
    "unit_num",
    "unit_den",
    "max_units",
    "static_only",
    "dbg",
    "partial_slots",
    "fd_tiles_max",
    "fd_num_heads",
    "fd_cp_world",
    "fd_num_items",
    "fd_unit_min",
    "fd_static_tiles",
    "fd_unit_den",
    "fd_clusters",
    "grid",
)
MERGE_KWARGS = (
    "partial_O",
    "partial_lse",
    "O",
    "LSE",
    "unit_flags",
    "split_meta",
    "merge_ctl",
    "cum_seq_lens_q",
    "num_heads",
    "tiles_max",
    "max_records",
    "grid",
)


def _fast_divmod(divisor: int) -> tvm_ffi.Shape:
    """Three-field carrier of a generated ``LoomFastDivmod`` parameter: the
    divisor with its CUTLASS FastDivmod multiplier and shift (host-derived so
    the kernel's single-warp prologue passes divide with one umulhi + shift)."""
    divisor = int(divisor)
    if not 1 <= divisor <= 0x7FFF_FFFF:
        raise ValueError(f"fast divmod divisor must be in [1, 2147483647], got {divisor}")
    if divisor == 1:
        return tvm_ffi.Shape((1, 0, 0))
    p = 31 + (divisor - 1).bit_length()
    multiplier = (((1 << p) + divisor - 1) // divisor) & 0xFFFF_FFFF
    return tvm_ffi.Shape((divisor, multiplier, p - 32))


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# ---------------------------------------------------------------------------
# Host plan (port of loom.examples.weave.mla_varq_dcp_decode.plan_varq_dcp_decode)
# ---------------------------------------------------------------------------


def item_groups_for(items: int) -> int:
    """Scheduler capacity variant: 32 items per lane group.

    The small variant keeps the scheduler warp's per-group register arrays
    within its budget; the large one covers up to ``MAX_ITEMS`` items.
    """
    for groups in ITEM_GROUP_VARIANTS:
        if items <= 32 * groups:
            return groups
    raise ValueError(f"{items} items exceed the scheduler capacity {MAX_ITEMS}")


def plan_varq_dcp_decode(
    *,
    batch_size: int,
    max_q_len: int,
    num_heads: int,
    max_seq_len: int,
    num_sms: int,
    unit_min: Optional[int] = None,
    unit_ratio: Optional[tuple[int, int]] = None,
    static_tiles: Optional[int] = None,
    partition_mode: Optional[int] = None,
) -> dict[str, Any]:
    """Rectangular launch geometry for one (max_q_len, H, max local len).

    The kernel balances work on the device; the host only fixes the grid, the
    scheduler knobs and replay-safe buffer bounds.  Every rule and default is
    the production plan of the Cake source module.
    """
    if num_heads <= 0 or num_heads > TILE_Q:
        raise ValueError(f"num_heads must be in [1, {TILE_Q}], got {num_heads}")
    if max_q_len <= 0:
        raise ValueError(f"max_q_len must be positive, got {max_q_len}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_sms <= 0:
        raise ValueError(f"num_sms must be positive, got {num_sms}")
    unit_min = DEFAULT_UNIT_MIN if unit_min is None else int(unit_min)
    tiles_max = _ceil_div(max_q_len * num_heads, TILE_Q)
    max_local_tiles = max(1, _ceil_div(max(max_seq_len, 1), TILE_KV))
    if unit_ratio is None:
        # Long items take dynamic units of half a cluster's share so the last
        # units of the kernel end within U/2 of each other.
        long_items = (
            tiles_max == 1 and max_local_tiles >= LONG_ITEM_TILES
        ) or max_local_tiles >= 2 * LONG_ITEM_TILES
        unit_ratio = (1, 2) if long_items else DEFAULT_UNIT_RATIO
    unit_num, unit_den = int(unit_ratio[0]), int(unit_ratio[1])
    if unit_min <= 0 or unit_num <= 0 or unit_den <= 0:
        raise ValueError("unit_min / unit_ratio must be positive")
    items = batch_size * tiles_max
    if items > MAX_ITEMS:
        raise ValueError(
            f"batch_size * tiles_max = {items} exceeds the scheduler capacity {MAX_ITEMS}"
        )
    resident = max(1, num_sms // CLUSTER_SIZE)
    # Balanced static partition variant: at least two clusters per item when
    # items split, whole short items with at most one item per cluster, and a
    # scheduler pass that covers the items.
    whole_items_m1 = items <= resident and max_local_tiles <= NO_SPLIT_MAX_TILES
    split_items_m1 = (
        2 * items <= resident
        and max_local_tiles > NO_SPLIT_MAX_TILES
        and unit_min < max_local_tiles
    )
    eligible_m1 = (whole_items_m1 or split_items_m1) and items <= 32 * M1_GROUPS
    if partition_mode is None:
        partition_mode = int(eligible_m1)
    partition_mode = int(bool(partition_mode))
    if partition_mode and not eligible_m1:
        raise ValueError(
            "partition_mode 1 needs items <= clusters (2 * items when items split) "
            "and items <= 32 * M1_GROUPS"
        )
    if max_local_tiles <= NO_SPLIT_MAX_TILES or (
        items >= resident and max_local_tiles <= 2 * NO_SPLIT_MAX_TILES
    ):
        # Short items run whole: a unit at least as long as any item makes
        # that a plan-level fact (no partial buffers, no merge launch).
        unit_min = max(unit_min, max_local_tiles)
    max_units_per_item = max(1, _ceil_div(max_local_tiles, unit_min))
    max_tickets = items * max_units_per_item
    grid_clusters = max(1, min(resident, max_tickets))
    partial_slots = (
        2 * items + grid_clusters * _ceil_div(unit_den, unit_num) + grid_clusters
    )
    partial_slots = min(partial_slots, max_tickets)
    if static_tiles is None:
        static_tiles = max(
            unit_min, _ceil_div(items * max_local_tiles, 2 * grid_clusters)
        )
        if max_local_tiles <= NO_SPLIT_MAX_TILES or 2 * items >= 3 * grid_clusters:
            # short items, or at least 1.5 items per cluster: whole items
            static_tiles = max_local_tiles
        elif max_local_tiles >= 2 * LONG_ITEM_TILES:
            # 64-tile items: half the item as the static unit
            static_tiles = max(static_tiles, max_local_tiles // 2)
    static_tiles = int(static_tiles)
    if static_tiles <= 0:
        raise ValueError("static_tiles must be positive")
    if partition_mode:
        # One unit per cluster (partial slot = cluster id), no tickets.
        grid_clusters = resident
        partial_slots = grid_clusters
        max_tickets = 0
    return {
        "tiles_max": tiles_max,
        "items": items,
        "max_local_tiles": max_local_tiles,
        "unit_min": unit_min,
        "static_tiles": static_tiles,
        "unit_num": unit_num,
        "unit_den": unit_den,
        "grid_clusters": grid_clusters,
        "max_units": max_tickets + 1,
        "partition_mode": partition_mode,
        "merge_grid": max(1, min(items * MERGE_JOBS, num_sms)),
        "partial_rows": partial_slots * TILE_Q,
        "can_split": max_local_tiles > unit_min,
        "static_only": int(
            (
                not partition_mode
                and items <= grid_clusters
                and max_local_tiles <= static_tiles
            )
            or (partition_mode and max_local_tiles <= unit_min)
        ),
    }


def launches_merge(plan: dict[str, Any]) -> bool:
    """True when the plan launches the split-KV merge kernel behind the main kernel."""
    return bool(plan["can_split"]) and not bool(plan["partition_mode"])


# ---------------------------------------------------------------------------
# Workspace layout
# ---------------------------------------------------------------------------


def _align(nbytes: int) -> int:
    return (
        (nbytes + WORKSPACE_ALIGNMENT - 1) // WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT
    )


def _region_sizes(
    *, partial_rows: int, can_split: bool, partition_mode: int, items: int
) -> tuple[tuple[str, int], ...]:
    partial_slots = partial_rows // TILE_Q
    return (
        ("partial_o", partial_rows * HEAD_DIM_V * 2 if can_split else 0),
        ("partial_lse", partial_rows * 4 if can_split else 0),
        ("sched_counters", SCHED_COUNTER_WORDS * 4),
        ("unit_flags", (4 if partition_mode else 1) * partial_slots * 4),
        ("split_meta", META_WORDS * items * 4),
        ("merge_ctl", MERGE_CTL_WORDS * 4),
        ("dbg", DBG_WORDS * 8),
    )


def workspace_layout(plan: dict[str, Any]) -> dict[str, Any]:
    """Byte offsets and sizes of every workspace region plus ``"total"``.

    Regions (256-byte aligned): BF16 ``partial_o [partial_rows, 512]`` and
    FP32 ``partial_lse [partial_rows]`` (absent when the plan cannot split:
    the kernel then binds the output itself), the scheduler counters, the
    partial-slot flags (one word per slot, four per slot in the partition
    variant), the split table and the merge control words, and the one-word
    timestamp buffer of instrumented builds.
    """
    layout: dict[str, Any] = {}
    offset = 0
    for name, nbytes in _region_sizes(
        partial_rows=int(plan["partial_rows"]),
        can_split=bool(plan["can_split"]),
        partition_mode=int(plan["partition_mode"]),
        items=int(plan["items"]),
    ):
        layout[name] = (offset, nbytes)
        offset += _align(nbytes)
    layout["total"] = offset
    return layout


def cake_mla_varq_dcp_decode_workspace_size(
    *,
    batch_size: int,
    max_q_len: int,
    num_heads: int,
    max_seq_len: int,
    num_sms: int,
    unit_min: Optional[int] = None,
    unit_ratio: Optional[tuple[int, int]] = None,
    static_tiles: Optional[int] = None,
    partition_mode: Optional[int] = None,
) -> int:
    """Workspace bytes ``prepare`` needs for one host plan."""
    plan = plan_varq_dcp_decode(
        batch_size=batch_size,
        max_q_len=max_q_len,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        num_sms=num_sms,
        unit_min=unit_min,
        unit_ratio=unit_ratio,
        static_tiles=static_tiles,
        partition_mode=partition_mode,
    )
    return int(workspace_layout(plan)["total"])


def max_cake_mla_varq_dcp_decode_workspace_size(
    *, batch_size: int, max_q_len: int, num_heads: int, num_sms: int
) -> int:
    """Upper bound of the workspace for any ``max_seq_len`` with default knobs.

    Partial slots are bounded by ``2 * items + 3 * clusters`` (ticket variant:
    ``ceil(unit_den / unit_num) <= 2``) and by ``clusters`` (partition
    variant); the partition variant keeps four flag words per slot.
    """
    if num_heads <= 0 or num_heads > TILE_Q or max_q_len <= 0 or batch_size <= 0:
        raise ValueError(
            "batch_size, max_q_len must be positive and num_heads in [1, 128]"
        )
    items = batch_size * _ceil_div(max_q_len * num_heads, TILE_Q)
    if items > MAX_ITEMS:
        raise ValueError(
            f"batch_size * tiles_max = {items} exceeds the scheduler capacity {MAX_ITEMS}"
        )
    resident = max(1, num_sms // CLUSTER_SIZE)
    partial_slots = max(resident, 2 * items + 3 * resident)
    total = 0
    for _name, nbytes in _region_sizes(
        partial_rows=partial_slots * TILE_Q,
        can_split=True,
        partition_mode=1,
        items=items,
    ):
        total += _align(nbytes)
    return total


def _carve(flat: torch.Tensor, layout: dict[str, Any], name: str, dtype, shape):
    offset, nbytes = layout[name]
    region = flat[offset : offset + nbytes]
    region.zero_()
    return region.view(dtype).view(shape)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CakeMLAVarQDcpDecodeRunner:
    """Launch the prepared decode (and, when planned, the split-KV merge).

    Calling the runner or ``launch()`` writes the caller-owned ``out`` / ``lse``
    on the current stream with no CUDA allocation and no host synchronization
    and returns ``(out, lse)``.  The plan, the bindings and the workspace are
    fixed at preparation; the scheduler state inside the workspace is live
    across launches (it self-resets by launch parity), so never share one
    workspace between two live runners and prepare a new runner when shapes,
    lengths (``max_seq_len``) or tensor bindings change.  CUDA Graph capture
    belongs to the caller: the direct launches record as graph nodes.
    """

    arch: str
    dtype: str
    page_size: int
    item_groups: int
    plan: dict[str, Any]
    main_module: str
    merge_module: Optional[str]
    main_kwargs: dict[str, Any]
    merge_kwargs: Optional[dict[str, Any]]
    workspace: dict[str, torch.Tensor]
    out: torch.Tensor
    lse: torch.Tensor
    _main_entry: Callable[..., Any]
    _main_arguments: tuple
    _merge_entry: Optional[Callable[..., Any]]
    _merge_arguments: tuple
    _keep: tuple

    def launch(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Tensor maps are encoded by the host binding and passed by value.  The
        # main launch carries the programmatic-dependent-launch attribute and
        # waits (griddepcontrol.wait) for the preceding work; the merge launch
        # carries the attribute too and never waits: it polls the per-unit
        # partial flags published by the main kernel.
        with tvm_ffi.use_torch_stream():
            self._main_entry(*self._main_arguments)
            if self._merge_entry is not None:
                self._merge_entry(*self._merge_arguments)
        return self.out, self.lse

    __call__ = launch

    @property
    def launches_merge(self) -> bool:
        return self._merge_entry is not None

    @property
    def route(self) -> str:
        return route_name(
            self.arch,
            self.dtype,
            self.page_size,
            self.item_groups,
            self.plan["partition_mode"],
        )

    @property
    def route_metadata(self) -> dict[str, Any]:
        return dict(
            arch=self.arch,
            dtype=self.dtype,
            page_size=self.page_size,
            item_groups=self.item_groups,
            partition_mode=int(self.plan["partition_mode"]),
            launches_merge=self.launches_merge,
            main_module=self.main_module,
            merge_module=self.merge_module,
            grid=tuple(self.main_kwargs["grid"]),
            merge_grid=tuple(self.merge_kwargs["grid"]) if self.merge_kwargs else None,
        )


def _bind_stage(
    module_name: str, kwargs: dict[str, Any]
) -> tuple[Callable[..., Any], tuple]:
    """Order ``kwargs`` by the generated argument plan of ``module_name``."""
    record = MODULES[module_name]
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), kwargs["grid"], strict=True))
    arguments = []
    for kind, name in record["arg_plan"]:
        if kind == "grid":
            arguments.append(grid[name])
        elif name in kwargs:
            arguments.append(kwargs[name])
        else:
            raise KeyError(
                f"generated module {module_name!r} expects argument {name!r} "
                f"({kind}); host binding provides {sorted(kwargs)}"
            )
    module = load_cake_mla_varq_dcp_decode_module(module_name)
    return getattr(module, record["ffi_entry"]), tuple(arguments)


def _arch_for(device: torch.device) -> str:
    capability = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            "Cake MLA var-Q DCP decode requires compute capability 10.0 or 10.3 "
            f"(got {capability[0]}.{capability[1]})"
        )
    return arch


def generated_program_available(
    device: torch.device,
    *,
    dtype: str = "bf16",
    page_size: int = 64,
    item_groups: int = 4,
    partition_mode: int = 0,
) -> bool:
    """True when this checkout registers the main variant for ``device``."""
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))
    return arch is not None and main_route_available(
        arch, dtype, page_size, item_groups, partition_mode
    )


# ---------------------------------------------------------------------------
# Validation and preparation
# ---------------------------------------------------------------------------


def _check_int32_vector(name: str, tensor: Any, numel: int) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype not in (torch.int32, torch.int64) or tensor.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional int32 tensor")
    if int(tensor.numel()) != numel:
        raise ValueError(f"{name} must have {numel} entries, got {int(tensor.numel())}")


def validate_cake_mla_varq_dcp_decode_inputs(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    *,
    max_q_len: int,
    max_seq_len: int,
    cp_world: int,
    cp_rank: int,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Shape / dtype validation shared by ``prepare``.

    Uses only tensor metadata and host scalars, so it also runs on host
    tensors; device placement and compute capability are checked separately.
    Returns ``dtype`` (``"bf16"`` / ``"fp8"``), ``batch_size``, ``num_heads``,
    ``total_q``, ``page_size``, ``num_pages`` and ``max_pages``.
    """
    for name, value in (
        ("query", query),
        ("kv_cache", kv_cache),
        ("page_table", page_table),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    dtype = KV_DTYPES.get(query.dtype)
    if dtype is None:
        raise ValueError(
            f"unsupported query / KV dtype {query.dtype}; expected bfloat16 or float8_e4m3fn"
        )
    if kv_cache.dtype != query.dtype:
        raise ValueError("query and kv_cache must share one dtype")
    if query.ndim != 3 or int(query.shape[-1]) != HEAD_DIM_QK:
        raise ValueError(f"query must have shape [total_q, num_heads, {HEAD_DIM_QK}]")
    total_q, num_heads = int(query.shape[0]), int(query.shape[1])
    if total_q <= 0:
        raise ValueError("query must hold at least one compact token row")
    if num_heads <= 0 or num_heads > TILE_Q:
        raise ValueError(f"num_heads must be in [1, {TILE_Q}], got {num_heads}")
    if kv_cache.ndim == 4:
        if int(kv_cache.shape[1]) != 1:
            raise ValueError("a 4D kv_cache must be [num_pages, 1, page_size, 576]")
        kv_shape = (
            int(kv_cache.shape[0]),
            int(kv_cache.shape[2]),
            int(kv_cache.shape[3]),
        )
    elif kv_cache.ndim == 3:
        kv_shape = tuple(int(v) for v in kv_cache.shape)  # type: ignore[assignment]
    else:
        raise ValueError(f"kv_cache must be [num_pages, page_size, {HEAD_DIM_QK}]")
    num_pages, page_size, kv_dim = kv_shape
    if kv_dim != HEAD_DIM_QK:
        raise ValueError(
            f"kv_cache rows must be {HEAD_DIM_QK} wide (512 latent + 64 rope)"
        )
    if page_size not in SUPPORTED_PAGE_SIZES:
        raise ValueError(
            f"page_size must be one of {SUPPORTED_PAGE_SIZES}, got {page_size}"
        )
    if num_pages <= 0:
        raise ValueError("kv_cache must hold at least one page")
    if page_table.ndim != 2 or page_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("page_table must be an int32 [batch_size, max_pages] tensor")
    batch_size, max_pages = int(page_table.shape[0]), int(page_table.shape[1])
    if batch_size <= 0 or max_pages <= 0:
        raise ValueError("page_table must have at least one request and one page slot")
    _check_int32_vector("seq_lens", seq_lens, batch_size)
    _check_int32_vector("cum_seq_lens_q", cum_seq_lens_q, batch_size + 1)
    if type(max_q_len) is not int or max_q_len <= 0:
        raise ValueError(
            "max_q_len must be a positive int (static per-request query capacity)"
        )
    if total_q > batch_size * max_q_len:
        raise ValueError(
            f"query holds {total_q} rows but batch_size * max_q_len = {batch_size * max_q_len}"
        )
    if type(max_seq_len) is not int or max_seq_len <= 0:
        raise ValueError(
            "max_seq_len must be a positive int (largest rank-local length)"
        )
    if max_seq_len > max_pages * page_size:
        raise ValueError(
            f"max_seq_len={max_seq_len} exceeds the page table capacity {max_pages * page_size}"
        )
    if type(cp_world) is not int or type(cp_rank) is not int or cp_world <= 0:
        raise ValueError("cp_world / cp_rank must be ints with cp_world >= 1")
    if not 0 <= cp_rank < cp_world:
        raise ValueError(
            f"cp_rank must satisfy 0 <= cp_rank < cp_world, got {cp_rank}/{cp_world}"
        )
    if causal_seqlens_kv_global is None:
        if cp_world != 1:
            raise ValueError("causal_seqlens_kv_global is required when cp_world > 1")
    else:
        _check_int32_vector(
            "causal_seqlens_kv_global", causal_seqlens_kv_global, batch_size
        )
    if out is not None and (
        tuple(out.shape) != (total_q, num_heads, HEAD_DIM_V)
        or out.dtype != torch.bfloat16
    ):
        raise ValueError(
            f"out must be a bfloat16 [total_q, num_heads, {HEAD_DIM_V}] tensor"
        )
    if lse is not None and (
        tuple(lse.shape) != (total_q, num_heads) or lse.dtype != torch.float32
    ):
        raise ValueError("lse must be a float32 [total_q, num_heads] tensor")
    return dict(
        dtype=dtype,
        batch_size=batch_size,
        num_heads=num_heads,
        total_q=total_q,
        page_size=page_size,
        num_pages=num_pages,
        max_pages=max_pages,
    )


def prepare_cake_mla_varq_dcp_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    max_q_len: int,
    *,
    max_seq_len: int,
    softmax_scale: float,
    workspace_buffer: torch.Tensor,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
    cp_world: int = 1,
    cp_rank: int = 0,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    backend: str = "cake",
    unit_min: Optional[int] = None,
    unit_ratio: Optional[tuple[int, int]] = None,
    static_tiles: Optional[int] = None,
    partition_mode: Optional[int] = None,
) -> CakeMLAVarQDcpDecodeRunner:
    """Validate, plan and bind one compact var-Q (+DCP) MLA decode problem.

    Every allocation happens here (the optional ``out`` / ``lse``, int32
    copies of non-int32 index tensors); the workspace regions are carved out
    of the caller's ``workspace_buffer`` and the scheduler state is zeroed
    once.  The returned runner launches with no allocation.  ``unit_min``,
    ``unit_ratio``, ``static_tiles`` and ``partition_mode`` expose the host
    plan knobs for tests and benchmarks; ``None`` selects the production
    defaults.  Nothing reads a CUDA tensor's contents: ``max_seq_len`` is the
    caller-known largest rank-local length (as in ``cute_dsl_mla_decode``).
    """
    if backend != "cake":
        raise ValueError("Cake MLA var-Q DCP decode supports backend='cake'")
    meta = validate_cake_mla_varq_dcp_decode_inputs(
        query,
        kv_cache,
        page_table,
        seq_lens,
        cum_seq_lens_q,
        max_q_len=max_q_len,
        max_seq_len=max_seq_len,
        cp_world=cp_world,
        cp_rank=cp_rank,
        causal_seqlens_kv_global=causal_seqlens_kv_global,
        out=out,
        lse=lse,
    )
    dtype = meta["dtype"]
    batch_size, num_heads, total_q = (
        meta["batch_size"],
        meta["num_heads"],
        meta["total_q"],
    )
    page_size, max_pages = meta["page_size"], meta["max_pages"]
    device = query.device
    tensors = [query, kv_cache, page_table, seq_lens, cum_seq_lens_q, workspace_buffer]
    tensors += [t for t in (causal_seqlens_kv_global, out, lse) if t is not None]
    if not all(t.is_cuda and t.device == device for t in tensors):
        raise ValueError("Expected all tensors on one CUDA device")
    if workspace_buffer.dtype != torch.uint8 or not workspace_buffer.is_contiguous():
        raise ValueError("workspace_buffer must be a contiguous uint8 CUDA tensor")
    arch = _arch_for(device)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    plan = plan_varq_dcp_decode(
        batch_size=batch_size,
        max_q_len=max_q_len,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        num_sms=num_sms,
        unit_min=unit_min,
        unit_ratio=unit_ratio,
        static_tiles=static_tiles,
        partition_mode=partition_mode,
    )
    item_groups = item_groups_for(int(plan["items"]))
    main_module = select_main_module(
        arch, dtype, page_size, item_groups, plan["partition_mode"]
    )
    merge_module = select_merge_module(arch) if launches_merge(plan) else None

    if kv_cache.ndim == 4:
        kv_cache = kv_cache.squeeze(1)
    q_flat = query.reshape(-1, HEAD_DIM_QK)
    if not q_flat.is_contiguous():
        q_flat = q_flat.contiguous()
    kv_cache = kv_cache.contiguous()
    if dtype == "fp8":
        # The FP8 descriptors are byte-typed (u8 carrier over the e4m3 payload).
        q_flat = q_flat.view(torch.uint8)
        kv_cache = kv_cache.view(torch.uint8)
    if out is None:
        out = torch.empty(
            (total_q, num_heads, HEAD_DIM_V), dtype=torch.bfloat16, device=device
        )
    if lse is None:
        lse = torch.empty((total_q, num_heads), dtype=torch.float32, device=device)
    o_flat = out.reshape(-1, HEAD_DIM_V)
    lse_flat = lse.reshape(-1)
    if not o_flat.is_contiguous() or not lse_flat.is_contiguous():
        raise ValueError("out / lse must be contiguous compact tensors")
    page_table_i32 = page_table.to(torch.int32).contiguous()
    seq_lens_i32 = seq_lens.to(torch.int32).contiguous()
    cum_i32 = cum_seq_lens_q.to(torch.int32).contiguous()
    causal_source = (
        seq_lens if causal_seqlens_kv_global is None else causal_seqlens_kv_global
    )
    causal_i32 = causal_source.to(torch.int32).contiguous()

    layout = workspace_layout(plan)
    flat = workspace_buffer.view(-1)
    if int(flat.numel()) < layout["total"]:
        raise ValueError(
            f"workspace_buffer needs {layout['total']} bytes for this plan "
            f"({plan['partial_rows']} partial rows), got {int(flat.numel())}"
        )
    partial_rows = int(plan["partial_rows"])
    partial_slots = partial_rows // TILE_Q
    items = int(plan["items"])
    if plan["can_split"]:
        partial_o = _carve(
            flat, layout, "partial_o", torch.bfloat16, (partial_rows * HEAD_DIM_V,)
        )
        partial_lse = _carve(
            flat, layout, "partial_lse", torch.float32, (partial_rows,)
        )
    else:
        # No item can split: the kernel never stores a partial; bind the
        # output itself exactly like the production runner.
        partial_o = o_flat
        partial_lse = lse_flat
    sched_counters = _carve(
        flat, layout, "sched_counters", torch.uint32, (SCHED_COUNTER_WORDS,)
    )
    unit_flags = _carve(
        flat,
        layout,
        "unit_flags",
        torch.uint32,
        ((4 if plan["partition_mode"] else 1) * partial_slots,),
    )
    split_meta = _carve(flat, layout, "split_meta", torch.uint32, (META_WORDS * items,))
    merge_ctl = _carve(flat, layout, "merge_ctl", torch.uint32, (MERGE_CTL_WORDS,))
    dbg = _carve(flat, layout, "dbg", torch.uint64, (DBG_WORDS,))
    workspace = dict(
        partial_o=partial_o,
        partial_lse=partial_lse,
        sched_counters=sched_counters,
        unit_flags=unit_flags,
        split_meta=split_meta,
        merge_ctl=merge_ctl,
        dbg=dbg,
    )

    main_kwargs: dict[str, Any] = dict(
        tmap_q=q_flat,
        tmap_k=kv_cache,
        tmap_v=kv_cache,
        tmap_o=o_flat,
        tmap_po=partial_o.reshape(-1, HEAD_DIM_V),
        O=o_flat,
        LSE=lse_flat,
        partial_O=partial_o,
        partial_lse=partial_lse,
        page_table=page_table_i32,
        seq_lens=seq_lens_i32,
        cum_seq_lens_q=cum_i32,
        causal_global=causal_i32,
        sched_counters=sched_counters,
        unit_flags=unit_flags,
        split_meta=split_meta,
        merge_ctl=merge_ctl,
        softmax_scale_log2=float(softmax_scale) * LOG2E,
        tiles_max=int(plan["tiles_max"]),
        num_heads=num_heads,
        max_pages=max_pages,
        cp_world=int(cp_world),
        cp_rank=int(cp_rank),
        num_items=items,
        unit_min=int(plan["unit_min"]),
        static_tiles=int(plan["static_tiles"]),
        unit_num=int(plan["unit_num"]),
        unit_den=int(plan["unit_den"]),
        max_units=int(plan["max_units"]),
        static_only=int(plan["static_only"]),
        dbg=dbg,
        partial_slots=partial_slots,
        fd_tiles_max=_fast_divmod(plan["tiles_max"]),
        fd_num_heads=_fast_divmod(num_heads),
        fd_cp_world=_fast_divmod(cp_world),
        fd_num_items=_fast_divmod(items),
        fd_unit_min=_fast_divmod(plan["unit_min"]),
        fd_static_tiles=_fast_divmod(plan["static_tiles"]),
        fd_unit_den=_fast_divmod(plan["unit_den"]),
        fd_clusters=_fast_divmod(plan["grid_clusters"]),
        grid=(int(plan["grid_clusters"]) * CLUSTER_SIZE, 1, 1),
    )
    assert tuple(main_kwargs) == MAIN_KWARGS
    merge_kwargs: Optional[dict[str, Any]] = None
    if merge_module is not None:
        if plan["static_only"]:
            raise AssertionError("a static-only plan cannot split")
        merge_kwargs = dict(
            partial_O=partial_o,
            partial_lse=partial_lse,
            O=o_flat,
            LSE=lse_flat,
            unit_flags=unit_flags,
            split_meta=split_meta,
            merge_ctl=merge_ctl,
            cum_seq_lens_q=cum_i32,
            num_heads=num_heads,
            tiles_max=int(plan["tiles_max"]),
            max_records=items,
            grid=(int(plan["merge_grid"]), 1, 1),
        )
        assert tuple(merge_kwargs) == MERGE_KWARGS
    main_entry, main_arguments = _bind_stage(main_module, main_kwargs)
    merge_entry: Optional[Callable[..., Any]] = None
    merge_arguments: tuple = ()
    if merge_module is not None and merge_kwargs is not None:
        merge_entry, merge_arguments = _bind_stage(merge_module, merge_kwargs)
    return CakeMLAVarQDcpDecodeRunner(
        arch,
        dtype,
        page_size,
        item_groups,
        plan,
        main_module,
        merge_module,
        main_kwargs,
        merge_kwargs,
        workspace,
        out,
        lse,
        main_entry,
        main_arguments,
        merge_entry,
        merge_arguments,
        (q_flat, kv_cache, page_table_i32, seq_lens_i32, cum_i32, causal_i32),
    )


def cake_mla_varq_dcp_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    softmax_scale: float,
    *,
    cum_seq_lens_q: torch.Tensor,
    max_q_len: int,
    enable_dcp: bool = False,
    cp_world: int = 1,
    cp_rank: int = 0,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = True,
    backend: str = "cake",
):
    """Compact variable-Q MLA decode of one DCP rank (prepare + one launch).

    Mirrors the argument names of ``cute_dsl_mla_decode(..., is_var_seq=True,
    return_lse=True, cum_seq_lens_q=..., max_q_len=..., enable_dcp=...,
    cp_world=..., cp_rank=..., causal_seqlens_kv_global=...)``.  Without DCP
    (``enable_dcp=False``) the rank-local lengths are the causal bounds
    (request-local causal decode with the newest token at the tail).  Returns
    ``(out, lse)`` with ``return_lse=True`` (default; the LSE is always
    produced) or ``out``.  Use ``prepare_cake_mla_varq_dcp_decode`` for a
    launch-only runner.
    """
    if not enable_dcp:
        if cp_world != 1 or cp_rank != 0 or causal_seqlens_kv_global is not None:
            raise ValueError(
                "cp_world / cp_rank / causal_seqlens_kv_global require enable_dcp=True"
            )
    elif causal_seqlens_kv_global is None:
        raise ValueError("causal_seqlens_kv_global is required when enable_dcp=True")
    runner = prepare_cake_mla_varq_dcp_decode(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        cum_seq_lens_q,
        max_q_len,
        max_seq_len=max_seq_len,
        softmax_scale=softmax_scale,
        workspace_buffer=workspace_buffer,
        causal_seqlens_kv_global=causal_seqlens_kv_global,
        cp_world=cp_world,
        cp_rank=cp_rank,
        out=out,
        lse=lse,
        backend=backend,
    )
    out, lse = runner.launch()
    return (out, lse) if return_lse else out


__all__ = [
    "CakeMLAVarQDcpDecodeRunner",
    "HEAD_DIM_QK",
    "HEAD_DIM_V",
    "ITEM_GROUP_VARIANTS",
    "MAIN_KWARGS",
    "MAX_ITEMS",
    "MERGE_KWARGS",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "SUPPORTED_PAGE_SIZES",
    "TILE_KV",
    "TILE_Q",
    "cake_mla_varq_dcp_decode",
    "cake_mla_varq_dcp_decode_workspace_size",
    "generated_program_available",
    "item_groups_for",
    "launches_merge",
    "max_cake_mla_varq_dcp_decode_workspace_size",
    "plan_varq_dcp_decode",
    "prepare_cake_mla_varq_dcp_decode",
    "validate_cake_mla_varq_dcp_decode_inputs",
    "workspace_layout",
]
