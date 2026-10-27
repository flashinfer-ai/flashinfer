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

# Host plan of the Cake W4A16 MegaMoE route (BF16 activations x NVFP4 expert
# weights, EP world 1 or 8): the route's geometry rules, the shipped cells and
# their selection, the per-workspace buffer table and the programs' argument
# names. Everything here is a host-known scalar; nothing queries a device.
#
# One forward is two programs on the current stream: the fused program
# (publish this rank's rows, one publication rendezvous, pull the routed rows
# into an expert-major pool, FC1 + SwiGLU -> BF16 handoff, FC2 -> BF16 route
# terms stored into the source rank's ``route_terms`` slot, completion signal)
# and the combine program (PDL-chained; waits on the local completion counter,
# sums the eight slots of every token in fixed slot order in FP32 and stores
# BF16). Both take their arguments by name: a binding's positional call is
# assembled from the program's recorded argument names (:data:`FUSED_ARGS`,
# :data:`COMBINE_ARGS` list what this host supplies), so a program may take any
# subset of them in any order; a program that names an argument this host does
# not supply is refused when it is loaded.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

KERNEL_NAME = "sm103_bf16_nvfp4_bf16_cake"

# ---------------------------------------------------------------------------
# Geometry rules (fixed by the device programs).
# ---------------------------------------------------------------------------
TOP_K = 8
# Both GEMMs stream their K dimension in 256-element stages: the FC1
# reduction length (hidden) and the FC2 reduction length (intermediate) are
# multiples of 256. FC1 tiles its M (the gate/up features) by 64 and FC2 its
# M (hidden) by 128.
HIDDEN_ALIGN = 256
INTERMEDIATE_ALIGN = 256
FC1_M_TILE = 64
FC2_M_TILE = 128
MAX_LOCAL_EXPERTS = 256
# The programs ship for these EP world sizes (``world`` is a build keyword).
SUPPORTED_WORLD_SIZES = (1, 8)
# Peer tables are by-value aggregates of eight slots (world <= 8); a table
# crosses the binding as a CPU int64 tensor of exactly this many addresses,
# the entries at or above the world size zero.
PEER_TABLE_CAPACITY = 8
# Process-group name of the programs: ``<name>_world``, ``<name>_rank`` and
# ``<name>_flags`` are the three implicit distributed parameters.
PROCESS_GROUP = "mega_pg"
# u32 words of one rank's process-group flag pad: two epoch words plus two
# phases x two banks x one mailbox per peer-table slot. Zeroed at allocation,
# never reset.
PG_FLAG_WORDS = 2 + 2 * 2 * PEER_TABLE_CAPACITY
# NVFP4 packing of the canonical weights.
FP4_PER_BYTE = 2
SCALE_BLOCK = 16

# Published-count header (per rank, symmetric): word 0 = this rank's live
# token count of the forward; the remaining words are reserved (zero).
PUBLISHED_META_WORDS = 8
# Completion counter (per rank, symmetric): word 0 = monotonic done counter
# (+1 from every rank per forward); the remaining words are reserved.
COMPLETION_WORDS = 8
# Rank-local counters of the fused program (device-resident, never written by
# the host after allocation): the intra-rank barrier words, which the program
# resets at its tail, and the launch epoch, which it increments.
SYNC_WORDS = 8
EPOCH_WORDS = 16
TILE_COUNT_WORDS = 4
# CTAs of the per-CTA route histogram scratch: the largest grid any device
# can launch (one CTA per SM, two per cluster slot at most). The fused program
# reads the histogram in rounds of 32 CTAs, so the scratch holds at least
# ``round_up(grid, 32)`` rows; two rows per SM cover that for every SM count
# of 32 or more.
MAX_CTAS_PER_SM = 2
# The combine program: ``COMBINE_THREADS`` threads of one 16-byte BF16 vector each,
# one CTA per (token row, ``COMBINE_HIDDEN_PER_CTA``-column slice); its hidden size
# is a run-time argument, a multiple of the slice.
COMBINE_THREADS = 128
COMBINE_HIDDEN_PER_CTA = COMBINE_THREADS * 8
# The combine grid's row cap: the CTA of grid row ``y`` walks the tokens ``y, y + rows,
# ...`` below the rank's live count, so a capacity above this many rows gives every
# CTA several tokens behind one completion wait (4 at a capacity of 4096), and one
# token per CTA stays the rule up to it.
COMBINE_MAX_ROWS = 1024
#: The fused program's ``h_max`` build keyword (the pull slot's row): its default,
#: its step and its limit. A hidden size above the default selects an ``h_max``
#: variant of every fused program (identifier suffix ``_H{h_max}``).
DEFAULT_FUSED_H_MAX = 6144
FUSED_H_STEP = 256
FUSED_H_LIMIT = 8192
#: Hidden sizes the loader enumerates (other multiples of 1024 up to the limit
#: are served by the same programs or by their ``h_max`` variants).
KNOWN_HIDDEN_SIZES = (6144, 7168)

#: Program-form switches of the route; the host and the programs they describe
#: land together, so each is flipped once with its programs. ``ROUTE_TABLES``:
#: at a world size above one the fused program publishes its own routes as the
#: owner-side route tables (``route_offs`` / ``route_list``, symmetric, banked
#: by the launch parity) and derives the placement from the peers' tables; the
#: global expert count is then bounded by :data:`MAX_TOTAL_EXPERTS_ROUTE_TABLES`.
#: ``FUSED_COMBINE``: the fused program sums the route terms into the output in
#: its tail; the combine program is not shipped, a forward is one launch and
#: the hidden size no longer has to be a multiple of the combine slice.
ROUTE_TABLES = False
FUSED_COMBINE = False
MAX_TOTAL_EXPERTS_ROUTE_TABLES = 256

# Program identifiers: the fused program per (cell, h_max variant), one combine
# program (world size and hidden size are run-time arguments of both).
FUSED = "FUSED"
COMBINE = "COMBINE"
KERNELS = (FUSED, COMBINE)
#: The three trailing grid arguments of every binding.
GRID_ARGS: Tuple[str, str, str] = ("grid_x", "grid_y", "grid_z")


# ---------------------------------------------------------------------------
# Cells.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    """One shipped fused-program configuration: ``n`` routed rows per route
    tile (the ``tcgen05`` N of the swap-AB MMA and the pool segment padding)
    and the CTA group (1, or 2 for the 2-CTA ``M = 256`` instruction)."""

    n: int
    cta_group: int = 1

    def __post_init__(self) -> None:
        if self.cta_group not in (1, 2):
            raise ValueError(f"cell cta_group must be 1 or 2, got {self.cta_group}")
        if self.n not in (32, 64, 128):
            raise ValueError(f"the cells ship for n = 32, 64 and 128, got n = {self.n}")
        if self.n == 128 and self.cta_group != 2:
            raise ValueError("the n = 128 cell ships as a 2-CTA cell only")
        if self.n == 32 and self.cta_group != 1:
            raise ValueError("the n = 32 cell ships as a single-CTA cell only")

    @property
    def suffix(self) -> str:
        return f"N{self.n}" + ("_CG2" if self.cta_group == 2 else "")

    @property
    def cluster_ctas(self) -> int:
        return self.cta_group

    def as_knobs(self) -> dict:
        return {"n": self.n, "cta_group": self.cta_group}


#: The cells the route ships, in loader order.
CELLS: Tuple[Cell, ...] = (Cell(32), Cell(64), Cell(64, 2), Cell(128, 2))
#: The cells the host selects from (one per n; the 2-CTA n = 64 cell is pinned
#: through ``knobs`` only).
SELECTABLE_CELLS: Tuple[Cell, ...] = (Cell(32), Cell(64), Cell(128, 2))
# Deviation of the per-expert row count above which the one-tile-per-expert rule
# picks the next n: rows per expert are ~Binomial(R, 1/E), and n >= mean + 3.1 sd
# keeps P(rows > n) below 1e-3.
_TAIL_SIGMAS = 3.1


def select_cell(
    tokens_per_rank: int, world_size: int, top_k: int, num_experts: int
) -> Cell:
    """The cell of a forward: a host function of the rank-uniform token bound
    (the graph bucket or the workspace capacity) and the routing geometry only
    (never of the routing itself), so the choice is CUDA-graph stable and the
    same on every rank.

    With ``m = world_size * tokens_per_rank * top_k / num_experts`` expected
    rows per expert, take the smallest ``n`` in 32, 64 with
    ``n >= m + 3.1 sqrt(m)``; above that, the 2-CTA ``n = 128`` cell.
    """

    if tokens_per_rank < 0 or world_size <= 0 or top_k <= 0 or num_experts <= 0:
        raise ValueError(
            "tokens_per_rank must be >= 0 and world_size, top_k, num_experts positive"
        )
    mean_rows = world_size * tokens_per_rank * top_k / num_experts
    needed = math.ceil(mean_rows + _TAIL_SIGMAS * math.sqrt(mean_rows))
    for cell in SELECTABLE_CELLS:
        if needed <= cell.n:
            return cell
    return SELECTABLE_CELLS[-1]


def cell_from_knobs(knobs: dict | str | None) -> Cell | None:
    """The cell a knobs dict pins (``{"n": .., "cta_group": ..}``), or ``None``
    (``None`` and ``"auto"`` leave the choice to :func:`select_cell`)."""

    if not isinstance(knobs, dict) or "n" not in knobs:
        return None
    return Cell(int(knobs["n"]), int(knobs.get("cta_group", 1)))


# ---------------------------------------------------------------------------
# Sizes.
# ---------------------------------------------------------------------------
def max_tiles(world_size: int, capacity: int, local_experts: int, n: int) -> int:
    """Route tiles of ``n`` rows the expert-major pool can need: every route
    of every rank, padded per expert to whole tiles (the sum of the padded
    segments is a multiple of ``n`` and at most ``routes + (n - 1) * experts``)."""

    return (world_size * capacity * TOP_K + (n - 1) * local_experts) // n


def pool_rows(world_size: int, capacity: int, local_experts: int, n: int) -> int:
    """Rows of the expert-major pool: :func:`max_tiles` whole tiles."""

    return max_tiles(world_size, capacity, local_experts, n) * n


def combine_grid(capacity: int, hidden: int) -> Tuple[int, int, int]:
    """Grid of the combine program: ``(hidden / 1024, min(cap, 1024), 1)`` -- one
    CTA per (``COMBINE_HIDDEN_PER_CTA``-column slice, token row); the CTA of row
    ``y`` combines the tokens ``y, y + rows, ...`` below the rank's live count
    (one token per CTA up to a capacity of ``COMBINE_MAX_ROWS``). A CTA whose
    first token is at or past the live count exits, so the launch is the same
    for every token bound (at least one token row, so a capacity of zero never
    happens here)."""

    return (
        hidden // COMBINE_HIDDEN_PER_CTA,
        min(max(capacity, 1), COMBINE_MAX_ROWS),
        1,
    )


def fused_grid(
    sm_count: int, cluster_ctas: int, max_active_clusters: Optional[int] = None
) -> Tuple[int, int, int]:
    """Grid of the fused program on one rank: ``sm_count // cluster_ctas``
    clusters (at least one), capped by the binding's residency probe when it
    exports a positive one, times the cluster size -- the persistent grid whose
    CTAs must all be co-resident. The backend takes the minimum over the EP
    ranks so every rank launches the same grid."""

    clusters = max(1, int(sm_count) // int(cluster_ctas))
    if max_active_clusters is not None and int(max_active_clusters) > 0:
        clusters = min(clusters, int(max_active_clusters))
    return (clusters * int(cluster_ctas), 1, 1)


def fused_h_max(hidden: int) -> int:
    """The ``h_max`` of the fused program serving ``hidden``: the default when the
    hidden size fits it, else the hidden size itself (a multiple of 256)."""

    return max(DEFAULT_FUSED_H_MAX, int(hidden))


def validate_geometry(
    *,
    world_size: int,
    num_experts: int,
    hidden: int,
    intermediate: int,
    top_k: int,
    capacity: int,
    route_tables: bool = ROUTE_TABLES,
    fused_combine: bool = FUSED_COMBINE,
) -> None:
    """Raise ``ValueError`` naming the first rule a geometry breaks."""

    checks = (
        (
            world_size in SUPPORTED_WORLD_SIZES,
            f"EP world size must be one of {SUPPORTED_WORLD_SIZES}, got {world_size}",
        ),
        (
            num_experts > 0 and num_experts % world_size == 0,
            f"num_experts ({num_experts}) must be a positive multiple of the "
            f"world size ({world_size})",
        ),
        (
            num_experts // max(world_size, 1) <= MAX_LOCAL_EXPERTS,
            f"at most {MAX_LOCAL_EXPERTS} experts per rank, got "
            f"{num_experts // max(world_size, 1)}",
        ),
        (
            hidden > 0 and hidden % HIDDEN_ALIGN == 0,
            f"token_hidden_size ({hidden}) must be a positive multiple of {HIDDEN_ALIGN}",
        ),
        (
            intermediate > 0 and intermediate % INTERMEDIATE_ALIGN == 0,
            f"intermediate_size ({intermediate}) must be a positive multiple of "
            f"{INTERMEDIATE_ALIGN}",
        ),
        (
            fused_combine or hidden % COMBINE_HIDDEN_PER_CTA == 0,
            f"token_hidden_size ({hidden}) must be a multiple of the combine program's "
            f"{COMBINE_HIDDEN_PER_CTA}-column slice",
        ),
        (
            not route_tables
            or world_size == 1
            or num_experts <= MAX_TOTAL_EXPERTS_ROUTE_TABLES,
            f"at most {MAX_TOTAL_EXPERTS_ROUTE_TABLES} experts in total at an EP world "
            f"size above one (the route tables span the global experts), got "
            f"{num_experts}",
        ),
        (
            hidden <= FUSED_H_LIMIT,
            f"token_hidden_size ({hidden}) exceeds the fused program's row limit "
            f"{FUSED_H_LIMIT}",
        ),
        (top_k == TOP_K, f"top_k must be {TOP_K}, got {top_k}"),
        (capacity >= 1, f"max_tokens_per_rank must be >= 1, got {capacity}"),
    )
    for ok, reason in checks:
        if not ok:
            raise ValueError(f"{KERNEL_NAME}: {reason}")


# ---------------------------------------------------------------------------
# Buffer table of one workspace.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Geometry:
    """The sizes one workspace is allocated for."""

    world_size: int
    rank: int
    num_experts: int
    hidden: int
    intermediate: int
    top_k: int
    capacity: int
    sm_count: int

    @property
    def local_experts(self) -> int:
        return self.num_experts // self.world_size

    def max_tiles(self, cell: Cell) -> int:
        return max_tiles(self.world_size, self.capacity, self.local_experts, cell.n)

    def pool_rows(self, cell: Cell) -> int:
        return self.max_tiles(cell) * cell.n

    @property
    def max_pool_rows(self) -> int:
        return max(self.pool_rows(cell) for cell in CELLS)

    @property
    def max_route_tiles(self) -> int:
        return max(self.max_tiles(cell) for cell in CELLS)

    @property
    def max_grid_ctas(self) -> int:
        return self.sm_count * MAX_CTAS_PER_SM

    @property
    def fc1_m_tiles(self) -> int:
        return self.intermediate // FC1_M_TILE

    @property
    def fc2_m_tiles(self) -> int:
        return self.hidden // FC2_M_TILE


# Symmetric (peer-mapped) buffers: name -> (shape, dtype). Each is bound to the
# programs as ``<name>`` (the local tensor) and ``<name>_peers`` (its
# peer-address table). Allocated zeroed; the host never writes them. The route
# tables are two banks (the launch parity selects one) of the owner's exclusive
# per-global-expert route prefix and of its ``token * 8 + slot`` route entries;
# the published ids and weights have no reader once the route tables carry the
# routing and the combine is fused.
def symmetric_buffers(
    g: Geometry,
    *,
    route_tables: bool = ROUTE_TABLES,
    fused_combine: bool = FUSED_COMBINE,
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    buffers: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        "published_hidden": ((g.capacity, g.hidden), torch.bfloat16),
        "published_ids": ((g.capacity, g.top_k), torch.int32),
        "published_weights": ((g.capacity, g.top_k), torch.float32),
        "published_meta": ((PUBLISHED_META_WORDS,), torch.int32),
        "route_terms": ((g.capacity, g.top_k, g.hidden), torch.bfloat16),
        "completion": ((COMPLETION_WORDS,), torch.uint32),
    }
    if route_tables:
        buffers["route_offs"] = (
            (ROUTE_TABLE_BANKS, MAX_TOTAL_EXPERTS_ROUTE_TABLES + 1),
            torch.int32,
        )
        buffers["route_list"] = (
            (ROUTE_TABLE_BANKS, g.capacity * g.top_k),
            torch.int32,
        )
        if fused_combine:
            del buffers["published_ids"], buffers["published_weights"]
    return buffers


#: The two banks of the route tables: a launch writes the bank of its own parity
#: and a peer still reading the other bank is never overtaken. ``route_offs``
#: holds the fixed 257 words of the largest global expert count per bank (the
#: programs read ``num_experts + 1`` of them), so its stride is world-independent.
ROUTE_TABLE_BANKS = 2


# Rank-local buffers: name -> (shape, dtype). The pool, the handoff and the
# per-tile tables are sized for the largest cell; a launch binds the leading
# rows its cell needs. ``cta_hist`` is sized for the device's largest grid.
def local_buffers(g: Geometry) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    rows, tiles = g.max_pool_rows, g.max_route_tiles
    return {
        "pool": ((rows, g.hidden), torch.bfloat16),
        "fc1_out": ((rows, g.intermediate), torch.bfloat16),
        "route_src": ((rows,), torch.int32),
        "tile_expert": ((tiles,), torch.int32),
        "tile_mn_limit": ((tiles,), torch.int32),
        "tile_count": ((TILE_COUNT_WORDS,), torch.int32),
        "expert_seg": ((g.local_experts + 1,), torch.int32),
        "cta_hist": ((g.max_grid_ctas, g.local_experts), torch.int32),
        "sync_words": ((SYNC_WORDS,), torch.uint32),
        "epoch": ((EPOCH_WORDS,), torch.uint32),
        "fc1_ready": ((tiles,), torch.uint32),
        # per route tile (B1_ABI v4): FC1 items done / FC2 items stored of tile t
        "fc1_done": ((tiles,), torch.uint32),
        "fc2_done": ((tiles,), torch.uint32),
        "fc1_alpha": ((g.local_experts,), torch.float32),
        "fc2_alpha": ((g.local_experts,), torch.float32),
        "output": ((g.capacity, g.hidden), torch.bfloat16),
        # Staging copies, used only when the caller's tensors do not satisfy
        # the programs' pointer contract (contiguous bf16 / int32 / float32 of
        # the token bound).
        "staged_hidden": ((g.capacity, g.hidden), torch.bfloat16),
        "staged_ids": ((g.capacity, g.top_k), torch.int32),
        "staged_weights": ((g.capacity, g.top_k), torch.float32),
    }


#: The id / weight publications: no program reads them once the route tables
#: carry the routing and the combine is fused, so the fused program's argument
#: list and ``symmetric_buffers`` drop them together.
_PUBLISHED_ROUTING_ARGS: Tuple[str, ...] = (
    "published_ids",
    "published_ids_peers",
    "published_weights",
    "published_weights_peers",
)


# Argument names this host supplies to the fused program (any subset, any
# order): the caller's rows, the canonical weights (TMA descriptors encoded by
# the binding from the tensors), the local buffers, every symmetric buffer as
# ``<name>`` + ``<name>_peers``, the process group, the scalars, and -- under
# the program-form switches -- the route tables and the output.
def fused_args(
    *, route_tables: bool = ROUTE_TABLES, fused_combine: bool = FUSED_COMBINE
) -> Tuple[str, ...]:
    base = FUSED_BASE_ARGS[: -len(GRID_ARGS)]
    if route_tables and fused_combine:
        base = tuple(name for name in base if name not in _PUBLISHED_ROUTING_ARGS)
    return (
        *base,
        *(
            ("route_offs", "route_offs_peers", "route_list", "route_list_peers")
            if route_tables
            else ()
        ),
        *(("output",) if fused_combine else ()),
        *GRID_ARGS,
    )


FUSED_BASE_ARGS: Tuple[str, ...] = (
    "hidden",
    "topk_ids",
    "topk_weights",
    "W13",
    "S13",
    "W2",
    "S2",
    "pool",
    "pool_ptr",
    "fc1_out",
    "fc1_out_st",
    "fc1_out_ld",
    "route_src",
    "tile_expert",
    "tile_mn_limit",
    "tile_count",
    "expert_seg",
    "cta_hist",
    "sync_words",
    "epoch",
    "fc1_ready",
    "fc1_done",
    "fc2_done",
    "fc1_alpha",
    "fc2_alpha",
    "published_hidden",
    "published_hidden_peers",
    "published_ids",
    "published_ids_peers",
    "published_weights",
    "published_weights_peers",
    "published_meta",
    "published_meta_peers",
    "route_terms",
    "route_terms_peers",
    "completion",
    "completion_peers",
    f"{PROCESS_GROUP}_world",
    f"{PROCESS_GROUP}_rank",
    f"{PROCESS_GROUP}_flags",
    "swiglu_limit",
    "T_r",
    "cap",
    "H",
    "I",
    "E_local",
    "fc1_m_tiles",
    "fc1_m_div",
    "fc2_m_tiles",
    "fc2_m_div",
    *GRID_ARGS,
)
FUSED_ARGS: Tuple[str, ...] = fused_args()

# Argument names this host supplies to the combine program (``tokens_per_rank``
# is the rank's ``published_meta`` buffer: word 0 holds the live token count the
# fused program stored, so a captured graph replays at any count up to the
# grid's capacity).
COMBINE_ARGS: Tuple[str, ...] = (
    "route_terms",
    "route_terms_peers",
    "published_ids",
    "published_ids_peers",
    "published_weights",
    "published_weights_peers",
    "published_meta",
    "published_meta_peers",
    "tokens_per_rank",
    "hidden",  # the i32 hidden size (the fused program's ``hidden`` is the activations)
    "topk_ids",
    "topk_weights",
    "output",
    "completion",
    "completion_peers",
    "epoch",
    f"{PROCESS_GROUP}_world",
    f"{PROCESS_GROUP}_rank",
    f"{PROCESS_GROUP}_flags",
    "T_r",
    "cap",
    "H",
    *GRID_ARGS,
)

SUPPLIED_ARGS = {FUSED: FUSED_ARGS, COMBINE: COMBINE_ARGS}

#: Arguments whose value changes between forwards (the caller's tensors and
#: the output); every other argument of a launch is fixed per (workspace,
#: weights, token bound).
PER_FORWARD_ARGS = frozenset({"hidden", "topk_ids", "topk_weights", "output"})


#: ... per program: the fused program reads the activations and routing (and
#: writes the output when the combine is fused); the combine program reads the
#: routing and writes the output (its ``hidden`` is the i32 hidden size, a plan
#: constant).
def fused_per_forward_args(*, fused_combine: bool = FUSED_COMBINE) -> frozenset:
    return frozenset(
        {"hidden", "topk_ids", "topk_weights", *(("output",) if fused_combine else ())}
    )


FUSED_PER_FORWARD_ARGS = fused_per_forward_args()
COMBINE_PER_FORWARD_ARGS = frozenset({"topk_ids", "topk_weights", "output"})


def program_identifier(
    kernel: str, cell: Cell | None = None, hidden: int | None = None
) -> str:
    """Loader identifier of a program: ``FUSED_N32`` ... ``FUSED_N128_CG2_H7168``
    (the ``h_max`` variant when the hidden size exceeds the default), ``COMBINE``
    (one program; its world and hidden size are run-time arguments)."""

    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; the route has {KERNELS}")
    if kernel == COMBINE:
        return COMBINE
    if cell is None:
        raise ValueError("the fused program needs a cell")
    h_max = fused_h_max(DEFAULT_FUSED_H_MAX if hidden is None else hidden)
    if h_max % FUSED_H_STEP or h_max > FUSED_H_LIMIT:
        raise ValueError(
            f"hidden size {hidden} needs an h_max of {h_max}; the fused program takes "
            f"multiples of {FUSED_H_STEP} up to {FUSED_H_LIMIT}"
        )
    suffix = "" if h_max == DEFAULT_FUSED_H_MAX else f"_H{h_max}"
    return f"{FUSED}_{cell.suffix}{suffix}"


#: Loader identifier of every shipped program: the fused program of every cell
#: at every known pull-slot row, then the combine program unless the combine is
#: fused into the tail.
def program_identifiers(*, fused_combine: bool = FUSED_COMBINE) -> Tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            [program_identifier(FUSED, c, h) for h in KNOWN_HIDDEN_SIZES for c in CELLS]
            + ([] if fused_combine else [program_identifier(COMBINE)])
        )
    )


PROGRAM_IDENTIFIERS: Tuple[str, ...] = program_identifiers()

__all__ = [
    "CELLS",
    "COMBINE",
    "COMBINE_ARGS",
    "COMBINE_HIDDEN_PER_CTA",
    "COMBINE_MAX_ROWS",
    "COMBINE_PER_FORWARD_ARGS",
    "DEFAULT_FUSED_H_MAX",
    "COMBINE_THREADS",
    "COMPLETION_WORDS",
    "EPOCH_WORDS",
    "FC1_M_TILE",
    "FC2_M_TILE",
    "FP4_PER_BYTE",
    "FUSED",
    "FUSED_ARGS",
    "FUSED_BASE_ARGS",
    "FUSED_COMBINE",
    "FUSED_H_LIMIT",
    "FUSED_H_STEP",
    "FUSED_PER_FORWARD_ARGS",
    "KNOWN_HIDDEN_SIZES",
    "fused_h_max",
    "fused_grid",
    "GRID_ARGS",
    "HIDDEN_ALIGN",
    "INTERMEDIATE_ALIGN",
    "KERNELS",
    "KERNEL_NAME",
    "MAX_CTAS_PER_SM",
    "MAX_LOCAL_EXPERTS",
    "MAX_TOTAL_EXPERTS_ROUTE_TABLES",
    "PEER_TABLE_CAPACITY",
    "PER_FORWARD_ARGS",
    "PG_FLAG_WORDS",
    "PROCESS_GROUP",
    "PROGRAM_IDENTIFIERS",
    "PUBLISHED_META_WORDS",
    "ROUTE_TABLES",
    "ROUTE_TABLE_BANKS",
    "SCALE_BLOCK",
    "SELECTABLE_CELLS",
    "SUPPLIED_ARGS",
    "SUPPORTED_WORLD_SIZES",
    "SYNC_WORDS",
    "TILE_COUNT_WORDS",
    "TOP_K",
    "Cell",
    "Geometry",
    "cell_from_knobs",
    "combine_grid",
    "fused_args",
    "fused_per_forward_args",
    "local_buffers",
    "max_tiles",
    "pool_rows",
    "program_identifier",
    "program_identifiers",
    "select_cell",
    "symmetric_buffers",
    "validate_geometry",
]
