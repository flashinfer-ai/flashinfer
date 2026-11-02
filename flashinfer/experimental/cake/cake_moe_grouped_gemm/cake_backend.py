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

Cake backend: non-deterministic ragged BF16 MoE grouped GEMM (SM100 / SM103 / SM107).

The three operations of the deterministic package
``flashinfer.experimental.cake_moe_grouped_det_gemm``, over ``E`` groups of rows
described by an int32 **device** tensor ``offs[E]`` of cumulative end offsets
(never read on the host):

* ``fwd``:   ``Y[offs[e-1]:offs[e]]  = X[offs[e-1]:offs[e]] @ W[e].T``
* ``dgrad``: ``dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]``
* ``wgrad``: ``dW[e] = G[offs[e-1]:offs[e]].T @ X[offs[e-1]:offs[e]]`` (bf16 or fp32 output)

Numerical mode.  This package is the ``deterministic=False`` implementation:
every entry point (``prepare_grouped_gemm_*`` here, ``grouped_gemm_*`` and
``cake_grouped_mm`` of the package) takes ``deterministic: bool = True`` and
raises ``ValueError`` unless it is ``False``; the deterministic programs are the
deterministic package's (or those of the facade
``flashinfer.experimental.cake.grouped_mm`` with ``deterministic=True``).
``fwd``, ``dgrad`` and the weight gradient with the 256-wide k tile run this
package's copies of the deterministic programs and give their bits (the
256-wide k tile's ``main_wave`` program is the same kernel rendered from a newer
source: it takes two more buffer arguments, ``A_rows`` / ``B_rows``, which its
sm_100a / sm_103a builds never read, and it has no sm_107a build).  The
weight gradient with the 512-wide k tile runs the builds of the deterministic
package's k512 forms whose split tail (the tiles of the last partial wave, cut
into ``tail_splits`` k-step units over the clusters) is added in place: each
unit adds its fp32 accumulator into the tile's fp32 sum with unordered
reductions and the last unit to arrive rounds the sum once and stores it, in
the same launch.  Every whole tile is still reduced by one CTA pair in a fixed
order (the deterministic bits); only the order in which a split tail tile's
units are added varies, so only the run-to-run bitwise reproducibility of such
a ``dW`` is relaxed.  The operands (bf16), the accumulation (fp32), the one
round-to-nearest rounding to bf16 (or the fp32 store), the layouts, the
``offs`` contract and the exact zeros of empty groups are the deterministic
package's.  With ``torch.use_deterministic_algorithms(True)``,
``prepare_grouped_gemm_wgrad`` (hence ``grouped_gemm_wgrad`` and the backward of
``CakeGroupedMm``) raises PyTorch's non-determinism alert (``_alert_not_deterministic``:
a ``RuntimeError``, or a ``UserWarning`` under ``warn_only=True``) and never falls back;
``fwd`` and ``dgrad`` give the deterministic bits and do not alert.  The stable
``flashinfer.grouped_mm.grouped_mm_bf16(..., backend="cake")`` stays on the
deterministic package: this package has no handoff for it.

Every launch is one persistent cluster kernel (one CTA pair per SM pair); the
256-wide k tile's weight gradient adds one ordered fp32 tail-reduce kernel when
the tiles of its last partial wave are split over the clusters, and the
512-wide k tile's adds its split tail in place (one kernel).  Nothing in the
plan depends on the group sizes: the grid, the k tile, the tail split and the
raster follow from host-known scalars (tensor shapes, the SM count, the
architecture) so a prepared launch is CUDA-graph capturable and stays valid for
any ``offs`` written later.  The host planner below reproduces the
deterministic package's decisions exactly from ``HOST_PLAN_CONSTANTS``; only the
split of the unordered tail follows ``atomic_tail_plan``
(``wgrad_atomic_tail_plan``).

Program choice.  ``fwd`` and ``dgrad`` launch the 512-column output tile (the
route's ``main_wide`` stage) when the output columns are a multiple of
``wide_block_n`` and the architecture is listed in ``wide_fwd_archs`` /
``wide_dgrad_archs``, else the 256-column tile (``main``).  On the
architectures of ``dgrad_evict_first_archs`` the data gradient launches the
twin of its wide tile that loads G and stores dX with the L2 evict-first policy
(``main_wide_ef``) when the groups average at least
``dgrad_evict_first_min_rows`` rows (``sum_m // E``) and dX has at least
``dgrad_evict_first_min_k`` columns (``K``).  On the architectures of
``dgrad_wide_ts33_archs`` the data gradient launches the twin of its wide tile
that splits three head and three tail k steps of an item around the release
of the accumulator's halves instead of four and three (``main_wide_ts33``)
when the groups average at least the architecture's
``dgrad_wide_ts33_min_rows`` rows and dX has at least
``dgrad_wide_ts33_min_k`` columns; on those of ``fwd_wide_ts33_archs`` the
forward pass launches its twin when its reduction ``K`` is at most
``fwd_wide_ts33_max_k`` and the groups average at most
``fwd_wide_ts33_max_rows`` rows.  On the
architectures of
``fwd_lwh_ef_archs`` the forward pass launches the twin of its wide tile that
releases the last wave of tiles in halves and stores Y with the L2 evict-first
policy (``main_wide_ef``) when its reduction ``K`` (the columns of x) is at most
``fwd_lwh_ef_max_k``, the down projection of an MoE layer.  The weight gradient takes the
512-wide k tile when ``K`` is a multiple of 512, the architecture is listed in
``k512_archs`` and the groups average at least the architecture's
``k512_min_rows_per_group`` rows, else the 256-wide tile.  It launches the
wave-synchronised build (``main_wave``) from ``wave_sync_min_k`` on the
architectures ``wave_sync_archs`` lists for its k tile, with the
architecture's ``wave_sync_mode``; with the 512-wide k tile on the
architectures listed in ``k512_release_archs`` it launches the twin of that
build that releases the accumulator in two halves (``main_halves`` /
``main_wave_halves``).  On the architectures of ``k256_tail_archs``
``main_halves`` gives way to its twin that runs the tiles left after the full
waves of the 512-wide k tile as 256-column half tiles, with the split tail of
the 256-wide k tiling (``main_halves_k256t``), when ``K`` (the
columns of ``dW[e]`` and of x) is at most ``k256_tail_max_k`` and the groups
average at least ``k256_tail_min_rows`` rows (``sum_m // E``), the down
projection of an MoE layer over few groups.  Every choice falls back to the
plain stage where the route serves no such stage on the architecture.  The
512-wide k tile binds the unordered-tail builds on the architectures of
``unordered_tail_archs`` (``wgrad_unordered_tail``), which lists every
architecture of the k512 routes: the registry holds no ordered k512 build.  The
list is an extension point only: leaving an architecture out needs the
deterministic k512 builds and a ``tail_reduce`` stage on those routes in the
registry (a new render), which the host's ordered path (``partials`` and the
tail reduce, as for the 256-wide k tile) then binds.

Stream workspaces.  The forward and data-gradient programs take a scheduler
workspace (``sched``: ``sched_words`` u32 words whose first three rest at
``sched_rest``) and the split of their clusters over the two claim queues
(``c0 = clusters // 2``, ``c1 = clusters - c0``).  A cluster claims first from
the queue of the locality domain (die) whose memory homes the probe chunk of
``sched`` (the 4 KB chunk at the first 8 KB boundary 128 bytes past it): queue
0 when it is nearer than the next 4 KB chunk, else queue 1, and which die homes
a chunk of default memory depends on its physical page.  On the architectures
of ``domain_split_archs``, a launch of an op in ``domain_split_ops`` (the data
gradient) whose group weight has at most ``domain_split_max_weight_elems``
elements (``N * K``: the down projection of an MoE layer) splits the queues by
the SM pairs nearer each chunk of its own ``sched`` instead
(``plan["queue_split"] == "probed"``); the forward keeps the even split.  The
first such launch on a kept workspace enqueues its probe on the stream (five
rounds of one CTA per SM timing an add-zero atomic on a line of each chunk; an
SM pair takes the chunk most of its rounds found nearer) and copies the pair
counts to pinned host memory behind an event; a later launch passes them once
the event has completed.  Nothing waits for the probe: until it completes, and
where it cannot run (its kernel is compiled once per compute capability at
preparation with ``tvm_ffi.cpp.load_inline``, which needs a CUDA compiler) or
does not see every SM and both domains, a launch passes the even split.  The
split only divides the tiles between the queues; every tile is still computed
once with its k steps in order, so the output bits do not depend on it.  The
weight-gradient wave
programs take two wave counters (``wave_sync``, resting at zero) and their wave
mode.  Every launch leaves a workspace at its rest state (the kernels re-arm it
at the end), so one buffer serves every launch of one CUDA stream: the buffers
are kept per (device, stream), allocated and set to their rest state by the
first launch on a stream and reused by every later one, never shared by two
streams (launches on one stream are ordered, so a buffer is never live in two
launches at once).  A launch captured into a CUDA graph, or issued for a
device other than the current one, takes buffers of its own at their rest
state, set at the launch (a capture records that set with the launch and
allocates the buffers from the graph's pool) and the even split.  The static
sm_107a forward and data-gradient programs take ``sched``, ``c0`` and ``c1`` and
ignore them.

The 512-wide k tile's weight-gradient programs take the fp32 sums of the split
tail tiles (``tail_sums``: ``num_tail * cluster_m * tail_tile_k`` floats) and
their arrival tickets (``tail_tickets``: ``num_tail * cta_group *
TAIL_TICKETS_PER_CTA`` u32 words, one per epilogue warp of a CTA), both resting
at zero: a launch needs them zero at its start and leaves them zero (the last
unit of a tile clears its sum, the tickets wrap).  They are stream workspaces
too: kept per (device, stream), allocated zeroed by the first launch on a
stream that splits its tail and replaced by a larger zeroed buffer when a later
launch needs more (launches on one stream are ordered, so the replaced buffer
is idle); a captured launch, or one for another device, takes zeroed buffers of
its own of the size it needs.  A launch whose tail is not split
(``tail_splits == 1``) passes one-element placeholders that the kernel does not
touch.  The weight gradient keeps no per-preparation partials on the 512-wide k
tile.
"""

from __future__ import annotations

import functools
import inspect
import math
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from .cake_jit import (
    HOST_PLAN_CONSTANTS,
    PROGRAMS,
    ROUTES,
    load_module,
    program_registered,
    select_module,
    stage_binding,
    stage_served,
)

SUPPORTED_COMPUTE_CAPABILITIES = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
DESCRIPTOR_ALIGN = 128  # bytes; CUtensorMap slots of the caller-owned workspaces
# The k width of the half tiles the weight gradient's ``main_halves_k256t`` stage runs
# after the full waves of its 512-wide k tile: its split tail is the 256-wide k tile's,
# summed by the tail-reduce stage of that tile's route.
K256_TAIL_TILE_K = 256
# The tickets of one split tail tile per CTA of its pair: one per epilogue warp of the widest
# (half-release, eight epilogue warps) build; the four-warp builds use the first four.
TAIL_TICKETS_PER_CTA = 8
_DTYPE_NAMES = {torch.bfloat16: "bfloat16", torch.float32: "float32"}


# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------


def host_plan_constants() -> dict[str, Any]:
    """The production planner constants delivered with the generated programs."""
    return HOST_PLAN_CONSTANTS


def _device_index(device: torch.device) -> int:
    index = device.index
    return torch.cuda.current_device() if index is None else int(index)


@functools.cache
def _device_facts(index: int) -> tuple[Optional[str], int]:
    """(exact architecture or None, SM count) of CUDA device ``index``: queried once per process."""
    capability = torch.cuda.get_device_capability(index)
    properties = torch.cuda.get_device_properties(index)
    return SUPPORTED_COMPUTE_CAPABILITIES.get(capability), int(
        properties.multi_processor_count
    )


def arch_for(device: torch.device) -> Optional[str]:
    """Exact architecture name of ``device`` or ``None`` when unsupported."""
    return _device_facts(_device_index(device))[0]


def sm_count(device: torch.device) -> int:
    return _device_facts(_device_index(device))[1]


def _require_deterministic_false(deterministic: object, entry: str) -> None:
    """Refuse every numerical mode but ``deterministic=False``: this package's weight gradient adds its split
    tail with unordered reductions, and its deterministic counterpart is another package."""
    if not isinstance(deterministic, bool) or deterministic:
        raise ValueError(
            f"{entry} is the non-deterministic ragged grouped GEMM (the 512-wide k tile's weight gradient adds "
            "its split tail with unordered fp32 reductions) and needs deterministic=False, got "
            f"deterministic={deterministic!r}; the deterministic programs are "
            "flashinfer.experimental.cake_moe_grouped_det_gemm (or the facade "
            "flashinfer.experimental.cake.grouped_mm with deterministic=True)"
        )


# This package's module prefix (the vendored name of a vendored copy).
_PACKAGE = __name__.rpartition(".")[0]


def _caller_stacklevel() -> int:
    """The ``stacklevel`` of the nearest frame outside this package and flashinfer, for a warning issued by
    the function that calls this one: the caller of the entry point, also through the facade and its
    decorators."""
    frame = inspect.currentframe()
    level = 0
    while frame is not None:
        parent = frame.f_back
        level += 1
        if parent is not None:
            name = str(parent.f_globals.get("__name__", ""))
            if not (
                name == _PACKAGE
                or name.startswith(_PACKAGE + ".")
                or name == "flashinfer"
                or name.startswith("flashinfer.")
            ):
                return level
        frame = parent
    return 2


def _alert_not_deterministic(entry: str) -> None:
    """PyTorch's alert for an operation without a deterministic implementation, when torch's deterministic
    algorithms are enabled: a ``RuntimeError``, or a ``UserWarning`` (naming the caller's frame) under
    ``torch.use_deterministic_algorithms(True, warn_only=True)``; nothing otherwise.  Called by the entry points
    that can launch the unordered weight gradient, before they prepare it; there is no fallback.  The alert is
    raised here because PyTorch exposes no public Python entry for it."""
    if not torch.are_deterministic_algorithms_enabled():
        return
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(
            f"{entry} does not have a deterministic implementation, but you set "
            "'torch.use_deterministic_algorithms(True, warn_only=True)'; "
            "flashinfer.experimental.cake_moe_grouped_det_gemm is the deterministic package",
            UserWarning,
            stacklevel=_caller_stacklevel(),
        )
        return
    raise RuntimeError(
        f"{entry} does not have a deterministic implementation, but you set "
        "'torch.use_deterministic_algorithms(True)'; flashinfer.experimental.cake_moe_grouped_det_gemm is "
        "the deterministic package, or use the 'warn_only=True' option if that is acceptable for your "
        "application"
    )


def generated_program_available(
    device: torch.device, op: str = "fwd", out_dtype: Optional[torch.dtype] = None
) -> bool:
    """True when this checkout registers the program of ``op`` for ``device``."""
    arch = arch_for(device)
    if arch is None:
        return False
    name = _DTYPE_NAMES.get(out_dtype) if out_dtype is not None else None
    return program_registered(op, arch, out_dtype=name)


# ---------------------------------------------------------------------------
# Host plan (pure functions; every value comes from HOST_PLAN_CONSTANTS)
# ---------------------------------------------------------------------------


def persistent_grid(sm_count: int, max_cluster_tiles: int, *, cta_group: int) -> int:
    """Even CTA count: one cluster per SM pair, capped by the tile count."""
    pairs = max(1, min(sm_count // cta_group, max(1, max_cluster_tiles)))
    return pairs * cta_group


def max_cluster_tiles_upper_bound(
    sum_m: int, num_groups: int, cols: int, *, cluster_m: int, block_n: int
) -> int:
    """fwd / dgrad: every group can add one partial cluster-row tile."""
    return (sum_m // cluster_m + num_groups) * (cols // block_n)


def fwd_dgrad_tile_n(
    op: str,
    n: int,
    k: int,
    arch: str,
    constants: Optional[dict[str, Any]] = None,
) -> int:
    """Output-column tile of ``fwd`` / ``dgrad``: ``wide_block_n`` when the output columns are a
    multiple of it on the architectures of the operation's ``wide_fwd_archs`` /
    ``wide_dgrad_archs``, else ``block_n``."""
    c = constants or host_plan_constants()
    cols = n if op == "fwd" else k
    wide = int(c["wide_block_n"])
    if cols % wide == 0 and arch in c[f"wide_{op}_archs"]:
        return wide
    return int(c["block_n"])


def wgrad_tile_k(
    k: int,
    arch: str,
    avg_rows: float,
    *,
    k512_archs: tuple[str, ...],
    k512_min_rows_per_group: int,
) -> int:
    """Weight-gradient k tile: 512 for long, 512-aligned reductions on the listed architectures, else 256."""
    prefer_512 = (
        k % 512 == 0 and arch in k512_archs and avg_rows >= k512_min_rows_per_group
    )
    return 512 if prefer_512 else 256


def wgrad_tail_tiles(
    num_groups: int, n: int, k: int, tile_k: int, num_clusters: int, *, block_n: int
) -> int:
    """Tiles left after the last full wave (0 = no tail phase, no reduce launch)."""
    return (num_groups * (n // block_n) * (k // tile_k)) % num_clusters


def wgrad_tail_plan(
    num_tail: int,
    num_clusters: int,
    sum_m: int,
    num_groups: int,
    tile_k: int,
    *,
    block_k: int,
    cluster_m: int,
    max_splits: int,
    sm_clock_ghz: float,
    dram_tbps: float,
    launch_us: float,
    cycles_per_step: dict,
) -> int:
    """Chunks per tail tile: minimise rounds(S) * steps(S) + partial traffic + one launch over S."""
    if num_tail == 0:
        return 1
    kb = max(1, math.ceil(sum_m / max(num_groups, 1) / block_k))
    cyc_step = _cycles_per_step(cycles_per_step, tile_k)
    tile_bytes = cluster_m * tile_k * 4
    best_s, best_t = 1, kb * cyc_step / (sm_clock_ghz * 1e3)
    for s in range(2, max_splits + 1):
        rounds = math.ceil(num_tail * s / num_clusters)
        t = rounds * math.ceil(kb / s) * cyc_step / (sm_clock_ghz * 1e3)
        t += 2 * num_tail * s * tile_bytes / (dram_tbps * 1e6) + launch_us
        if t < best_t:
            best_s, best_t = s, t
    return best_s


def wgrad_atomic_tail_plan(
    num_tail: int,
    num_clusters: int,
    sum_m: int,
    num_groups: int,
    tile_k: int,
    *,
    block_k: int,
    cluster_m: int,
    max_splits: int,
    sm_clock_ghz: float,
    dram_tbps: float,
    launch_us: float,
    cycles_per_step: dict,
) -> int:
    """Units per tail tile of the unordered split tail (``atomic_tail_plan`` constants).

    The cost model of :func:`wgrad_tail_plan`: rounds(S) * steps(S), plus the
    fp32 traffic of the units' reductions into ``tail_sums`` and the finishing
    read (``dram_tbps``), plus a fixed cost of splitting at all (``launch_us``:
    for this form the ticket count and the finishing pass; there is no second
    launch).  The constants are this package's, so the split can follow the
    unordered tail's measurements without moving the ordered tail's."""
    return wgrad_tail_plan(
        num_tail,
        num_clusters,
        sum_m,
        num_groups,
        tile_k,
        block_k=block_k,
        cluster_m=cluster_m,
        max_splits=max_splits,
        sm_clock_ghz=sm_clock_ghz,
        dram_tbps=dram_tbps,
        launch_us=launch_us,
        cycles_per_step=cycles_per_step,
    )


def _cycles_per_step(cycles_per_step: dict, tile_k: int) -> int:
    if tile_k in cycles_per_step:
        return int(cycles_per_step[tile_k])
    raise KeyError(f"tail cost model has no cycles_per_step entry for tile_k={tile_k}")


def wgrad_raster_rows(
    grid_n: int, grid_k: int, clusters: int, tile_k: int, *, block_n: int, block_k: int
) -> int:
    """Tile order inside a group by the per-wave operand footprint (1 = n blocks fastest)."""
    a_slab, b_slab = block_n * block_k * 2, tile_k * block_k * 2
    cols = math.ceil(clusters / grid_k) * a_slab + min(clusters, grid_k) * b_slab
    rows = min(clusters, grid_n) * a_slab + math.ceil(clusters / grid_n) * b_slab
    return 1 if rows < cols else 0


def wgrad_reduce_grid(
    num_tail: int, tile_k: int, *, reduce_threads: int, epi_chunk: int, cluster_m: int
) -> int:
    """CTAs of the tail-reduce launch: one per tail tile and row block."""
    rows_per_cta = reduce_threads // (tile_k // epi_chunk)
    return num_tail * (cluster_m // rows_per_cta)


def select_wgrad_tile(
    k: int,
    arch: str,
    sum_m: int,
    num_groups: int,
    constants: Optional[dict[str, Any]] = None,
) -> int:
    """The production k-tile rule for a weight gradient of shape ``(sum_m, num_groups, k)``:
    the row threshold is the architecture's ``k512_min_rows_per_group``."""
    c = constants or host_plan_constants()
    k512_archs = tuple(c["k512_archs"])
    min_rows = c["k512_min_rows_per_group"]
    return wgrad_tile_k(
        k,
        arch,
        sum_m / max(num_groups, 1),
        k512_archs=k512_archs,
        k512_min_rows_per_group=int(min_rows[arch]) if arch in k512_archs else 0,
    )


def plan_fwd_dgrad(
    op: str,
    *,
    sum_m: int,
    num_groups: int,
    n: int,
    k: int,
    sm_count: int,
    tile_n: Optional[int] = None,
    constants: Optional[dict[str, Any]] = None,
    arch: Optional[str] = None,
) -> dict[str, Any]:
    """Launch plan of ``fwd`` / ``dgrad``: one persistent launch over output tiles ``tile_n``
    columns wide (``block_n`` by default).  ``c0`` / ``c1`` are the even split of the
    clusters over the claim queues; ``queue_split`` is ``"probed"`` where a launch of
    ``op`` on ``arch`` passes the probed split of its scheduler workspace instead
    (module docstring), else ``"even"``."""
    c = constants or host_plan_constants()
    cols = n if op == "fwd" else k
    tile_n = int(c["block_n"]) if tile_n is None else int(tile_n)
    tiles = max_cluster_tiles_upper_bound(
        sum_m,
        num_groups,
        cols,
        cluster_m=int(c["cluster_m"]),
        block_n=tile_n,
    )
    cta_group = int(c["cta_group"])
    grid_x = persistent_grid(sm_count, tiles, cta_group=cta_group)
    clusters = grid_x // cta_group
    c0 = clusters // 2
    probed = (
        op in c["domain_split_ops"]
        and arch in c["domain_split_archs"]
        and n * k <= int(c["domain_split_max_weight_elems"])
    )
    return dict(
        grid=[grid_x, 1, 1],
        clusters=clusters,
        c0=c0,
        c1=clusters - c0,
        queue_split="probed" if probed else "even",
        launches=1,
        tile_n=tile_n,
        tile_k=None,
    )


def plan_wgrad(
    *,
    sum_m: int,
    num_groups: int,
    n: int,
    k: int,
    tile_k: int,
    sm_count: int,
    reduce_threads: int,
    k256_tail: bool = False,
    unordered_tail: bool = False,
    constants: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Launch plan of ``wgrad`` for the selected ``tile_k``.

    ``reduce_threads`` is the thread count of the tail-reduce program (its
    launch block); it sizes the reduce grid.  ``k256_tail`` plans the 512-wide
    k tile's ``main_halves_k256t`` stage: the grid of the 512-wide tile, whose
    full waves cover ``[0, total // clusters * clusters)`` of its tiles; the
    tiles left run n-major (``raster_rows`` 0) as the ``K256_TAIL_TILE_K``-wide
    half tiles of the 256-wide tiling, and the last partial wave of that tiling
    is the tail (``tail_tile_k``), split as the 256-wide k tile's.

    ``unordered_tail`` plans a program that adds its split tail in place (the
    512-wide k tile's): the split comes from ``atomic_tail_plan``
    (:func:`wgrad_atomic_tail_plan`), there is no reduce launch
    (``tail_reduce_grid`` None, one launch) and, when the tail is split,
    ``tail_sum_elems`` / ``tail_ticket_words`` size its ``tail_sums`` /
    ``tail_tickets`` stream workspaces (0 otherwise).  Otherwise the split
    comes from ``tail_plan`` and a split tail is summed by the ordered
    tail-reduce launch, as in the deterministic package.
    """
    c = constants or host_plan_constants()
    block_n, block_k = int(c["block_n"]), int(c["block_k"])
    cluster_m, cta_group = int(c["cluster_m"]), int(c["cta_group"])
    tiles = num_groups * (n // block_n) * (k // tile_k)
    grid_x = persistent_grid(sm_count, tiles, cta_group=cta_group)
    clusters = grid_x // cta_group
    tail_tile_k = K256_TAIL_TILE_K if k256_tail else tile_k
    num_tail = 0
    if int(c["wgrad_tail_split"]):
        num_tail = wgrad_tail_tiles(
            num_groups, n, k, tail_tile_k, clusters, block_n=block_n
        )
    splits = 1
    if num_tail:
        tail = c["atomic_tail_plan" if unordered_tail else "tail_plan"]
        tail_plan = wgrad_atomic_tail_plan if unordered_tail else wgrad_tail_plan
        splits = tail_plan(
            num_tail,
            clusters,
            sum_m,
            num_groups,
            tail_tile_k,
            block_k=block_k,
            cluster_m=cluster_m,
            max_splits=int(tail["max_splits"]),
            sm_clock_ghz=float(tail["sm_clock_ghz"]),
            dram_tbps=float(tail["dram_tbps"]),
            launch_us=float(tail["launch_us"]),
            cycles_per_step=tail["cycles_per_step"],
        )
    raster_rows = (
        0
        if k256_tail
        else wgrad_raster_rows(
            n // block_n,
            k // tile_k,
            clusters,
            tile_k,
            block_n=block_n,
            block_k=block_k,
        )
    )
    reduce_grid = None
    if splits > 1 and not unordered_tail:
        reduce_grid = [
            wgrad_reduce_grid(
                num_tail,
                tail_tile_k,
                reduce_threads=reduce_threads,
                epi_chunk=int(c["epi_chunk"]),
                cluster_m=cluster_m,
            ),
            1,
            1,
        ]
    split = splits > 1 and unordered_tail
    return dict(
        grid=[grid_x, 1, 1],
        clusters=clusters,
        tile_k=tile_k,
        tail_tile_k=tail_tile_k,
        num_tail=num_tail,
        tail_splits=splits,
        raster_rows=raster_rows,
        tail_reduce_grid=reduce_grid,
        unordered_tail=bool(unordered_tail),
        tail_sum_elems=num_tail * cluster_m * tail_tile_k if split else 0,
        tail_ticket_words=num_tail * cta_group * TAIL_TICKETS_PER_CTA if split else 0,
        launches=1 + int(reduce_grid is not None),
    )


def dgrad_evict_first(
    k: int,
    arch: str,
    sum_m: int,
    num_groups: int,
    constants: Optional[dict[str, Any]] = None,
) -> bool:
    """Whether the data gradient's wide tile takes its L2 evict-first twin.

    ``k`` is the dgrad ``K``, the columns of ``dX`` (``w.shape[2]``); the reduction runs
    over ``N`` (``w.shape[1]``).  The twin keeps each group's weights resident in L2 when a
    group spans many 256-row blocks: the groups average at least
    ``dgrad_evict_first_min_rows`` rows (``sum_m // num_groups``) and every 256-row block
    has at least ``dgrad_evict_first_min_k / wide_block_n`` column tiles."""
    c = constants or host_plan_constants()
    return (
        arch in c["dgrad_evict_first_archs"]
        and sum_m // max(num_groups, 1) >= int(c["dgrad_evict_first_min_rows"])
        and k >= int(c["dgrad_evict_first_min_k"])
    )


def dgrad_wide_ts33(
    k: int,
    arch: str,
    sum_m: int,
    num_groups: int,
    constants: Optional[dict[str, Any]] = None,
) -> bool:
    """Whether the data gradient's wide tile takes its twin with three head and three
    tail k steps around the release of the accumulator's halves (``main_wide_ts33``).

    ``k`` is the dgrad ``K``, the columns of ``dX`` (``w.shape[2]``).  The twin serves
    the gate/up projection's data gradient over few groups: the groups average at least
    the architecture's ``dgrad_wide_ts33_min_rows`` rows (``sum_m // num_groups``) and
    ``K >= dgrad_wide_ts33_min_k``, on the architectures of ``dgrad_wide_ts33_archs``.
    Every element accumulates its k steps in the same order, so the twin gives the bits
    of ``main_wide``."""
    c = constants or host_plan_constants()
    return (
        arch in c["dgrad_wide_ts33_archs"]
        and k >= int(c["dgrad_wide_ts33_min_k"])
        and sum_m // max(num_groups, 1) >= int(c["dgrad_wide_ts33_min_rows"][arch])
    )


def fwd_wide_ts33(
    k: int,
    arch: str,
    sum_m: int,
    num_groups: int,
    constants: Optional[dict[str, Any]] = None,
) -> bool:
    """Whether the forward pass's wide tile takes its twin with three head and three
    tail k steps around the release of the accumulator's halves (``main_wide_ts33``).

    ``k`` is the forward ``K``, the reduction dimension (``w.shape[2]``).  On the
    architectures of ``fwd_wide_ts33_archs`` the twin serves the short reductions of
    the down projection over many groups (``K <= fwd_wide_ts33_max_k`` and at most
    ``fwd_wide_ts33_max_rows`` rows per group, ``sum_m // num_groups``).  It gives
    the bits of ``main_wide``."""
    c = constants or host_plan_constants()
    rows = sum_m // max(num_groups, 1)
    return (
        arch in c["fwd_wide_ts33_archs"]
        and k <= int(c["fwd_wide_ts33_max_k"])
        and rows <= int(c["fwd_wide_ts33_max_rows"])
    )


def fwd_lwh_ef(
    k: int,
    arch: str,
    constants: Optional[dict[str, Any]] = None,
) -> bool:
    """Whether the forward pass's wide tile takes its twin with the half-release last
    wave and evict-first Y stores.

    ``k`` is the forward ``K``, the reduction dimension: the columns of ``x``
    (``x.shape[1]``, equal to ``w.shape[2]``); the output columns are ``N``
    (``w.shape[1]``).  The twin serves the short reductions of the down projection,
    ``K <= fwd_lwh_ef_max_k``, on the architectures of ``fwd_lwh_ef_archs``; a longer
    reduction (the gate/up projection) keeps the plain wide tile."""
    c = constants or host_plan_constants()
    return arch in c["fwd_lwh_ef_archs"] and k <= int(c["fwd_lwh_ef_max_k"])


def wgrad_k256_tail(
    k: int,
    arch: str,
    sum_m: int,
    num_groups: int,
    constants: Optional[dict[str, Any]] = None,
) -> bool:
    """Whether the weight gradient's half-release 512-wide k tile takes its twin that
    runs the tiles left after its full waves as 256-column half tiles
    (``main_halves_k256t``).

    ``k`` is the weight-gradient ``K``: the columns of ``dW[e]`` and of ``x``
    (``x.shape[1]``), the dimension the k tile cuts.  ``N``, the rows of ``dW[e]``
    (the columns of ``g``), does not enter the choice, and the reduction runs over
    the ``sum_m`` routed rows.  The twin serves the short outputs of the down
    projection, ``K <= k256_tail_max_k``, when the groups average at least
    ``k256_tail_min_rows`` rows (``sum_m // num_groups``), on the architectures of
    ``k256_tail_archs``; a longer ``K`` (the gate/up projection) or fewer rows per
    group (more groups) keeps ``main_halves``."""
    c = constants or host_plan_constants()
    return (
        arch in c["k256_tail_archs"]
        and k <= int(c["k256_tail_max_k"])
        and sum_m // max(num_groups, 1) >= int(c["k256_tail_min_rows"])
    )


def select_row_stage(
    name: str,
    op: str,
    n: int,
    k: int,
    arch: str,
    constants: Optional[dict[str, Any]] = None,
    *,
    sum_m: int,
    num_groups: int,
) -> tuple[str, int]:
    """``(stage, tile_n)`` of route ``name`` for ``fwd`` / ``dgrad``: ``main_wide`` with the wide
    tile (``fwd_dgrad_tile_n``) where the route serves it on ``arch``, else ``main`` with
    ``block_n``; a data gradient takes ``main_wide_ef`` in place of ``main_wide`` where
    ``dgrad_evict_first`` holds, and a forward pass where ``fwd_lwh_ef`` holds, when the
    route serves that stage on ``arch``; else ``main_wide_ts33`` where
    ``dgrad_wide_ts33`` (a data gradient) or ``fwd_wide_ts33`` (a forward pass) holds
    and the route serves it on ``arch``."""
    c = constants or host_plan_constants()
    tile_n = fwd_dgrad_tile_n(op, n, k, arch, c)
    if tile_n != int(c["block_n"]) and stage_served(name, "main_wide", arch):
        if (
            (op == "dgrad" and dgrad_evict_first(k, arch, sum_m, num_groups, c))
            or (op == "fwd" and fwd_lwh_ef(k, arch, c))
        ) and stage_served(name, "main_wide_ef", arch):
            return "main_wide_ef", tile_n
        if (
            (op == "dgrad" and dgrad_wide_ts33(k, arch, sum_m, num_groups, c))
            or (op == "fwd" and fwd_wide_ts33(k, arch, sum_m, num_groups, c))
        ) and stage_served(name, "main_wide_ts33", arch):
            return "main_wide_ts33", tile_n
        return "main_wide", tile_n
    return "main", int(c["block_n"])


def wgrad_unordered_tail(
    tile_k: int,
    arch: str,
    constants: Optional[dict[str, Any]] = None,
) -> bool:
    """Whether the weight gradient binds the builds that add their split tail in place with
    unordered reductions: the 512-wide k tile on the architectures of ``unordered_tail_archs``.
    Otherwise it binds the deterministic programs, whose split tail the ordered ``tail_reduce``
    stage sums; the registry has those for the 256-wide k tile only, so the list must name every
    architecture of the k512 routes (module docstring)."""
    c = constants or host_plan_constants()
    return tile_k == 512 and arch in c["unordered_tail_archs"]


def select_wgrad_stage(
    name: str,
    k: int,
    tile_k: int,
    arch: str,
    constants: Optional[dict[str, Any]] = None,
    *,
    sum_m: int,
    num_groups: int,
) -> tuple[str, bool]:
    """``(main stage, wave)`` of weight-gradient route ``name`` (module docstring, program choice).

    ``k`` is the weight-gradient ``K``, the columns of ``dW[e]`` and of ``x``; ``sum_m``
    and ``num_groups`` are the routed rows and the groups they are split over."""
    c = constants or host_plan_constants()
    wave = (
        k >= int(c["wave_sync_min_k"])
        and arch in c["wave_sync_archs"][tile_k]
        and stage_served(name, "main_wave", arch)
    )
    stage = "main_wave" if wave else "main"
    if tile_k == 512 and arch in c["k512_release_archs"]:
        halves = f"{stage}_halves"
        if stage_served(name, halves, arch):
            stage = halves
    if (
        stage == "main_halves"
        and wgrad_k256_tail(k, arch, sum_m, num_groups, c)
        and stage_served(name, "main_halves_k256t", arch)
    ):
        stage = "main_halves_k256t"
    return stage, wave


# ---------------------------------------------------------------------------
# Prepared launches
# ---------------------------------------------------------------------------


# The CUDA state queries of the per-launch path, bound to their C entries
# (``torch.cuda.current_stream`` constructs a ``Stream`` object per call), with
# the ``torch.cuda`` forms on a torch build without them.
try:
    _current_raw_stream = torch._C._cuda_getCurrentRawStream
except AttributeError:  # pragma: no cover - older torch

    def _current_raw_stream(device_index: int) -> int:
        return torch.cuda.current_stream(device_index).cuda_stream


try:
    _current_stream_capturing = torch._C._cuda_isCurrentStreamCapturing
except AttributeError:  # pragma: no cover - older torch
    _current_stream_capturing = torch.cuda.is_current_stream_capturing
try:
    _current_device = torch._C._cuda_getDevice
except AttributeError:  # pragma: no cover - older torch
    _current_device = torch.cuda.current_device

# The workspace kinds a program's argument plan may name, by argument name.
STREAM_WORKSPACES = ("sched", "wave_sync", "tail_sums", "tail_tickets")
# The kinds whose size is the launch's (the split tail's sums and tickets); the others have a fixed size.
SIZED_STREAM_WORKSPACES = ("tail_sums", "tail_tickets")
# The arguments each host binding passes to ``_bind`` (besides ``grid``): the forward pass and the data
# gradient (``_prepare_row_op``), the weight gradient's main stage and its ordered tail reduce
# (``_prepare_wgrad``).  ``_bind`` refuses any other.  A generated program's argument plan may name these,
# the stream workspaces (bound per launch), the row bases of ``ROW_BASES`` whose operand the binding passes,
# and ``GRID_ARGUMENTS`` (``bindable_arguments``).
HOST_ARGUMENTS: dict[str, frozenset[str]] = {
    "row_op": frozenset(
        ("A", "B", "C", "offs", "num_groups", "sum_m", "N", "K", "ldc", "c0", "c1")
    ),
    "wgrad": frozenset(
        (
            "A",
            "B",
            "C",
            "offs",
            "tensormap_workspace",
            "partials",
            "tail_sums",
            "tail_tickets",
            "tail_splits",
            "raster_rows",
            "num_groups",
            "sum_m",
            "N",
            "K",
            "ldc",
            "stride_e",
            "wave_mode",
        )
    ),
    "tail_reduce": frozenset(
        (
            "partials",
            "C",
            "offs",
            "num_groups",
            "N",
            "K",
            "ldc",
            "stride_e",
            "num_clusters",
            "tail_splits",
            "raster_rows",
        )
    ),
}
# Row-base buffers: a program may take the base of an operand again under this name, besides its TMA
# descriptor.  The kernel build reads rows through it only in its sm_107a override (``ROW_BASE_READ_ARCHS``),
# which takes the row pitch to be the row width, so the operand must have dense rows there.  On another
# architecture the build never reads it, and the host binds a one-element view of the operand: no copy, any
# row stride, safe under CUDA graph capture.  (Today only the 256-wide k tile's ``main_wave`` program on
# sm_100a / sm_103a takes row bases, and it does not read them.)
ROW_BASES = {"A_rows": "A", "B_rows": "B"}
ROW_BASE_READ_ARCHS = ("sm_107a",)
GRID_ARGUMENTS = ("grid_x", "grid_y", "grid_z")
# The caller's names of the operands ``A`` / ``B`` per operation, for error messages.
_OPERAND_NAMES = {
    "fwd": {"A": "x", "B": "w"},
    "dgrad": {"A": "g", "B": "w"},
    "wgrad": {"A": "g", "B": "x"},
}


def host_binding(op: str, stage: str) -> str:
    """The host binding (a key of ``HOST_ARGUMENTS``) that binds ``stage`` of a route of ``op``."""
    if stage == "tail_reduce":
        return "tail_reduce"
    return "wgrad" if op == "wgrad" else "row_op"


def bindable_arguments(op: str, stage: str) -> frozenset[str]:
    """Every argument name the host can give a program bound as ``stage`` of a route of ``op``: the binding's
    arguments, the stream workspaces, the row bases of the operands it passes and the grid."""
    host = HOST_ARGUMENTS[host_binding(op, stage)]
    rows = {name for name, operand in ROW_BASES.items() if operand in host}
    return host | set(STREAM_WORKSPACES) | rows | set(GRID_ARGUMENTS)


def _row_base(
    t: torch.Tensor, op: str, operand: str, program: str, stage: str, arch: str
) -> torch.Tensor:
    """The row-base argument of ``operand`` (``t``, two or more dimensions) for a launch on ``arch``: ``t``
    itself where the build reads rows through it (``ROW_BASE_READ_ARCHS``), refused unless its rows are dense
    (row stride = row width; a single row has any stride); else a one-element view at ``t``'s first element."""
    if arch not in ROW_BASE_READ_ARCHS:
        return t.as_strided((1,), (1,))
    if t.size(-2) > 1 and t.stride(-2) != t.size(-1):
        label = _OPERAND_NAMES.get(op, {}).get(operand, operand)
        raise ValueError(
            f"{op}: {label} must have dense rows (row stride {t.size(-1)}, got {t.stride(-2)}): the {arch} "
            f"build of the {stage!r} stage ({program}) also reads its rows through a plain pointer; pass "
            f"{label}.contiguous()"
        )
    return t


# Stream workspaces (module docstring): one buffer per (kind, device index,
# CUDA stream handle), created on the first launch on that stream that takes
# it (a sized kind: replaced by a larger one when a launch needs more).
_stream_workspaces: dict[tuple[str, int, int], torch.Tensor] = {}
_STREAM_WORKSPACES_LOCK = threading.Lock()


def _rest_workspace(kind: str, device: torch.device, size: int = 0) -> torch.Tensor:
    """A new ``kind`` workspace on ``device`` at its rest state (contiguous; u32 words, or fp32 for
    ``tail_sums``), set on the current stream.  ``size`` is the element count of a sized kind."""
    c = host_plan_constants()
    if kind == "tail_sums":
        return torch.zeros(int(size), dtype=torch.float32, device=device)
    if kind == "tail_tickets":
        return torch.zeros(int(size), dtype=torch.int32, device=device).view(
            torch.uint32
        )
    if kind == "sched":
        words = torch.zeros(int(c["sched_words"]), dtype=torch.int32, device=device)
        for index, value in enumerate(c["sched_rest"]):
            if value:
                # Two's-complement spelling of the u32 rest word in the int32 view,
                # written by a device fill so the set-up is legal under graph capture.
                words.narrow(0, index, 1).fill_(
                    int(value) - (1 << 32) if value >= 1 << 31 else int(value)
                )
        return words.view(torch.uint32)
    if kind == "wave_sync":
        return torch.zeros(
            int(c["wave_sync_words"]), dtype=torch.int32, device=device
        ).view(torch.uint32)
    raise KeyError(
        f"unknown stream workspace {kind!r}; expected one of {STREAM_WORKSPACES}"
    )


def _keeps_workspaces(device_index: int) -> bool:
    """Whether a launch for device ``device_index`` on the current stream takes the kept
    stream workspaces: a capture records no buffer the host keeps, and the capture state
    is readable for the current device's stream only, so a launch for another device
    takes buffers of its own too."""
    return device_index == _current_device() and not _current_stream_capturing()


def _launch_workspace(
    kind: str, device: torch.device, device_index: int, kept: bool, size: int = 0
) -> torch.Tensor:
    """The ``kind`` workspace of a launch on the current stream of ``device`` (module docstring);
    ``kept`` is ``_keeps_workspaces(device_index)`` and ``size`` the element count a sized kind
    needs (the kept buffer may be larger; its prefix is used)."""
    if not kept:
        return _rest_workspace(kind, device, size)
    key = (kind, device_index, _current_raw_stream(device_index))
    buffer = _stream_workspaces.get(key)
    if buffer is None or buffer.numel() < size:
        with _STREAM_WORKSPACES_LOCK:
            buffer = _stream_workspaces.get(key)
            if buffer is None or buffer.numel() < size:
                # A replaced buffer was last used by a launch on this stream; the caching
                # allocator hands its memory to later work of this stream only, in order.
                buffer = _stream_workspaces[key] = _rest_workspace(kind, device, size)
    return buffer


def stream_workspace(
    kind: str, device: torch.device, stream: Optional[torch.cuda.Stream] = None
) -> Optional[torch.Tensor]:
    """The ``kind`` (``"sched"`` / ``"wave_sync"`` / ``"tail_sums"`` / ``"tail_tickets"``) workspace kept for
    ``stream`` (the current stream of ``device`` by default), or ``None`` before a launch on that stream needed
    one."""
    if kind not in STREAM_WORKSPACES:
        raise KeyError(
            f"unknown stream workspace {kind!r}; expected one of {STREAM_WORKSPACES}"
        )
    index = _device_index(device)
    handle = _current_raw_stream(index) if stream is None else int(stream.cuda_stream)
    return _stream_workspaces.get((kind, index, handle))


# Probed queue split (module docstring).  Per (device index, data pointer) of a kept
# scheduler workspace: the probe in flight, then the (c0, c1) it measured, or None where
# it could not run or did not see every SM and both domains.  Kept workspaces live as long
# as the process, so a data pointer never names two of them.
_queue_splits: dict[tuple[int, int], Any] = {}
_QUEUE_PROBE_ROUNDS = 5
_QUEUE_PROBE_SMS = 1024  # %smid values the probe records
_QUEUE_PROBE_SOURCE = r"""
// One 32-thread CTA per SM (the launch asks for an SM's whole shared memory): lane 0
// times an add-zero atomic on a line of the 4 KB chunk at the first 8 KB boundary 128 B
// past sched (the programs' probe chunk) and on the same line of the next chunk, the
// minimum of two rounds each, and stores both under its %smid.
__global__ void __launch_bounds__(32, 1) cake_queue_probe_kernel(unsigned* sched, long long* lat) {
  extern __shared__ unsigned cell[];
  unsigned smid, s = (unsigned)__cvta_generic_to_shared(cell);
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  if (threadIdx.x != 0 || smid >= 1024) return;
  unsigned long long base = (unsigned long long)sched;
  unsigned word = (unsigned)((((base + 8319ull) & ~8191ull) - base) >> 2) + (smid / 2u) % 32u * 32u;
  long long lat_a = 0x7fffffffffffffffll, lat_b = 0x7fffffffffffffffll;
  for (int round = 0; round < 2; ++round) {
    long long t0, t1, t2;
    unsigned v;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) : : "memory");
    asm volatile("atom.global.add.u32 %0, [%1], 0;" : "=r"(v) : "l"(sched + word) : "memory");
    asm volatile("st.volatile.shared.u32 [%2], %1; mov.u64 %0, %%clock64;"
                 : "=l"(t1) : "r"(v), "r"(s) : "memory");
    asm volatile("atom.global.add.u32 %0, [%1], 0;" : "=r"(v) : "l"(sched + word + 1024) : "memory");
    asm volatile("st.volatile.shared.u32 [%2], %1; mov.u64 %0, %%clock64;"
                 : "=l"(t2) : "r"(v), "r"(s) : "memory");
    lat_a = t1 - t0 < lat_a ? t1 - t0 : lat_a;
    lat_b = t2 - t1 < lat_b ? t2 - t1 : lat_b;
  }
  lat[2 * smid] = lat_a;
  lat[2 * smid + 1] = lat_b;
}

void cake_queue_probe(tvm::ffi::TensorView sched, tvm::ffi::TensorView lat, int64_t ctas,
                      int64_t smem) {
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(sched.device().device_type, sched.device().device_id));
  TVM_FFI_ICHECK(cudaFuncSetAttribute(cake_queue_probe_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem)) == cudaSuccess);
  cake_queue_probe_kernel<<<static_cast<unsigned>(ctas), 32, static_cast<size_t>(smem),
                            stream>>>(static_cast<unsigned*>(sched.data_ptr()),
                                      static_cast<long long*>(lat.data_ptr()));
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess);
}
"""
_UNPROBED = object()


@functools.cache
def _queue_probe_module(capability: tuple[int, int]) -> Optional[Any]:
    """The queue probe built for compute capability ``capability`` (``tvm_ffi.cpp.load_inline``,
    nvcc), or None when it cannot be built here."""
    try:
        import tvm_ffi.cpp

        major, minor = capability
        return tvm_ffi.cpp.load_inline(
            name=f"cake_moe_queue_probe_sm{major}{minor}",
            cuda_sources=_QUEUE_PROBE_SOURCE,
            functions=["cake_queue_probe"],
            extra_cuda_cflags=[
                f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
            ],
        )
    except Exception:  # noqa: BLE001 - no CUDA compiler here: the even split serves
        return None


@dataclass(frozen=True)
class _QueueProbe:
    """A queue probe in flight: ``counts`` (pinned host memory) holds the SM pairs nearer
    the probe chunk, the pairs nearer the next chunk and the SMs seen once ``done`` has
    completed."""

    counts: torch.Tensor
    done: torch.cuda.Event
    sms: int

    def split(self) -> Optional[tuple[int, int]]:
        near_a, near_b, seen = (int(v) for v in self.counts.tolist())
        if seen != self.sms or min(near_a, near_b) == 0:
            return None
        return near_a, near_b


def _start_queue_probe(sched: torch.Tensor, index: int) -> Optional[_QueueProbe]:
    """Enqueue the probe of the kept scheduler workspace ``sched`` of device ``index`` on
    the current stream (module docstring); None when it cannot run."""
    module = _queue_probe_module(torch.cuda.get_device_capability(index))
    sms = sm_count(sched.device)
    if module is None or sms > _QUEUE_PROBE_SMS:
        return None
    smem = int(torch.cuda.get_device_properties(index).shared_memory_per_block_optin)
    try:
        lat = torch.full(
            (_QUEUE_PROBE_ROUNDS, 2 * _QUEUE_PROBE_SMS),
            -1,
            dtype=torch.int64,
            device=sched.device,
        )
        for row in lat:
            module.cake_queue_probe(sched, row, sms, smem)
        near, far = lat[:, 0::2], lat[:, 1::2]
        seen = (near >= 0).all(0)
        # Rounds in which the next chunk was nearer, per SM pair (2i, 2i + 1: one die),
        # over its seen members; a pair takes the next chunk's queue when most did.
        votes = ((far < near) & seen).sum(0).view(-1, 2).sum(1)
        members = seen.view(-1, 2).sum(1)
        queue_1 = 2 * votes > _QUEUE_PROBE_ROUNDS * members
        counts = torch.empty(3, dtype=torch.int64, pin_memory=True)
        counts.copy_(
            torch.stack(
                (
                    ((members > 0) & ~queue_1).sum(),
                    ((members > 0) & queue_1).sum(),
                    seen.sum(),
                )
            ),
            non_blocking=True,
        )
        done = torch.cuda.Event()
        done.record()
    except Exception:  # noqa: BLE001 - the even split serves when the probe cannot run
        return None
    return _QueueProbe(counts=counts, done=done, sms=sms)


def _probed_queue_split(sched: torch.Tensor, index: int) -> Optional[tuple[int, int]]:
    """``(c0, c1)`` of a launch on the kept scheduler workspace ``sched`` of device
    ``index``: the probed pair counts once its probe has completed, else None (the even
    split).  The first call for ``sched`` enqueues its probe; none waits for it."""
    key = (index, sched.data_ptr())
    state: Any = _queue_splits.get(key, _UNPROBED)
    if state is _UNPROBED:
        with _STREAM_WORKSPACES_LOCK:
            if key not in _queue_splits:
                _queue_splits[key] = _start_queue_probe(sched, index)
        return None
    if isinstance(state, _QueueProbe):
        if not state.done.query():
            return None
        state = _queue_splits[key] = state.split()
    return state


def stream_queue_split(
    device: torch.device, stream: Optional[torch.cuda.Stream] = None
) -> Optional[tuple[int, int]]:
    """The probed ``(c0, c1)`` that launches on ``stream`` (the current stream of
    ``device`` by default) pass where ``plan["queue_split"]`` is ``"probed"``, or ``None``
    while they pass the even split: before a probed launch on that stream, while its
    probe runs, or where the probe could not run (module docstring)."""
    sched = stream_workspace("sched", device, stream)
    if sched is None:
        return None
    state = _queue_splits.get((_device_index(device), sched.data_ptr()))
    if isinstance(state, _QueueProbe):
        return state.split() if state.done.query() else None
    return state


@dataclass(frozen=True)
class _Launch:
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)
    # (argument position, workspace kind, elements) of the stream workspaces the
    # program takes (elements: the size a sized kind needs, else 0); their
    # placeholders in ``arguments`` are filled per launch.
    workspace_slots: tuple[tuple[int, str, int], ...] = ()
    # (sched, c0, c1) argument positions of a launch that passes the probed queue
    # split of its kept scheduler workspace (module docstring), else empty.
    split_slots: tuple[int, ...] = ()


@dataclass(frozen=True)
class GroupedGemmLaunch:
    """One prepared ragged grouped GEMM of the non-deterministic package.

    ``launch()`` runs the bound program on the current torch stream into the
    caller-owned ``out`` with no CUDA allocation and no host synchronisation
    and returns ``out``.  Each launch passes the operands' TMA descriptors by
    value (encoded by the binding from the bound tensors' metadata), and the
    kernels read ``offs`` on device at every launch, so the same prepared
    launch (or a CUDA Graph capturing it) stays valid when the caller writes
    new group boundaries into ``offs`` or new values into the operands.
    Prepare a new launch when a shape, dtype or tensor binding changes.
    ``plan`` records the host decisions (grid, queue split, k tile, tail
    split, whether the split tail is unordered and the sizes of its sums and
    tickets, raster, wave mode, launch count); ``stages`` names the bound stages
    of the route in launch order.  The scheduler, wave, tail-sum and tail-ticket
    workspaces are the current stream's (module docstring), resolved at every
    launch, as is the probed queue split where ``plan["queue_split"]`` is
    ``"probed"``.

    A weight-gradient launch checks torch's deterministic mode at every
    ``launch()`` (a host flag read, no synchronisation) and raises the
    non-determinism alert when it is on, also when it was switched on after
    the preparation.  Under CUDA graph capture the check runs when the launch
    is captured; a replay of the graph does not check again.
    """

    op: str
    record_name: str
    stages: tuple[str, ...]
    plan: dict[str, Any]
    out: torch.Tensor
    launches: tuple[_Launch, ...] = field(repr=False)
    zero_fill: bool = False
    # Caller-owned buffers of this launch (weight gradient): the per-CTA
    # descriptor slots and, on the 256-wide k tile, the fp32 tail partials (on
    # the 512-wide one without a split tail, the one-element tail placeholders).
    # Kept alive with the launch.
    workspaces: tuple[torch.Tensor, ...] = field(default=(), repr=False)
    device_index: int = -1

    def launch(self) -> torch.Tensor:
        if self.op == "wgrad":
            _alert_not_deterministic("GroupedGemmLaunch.launch")
        return self._run()

    def _run(self) -> torch.Tensor:
        """``launch()`` without the deterministic-mode check, for entry points that raised the alert."""
        if self.zero_fill:
            self.out.zero_()
        with tvm_ffi.use_torch_stream():
            for item in self.launches:
                arguments: tuple | list[Any] = item.arguments
                if item.workspace_slots:
                    bound = list(item.arguments)
                    kept = _keeps_workspaces(self.device_index)
                    for position, kind, size in item.workspace_slots:
                        bound[position] = _launch_workspace(
                            kind, self.out.device, self.device_index, kept, size
                        )
                    if item.split_slots and kept:
                        sched, c0, c1 = item.split_slots
                        split = _probed_queue_split(bound[sched], self.device_index)
                        if split is not None:
                            bound[c0], bound[c1] = split
                    arguments = bound
                item.entry(*arguments)
        return self.out

    __call__ = launch


def _span_view(t: torch.Tensor) -> torch.Tensor:
    """Contiguous 1-D view over the memory span of a (possibly row-padded) output.

    The kernels address the output through ``ldc`` / ``stride_e``; the binding
    accepts contiguous pointer arguments, so it receives the flat span starting
    at the tensor's first element.
    """
    if t.is_contiguous():
        return t
    span = 1 + sum(
        (size - 1) * stride
        for size, stride in zip(t.shape, t.stride(), strict=True)
        if size > 0
    )
    return t.as_strided((span,), (1,))


def _end_offsets(offs: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Accept ``offs[E]`` (cumulative end offsets) or ``m_indptr[E + 1]`` (leading 0).

    Both forms are the ``torch._grouped_mm`` / ``grouped_mm_bf16`` conventions;
    the ``[E + 1]`` form is reduced to its ``[E]`` end-offset view on the device
    (no copy, no host read).
    """
    if offs.dtype == torch.int32 and offs.ndim == 1 and offs.numel() == num_groups + 1:
        return offs[1:]
    return offs


def _check_offs(offs: torch.Tensor, num_groups: int) -> None:
    if offs.dtype != torch.int32 or offs.ndim != 1 or offs.numel() != num_groups:
        raise ValueError(
            f"offs must be a one-dimensional int32 tensor of {num_groups} cumulative "
            f"end offsets, or an m_indptr of {num_groups + 1} entries "
            f"(got {offs.dtype}, shape {tuple(offs.shape)})"
        )


def _check_device(tensors: dict[str, torch.Tensor]) -> torch.device:
    device = next(iter(tensors.values())).device
    for name, t in tensors.items():
        if not t.is_cuda or t.device != device:
            raise ValueError(
                f"{name} must be on the CUDA device of the first operand ({device})"
            )
    arch = arch_for(device)
    if arch is None:
        cc = torch.cuda.get_device_capability(device)
        raise ValueError(
            "the ragged grouped GEMM programs require compute capability 10.0, 10.3 "
            f"or 10.7 (got {cc[0]}.{cc[1]})"
        )
    return device


def _aligned_bytes(nbytes: int, device: torch.device, what: str) -> torch.Tensor:
    ws = torch.empty(nbytes, dtype=torch.uint8, device=device)
    if ws.data_ptr() % DESCRIPTOR_ALIGN:
        raise RuntimeError(f"{what} must be {DESCRIPTOR_ALIGN}-byte aligned")
    return ws


def _bind(
    name: str,
    stage: str,
    kwargs: dict[str, Any],
    arch: str,
    *,
    split: bool = False,
    sizes: Optional[dict[str, int]] = None,
) -> _Launch:
    """Order ``kwargs`` by the generated argument plan of ``stage`` of route ``name`` on ``arch`` and load
    its entry (the binding encodes the operands' TMA descriptors by value at every launch).  A buffer the
    plan names after a stream workspace kind and ``kwargs`` does not bind is left to
    ``GroupedGemmLaunch.launch`` (a sized kind with its element count from ``sizes``); with ``split`` so
    are ``c0`` / ``c1``, which a launch replaces by the probed split of its ``sched`` once known, and the
    queue probe is built here."""
    binding = stage_binding(name, stage, arch)
    program = PROGRAMS[binding["program"]]
    op = ROUTES[name]["op"]
    unlisted = set(kwargs) - {"grid"} - HOST_ARGUMENTS[host_binding(op, stage)]
    if unlisted:
        raise KeyError(
            f"the host binding of route {name!r}, stage {stage!r} passes {sorted(unlisted)}, which "
            f"HOST_ARGUMENTS[{host_binding(op, stage)!r}] does not list"
        )
    grid = dict(zip(GRID_ARGUMENTS, kwargs["grid"], strict=True))
    arguments: list[Any] = []
    slots: list[tuple[int, str, int]] = []
    for kind, arg in program["arg_plan"]:
        if kind == "grid":
            arguments.append(int(grid[arg]))
        elif kind == "buffer" and arg in STREAM_WORKSPACES and arg not in kwargs:
            size = int((sizes or {}).get(arg, 0))
            if arg in SIZED_STREAM_WORKSPACES and size <= 0:
                raise KeyError(
                    f"generated program {binding['program']!r} (route {name!r}, stage {stage!r}) "
                    f"takes the stream workspace {arg!r}; the host binding gives neither a buffer nor its size"
                )
            slots.append((len(arguments), arg, size))
            arguments.append(None)
        elif kind == "buffer" and arg in ROW_BASES and ROW_BASES[arg] in kwargs:
            arguments.append(
                _row_base(
                    kwargs[ROW_BASES[arg]],
                    op,
                    ROW_BASES[arg],
                    binding["program"],
                    stage,
                    arch,
                )
            )
        elif arg in kwargs:
            arguments.append(kwargs[arg])
        else:
            raise KeyError(
                f"generated program {binding['program']!r} (route {name!r}, stage {stage!r}) "
                f"expects argument {arg!r} ({kind}); the host binding provides {sorted(kwargs)}"
            )
    module = load_module(binding["program"], arch, binding["specializations"])
    split_slots: tuple[int, ...] = ()
    if split:
        names = [arg for _kind, arg in program["arg_plan"]]
        split_slots = (names.index("sched"), names.index("c0"), names.index("c1"))
        _queue_probe_module(torch.cuda.get_device_capability(kwargs["A"].device))
    return _Launch(
        entry=getattr(module, program["ffi_entry"]),
        arguments=tuple(arguments),
        workspace_slots=tuple(slots),
        split_slots=split_slots,
    )


def _finish(
    op: str,
    name: str,
    plan: dict[str, Any],
    out: torch.Tensor,
    stages: tuple[str, ...],
    launches: list[_Launch],
    workspaces: list,
    *,
    zero_fill: bool = False,
) -> GroupedGemmLaunch:
    return GroupedGemmLaunch(
        op=op,
        record_name=name,
        stages=stages,
        plan=plan,
        out=out,
        launches=tuple(launches),
        zero_fill=zero_fill,
        workspaces=tuple(workspaces),
        device_index=_device_index(out.device),
    )


def _prepare_row_op(
    op: str,
    a: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor],
) -> GroupedGemmLaunch:
    c = host_plan_constants()
    block_n, block_k = int(c["block_n"]), int(c["block_k"])
    a_name = "x" if op == "fwd" else "g"
    if a.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise ValueError(f"{a_name} and w must be bfloat16")
    if a.dim() != 2 or w.dim() != 3 or a.stride(1) != 1 or w.stride(2) != 1:
        raise ValueError(
            f"{a_name} must be a [sum_m, cols] tensor and w an [E, N, K] tensor, both "
            "with unit stride along the last dimension"
        )
    num_groups = int(w.shape[0])
    if op == "fwd":
        sum_m, k = (int(v) for v in a.shape)
        n = int(w.shape[1])
        if int(w.shape[2]) != k:
            raise ValueError(
                f"x has K={k} columns but w is [E, N, K] with K={int(w.shape[2])}"
            )
        if n % block_n or k % block_k:
            raise ValueError(
                f"fwd needs N % {block_n} == 0 and K % {block_k} == 0 (got N={n}, K={k})"
            )
        out_cols = n
    else:
        sum_m, n = (int(v) for v in a.shape)
        k = int(w.shape[2])
        if int(w.shape[1]) != n:
            raise ValueError(
                f"g has N={n} columns but w is [E, N, K] with N={int(w.shape[1])}"
            )
        if k % block_n or n % block_k:
            raise ValueError(
                f"dgrad needs K % {block_n} == 0 and N % {block_k} == 0 (got N={n}, K={k})"
            )
        out_cols = k
    offs = _end_offsets(offs, num_groups)
    _check_offs(offs, num_groups)
    device = _check_device({a_name: a, "w": w, "offs": offs})
    if out is None:
        out = torch.empty(sum_m, out_cols, dtype=torch.bfloat16, device=device)
    if (
        tuple(out.shape) != (sum_m, out_cols)
        or out.dtype != torch.bfloat16
        or out.stride(1) != 1
    ):
        raise ValueError(
            f"out must be a bfloat16 [{sum_m}, {out_cols}] tensor with unit column stride"
        )
    if out.device != device:
        raise ValueError("out must be on the device of the operands")
    if sum_m * out.stride(0) > int(c["int32_elems"]):
        raise ValueError(
            "the output span exceeds the 32-bit element addressing of the kernel"
        )
    arch = arch_for(device)
    name = select_module(op, arch)
    stage, tile_n = select_row_stage(
        name, op, n, k, arch, c, sum_m=sum_m, num_groups=num_groups
    )
    plan = plan_fwd_dgrad(
        op,
        sum_m=sum_m,
        num_groups=num_groups,
        n=n,
        k=k,
        sm_count=sm_count(device),
        tile_n=tile_n,
        constants=c,
        arch=arch,
    )
    if sum_m == 0:
        return _finish(op, name, plan, out, (), [], [])
    kwargs = dict(
        A=a,
        B=w,
        C=_span_view(out),
        offs=offs,
        num_groups=num_groups,
        sum_m=sum_m,
        N=n,
        K=k,
        ldc=int(out.stride(0)),
        c0=plan["c0"],
        c1=plan["c1"],
        grid=plan["grid"],
    )
    launch = _bind(name, stage, kwargs, arch, split=plan["queue_split"] == "probed")
    return _finish(op, name, plan, out, (stage,), [launch], [])


def prepare_grouped_gemm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    deterministic: bool = True,
) -> GroupedGemmLaunch:
    """Prepare ``Y[sum_m, N] = grouped X @ W[e].T`` (bf16 in, bf16 out).

    ``x``: ``[sum_m, K]`` bf16; ``w``: ``[E, N, K]`` bf16; ``offs``: int32
    ``[E]`` device end offsets; ``N % 256 == 0``, ``K % 64 == 0`` (the
    512-column tile when ``N % 512 == 0`` on the architectures of
    ``wide_fwd_archs``, its 3 / 3 head / tail twin where ``fwd_wide_ts33`` holds).
    ``out`` may be row-padded (any row stride, unit column
    stride).  ``deterministic`` must be ``False`` (module docstring); the
    programs are copies of the deterministic package's and give its bits.
    """
    _require_deterministic_false(deterministic, "prepare_grouped_gemm_fwd")
    return _prepare_row_op("fwd", x, w, offs, out)


def prepare_grouped_gemm_dgrad(
    g: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    deterministic: bool = True,
) -> GroupedGemmLaunch:
    """Prepare ``dX[sum_m, K] = grouped G @ W[e]`` (W read in place; bf16 in, bf16 out).

    ``g``: ``[sum_m, N]`` bf16; ``w``: ``[E, N, K]`` bf16; ``offs``: int32
    ``[E]`` device end offsets; ``K % 256 == 0``, ``N % 64 == 0`` (the
    512-column tile when ``K % 512 == 0`` on the architectures of
    ``wide_dgrad_archs``, its L2 evict-first twin where ``dgrad_evict_first``
    holds for ``K``, ``sum_m`` and ``E``, its 3 / 3 head / tail twin where
    ``dgrad_wide_ts33`` holds).  ``deterministic`` must be ``False`` (module
    docstring); the programs are copies of the deterministic package's and give
    its bits.
    """
    _require_deterministic_false(deterministic, "prepare_grouped_gemm_dgrad")
    return _prepare_row_op("dgrad", g, w, offs, out)


def _prepare_wgrad(
    g: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
    num_groups: Optional[int],
) -> GroupedGemmLaunch:
    """The weight-gradient preparation of :func:`prepare_grouped_gemm_wgrad`, after its mode check and alert."""
    c = host_plan_constants()
    block_n = int(c["block_n"])
    if g.dtype != torch.bfloat16 or x.dtype != torch.bfloat16:
        raise ValueError("g and x must be bfloat16")
    if g.dim() != 2 or x.dim() != 2 or g.stride(1) != 1 or x.stride(1) != 1:
        raise ValueError(
            "g must be a [sum_m, N] tensor and x a [sum_m, K] tensor, both with unit "
            "stride along the last dimension"
        )
    sum_m, n = (int(v) for v in g.shape)
    sum_mx, k = (int(v) for v in x.shape)
    if sum_mx != sum_m:
        raise ValueError(f"g has {sum_m} rows but x has {sum_mx}")
    if num_groups is None:
        num_groups = int(offs.numel())
    else:
        offs = _end_offsets(offs, int(num_groups))
    _check_offs(offs, num_groups)
    device = _check_device({"g": g, "x": x, "offs": offs})
    out_dtype = out_dtype or (out.dtype if out is not None else torch.bfloat16)
    if out_dtype not in _DTYPE_NAMES:
        raise ValueError(
            f"wgrad output dtype must be bfloat16 or float32, got {out_dtype}"
        )
    if out is None:
        out = torch.empty(num_groups, n, k, dtype=out_dtype, device=device)
    if (
        tuple(out.shape) != (num_groups, n, k)
        or out.dtype != out_dtype
        or out.stride(2) != 1
    ):
        raise ValueError(
            f"out must be a {_DTYPE_NAMES[out_dtype]} [{num_groups}, {n}, {k}] tensor "
            "with unit stride along K"
        )
    if out.device != device:
        raise ValueError("out must be on the device of the operands")
    if num_groups * out.stride(0) > int(c["int32_elems"]):
        raise ValueError(
            "the output span exceeds the 32-bit element addressing of the kernel"
        )
    arch = arch_for(device)
    tile_k = select_wgrad_tile(k, arch, sum_m, num_groups, c)
    if n % block_n or k % tile_k:
        raise ValueError(
            f"wgrad needs N % {block_n} == 0 and K % {tile_k} == 0 (got N={n}, K={k})"
        )
    name = select_module(
        "wgrad", arch, out_dtype=_DTYPE_NAMES[out_dtype], tile_k=tile_k
    )
    main_stage, wave = select_wgrad_stage(
        name, k, tile_k, arch, c, sum_m=sum_m, num_groups=num_groups
    )
    k256_tail = main_stage == "main_halves_k256t"
    # The unordered-tail programs add their split tail in place (they take
    # ``tail_sums``); the deterministic programs' split tail is summed in order by
    # the tail-reduce stage.
    unordered = wgrad_unordered_tail(tile_k, arch, c)
    reduce_name = None
    reduce_threads = 0
    if not unordered:
        # An ordered k256-tail stage's split tail is the 256-wide k tile's: the
        # tail-reduce stage of that tile's route sums its partials.
        reduce_name = (
            select_module(
                "wgrad",
                arch,
                out_dtype=_DTYPE_NAMES[out_dtype],
                tile_k=K256_TAIL_TILE_K,
            )
            if k256_tail
            else name
        )
        if stage_served(reduce_name, "tail_reduce", arch):
            reduce_program = PROGRAMS[
                stage_binding(reduce_name, "tail_reduce", arch)["program"]
            ]
            reduce_threads = int(reduce_program["launch"]["block"][0])
    plan = plan_wgrad(
        sum_m=sum_m,
        num_groups=num_groups,
        n=n,
        k=k,
        tile_k=tile_k,
        sm_count=sm_count(device),
        reduce_threads=reduce_threads,
        k256_tail=k256_tail,
        unordered_tail=unordered,
        constants=c,
    )
    plan["wave_mode"] = int(c["wave_sync_mode"][arch]) if wave else 0
    if sum_m == 0:
        return _finish("wgrad", name, plan, out, (), [], [], zero_fill=True)
    grid_x, clusters, splits = plan["grid"][0], plan["clusters"], plan["tail_splits"]
    slots = _aligned_bytes(
        grid_x * int(c["tmap_slots_per_cta"]) * int(c["tmap_slot_bytes"]),
        device,
        "per-CTA descriptor slots",
    )
    workspaces: list = [slots]
    c_view = _span_view(out)
    main_kwargs = dict(
        A=g,
        B=x,
        C=c_view,
        offs=offs,
        tensormap_workspace=slots,
        tail_splits=splits,
        raster_rows=plan["raster_rows"],
        num_groups=num_groups,
        sum_m=sum_m,
        N=n,
        K=k,
        ldc=int(out.stride(1)),
        stride_e=int(out.stride(0)),
        wave_mode=plan["wave_mode"],
        grid=plan["grid"],
    )
    sizes: dict[str, int] = {}
    partials: Optional[torch.Tensor] = None
    if unordered and splits > 1:
        # The split tail's sums and tickets: the current stream's, resolved per launch.
        sizes = {
            "tail_sums": plan["tail_sum_elems"],
            "tail_tickets": plan["tail_ticket_words"],
        }
    elif unordered:
        # No split tail: one-element placeholders the kernel never touches.
        tail_sums = torch.zeros(1, dtype=torch.float32, device=device)
        tail_tickets = torch.zeros(1, dtype=torch.int32, device=device).view(
            torch.uint32
        )
        main_kwargs.update(tail_sums=tail_sums, tail_tickets=tail_tickets)
        workspaces += [tail_sums, tail_tickets]
    else:
        units = plan["num_tail"] * splits if splits > 1 else 0
        partials = torch.empty(
            max(units, 1) * int(c["cluster_m"]) * plan["tail_tile_k"],
            dtype=torch.float32,
            device=device,
        )
        main_kwargs["partials"] = partials
        workspaces.append(partials)
    stages = [main_stage]
    launches = [_bind(name, main_stage, main_kwargs, arch, sizes=sizes)]
    if plan["tail_reduce_grid"] is not None:
        reduce_kwargs = dict(
            partials=partials,
            C=c_view,
            offs=offs,
            num_groups=num_groups,
            N=n,
            K=k,
            ldc=int(out.stride(1)),
            stride_e=int(out.stride(0)),
            num_clusters=clusters,
            tail_splits=splits,
            raster_rows=plan["raster_rows"],
            grid=plan["tail_reduce_grid"],
        )
        stages.append("tail_reduce")
        launches.append(_bind(reduce_name, "tail_reduce", reduce_kwargs, arch))
    return _finish("wgrad", name, plan, out, tuple(stages), launches, workspaces)


def prepare_grouped_gemm_wgrad(
    g: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    num_groups: Optional[int] = None,
    *,
    deterministic: bool = True,
) -> GroupedGemmLaunch:
    """Prepare ``dW[E, N, K] = grouped G[e].T @ X[e]`` (bf16 in; bf16 or fp32 out), non-deterministic.

    ``g``: ``[sum_m, N]`` bf16; ``x``: ``[sum_m, K]`` bf16; ``offs``: int32
    ``[E]`` device end offsets (``E = offs.numel()`` unless ``num_groups`` is
    given, in which case an ``m_indptr`` of ``E + 1`` entries is accepted too);
    ``N % 256 == 0`` and ``K`` a multiple of the selected k tile (256, or 512
    when it is chosen).  Empty groups produce exact zeros.  ``out_dtype``
    defaults to the dtype of ``out`` (bf16 when neither is given).

    ``deterministic`` must be ``False``.  With the 512-wide k tile a split tail
    is added in place with unordered fp32 reductions (``plan["unordered_tail"]``
    and ``plan["tail_splits"] > 1``), so ``dW`` is not bitwise reproducible run
    to run; with the 256-wide k tile the deterministic program and its ordered
    tail reduce run.  Under ``torch.use_deterministic_algorithms(True)`` this
    function raises the non-determinism alert (module docstring).
    """
    _require_deterministic_false(deterministic, "prepare_grouped_gemm_wgrad")
    _alert_not_deterministic("prepare_grouped_gemm_wgrad")
    return _prepare_wgrad(g, x, offs, out, out_dtype, num_groups)
