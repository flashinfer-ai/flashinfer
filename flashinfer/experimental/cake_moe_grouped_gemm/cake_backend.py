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

Cake backend: ragged BF16 MoE grouped GEMM (SM100 / SM103 / SM107).

Three operations over ``E`` groups of rows described by an int32 **device**
tensor ``offs[E]`` of cumulative end offsets (never read on the host):

* ``fwd``:   ``Y[offs[e-1]:offs[e]]  = X[offs[e-1]:offs[e]] @ W[e].T``
* ``dgrad``: ``dX[offs[e-1]:offs[e]] = G[offs[e-1]:offs[e]] @ W[e]``
* ``wgrad``: ``dW[e] = G[offs[e-1]:offs[e]].T @ X[offs[e-1]:offs[e]]`` (bf16 or fp32 output)

Every launch is one persistent cluster kernel (one CTA pair per SM pair); the
weight gradient adds one ordered fp32 tail-reduce kernel when the tiles of its
last partial wave are split over the clusters.  Nothing in the plan depends on
the group sizes: the grid, the k tile, the tail split and the raster follow
from host-known scalars (tensor shapes, the SM count, the architecture) so a
prepared launch is CUDA-graph capturable and stays valid for any ``offs``
written later.  The host planner below reproduces the production launchers'
decisions exactly from ``HOST_PLAN_CONSTANTS``.

Program choice.  ``fwd`` and ``dgrad`` launch the 512-column output tile (the
route's ``main_wide`` stage) when the output columns are a multiple of
``wide_block_n`` and the architecture is listed in ``wide_fwd_archs`` /
``wide_dgrad_archs``, else the 256-column tile (``main``).  On the
architectures of ``dgrad_evict_first_archs`` the data gradient launches the
twin of its wide tile that loads G and stores dX with the L2 evict-first policy
(``main_wide_ef``) when the groups average at least
``dgrad_evict_first_min_rows`` rows (``sum_m // E``) and dX has at least
``dgrad_evict_first_min_k`` columns (``K``).  On the architectures of
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
waves of the 512-wide k tile as 256-column half tiles, with the split tail and
tail reduce of the 256-wide k tile (``main_halves_k256t``), when ``K`` (the
columns of ``dW[e]`` and of x) is at most ``k256_tail_max_k`` and the groups
average at least ``k256_tail_min_rows`` rows (``sum_m // E``), the down
projection of an MoE layer over few groups.  Every choice falls back to the
plain stage where the route serves no such stage on the architecture.

Stream workspaces.  The forward and data-gradient programs take a scheduler
workspace (``sched``: ``sched_words`` u32 words whose first three rest at
``sched_rest``) and the split of their clusters over the two claim queues
(``c0 = clusters // 2``, ``c1 = clusters - c0``); the weight-gradient wave
programs take two wave counters (``wave_sync``, resting at zero) and their wave
mode.  Every launch leaves a workspace at its rest state (the kernels re-arm it
at the end), so one buffer serves every launch of one CUDA stream: the buffers
are kept per (device, stream), allocated and set to their rest state by the
first launch on a stream and reused by every later one, never shared by two
streams (launches on one stream are ordered, so a buffer is never live in two
launches at once).  A launch captured into a CUDA graph, or issued for a
device other than the current one, takes buffers of its own at their rest
state, set at the launch (a capture records that set with the launch and
allocates the buffers from the graph's pool).  The static sm_107a forward and
data-gradient programs take ``sched``, ``c0`` and ``c1`` and ignore them.
"""

from __future__ import annotations

import functools
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from .cake_jit import (
    HOST_PLAN_CONSTANTS,
    PROGRAMS,
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
) -> dict[str, Any]:
    """Launch plan of ``fwd`` / ``dgrad``: one persistent launch over output tiles ``tile_n``
    columns wide (``block_n`` by default)."""
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
    return dict(
        grid=[grid_x, 1, 1],
        clusters=clusters,
        c0=c0,
        c1=clusters - c0,
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
    constants: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Launch plan of ``wgrad`` for the selected ``tile_k``.

    ``reduce_threads`` is the thread count of the tail-reduce program (its
    launch block); it sizes the reduce grid.  ``k256_tail`` plans the 512-wide
    k tile's ``main_halves_k256t`` stage: the grid of the 512-wide tile, whose
    full waves cover ``[0, total // clusters * clusters)`` of its tiles; the
    tiles left run n-major (``raster_rows`` 0) as the ``K256_TAIL_TILE_K``-wide
    half tiles of the 256-wide tiling, and the last partial wave of that tiling
    is the tail (``tail_tile_k``), split and reduced as the 256-wide k tile's.
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
        tail = c["tail_plan"]
        splits = wgrad_tail_plan(
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
    if splits > 1:
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
    return dict(
        grid=[grid_x, 1, 1],
        clusters=clusters,
        tile_k=tile_k,
        tail_tile_k=tail_tile_k,
        num_tail=num_tail,
        tail_splits=splits,
        raster_rows=raster_rows,
        tail_reduce_grid=reduce_grid,
        launches=1 + int(splits > 1),
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
    route serves that stage on ``arch``."""
    c = constants or host_plan_constants()
    tile_n = fwd_dgrad_tile_n(op, n, k, arch, c)
    if tile_n != int(c["block_n"]) and stage_served(name, "main_wide", arch):
        if (
            (op == "dgrad" and dgrad_evict_first(k, arch, sum_m, num_groups, c))
            or (op == "fwd" and fwd_lwh_ef(k, arch, c))
        ) and stage_served(name, "main_wide_ef", arch):
            return "main_wide_ef", tile_n
        return "main_wide", tile_n
    return "main", int(c["block_n"])


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
STREAM_WORKSPACES = ("sched", "wave_sync")
# Stream workspaces (module docstring): one buffer per (kind, device index,
# CUDA stream handle), created on the first launch on that stream.
_stream_workspaces: dict[tuple[str, int, int], torch.Tensor] = {}
_STREAM_WORKSPACES_LOCK = threading.Lock()


def _rest_workspace(kind: str, device: torch.device) -> torch.Tensor:
    """A new ``kind`` workspace on ``device`` at its rest state (u32 words, contiguous), set on the current stream."""
    c = host_plan_constants()
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


def _launch_workspace(
    kind: str, device: torch.device, device_index: int
) -> torch.Tensor:
    """The ``kind`` workspace of a launch on the current stream of ``device`` (module docstring)."""
    if device_index != _current_device() or _current_stream_capturing():
        # A capture records no buffer the host keeps; the capture state is readable for the
        # current device's stream only, so a launch for another device takes this path too.
        return _rest_workspace(kind, device)
    key = (kind, device_index, _current_raw_stream(device_index))
    buffer = _stream_workspaces.get(key)
    if buffer is None:
        with _STREAM_WORKSPACES_LOCK:
            buffer = _stream_workspaces.get(key)
            if buffer is None:
                buffer = _stream_workspaces[key] = _rest_workspace(kind, device)
    return buffer


def stream_workspace(
    kind: str, device: torch.device, stream: Optional[torch.cuda.Stream] = None
) -> Optional[torch.Tensor]:
    """The ``kind`` (``"sched"`` / ``"wave_sync"``) workspace kept for ``stream`` (the current stream of
    ``device`` by default), or ``None`` before a launch on that stream needed one."""
    if kind not in STREAM_WORKSPACES:
        raise KeyError(
            f"unknown stream workspace {kind!r}; expected one of {STREAM_WORKSPACES}"
        )
    index = _device_index(device)
    handle = _current_raw_stream(index) if stream is None else int(stream.cuda_stream)
    return _stream_workspaces.get((kind, index, handle))


@dataclass(frozen=True)
class _Launch:
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)
    # (argument position, workspace kind) of the stream workspaces the program
    # takes; their placeholders in ``arguments`` are filled per launch.
    workspace_slots: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class GroupedGemmLaunch:
    """One prepared ragged grouped GEMM.

    ``launch()`` runs the bound program on the current torch stream into the
    caller-owned ``out`` with no CUDA allocation and no host synchronisation
    and returns ``out``.  Each launch passes the operands' TMA descriptors by
    value (encoded by the binding from the bound tensors' metadata), and the
    kernels read ``offs`` on device at every launch, so the same prepared
    launch (or a CUDA Graph capturing it) stays valid when the caller writes
    new group boundaries into ``offs`` or new values into the operands.
    Prepare a new launch when a shape, dtype or tensor binding changes.
    ``plan`` records the host decisions (grid, queue split, k tile, tail
    split, raster, wave mode, launch count); ``stages`` names the bound stages
    of the route in launch order.  The scheduler and wave workspaces are the
    current stream's (module docstring), resolved at every launch.
    """

    op: str
    record_name: str
    stages: tuple[str, ...]
    plan: dict[str, Any]
    out: torch.Tensor
    launches: tuple[_Launch, ...] = field(repr=False)
    zero_fill: bool = False
    # Caller-owned buffers of this launch (weight gradient): the per-CTA
    # descriptor slots and the fp32 tail partials.  Kept alive with the launch.
    workspaces: tuple[torch.Tensor, ...] = field(default=(), repr=False)
    device_index: int = -1

    def launch(self) -> torch.Tensor:
        if self.zero_fill:
            self.out.zero_()
        with tvm_ffi.use_torch_stream():
            for item in self.launches:
                arguments: tuple | list[Any] = item.arguments
                if item.workspace_slots:
                    bound = list(item.arguments)
                    for position, kind in item.workspace_slots:
                        bound[position] = _launch_workspace(
                            kind, self.out.device, self.device_index
                        )
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


def _bind(name: str, stage: str, kwargs: dict[str, Any], arch: str) -> _Launch:
    """Order ``kwargs`` by the generated argument plan of ``stage`` of route ``name`` on ``arch`` and load
    its entry (the binding encodes the operands' TMA descriptors by value at every launch).  A buffer the
    plan names after a stream workspace kind is left to ``GroupedGemmLaunch.launch``."""
    binding = stage_binding(name, stage, arch)
    program = PROGRAMS[binding["program"]]
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), kwargs["grid"], strict=True))
    arguments: list[Any] = []
    slots: list[tuple[int, str]] = []
    for kind, arg in program["arg_plan"]:
        if kind == "grid":
            arguments.append(int(grid[arg]))
        elif kind == "buffer" and arg in STREAM_WORKSPACES:
            slots.append((len(arguments), arg))
            arguments.append(None)
        elif arg in kwargs:
            arguments.append(kwargs[arg])
        else:
            raise KeyError(
                f"generated program {binding['program']!r} (route {name!r}, stage {stage!r}) "
                f"expects argument {arg!r} ({kind}); the host binding provides {sorted(kwargs)}"
            )
    module = load_module(binding["program"], arch, binding["specializations"])
    return _Launch(
        entry=getattr(module, program["ffi_entry"]),
        arguments=tuple(arguments),
        workspace_slots=tuple(slots),
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
    return _finish(
        op, name, plan, out, (stage,), [_bind(name, stage, kwargs, arch)], []
    )


def prepare_grouped_gemm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> GroupedGemmLaunch:
    """Prepare ``Y[sum_m, N] = grouped X @ W[e].T`` (bf16 in, bf16 out).

    ``x``: ``[sum_m, K]`` bf16; ``w``: ``[E, N, K]`` bf16; ``offs``: int32
    ``[E]`` device end offsets; ``N % 256 == 0``, ``K % 64 == 0`` (the
    512-column tile when ``N % 512 == 0`` on the architectures of
    ``wide_fwd_archs``).  ``out`` may be row-padded (any row stride, unit column
    stride).
    """
    return _prepare_row_op("fwd", x, w, offs, out)


def prepare_grouped_gemm_dgrad(
    g: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> GroupedGemmLaunch:
    """Prepare ``dX[sum_m, K] = grouped G @ W[e]`` (W read in place; bf16 in, bf16 out).

    ``g``: ``[sum_m, N]`` bf16; ``w``: ``[E, N, K]`` bf16; ``offs``: int32
    ``[E]`` device end offsets; ``K % 256 == 0``, ``N % 64 == 0`` (the
    512-column tile when ``K % 512 == 0`` on the architectures of
    ``wide_dgrad_archs``, its L2 evict-first twin where ``dgrad_evict_first``
    holds for ``K``, ``sum_m`` and ``E``).
    """
    return _prepare_row_op("dgrad", g, w, offs, out)


def prepare_grouped_gemm_wgrad(
    g: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    num_groups: Optional[int] = None,
) -> GroupedGemmLaunch:
    """Prepare ``dW[E, N, K] = grouped G[e].T @ X[e]`` (bf16 in; bf16 or fp32 out), deterministic.

    ``g``: ``[sum_m, N]`` bf16; ``x``: ``[sum_m, K]`` bf16; ``offs``: int32
    ``[E]`` device end offsets (``E = offs.numel()`` unless ``num_groups`` is
    given, in which case an ``m_indptr`` of ``E + 1`` entries is accepted too);
    ``N % 256 == 0`` and ``K`` a multiple of the selected k tile (256, or 512
    when it is chosen).  Empty groups produce exact zeros.  ``out_dtype``
    defaults to the dtype of ``out`` (bf16 when neither is given).
    """
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
    # The k256-tail stage's split tail is the 256-wide k tile's: the tail-reduce
    # stage of that tile's route sums its partials.
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
    reduce_threads = 0
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
    units = plan["num_tail"] * splits if splits > 1 else 0
    partials = torch.empty(
        max(units, 1) * int(c["cluster_m"]) * plan["tail_tile_k"],
        dtype=torch.float32,
        device=device,
    )
    workspaces: list = [slots, partials]
    c_view = _span_view(out)
    main_kwargs = dict(
        A=g,
        B=x,
        C=c_view,
        offs=offs,
        tensormap_workspace=slots,
        partials=partials,
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
    stages = [main_stage]
    launches = [_bind(name, main_stage, main_kwargs, arch)]
    if splits > 1:
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


# ---------------------------------------------------------------------------
# Stable-API handoff: flashinfer.grouped_mm.grouped_mm_bf16(..., backend="cake")
# ---------------------------------------------------------------------------


def grouped_mm_bf16_cake(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    tactic: int = -1,
) -> torch.Tensor:
    """``grouped_mm_bf16`` forward projection on the Cake backend.

    ``out[m_indptr[e]:m_indptr[e + 1]] = a[m_indptr[e]:m_indptr[e + 1]] @ b[e].T``
    with ``a`` ``[cum_m, K]`` bf16, ``b`` ``[E, N, K]`` bf16 and ``m_indptr``
    ``[E + 1]`` int32 on the device (read by the kernel, never on the host).
    Only bfloat16 outputs are produced and there is no tactic index; rows of
    ``a`` past ``m_indptr[-1]`` are left untouched in ``out``.  The
    experimental-backend warning is emitted by
    ``flashinfer.grouped_mm.grouped_mm_bf16`` when the backend is named; this
    function emits none.
    """
    if out is not None:
        out_dtype = out.dtype
    if out_dtype != torch.bfloat16:
        raise ValueError(
            "the cake backend of grouped_mm_bf16 produces bfloat16 outputs only "
            f"(got out_dtype={out_dtype})"
        )
    if tactic != -1:
        raise ValueError(
            "the cake backend of grouped_mm_bf16 has a single generated program "
            f"per architecture; tactic must be -1 (got {tactic})"
        )
    return prepare_grouped_gemm_fwd(a, b, m_indptr, out=out).launch()
