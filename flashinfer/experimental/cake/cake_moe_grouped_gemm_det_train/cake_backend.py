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

The slice lower bound is ``0`` for ``e == 0`` and ``offs[e-1]`` for ``e > 0``.
For ``m_indptr`` input, ``offs`` in these formulas means ``m_indptr[1:]``.

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
(``main_wide_ef``) when the groups average at least the architecture's
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
``fwd_lwh_ef_max_k``, the down projection of an MoE layer; on those of
``fwd_evict_first_archs`` it launches the twin of its wide tile that loads X and
stores Y with that policy (``main_wide_ef`` too) when ``K`` is at least
``fwd_evict_first_min_k``, the gate/up projection.  The weight gradient takes the
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

Accumulate mode.  ``cake_moe_grouped_gemm_det_train_prepare_wgrad(..., out=out, accumulate=True)``
(also ``cake_moe_grouped_gemm_det_train_wgrad`` and the autograd wrapper's
``fuse_wgrad_accumulation``) adds the weight gradient into ``out``:
``out[e] <- out[e] + G_e^T X_e``, Megatron's gradient-accumulation fusion into an
fp32 or bf16 ``main_grad``.  The host plans the launch exactly as the overwrite of
``out.dtype`` (route, k tile, stage, grid, tail split, raster, wave mode);
``plan["accumulate"]`` records how the launch adds (``"off"`` for an overwrite).
``"fused"``: the 512-wide k tile, where the route serves the main stage's
accumulate build (the stage of its name with ``ACC_SUFFIX``: the same argument
plan, launch shape and host plan, an epilogue that adds each element of C once
in place) on the device's architecture and, when the plan splits its tail
(``tail_splits > 1``), the tail-reduce stage's accumulate build
(``TAIL_REDUCE_ACC``) on the route of that tail (the 256-wide k tile's for
``main_halves_k256t``), which sums the ordered partials in their fixed order and
adds the sum into C once; every such build's library can load by the loader's
static rule (its prebuilt file under ``aot/``, else a FlashInfer JIT toolkit with
``nvcc``, JIT not disabled, at its ``build.json`` ``min_cuda``:
:func:`acc_library_loadable`, nothing built or probed), and C is
``ACC_ALIGN``-byte aligned (its base, row pitch and expert pitch).  The programs
write ``out`` in place; no scratch.  ``"composed"``
(``plan["accumulate_reason"]``, the first that fails in this order:
``"tile_k"`` the 256-wide k tile, ``"stage"`` a main stage without an
accumulate build (every bf16 ``out``), ``"tail"`` a split tail whose route
serves no ``TAIL_REDUCE_ACC``, ``"min_cuda"`` a build that cannot load,
``"alignment"``; the last two with one ``UserWarning`` per process each): the
bound programs write the overwrite into a contiguous scratch of ``out.dtype``
(``E * N * K`` elements) that the preparation allocates and the launch owns, and
the launch adds it with ``out.add_`` on the same stream.  Both routes give
``out`` AccumulateGrad's bits of the deterministic weight gradient in its dtype,
the same bits: one round-to-nearest add of the overwrite's ``dW`` per element of
``out`` (an empty group adds exact zeros, so ``-0.0`` becomes ``+0.0``), bitwise
reproducible run to run.  The checks before anything is allocated or bound:
``out`` given, ``out_dtype`` None or ``out.dtype``, bfloat16 or float32, three
dimensions with unit stride along ``K``, rows and experts that cannot overlap
(row stride at least ``K``, expert stride at least ``N`` row strides) and no byte
shared with ``g``, ``x`` or ``offs``.  ``sum_m == 0`` launches nothing and
leaves ``out`` unchanged.  A launch adds once more at every ``launch()`` and
every CUDA graph replay.
"""

from __future__ import annotations

import functools
import inspect
import math
import os
import sys
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from .cake_build import (
    _jit_toolchain,
    _local_cuda_version,
    _version,
    libraries_by_name,
    prebuilt_path,
)
from .cake_jit import (
    HOST_PLAN_CONSTANTS,
    PROGRAMS,
    ROUTES,
    library_name,
    load_module,
    program_registered,
    record_name,
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
# Accumulate mode (module docstring): the stage that adds into C is the overwrite stage's name with this
# suffix, on the same weight-gradient route.
ACC_SUFFIX = "_acc"
# The tail-reduce stage of a weight-gradient route and its accumulate build: the ordered sum of a split
# tail's partials, stored into C or added into C once.
TAIL_REDUCE = "tail_reduce"
TAIL_REDUCE_ACC = TAIL_REDUCE + ACC_SUFFIX
# The byte alignment of C's base, row pitch and expert pitch the accumulate builds need (the width of
# their vector accesses of C); a less aligned ``out`` takes the composed route.
ACC_ALIGN = 32
# ``plan["accumulate"]`` of a weight gradient: an overwrite, the accumulate builds adding into ``out`` in
# place, or the overwrite into a scratch and an add (module docstring, accumulate mode).
ACCUMULATE_MODES = ("off", "fused", "composed")
# The once-per-process warnings of this module that were issued (the composed route taken because an
# accumulate build cannot load, or for the alignment).
_WARNED: set[str] = set()


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


def _warn_once(key: str, message: str) -> None:
    """A ``UserWarning`` naming the caller's frame, the first time ``key`` warns in this process."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(message, UserWarning, stacklevel=_caller_stacklevel())


def _acc_stages(out_dtype: str, arch: str) -> tuple[str, tuple[str, ...]]:
    """``(route, stages)``: the 512-wide k tile weight-gradient route of ``out_dtype`` (``"bfloat16"`` /
    ``"float32"``) and the accumulate builds of its main stages (``<twin>`` + ``ACC_SUFFIX``, not the
    tail reduce's) it serves on ``arch``."""
    name = record_name("wgrad", out_dtype=out_dtype, tile_k=512)
    route = ROUTES.get(name)
    if route is None or arch not in route["arches"]:
        return name, ()
    return name, tuple(
        dict.fromkeys(
            item["name"]
            for item in route["stages"]
            if item["name"].endswith(ACC_SUFFIX)
            and item["name"] != TAIL_REDUCE_ACC
            and stage_served(name, item["name"], arch)
        )
    )


def acc_stage_served(out_dtype: str, arch: str) -> bool:
    """True when the 512-wide k tile route of ``out_dtype`` (``"bfloat16"`` / ``"float32"``) serves the
    accumulate build of a main stage (a stage named ``<twin>`` + ``ACC_SUFFIX``) on ``arch`` (the registry
    alone; whether its library can load is :func:`acc_library_loadable`)."""
    return bool(_acc_stages(out_dtype, arch)[1])


@functools.cache
def _jit_cuda_version() -> Optional[tuple[int, int]]:
    """``(major, minor)`` of the toolkit FlashInfer's JIT builds with, the version ``cake_build.jit_spec``
    holds an entry's ``min_cuda`` against; None when the loader cannot build: FlashInfer's JIT is not
    importable, ``FLASHINFER_DISABLE_JIT`` is set, or the CUDA path has no ``bin/nvcc`` (FlashInfer's
    version query then reports PyTorch's CUDA version, which compiles nothing) or cannot be resolved.
    Asked once per process."""
    if os.environ.get("FLASHINFER_DISABLE_JIT"):
        return None
    try:
        # Any manifest library: the name only labels the error when the JIT is not importable.
        _gen_jit_spec, get_cuda_version = _jit_toolchain(
            next(iter(libraries_by_name()))
        )
        # _jit_toolchain imported FlashInfer's toolchain module; the package's only flashinfer
        # import stays the JIT fallback of cake_build.py.
        cpp_ext = sys.modules["flashinfer.jit.cpp_ext"]
        if not Path(cpp_ext.get_cuda_path(), "bin", "nvcc").is_file():
            return None
        return _local_cuda_version(get_cuda_version)
    except (RuntimeError, OSError):
        return None


def acc_library_loadable(name: str, stage: str, arch: str) -> bool:
    """Whether the library of ``stage`` of route ``name`` on ``arch`` can load, by the loader's own static rule
    (``cake_build.load_library``): its prebuilt file under ``aot/`` exists, or FlashInfer's JIT toolkit
    (``nvcc`` present, JIT not disabled: :func:`_jit_cuda_version`) reaches the entry's ``build.json``
    ``min_cuda`` (``cake_build.jit_spec`` refuses a lower one).  Nothing is
    built, loaded or probed: an ahead-of-time build below an entry's floor ships no file for it, and a JIT
    toolkit below it would raise at the preparation, so such an accumulate build is not bound (the composed
    route, ``plan["accumulate_reason"] == "min_cuda"``)."""
    binding = stage_binding(name, stage, arch)
    library = library_name(binding["program"], arch, binding["specializations"])
    if prebuilt_path(library).is_file():
        return True
    toolkit = _jit_cuda_version()
    return toolkit is not None and toolkit >= _version(
        libraries_by_name()[library]["min_cuda"]
    )


def generated_program_available(
    device: torch.device,
    op: str = "fwd",
    out_dtype: Optional[torch.dtype] = None,
    *,
    accumulate: bool = False,
) -> bool:
    """True when this checkout registers the program of ``op`` for ``device``.

    With ``accumulate=True`` (weight gradient only): True when the 512-wide k tile route of
    ``out_dtype`` (either output dtype when None; False for a dtype the weight gradient does not
    write) serves the accumulate build of a main stage on the device's architecture whose library
    can load (:func:`acc_library_loadable`), that is when some weight gradient into an ``out`` of
    that dtype can take the fused route (``plan["accumulate"] == "fused"``; a call whose plan splits
    its tail also needs the ``TAIL_REDUCE_ACC`` build of its tail's route, :func:`wgrad_accumulate_mode`);
    the composed route needs only the overwrite programs."""
    _require_flag(accumulate, "accumulate", "generated_program_available")
    if accumulate and op != "wgrad":
        raise ValueError(
            f"generated_program_available: accumulate=True applies to the weight gradient, got op={op!r}"
        )
    arch = arch_for(device)
    if arch is None:
        return False
    name = _DTYPE_NAMES.get(out_dtype) if out_dtype is not None else None
    if not accumulate:
        return program_registered(op, arch, out_dtype=name)
    if out_dtype is not None and name is None:
        return False
    for dtype in (name,) if name else ("bfloat16", "float32"):
        route, stages = _acc_stages(dtype, arch)
        if any(acc_library_loadable(route, stage, arch) for stage in stages):
            return True
    return False


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
    group spans many 256-row blocks: on the architectures of ``dgrad_evict_first_archs``,
    the groups average at least the architecture's ``dgrad_evict_first_min_rows`` rows
    (``sum_m // num_groups``) and every 256-row block has at least
    ``dgrad_evict_first_min_k / wide_block_n`` column tiles (the gate/up projection)."""
    c = constants or host_plan_constants()
    return (
        arch in c["dgrad_evict_first_archs"]
        and sum_m // max(num_groups, 1) >= int(c["dgrad_evict_first_min_rows"][arch])
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


def fwd_evict_first(
    k: int,
    arch: str,
    constants: Optional[dict[str, Any]] = None,
) -> bool:
    """Whether the forward pass's wide tile takes its twin that loads X and stores Y
    with the L2 evict-first policy.

    ``k`` is the forward ``K``, the reduction dimension (``x.shape[1]``, equal to
    ``w.shape[2]``).  The twin serves the long reductions of the gate/up projection,
    ``K >= fwd_evict_first_min_k``, on the architectures of ``fwd_evict_first_archs``,
    whatever the group sizes; a shorter reduction (the down projection) keeps the
    plain wide tile."""
    c = constants or host_plan_constants()
    return arch in c["fwd_evict_first_archs"] and k >= int(c["fwd_evict_first_min_k"])


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
    ``dgrad_evict_first`` holds, and a forward pass where ``fwd_lwh_ef`` or
    ``fwd_evict_first`` holds, when the route serves that stage on ``arch``; else
    ``main_wide_ts33`` where ``dgrad_wide_ts33`` (a data gradient) or ``fwd_wide_ts33``
    (a forward pass) holds and the route serves it on ``arch``."""
    c = constants or host_plan_constants()
    tile_n = fwd_dgrad_tile_n(op, n, k, arch, c)
    if tile_n != int(c["block_n"]) and stage_served(name, "main_wide", arch):
        if (
            (op == "dgrad" and dgrad_evict_first(k, arch, sum_m, num_groups, c))
            or (op == "fwd" and (fwd_lwh_ef(k, arch, c) or fwd_evict_first(k, arch, c)))
        ) and stage_served(name, "main_wide_ef", arch):
            return "main_wide_ef", tile_n
        if (
            (op == "dgrad" and dgrad_wide_ts33(k, arch, sum_m, num_groups, c))
            or (op == "fwd" and fwd_wide_ts33(k, arch, sum_m, num_groups, c))
        ) and stage_served(name, "main_wide_ts33", arch):
            return "main_wide_ts33", tile_n
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


def wgrad_accumulate_mode(
    name: str,
    stage: str,
    tile_k: int,
    arch: str,
    *,
    reduce_name: Optional[str] = None,
    aligned: bool = True,
) -> tuple[str, Optional[str]]:
    """``(plan["accumulate"], plan["accumulate_reason"])`` of a weight gradient that adds into its output,
    for the route ``name``, main stage ``stage`` and k tile ``tile_k`` the overwrite of the same output dtype
    selects on ``arch`` (module docstring, accumulate mode).  ``reduce_name`` is the route whose
    tail-reduce stage sums the plan's split tail (``tail_splits > 1``), None when the tail is not split.

    ``("fused", None)`` when every condition holds; else ``("composed", reason)`` with the first that fails:
    ``"tile_k"`` (not the 512-wide k tile), ``"stage"`` (the route serves no ``stage + ACC_SUFFIX`` on
    ``arch``), ``"tail"`` (a split tail whose route serves no ``TAIL_REDUCE_ACC`` on ``arch``: the
    overwriting tail reduce would store over C), ``"min_cuda"`` (the library of one of those builds cannot
    load, :func:`acc_library_loadable`) or ``"alignment"`` (``out`` is not ``ACC_ALIGN``-byte aligned,
    ``aligned``)."""
    if tile_k != 512:
        return "composed", "tile_k"
    acc_stage = stage + ACC_SUFFIX
    if not stage_served(name, acc_stage, arch):
        return "composed", "stage"
    builds = [(name, acc_stage)]
    if reduce_name is not None:
        if not stage_served(reduce_name, TAIL_REDUCE_ACC, arch):
            return "composed", "tail"
        builds.append((reduce_name, TAIL_REDUCE_ACC))
    if not all(acc_library_loadable(route, build, arch) for route, build in builds):
        return "composed", "min_cuda"
    if not aligned:
        return "composed", "alignment"
    return "fused", None


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
# The version-counter bump of a tensor written in place outside torch (a fused accumulate launch).
try:
    _increment_version = torch.autograd.graph.increment_version
except AttributeError:  # pragma: no cover - older torch
    _increment_version = torch._C._increment_version

# The workspace kinds a program's argument plan may name, by argument name.
STREAM_WORKSPACES = ("sched", "wave_sync")
# Arguments a generated program may name beside the operands it addresses through TMA descriptors: the
# operands' row-addressed bases, the same tensors as ``A`` / ``B`` (``_bind`` passes them where the
# argument plan names them).  A build on an architecture outside ``ROW_BASE_READ_ARCHS`` never reads its
# row bases, and the host binds a one-element view of the operand there: no copy, any row stride, safe
# under CUDA graph capture.  (Today only the 256-wide k tile's ``main_wave`` program on sm_100a /
# sm_103a takes row bases, and it does not read them.)
OPERAND_ALIASES = {"A_rows": "A", "B_rows": "B"}
ROW_BASE_READ_ARCHS = ("sm_107a",)
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


def _keeps_workspaces(device_index: int) -> bool:
    """Whether a launch for device ``device_index`` on the current stream takes the kept
    stream workspaces: a capture records no buffer the host keeps, and the capture state
    is readable for the current device's stream only, so a launch for another device
    takes buffers of its own too."""
    return device_index == _current_device() and not _current_stream_capturing()


def _launch_workspace(
    kind: str, device: torch.device, device_index: int, kept: bool
) -> torch.Tensor:
    """The ``kind`` workspace of a launch on the current stream of ``device`` (module docstring);
    ``kept`` is ``_keeps_workspaces(device_index)``."""
    if not kept:
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
    # (argument position, workspace kind) of the stream workspaces the program
    # takes; their placeholders in ``arguments`` are filled per launch.
    workspace_slots: tuple[tuple[int, str], ...] = ()
    # (sched, c0, c1) argument positions of a launch that passes the probed queue
    # split of its kept scheduler workspace (module docstring), else empty.
    split_slots: tuple[int, ...] = ()


@dataclass(frozen=True)
class CakeMoEGroupedGemmDetLaunch:
    """One prepared ragged grouped GEMM.

    ``launch()`` runs the bound program on the current torch stream into the
    caller-owned ``out`` with no host synchronisation and returns ``out``.  It
    allocates only the stream workspaces (module docstring): the first launch
    on a stream allocates that stream's buffers, later launches on it allocate
    nothing, and a launch under capture or for another device allocates a set
    of its own.  Each launch passes the operands' TMA descriptors by
    value (encoded by the binding from the bound tensors' metadata), and the
    kernels read ``offs`` on device at every launch, so the same prepared
    launch (or a CUDA Graph capturing it) stays valid when the caller writes
    new group boundaries into ``offs`` or new values into the operands.
    Prepare a new launch when a shape, dtype or tensor binding changes.
    ``plan`` records the host decisions (grid, queue split, k tile, tail
    split, raster, wave mode, launch count); ``stages`` names the bound stages
    of the route in launch order.  The scheduler and wave workspaces are the
    current stream's (module docstring), resolved at every launch, as is the
    probed queue split where ``plan["queue_split"]`` is ``"probed"``.

    A weight gradient prepared with ``accumulate=True`` adds into ``out`` at
    every ``launch()`` and every replay of a graph that captured it (it is not
    idempotent).  On the fused route (``plan["accumulate"] == "fused"``,
    ``bump_version``) the bound accumulate builds add into ``out`` in place and
    the launch bumps ``out._version`` with
    ``torch.autograd.graph.increment_version`` (not for an inference tensor).
    On the composed route (``plan["accumulate"] == "composed"``) ``add_from``
    is the scratch the bound programs write, and the launch adds it into
    ``out`` with ``out.add_`` on the same stream, outside autograd.  Either
    route bumps ``out._version`` at ``launch()`` and at capture, not at a
    graph replay.  The composed scratch is owned by the prepared launch, as its
    descriptor slots are, and is allocated on the stream current at the
    preparation (no ``record_stream``), so one prepared accumulate launch must
    not be in flight on two streams at once, the launch must be kept alive
    while any graph that captured it may replay (a graph does not keep the
    scratch alive), and after a launch on another stream it must be kept alive
    until that launch completes.  A launch prepared inside a captured backward
    allocates the scratch from, and returns it to, the graph's pool within that
    backward.
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
    # Composed accumulate: the contiguous scratch of ``out.dtype`` the bound programs write, added into
    # ``out`` after them; None otherwise.
    add_from: Optional[torch.Tensor] = field(default=None, repr=False)
    # Fused accumulate: the programs wrote ``out`` in place; bump its version counter after the launch.
    bump_version: bool = False

    def launch(self) -> torch.Tensor:
        if self.zero_fill:
            self.out.zero_()
        with tvm_ffi.use_torch_stream():
            for item in self.launches:
                arguments: tuple | list[Any] = item.arguments
                if item.workspace_slots:
                    bound = list(item.arguments)
                    kept = _keeps_workspaces(self.device_index)
                    for position, kind in item.workspace_slots:
                        bound[position] = _launch_workspace(
                            kind, self.out.device, self.device_index, kept
                        )
                    if item.split_slots and kept:
                        sched, c0, c1 = item.split_slots
                        split = _probed_queue_split(bound[sched], self.device_index)
                        if split is not None:
                            bound[c0], bound[c1] = split
                    arguments = bound
                item.entry(*arguments)
        if self.add_from is not None:
            # Outside autograd, as the fused route's in-place write: ``out`` may be a leaf that requires
            # grad, or an inference tensor (updated in place only in inference mode).
            mode = torch.inference_mode if self.out.is_inference() else torch.no_grad
            with mode():
                self.out.add_(self.add_from)
        elif self.bump_version and not self.out.is_inference():
            _increment_version(self.out)
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


def _row_base(t: torch.Tensor, operand: str, program: str, arch: str) -> torch.Tensor:
    """The row-base argument of ``operand`` (``t``, two or more dimensions) for a launch on ``arch``: ``t``
    itself where the build reads rows through it (``ROW_BASE_READ_ARCHS``), refused unless its rows are
    dense (row stride = row width; a single row has any stride); else a one-element view at ``t``'s first
    element."""
    if arch not in ROW_BASE_READ_ARCHS:
        return t.as_strided((1,), (1,))
    if t.size(-2) > 1 and t.stride(-2) != t.size(-1):
        raise ValueError(
            f"{operand} must have dense rows (row stride {t.size(-1)}, got {t.stride(-2)}): the {arch} "
            f"build {program} also reads its rows through a plain pointer; pass a contiguous tensor"
        )
    return t


def _bind(
    name: str, stage: str, kwargs: dict[str, Any], arch: str, *, split: bool = False
) -> _Launch:
    """Order ``kwargs`` by the generated argument plan of ``stage`` of route ``name`` on ``arch`` and load
    its entry (the binding encodes the operands' TMA descriptors by value at every launch).  A buffer the
    plan names after a stream workspace kind is left to ``CakeMoEGroupedGemmDetLaunch.launch``; with ``split`` so
    are ``c0`` / ``c1``, which a launch replaces by the probed split of its ``sched`` once known, and the
    queue probe is built here.  An argument of ``OPERAND_ALIASES`` the plan names gets the operand it
    aliases (``_row_base``)."""
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
        elif OPERAND_ALIASES.get(arg) in kwargs:
            arguments.append(
                _row_base(
                    kwargs[OPERAND_ALIASES[arg]],
                    OPERAND_ALIASES[arg],
                    binding["program"],
                    arch,
                )
            )
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
    add_from: Optional[torch.Tensor] = None,
    bump_version: bool = False,
) -> CakeMoEGroupedGemmDetLaunch:
    return CakeMoEGroupedGemmDetLaunch(
        op=op,
        record_name=name,
        stages=stages,
        plan=plan,
        out=out,
        launches=tuple(launches),
        zero_fill=zero_fill,
        workspaces=tuple(workspaces),
        device_index=_device_index(out.device),
        add_from=add_from,
        bump_version=bump_version,
    )


def _prepare_row_op(
    op: str,
    a: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor],
) -> CakeMoEGroupedGemmDetLaunch:
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


def cake_moe_grouped_gemm_det_train_prepare_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> CakeMoEGroupedGemmDetLaunch:
    """Prepare ``Y[sum_m, N] = grouped X @ W[e].T`` (bf16 in, bf16 out).

    ``x``: ``[sum_m, K]`` bf16; ``w``: ``[E, N, K]`` bf16; ``offs``: int32
    ``[E]`` device end offsets; ``N % 256 == 0``, ``K % 64 == 0`` (the
    512-column tile when ``N % 512 == 0`` on the architectures of
    ``wide_fwd_archs``, its L2 evict-first twin where ``fwd_lwh_ef`` or
    ``fwd_evict_first`` holds for ``K``, its 3 / 3 head / tail twin where
    ``fwd_wide_ts33`` holds).
    ``out`` may be row-padded (any row stride, unit column
    stride).
    """
    return _prepare_row_op("fwd", x, w, offs, out)


def cake_moe_grouped_gemm_det_train_prepare_dgrad(
    g: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> CakeMoEGroupedGemmDetLaunch:
    """Prepare ``dX[sum_m, K] = grouped G @ W[e]`` (W read in place; bf16 in, bf16 out).

    ``g``: ``[sum_m, N]`` bf16; ``w``: ``[E, N, K]`` bf16; ``offs``: int32
    ``[E]`` device end offsets; ``K % 256 == 0``, ``N % 64 == 0`` (the
    512-column tile when ``K % 512 == 0`` on the architectures of
    ``wide_dgrad_archs``, its L2 evict-first twin where ``dgrad_evict_first``
    holds for ``K``, ``sum_m`` and ``E``, its 3 / 3 head / tail twin where
    ``dgrad_wide_ts33`` holds).
    """
    return _prepare_row_op("dgrad", g, w, offs, out)


def _require_flag(value: object, keyword: str, entry: str) -> None:
    """Refuse a non-bool value of the switch ``keyword`` of ``entry`` (``ValueError``)."""
    if not isinstance(value, bool):
        raise ValueError(f"{entry}: {keyword} must be a bool, got {value!r}")


def _byte_span(t: torch.Tensor) -> tuple[int, int]:
    """The device bytes ``[start, end)`` the elements of ``t`` lie in (an empty range when it has none)."""
    if t.numel() == 0:
        return 0, 0
    extent = 1 + sum(
        (size - 1) * stride for size, stride in zip(t.shape, t.stride(), strict=True)
    )
    start = int(t.data_ptr())
    return start, start + extent * t.element_size()


def _check_accumulate_out(
    out: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
    operands: dict[str, torch.Tensor],
    entry: str,
) -> None:
    """The checks of a weight gradient that adds into ``out`` (module docstring, accumulate mode), made before
    anything is allocated or bound: ``out`` given, ``out_dtype`` None or ``out``'s dtype, bfloat16 or
    float32, three dimensions with unit stride along ``K``, rows and experts that cannot overlap (row stride
    at least ``K``, expert stride at least ``N`` row strides: an expanded ``out`` is refused), on the device of
    ``operands`` (name -> tensor) and sharing no byte with any of them."""
    if out is None:
        raise ValueError(
            f"{entry}: accumulate=True adds the weight gradient into out; pass the [E, N, K] tensor "
            "to add into as out"
        )
    if out_dtype is not None and out_dtype != out.dtype:
        raise ValueError(
            f"{entry}: accumulate=True adds in the dtype of out ({out.dtype}); out_dtype must be None or "
            f"that dtype, got {out_dtype}"
        )
    if out.dtype not in _DTYPE_NAMES:
        raise ValueError(
            f"{entry}: accumulate=True needs a bfloat16 or float32 out, got {out.dtype}"
        )
    if out.dim() != 3 or out.stride(2) != 1:
        raise ValueError(
            f"{entry}: out must be an [E, N, K] tensor with unit stride along K"
        )
    groups, n, k = (int(v) for v in out.shape)
    ldc, stride_e = int(out.stride(1)), int(out.stride(0))
    if n > 1 and ldc < k:
        raise ValueError(
            f"{entry}: accumulate=True reads and writes every row of out, so its rows must not overlap: "
            f"the row stride {ldc} is below K={k}"
        )
    if groups > 1 and stride_e < n * ldc:
        raise ValueError(
            f"{entry}: accumulate=True reads and writes every expert of out, so its experts must not "
            f"overlap: the expert stride {stride_e} is below N * row stride = {n * ldc} (an expanded "
            "out has expert stride 0)"
        )
    lo, hi = _byte_span(out)
    for name, t in operands.items():
        if t.device != out.device:
            raise ValueError(f"{entry}: out must be on the device of {name}")
        start, end = _byte_span(t)
        if lo < end and start < hi:
            raise ValueError(
                f"{entry}: out shares memory with {name}; accumulate=True writes out while the "
                f"kernel reads {name}"
            )


def _accumulate_aligned(out: torch.Tensor) -> bool:
    """Whether the base, the row pitch and (more than one expert) the expert pitch of ``out`` are
    ``ACC_ALIGN``-byte aligned, as the accumulate builds need."""
    size = out.element_size()
    if int(out.data_ptr()) % ACC_ALIGN or int(out.stride(1)) * size % ACC_ALIGN:
        return False
    return out.shape[0] <= 1 or int(out.stride(0)) * size % ACC_ALIGN == 0


def cake_moe_grouped_gemm_det_train_prepare_wgrad(
    g: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    num_groups: Optional[int] = None,
    *,
    accumulate: bool = False,
) -> CakeMoEGroupedGemmDetLaunch:
    """Prepare ``dW[E, N, K] = grouped G[e].T @ X[e]`` (bf16 in; bf16 or fp32 out), deterministic.

    ``g``: ``[sum_m, N]`` bf16; ``x``: ``[sum_m, K]`` bf16; ``offs``: int32
    ``[E]`` device end offsets (``E = offs.numel()`` unless ``num_groups`` is
    given, in which case an ``m_indptr`` of ``E + 1`` entries is accepted too);
    ``N % 256 == 0`` and ``K`` a multiple of the selected k tile (256, or 512
    when it is chosen).  Empty groups produce exact zeros.  ``out_dtype``
    defaults to the dtype of ``out`` (bf16 when neither is given).

    ``accumulate=True`` (keyword-only, default ``False``) adds instead: every
    launch computes ``out[e] <- out[e] + G[e].T @ X[e]`` into the caller's
    ``out`` (required, bf16 or fp32; ``out_dtype`` None or ``out.dtype``; rows
    and experts that do not overlap, no memory shared with ``g``, ``x`` or
    ``offs``; ``ValueError`` otherwise, before anything is allocated or
    bound), so ``out`` holds AccumulateGrad's bits of the deterministic weight
    gradient in its dtype, bitwise reproducible run to run (an empty group
    adds exact zeros: its values stay, ``-0.0`` becomes ``+0.0``).
    ``plan["accumulate"]`` is ``"fused"`` (the accumulate builds of the main
    stage and, for a split tail, of the tail reduce add into ``out`` in place:
    ``stages`` is the overwrite's with ``ACC_SUFFIX`` on each) or
    ``"composed"`` (the overwrite of this plan into a contiguous scratch of
    ``out.dtype``, ``E * N * K`` elements that this preparation allocates and
    the launch owns, then ``out.add_(scratch)`` on the same stream;
    ``plan["accumulate_reason"]`` says why); both give the same bits, and
    every other plan key is the overwrite's (module docstring, accumulate
    mode).  No row (``sum_m == 0``) launches nothing and leaves ``out``
    unchanged.  Each ``launch()`` and each replay of a graph that captured it
    adds again; keep the launch alive while such a graph may replay.
    """
    _require_flag(
        accumulate, "accumulate", "cake_moe_grouped_gemm_det_train_prepare_wgrad"
    )
    if accumulate:
        _check_accumulate_out(
            out,
            out_dtype,
            {"g": g, "x": x, "offs": offs},
            "cake_moe_grouped_gemm_det_train_prepare_wgrad",
        )
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
    # The kernels address C with 32-bit offsets over its whole span (an expert stride of 0 or a row
    # pitch above K included), so bound the span, not E times the expert stride; the binding also
    # takes each stride as a 32-bit value (a one-group out's expert stride adds nothing to the span).
    out_span = 1 + sum(
        (size - 1) * stride
        for size, stride in zip(out.shape, out.stride(), strict=True)
        if size > 0
    )
    if out_span > int(c["int32_elems"]) or max(out.stride()) > int(c["int32_elems"]):
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
    if stage_served(reduce_name, TAIL_REDUCE, arch):
        reduce_program = PROGRAMS[
            stage_binding(reduce_name, TAIL_REDUCE, arch)["program"]
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
    mode, reason = "off", None
    if accumulate:
        # The route, the tile, the stage and the plan are the overwrite's (the twin's); only the
        # binding differs (module docstring, accumulate mode).
        mode, reason = wgrad_accumulate_mode(
            name,
            main_stage,
            tile_k,
            arch,
            reduce_name=reduce_name if plan["tail_splits"] > 1 else None,
            aligned=_accumulate_aligned(out),
        )
        if reason == "min_cuda":
            _warn_once(
                "accumulate_min_cuda",
                "cake_moe_grouped_gemm_det_train_prepare_wgrad(accumulate=True): the accumulate build of the "
                f"{main_stage} stage or of its tail reduce on {arch} is neither prebuilt nor buildable by "
                "FlashInfer's JIT "
                "toolkit (not importable, FLASHINFER_DISABLE_JIT set, no nvcc, or below its build.json "
                "min_cuda), so the "
                "weight gradient is written to a "
                "scratch and added with out.add_ (the composed route, same bits, "
                'plan["accumulate"] == "composed"); this warning is issued once per process',
            )
        elif reason == "alignment":
            _warn_once(
                "accumulate_alignment",
                "cake_moe_grouped_gemm_det_train_prepare_wgrad(accumulate=True): out is not "
                f"{ACC_ALIGN}-byte aligned (base, row pitch or expert pitch), so the weight "
                "gradient is written to a scratch and added with out.add_ (the composed route, same "
                'bits, plan["accumulate"] == "composed"); this warning is issued once per process',
            )
    plan["accumulate"] = mode
    if reason is not None:
        plan["accumulate_reason"] = reason
    if sum_m == 0:
        # No row: an overwrite stores exact zeros, an accumulate leaves out as it is (no launch, no add).
        return _finish("wgrad", name, plan, out, (), [], [], zero_fill=not accumulate)
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
    # The tensor the bound programs write: out (an overwrite, or the fused route adding in place), or on
    # the composed route a contiguous scratch of its dtype (allocated here, owned by the launch; from the
    # graph's pool when the preparation runs under capture).
    add_from = None
    target = out
    if mode == "composed":
        add_from = target = torch.empty(
            num_groups, n, k, dtype=out.dtype, device=device
        )
    c_view = _span_view(target)
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
        ldc=int(target.stride(1)),
        stride_e=int(target.stride(0)),
        wave_mode=plan["wave_mode"],
        grid=plan["grid"],
    )
    suffix = ACC_SUFFIX if mode == "fused" else ""
    stages = [main_stage + suffix]
    launches = [_bind(name, stages[0], main_kwargs, arch)]
    if splits > 1:
        reduce_kwargs = dict(
            partials=partials,
            C=c_view,
            offs=offs,
            num_groups=num_groups,
            N=n,
            K=k,
            ldc=int(target.stride(1)),
            stride_e=int(target.stride(0)),
            num_clusters=clusters,
            tail_splits=splits,
            raster_rows=plan["raster_rows"],
            grid=plan["tail_reduce_grid"],
        )
        stages.append(TAIL_REDUCE + suffix)
        launches.append(_bind(reduce_name, stages[-1], reduce_kwargs, arch))
    return _finish(
        "wgrad",
        name,
        plan,
        out,
        tuple(stages),
        launches,
        workspaces,
        add_from=add_from,
        bump_version=mode == "fused",
    )


# ---------------------------------------------------------------------------
# Stable-API handoff: flashinfer.grouped_mm.grouped_mm_bf16(..., backend="cake")
# ---------------------------------------------------------------------------


def grouped_det_mm_bf16_cake(
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
    return cake_moe_grouped_gemm_det_train_prepare_forward(
        a, b, m_indptr, out=out
    ).launch()
