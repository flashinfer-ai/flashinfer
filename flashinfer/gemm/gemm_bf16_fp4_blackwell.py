# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Source-built Blackwell backends for the BF16 x FP4 GEMM.

Route selection, grid arithmetic and the split-K workspace live here; the
generated launchers in ``csrc/blackwell_bf16_fp4`` validate tensor metadata,
encode the TMA descriptors by value and launch exactly one kernel.
"""

from __future__ import annotations

import collections
import functools
from typing import Any, Literal, NamedTuple, Optional, Tuple

import torch

from ..utils import get_compute_capability, get_device_sm_count
from .gemm_bf16_fp4 import _unswizzle_sf_128x4
from .gemm_bf16_fp4_cute_dsl import (
    _cute_dsl_pack_fp4_weight,
    _e4m3_to_s0e5m3,
)

BlackwellBf16Fp4Backend = Literal["blackwell-native", "blackwell-tiled"]

TILE_M = 16
TILE_N = 64
NATIVE_TMA_K = 256
CUDA_MAX_GRID_Y = 65535
WARP_K = 128
# Round-1 routes (mirror of the exporter route table).
# smk register-MMA cells: a tiled row with N % 64 == 0, K % 64 == 0, M <= 32 is planned by the split-K
# policy as (TILE_M, K256 stages per split CTA, S planes) and routed when the plan is one of the cells
# below (one CTA of 64 * k_pairs threads per N(64 * n_pairs) tile and plane, no SM clamp).  The S > 1 cells accumulate FP32
# planes in a stream-private workspace and finalize in fixed plane order from the last-arriving CTA
# (zeroed completion counters, reset by the kernel), so replays and graph captures stay bitwise.
SMK_TILE_M_MAX = 32
SMK_TILE_N = 64
SMK_STAGE_K = 256
SMK_MAX_SPLIT = 16
SMK_SPLIT_POLICY = {
    "k_le_256": 0,
    "k_le_1024": 1,
    "k_gt_1024_m16": 2,
    "k_gt_1024_m32": 2,
}
SMK_CELLS = {
    "cute_smk_s1_bf16": (16, 1, 1),
    "cute_smk_s1_m16_z4_bf16": (16, 1, 4),
    "cute_smk_s2_p2_n2_m16_z8_bf16": (16, 2, 8),
    "cute_smk_s2_m32_z6_bf16": (32, 2, 6),
}
SMK_K_PAIRS = 4
# (k_pairs, n_pairs) per cell: candidate A (16, 2, 8) runs two warp pairs over two N64 sub-tiles (N % 128 == 0).
SMK_CELL_PAIRS = {"cute_smk_s2_p2_n2_m16_z8_bf16": (2, 2)}
SMK_COMPONENT_BY_CELL = {cell: component for component, cell in SMK_CELLS.items()}


# Planes completion-counter bindings of the tiled smk split programs and their counter dtypes: ``completion_u32``
# for the ``_a1`` programs, ``completion`` otherwise.  Each routed program's ``MODULES`` record decides (its
# ``arg_plan``), so the PDL and non-PDL programs of one cell may reduce differently.  A program whose plan binds no
# ``partials`` is a cluster form -- the DSMEM-mailbox ``_x1_d1`` S=4 programs, the reduce-scatter ``_r1`` (16, 2, 8)
# PDL programs: no planes workspace and no counter, the launch binds only the GEMM operands and the record's cluster
# dims fix the cluster attribute (like ``_z4c``).
SMK_PLANES_COUNTERS = {"completion": torch.int32, "completion_u32": torch.uint32}


def _smk_planes_counter(record: dict[str, Any]) -> Optional[str]:
    """The completion-counter binding of a planes program's record, ``None`` when it binds no planes workspace."""

    names = {name for kind, name in record["arg_plan"] if kind == "buffer"}
    if "partials" not in names:
        return None
    (counter,) = names & SMK_PLANES_COUNTERS.keys()
    return counter


def _smk_cell_pairs(component: str) -> Tuple[int, int]:
    return SMK_CELL_PAIRS.get(component, (SMK_K_PAIRS, 1))


SMK_WORKSPACE_LIMIT = 16
# cuDNN-ABI smk cells: native rows with M <= 16, K % 256 == 0 are planned as (K256 stages per CTA, S planes)
# (K <= 1024: S=1 burst; K > 1024: the most K per CTA of 4 / 2 / 1 stages whose split keeps at least
# SMK_CUDNN_SPLIT_MIN_CTAS CTAs, else the finest split) and routed when the plan is one of the cells below
# (the S=4 split cell for BF16 output only).
# The S > 1 cell reduces its four planes through a cluster DSMEM mailbox (cluster dims (1, 1, 4): the plane
# CTAs of one N64 tile form a cluster and post their FP32 partials into the leader, which sums them in
# fixed order), so it needs no planes workspace or completion counters; the registry record carries the
# cluster dims and the binding launches it with the cluster attribute (bitwise replays / graph captures).
SMK_CUDNN_MAX_M = 16
SMK_CUDNN_K_STAGES = 4
SMK_CUDNN_SPLIT_MIN_CTAS = 128
SMK_CUDNN_CELLS = {
    "cudnn_smk_k4_n16_bf16": (4, 1, False),
    "cudnn_smk_k4_n16_f16": (4, 1, True),
    "cudnn_smk_k4_z4c_bf16": (4, 4, False),
}
SMK_CUDNN_COMPONENTS = frozenset(SMK_CUDNN_CELLS)
# Tile width per cuDNN smk component (exporter ``SMK_CUDNN_CELL_ARMS[component][2]``, producer
# ``SMK_CUDNN_ROUTE_TILE_N``): the S=1 burst cells run the N16 quarter-tile arm (``_n16``, wave 4 round 7; N32 ``_n32`` in round 6),
# the split cell keeps the N64 tile.  ``SMK_TILE_N`` still sizes the planes completion counters of the tiled cells.
SMK_CUDNN_HALF_TILE_N = 32
SMK_CUDNN_QUARTER_TILE_N = 16
SMK_CUDNN_CELL_TILE_N = {
    "cudnn_smk_k4_n16_bf16": SMK_CUDNN_QUARTER_TILE_N,
    "cudnn_smk_k4_n16_f16": SMK_CUDNN_QUARTER_TILE_N,
    "cudnn_smk_k4_z4c_bf16": SMK_TILE_N,
}
# Split reduction of the cuDNN split cell (exporter ``SMK_CUDNN_REDUCE``): "dsm" = the cluster mailbox, which
# takes no planes workspace; "planes" would bind the FP32 planes + completion counters like the tiled cells.
SMK_CUDNN_REDUCE = "dsm"
SMK_CUDNN_COMPONENT_BY_CELL = {
    cell: component for component, cell in SMK_CUDNN_CELLS.items()
}
# tcgen05 pair kernel for the native BF16 (768, 2112, 2048) row (cluster (2, 1, 1), 256 x 96 pair tiles,
# persistent over grid_m x grid_n tiles with one cluster per SM pair).
TCGEN05_M128_SHAPE = (768, 2112, 2048)
TCGEN05_M128_CTA_GROUP = 2
TCGEN05_M128_TILE_N = 96
TCGEN05_M128_BLOCK_M = 128
# The exact (768, 2112, 2048) tiled row with alpha and PDL runs the same pair kernel's CuTe-ABI arm (TILE_N 128 =
# whole 64-column prepared tiles; wave 4 round 6); the delivered seed kernel is no longer routed.
TCGEN05_M128_CUTE_SHAPE = (768, 2112, 2048)
TCGEN05_M128_CUTE_TILE_N = 128

GRID_COMPONENTS = frozenset(
    {
        "cudnn_tma_bf16",
        "cudnn_tma_f16",
        "cudnn_cp_async_bf16",
        "cudnn_cp_async_f16",
        "cute_bf16",
    }
)
WARP_TILE_M = {
    "cute_warp_mma_m16_k16_bf16": 16,
    "cute_warp_mma_m16_k32_bf16": 16,
    "cute_warp_mma_m16_k48_bf16": 16,
    "cute_warp_mma_m16_bf16": 16,
    "cute_warp_mma_m32_bf16": 32,
    "cute_warp_mma_m64_bf16": 64,
}


def _require_blackwell_source_arch(device: torch.device) -> None:
    major, minor = get_compute_capability(device)
    if (major, minor) not in ((10, 0), (10, 3)):
        raise NotImplementedError(
            "the source-built BF16 x FP4 backends require SM100 or SM103; "
            f"got SM{major}{minor}"
        )


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _device_arch(device_index: int) -> str:
    from ..jit.blackwell_bf16_fp4 import supported_arch

    capability = get_compute_capability(torch.device("cuda", device_index))
    arch = supported_arch(capability)
    if arch is None:
        raise NotImplementedError(
            f"cuda:{device_index} (compute capability {capability[0]}.{capability[1]}) is not a "
            "Blackwell BF16 x FP4 build target; set FLASHINFER_CUDA_ARCH_LIST to include it"
        )
    return arch


@functools.cache
def _sm_count(device_index: int) -> int:
    return int(get_device_sm_count(torch.device("cuda", device_index)))


@functools.cache
def _unit_alpha(device_index: int) -> torch.Tensor:
    """The alpha operand of programs compiled without alpha; never read by them."""

    return torch.ones(
        (1,), dtype=torch.float32, device=torch.device("cuda", device_index)
    )


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# smk split-K planes workspaces: FP32 partials [S, M, N] + zero-initialised int32 completion counters
# (one per N64 tile), private to (device, stream, M, N, S) like the split-K2 workspace above.  The kernel
# resets every counter it consumes, so one zero fill at allocation is enough for repeated launches and
# graph replays; during capture the buffers come from the graph's private pool and the zero fill is
# captured with the launch.
# key = (device index, stream handle, M, N, S planes, plane N, counter dtype name)
_SMK_WORKSPACES: "collections.OrderedDict[Tuple[int, int, int, int, int, int, str], Tuple[torch.Tensor, torch.Tensor]]" = collections.OrderedDict()


# The current stream's raw handle without constructing a ``torch.cuda.Stream`` (the one per-call ``torch.cuda.*``
# call on the launch path; the Python ``Stream`` wrapper costs ~2.5 us per call).
_cuda_current_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
_COUNTER_DTYPE_KEY = {torch.int32: "torch.int32", torch.uint32: "torch.uint32"}


def _current_stream_handle(device_index: int) -> int:
    if _cuda_current_raw_stream is not None:
        return int(_cuda_current_raw_stream(device_index))
    return int(
        torch.cuda.current_stream(torch.device("cuda", device_index)).cuda_stream
    )


def _smk_split_workspace(
    a: torch.Tensor,
    m: int,
    n: int,
    split_s: int,
    plane_n: Optional[int] = None,
    counter_dtype: Optional[torch.dtype] = None,
    device_index: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Planes ``[S, M, plane_n]`` (``plane_n`` = N for the tiled cells, ceil4(N) for the cuDNN planes cell, which is
    unused while ``SMK_CUDNN_REDUCE == "dsm"``) + counters."""

    plane_n = n if plane_n is None else plane_n
    device_index = _device_index(a.device) if device_index is None else device_index
    counter_dtype = torch.int32 if counter_dtype is None else counter_dtype
    key = (
        device_index,
        _current_stream_handle(device_index),
        m,
        n,
        split_s,
        plane_n,
        _COUNTER_DTYPE_KEY.get(counter_dtype) or str(counter_dtype),
    )
    workspace = _SMK_WORKSPACES.get(key)
    if workspace is not None:
        _SMK_WORKSPACES.move_to_end(key)
        return workspace
    partials = torch.empty((split_s, m, plane_n), dtype=torch.float32, device=a.device)
    completion = torch.zeros(
        (_ceil_div(n, SMK_TILE_N),), dtype=counter_dtype, device=a.device
    )
    workspace = (partials, completion)
    if torch.cuda.is_current_stream_capturing():
        return workspace
    _SMK_WORKSPACES[key] = workspace
    while len(_SMK_WORKSPACES) > SMK_WORKSPACE_LIMIT:
        (
            (old_device, old_stream, _m, _n, _s, _p, _dt),
            (old_partials, old_completion),
        ) = _SMK_WORKSPACES.popitem(last=False)
        old_external = torch.cuda.ExternalStream(
            old_stream, device=torch.device("cuda", old_device)
        )
        old_partials.record_stream(old_external)
        old_completion.record_stream(old_external)
    return workspace


def _smk_policy_stages(m: int, k: int) -> int:
    """K256 stages per split CTA of the smk split-K policy (0 = the one-shot S=1 burst cell)."""

    if k <= SMK_STAGE_K:
        return SMK_SPLIT_POLICY["k_le_256"]
    if k <= 4 * SMK_STAGE_K:
        return SMK_SPLIT_POLICY["k_le_1024"]
    return SMK_SPLIT_POLICY["k_gt_1024_m16" if m <= 16 else "k_gt_1024_m32"]


def _smk_plan_cell(m: int, n: int, k: int) -> Optional[Tuple[int, int, int]]:
    """(TILE_M, K256 stages per CTA, S planes) the smk policy plans for a tiled row, or None."""

    if m > SMK_TILE_M_MAX or k % 64 or n % SMK_TILE_N:
        return None
    tile_m = 16 if m <= 16 else SMK_TILE_M_MAX
    stages = _smk_policy_stages(m, k)
    if stages == 0:
        return (tile_m, _ceil_div(k, SMK_STAGE_K), 1)
    split_s = _ceil_div(k, stages * SMK_STAGE_K)
    if split_s > SMK_MAX_SPLIT:
        return None
    return (tile_m, stages, split_s)


def _smk_component(m: int, n: int, k: int) -> Optional[str]:
    component = SMK_COMPONENT_BY_CELL.get(_smk_plan_cell(m, n, k))
    if component is None:
        return None
    _k_pairs, n_pairs = _smk_cell_pairs(component)
    if n % (SMK_TILE_N * n_pairs):
        return None
    return component


def _smk_cudnn_policy_stages(n: int, k: int) -> int:
    """K256 stages per split CTA of a cuDNN smk row (0 = the S=1 burst cell)."""

    if k <= SMK_CUDNN_K_STAGES * SMK_STAGE_K:
        return 0
    tiles = _ceil_div(n, SMK_TILE_N)
    for stages in (4, 2, 1):
        per_cta = stages * SMK_STAGE_K
        if k % per_cta:
            continue
        split_s = k // per_cta
        if split_s <= SMK_MAX_SPLIT and tiles * split_s >= SMK_CUDNN_SPLIT_MIN_CTAS:
            return stages
    return 1


def _smk_cudnn_plan_cell(m: int, n: int, k: int) -> Optional[Tuple[int, int]]:
    """(K256 stages per CTA, S planes) the cuDNN smk policy plans for a native row, or None."""

    if m > SMK_CUDNN_MAX_M or k <= 0 or k % SMK_STAGE_K:
        return None
    stages = _smk_cudnn_policy_stages(n, k)
    if stages == 0:
        return (k // SMK_STAGE_K, 1)
    per_cta = stages * SMK_STAGE_K
    if k % per_cta or k // per_cta > SMK_MAX_SPLIT:
        return None
    return (stages, k // per_cta)


def _smk_cudnn_component(m: int, n: int, k: int, output_bf16: bool) -> Optional[str]:
    cell = _smk_cudnn_plan_cell(m, n, k)
    if cell is None:
        return None
    return SMK_CUDNN_COMPONENT_BY_CELL.get((cell[0], cell[1], not output_bf16))


def _cell_key(
    component: str, has_alpha: bool, enable_pdl: bool, flat_grid: bool
) -> str:
    key = f"{component}:alpha{int(has_alpha)}:pdl{int(enable_pdl)}"
    if component in GRID_COMPONENTS:
        key += ":flat" if flat_grid else ":grid2d"
    return key


class _StageSpec(NamedTuple):
    """Tensor-independent part of one launch stage, resolved once per problem by ``_route_plan``."""

    cell: str
    module: str
    grid: Tuple[int, int, int]
    cluster_dims: Tuple[int, int, int]
    # (S planes, plane N, counter binding name, counter dtype) for the split-K planes cells, else None
    workspace: Optional[Tuple[int, int, str, torch.dtype]]
    # (grid_m, grid_n) bindings of the tcgen05 pair kernels, else None
    pair_tiles: Optional[Tuple[int, int]]


class _Stage(NamedTuple):
    cell: str
    module: str
    grid: Tuple[int, int, int]
    cluster_dims: Tuple[int, int, int]
    bindings: dict[str, Any]


def _cluster_aligned_grid(
    grid: Tuple[int, int, int], cluster_dims: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    """Round a CTA grid up to whole thread-block clusters (extra CTAs are zero-work tail CTAs)."""

    aligned = [
        _ceil_div(ctas, cluster) * cluster
        for ctas, cluster in zip(grid, cluster_dims, strict=True)
    ]
    return (aligned[0], aligned[1], aligned[2])


def _check_cluster_grid(
    cell: str, grid: Tuple[int, int, int], cluster_dims: Tuple[int, int, int]
) -> None:
    """The launch grid of a cluster program must be a whole number of clusters per axis.

    The generated binding repeats this check before ``cudaLaunchKernelExC``; raising here names the
    route instead of the kernel symbol.
    """

    if any(
        ctas % cluster != 0 for ctas, cluster in zip(grid, cluster_dims, strict=True)
    ):
        raise ValueError(
            f"route {cell} launches grid {tuple(grid)} which is not divisible by its "
            f"cluster dims {tuple(cluster_dims)}"
        )


class _Plan(NamedTuple):
    stages: Tuple[_Stage, ...]
    arch: str


def _tcgen05_m128_tiles(m: int, n: int) -> Tuple[int, int]:
    """Pair-tile grid ``(grid_m, grid_n)`` of the tcgen05 pair kernel (256 activation x 128 weight rows)."""

    return (
        _ceil_div(m, TCGEN05_M128_BLOCK_M * TCGEN05_M128_CTA_GROUP),
        _ceil_div(n, TCGEN05_M128_TILE_N),
    )


def _tcgen05_m128_grid(m: int, n: int, sm_count: int) -> Tuple[int, int, int]:
    """Persistent cluster grid: two CTAs per cluster, at most one cluster per SM pair, never more than tiles."""

    grid_m, grid_n = _tcgen05_m128_tiles(m, n)
    clusters = max(1, min(grid_m * grid_n, sm_count // TCGEN05_M128_CTA_GROUP))
    return (TCGEN05_M128_CTA_GROUP * clusters, 1, 1)


def _tcgen05_m128_cute_tiles(m: int, n: int) -> Tuple[int, int]:
    """Pair-tile grid of the pair kernel's CuTe-ABI arm (256 activation x 128 weight rows)."""

    return (
        _ceil_div(m, TCGEN05_M128_BLOCK_M * TCGEN05_M128_CTA_GROUP),
        _ceil_div(n, TCGEN05_M128_CUTE_TILE_N),
    )


def _tcgen05_m128_cute_grid(m: int, n: int, sm_count: int) -> Tuple[int, int, int]:
    """Persistent cluster grid of the CuTe-ABI arm: two CTAs per cluster, at most one cluster per SM pair."""

    grid_m, grid_n = _tcgen05_m128_cute_tiles(m, n)
    clusters = max(1, min(grid_m * grid_n, sm_count // TCGEN05_M128_CTA_GROUP))
    return (TCGEN05_M128_CTA_GROUP * clusters, 1, 1)


def _persistent_grid(
    component: str, m: int, n: int, sm_count: int
) -> Tuple[int, int, int]:
    """Launch grid of a persistent or seed program."""

    grid_n = _ceil_div(n, TILE_N)
    if component in SMK_CELLS:
        return (
            _ceil_div(n, TILE_N * _smk_cell_pairs(component)[1]),
            1,
            SMK_CELLS[component][2],
        )
    if component in SMK_CUDNN_COMPONENTS:
        # One 256-thread CTA per (ragged) N tile of the cell's width (N16 burst cells, N64 split cell) and K plane.
        return (
            _ceil_div(n, SMK_CUDNN_CELL_TILE_N[component]),
            1,
            SMK_CUDNN_CELLS[component][1],
        )
    if component == "cute_tcgen05_m128_cg2_n128_bf16":
        return _tcgen05_m128_cute_grid(m, n, sm_count)
    return (min(_ceil_div(m, WARP_TILE_M[component]) * grid_n, sm_count), 1, 1)


def _select_route(
    m: int,
    n: int,
    k: int,
    *,
    tiled: bool,
    output_bf16: bool,
    has_alpha: bool,
    enable_pdl: bool,
    sm_count: int,
) -> Tuple[Tuple[str, bool, bool, bool, Tuple[int, int, int]], ...]:
    """Stages ``(component, has_alpha, enable_pdl, flat_grid, grid)`` for one problem."""

    cp_async = not tiled and k % NATIVE_TMA_K != 0
    if not tiled and output_bf16 and not cp_async and (m, n, k) == TCGEN05_M128_SHAPE:
        grid = _tcgen05_m128_grid(m, n, sm_count)
        return (("cudnn_tcgen05_m128_cg2_bf16", has_alpha, enable_pdl, False, grid),)
    if not tiled:
        component = _smk_cudnn_component(m, n, k, output_bf16)
        if component is not None:
            return (
                (
                    component,
                    has_alpha,
                    enable_pdl,
                    False,
                    _persistent_grid(component, m, n, sm_count),
                ),
            )
    if tiled:
        component = _smk_component(m, n, k)
        if component is not None:
            pass
        elif (
            output_bf16
            and has_alpha
            and enable_pdl
            and (m, n, k) == TCGEN05_M128_CUTE_SHAPE
        ):
            grid = _tcgen05_m128_cute_grid(m, n, sm_count)
            return (
                ("cute_tcgen05_m128_cg2_n128_bf16", has_alpha, enable_pdl, False, grid),
            )
        elif k == 16:
            component = "cute_warp_mma_m16_k16_bf16"
        elif k == 32:
            component = "cute_warp_mma_m16_k32_bf16"
        elif k == 48:
            component = "cute_warp_mma_m16_k48_bf16"
        elif m <= 16 and k >= WARP_K and k % 64 == 0:
            component = "cute_warp_mma_m16_bf16"
        elif m <= 32 and k >= WARP_K and k % 64 == 0:
            component = "cute_warp_mma_m32_bf16"
        elif m >= 33 and k % WARP_K == 0:
            component = "cute_warp_mma_m64_bf16"
        if component is not None:
            return (
                (
                    component,
                    has_alpha,
                    enable_pdl,
                    False,
                    _persistent_grid(component, m, n, sm_count),
                ),
            )
    grid_m = _ceil_div(m, TILE_M)
    grid_n = _ceil_div(n, TILE_N)
    flat_grid = grid_m > CUDA_MAX_GRID_Y
    if tiled:
        component = "cute_bf16"
    elif cp_async:
        component = "cudnn_cp_async_bf16" if output_bf16 else "cudnn_cp_async_f16"
    else:
        component = "cudnn_tma_bf16" if output_bf16 else "cudnn_tma_f16"
    grid = (grid_m * grid_n, 1, 1) if flat_grid else (grid_n, grid_m, 1)
    return ((component, has_alpha, enable_pdl, flat_grid, grid),)


@functools.lru_cache(maxsize=4096)
def _route_plan(
    m: int,
    n: int,
    k: int,
    tiled: bool,
    output_bf16: bool,
    has_alpha: bool,
    enable_pdl: bool,
    sm_count: int,
) -> Tuple[_StageSpec, ...]:
    """Resolve the route of one problem once: cells, modules, grids, the cluster check and the workspace spec.

    Everything here depends only on the problem key, so repeated calls with the same shapes skip the route
    selection; ``_plan`` only binds the tensors (and the stream-private workspace) per call.
    """

    from ..jit.blackwell_bf16_fp4 import MODULES, ROUTES, cluster_dims

    specs = []
    for component, stage_alpha, pdl, flat_grid, grid in _select_route(
        m,
        n,
        k,
        tiled=tiled,
        output_bf16=output_bf16,
        has_alpha=has_alpha,
        enable_pdl=enable_pdl,
        sm_count=sm_count,
    ):
        cell = _cell_key(component, stage_alpha, pdl, flat_grid)
        module = ROUTES[cell]
        cluster = cluster_dims(module)
        _check_cluster_grid(cell, grid, cluster)
        workspace = None
        pair_tiles = None
        if component in (
            "cudnn_tcgen05_m128_cg2_bf16",
            "cute_tcgen05_m128_cg2_n128_bf16",
        ):
            # Same ABI for both pair-kernel arms (no workspace); the tiled row hands the u8 scales through as prepared.
            tiles = (
                _tcgen05_m128_tiles
                if component == "cudnn_tcgen05_m128_cg2_bf16"
                else _tcgen05_m128_cute_tiles
            )
            pair_tiles = tiles(m, n)
        elif (
            component in SMK_CELLS
            and SMK_CELLS[component][2] > 1
            and (counter_name := _smk_planes_counter(MODULES[module])) is not None
        ):
            # The planes program binds the FP32 planes and the counter its record names.
            workspace = (
                SMK_CELLS[component][2],
                n,
                counter_name,
                SMK_PLANES_COUNTERS[counter_name],
            )
        elif (
            component in SMK_CUDNN_COMPONENTS
            and SMK_CUDNN_CELLS[component][1] > 1
            and SMK_CUDNN_REDUCE == "planes"
        ):
            workspace = (
                SMK_CUDNN_CELLS[component][1],
                _ceil_div(n, 4) * 4,
                "completion",
                torch.int32,
            )
        specs.append(_StageSpec(cell, module, grid, cluster, workspace, pair_tiles))
    return tuple(specs)


def _plan(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    tiled: bool,
    enable_pdl: bool,
) -> _Plan:
    """Bind validated operands to the (cached) route of their problem."""

    index = _device_index(a.device)
    arch = _device_arch(index)
    m, k = int(a.shape[0]), int(a.shape[1])
    n = int(out.shape[1])
    specs = _route_plan(
        m,
        n,
        k,
        tiled,
        out.dtype == torch.bfloat16,
        alpha is not None,
        enable_pdl,
        _sm_count(index),
    )
    scale = b_descale if tiled else b_descale.view(torch.uint8)
    alpha_operand = alpha if alpha is not None else _unit_alpha(index)
    stages = []
    for spec in specs:
        bindings: dict[str, Any] = dict(
            A=a, B=b, B_descale=scale, alpha=alpha_operand, C=out, M=m, N=n, K=k
        )
        if spec.pair_tiles is not None:
            bindings["grid_m"], bindings["grid_n"] = spec.pair_tiles
        elif spec.workspace is not None:
            split_s, plane_n, counter_name, counter_dtype = spec.workspace
            partials, completion = _smk_split_workspace(
                a,
                m,
                n,
                split_s,
                plane_n=plane_n,
                counter_dtype=counter_dtype,
                device_index=index,
            )
            bindings["partials"] = partials
            bindings[counter_name] = completion
        stages.append(
            _Stage(spec.cell, spec.module, spec.grid, spec.cluster_dims, bindings)
        )
    return _Plan(tuple(stages), arch)


@functools.cache
def _launch_entry(module: str, arch: str):
    """The loaded FFI entry of ``module`` for ``arch`` plus its argument plan (grid axis index or binding name)."""

    from ..jit.blackwell_bf16_fp4 import MODULES, load_module

    record = MODULES[module]
    axes = {"grid_x": 0, "grid_y": 1, "grid_z": 2}
    arg_plan = tuple(
        (axes[name], None) if kind == "grid" else (None, name)
        for kind, name in record["arg_plan"]
    )
    return getattr(load_module(module, arch), record["ffi_entry"]), arg_plan


def _launch(plan: _Plan) -> None:
    for stage in plan.stages:
        entry, arg_plan = _launch_entry(stage.module, plan.arch)
        grid = stage.grid
        bindings = stage.bindings
        entry(
            *[
                grid[axis] if axis is not None else bindings[name]
                for axis, name in arg_plan
            ]
        )


def _prepare_blackwell_bf16_fp4(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    block_size: int,
    backend: BlackwellBf16Fp4Backend,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Prepare one of the two explicit source-kernel tensor layouts (one-time weight transform)."""

    if block_size != 16:
        raise ValueError(
            f"source-built BF16 x FP4 preparation requires block_size=16; got {block_size}"
        )
    _require_blackwell_source_arch(b.device)
    if b.device != b_descale.device:
        raise ValueError(
            "b and b_descale must be on the same device; got "
            f"{b.device} and {b_descale.device}"
        )
    if alpha is not None and alpha.device != b.device:
        raise ValueError(
            f"alpha must be on the same device as b ({b.device}); got {alpha.device}"
        )

    n = int(b.shape[0])
    k = int(b.shape[1]) * 2
    k_sf = k // block_size
    linear_sf = _unswizzle_sf_128x4(b_descale, n, k_sf).contiguous()

    if backend == "blackwell-native":
        return b.contiguous(), linear_sf.view(torch.float8_e4m3fn), alpha

    if backend == "blackwell-tiled":
        if n % 64 != 0:
            raise ValueError(
                f"blackwell-tiled requires N to be a multiple of 64; got N={n}"
            )
        b_kn = b.t().contiguous()
        b_packed = _cute_dsl_pack_fp4_weight(b_kn)
        scale_s0e5m3 = _e4m3_to_s0e5m3(linear_sf.t().contiguous())
        return b_packed, scale_s0e5m3, alpha

    raise ValueError(f"unknown source-built BF16 x FP4 backend {backend!r}")


def _validate_blackwell_bf16_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
    block_size: int,
    backend: BlackwellBf16Fp4Backend,
) -> Tuple[bool, int, int]:
    """Validate the prepared ABI of one call and return ``(tiled, M, N)``.

    The ``mm_bf16_fp4`` backend checker and :func:`_compute_blackwell_bf16_fp4`
    both run this, so ``skip_check=True`` skips a duplicate run, not a check.
    """

    if a.dim() != 2 or a.dtype != torch.bfloat16:
        raise ValueError(
            "source-built BF16 x FP4 GEMM expects a as bfloat16 [M, K]; "
            f"got dtype={a.dtype}, shape={tuple(a.shape)}"
        )
    k = int(a.shape[1])
    if block_size != 16:
        raise ValueError(
            f"source-built BF16 x FP4 GEMM requires block_size=16 and K%16=0; "
            f"got block_size={block_size}, K={k}"
        )
    if k % block_size != 0:
        raise ValueError(f"K={k} must be a multiple of block_size={block_size}")
    if backend == "blackwell-native":
        tiled = False
        if b.dim() != 2 or b.dtype != torch.uint8:
            raise ValueError(
                "blackwell-native expects the uint8 [N, K/2] weight returned by "
                "prepare_bf16_fp4_weights(..., backend='blackwell-native')"
            )
        n = int(b.shape[0])
        encoded_k = int(b.shape[1]) * 2
        if k != encoded_k:
            raise ValueError(f"a.shape[1]={k} but prepared b encodes K={encoded_k}")
        if b_descale.dtype != torch.float8_e4m3fn or tuple(b_descale.shape) != (
            n,
            k // block_size,
        ):
            raise ValueError(
                "blackwell-native expects linear float8_e4m3fn scales with shape "
                f"{(n, k // block_size)}; got dtype={b_descale.dtype}, "
                f"shape={tuple(b_descale.shape)}"
            )
        if out_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"blackwell-native supports bfloat16 or float16 output; got {out_dtype}"
            )
    elif backend == "blackwell-tiled":
        tiled = True
        if b.dim() != 2 or b.dtype != torch.int32 or int(b.shape[1]) % 2 != 0:
            raise ValueError(
                "blackwell-tiled expects the int32 [K/16, N*2] weight returned by "
                "prepare_bf16_fp4_weights(..., backend='blackwell-tiled')"
            )
        k_tiles = int(b.shape[0])
        n = int(b.shape[1]) // 2
        encoded_k = k_tiles * block_size
        if k != encoded_k:
            raise ValueError(f"a.shape[1]={k} but prepared b encodes K={encoded_k}")
        if n % 64 != 0:
            raise ValueError(
                f"blackwell-tiled requires N to be a multiple of 64; got N={n}"
            )
        if b_descale.dtype != torch.uint8 or tuple(b_descale.shape) != (k_tiles, n):
            raise ValueError(
                "blackwell-tiled expects S0E5M3 uint8 scales with shape "
                f"{(k_tiles, n)}; got dtype={b_descale.dtype}, "
                f"shape={tuple(b_descale.shape)}"
            )
        if out_dtype != torch.bfloat16:
            raise ValueError(
                f"blackwell-tiled requires bfloat16 output; got {out_dtype}"
            )
    else:
        raise ValueError(f"unknown source-built BF16 x FP4 backend {backend!r}")
    if not b.is_contiguous() or not b_descale.is_contiguous():
        raise ValueError(f"{backend} requires contiguous b and b_descale")
    if b.device != a.device or b_descale.device != a.device:
        raise ValueError("a, b, and b_descale must be on the same device")
    m = int(a.shape[0])
    if out is not None and (
        tuple(out.shape) != (m, n)
        or out.dtype != out_dtype
        or out.device != a.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be contiguous, on the same device, and have shape/dtype "
            f"{(m, n)}/{out_dtype}"
        )
    if not a.is_contiguous():
        raise ValueError("a, b, and b_descale must be contiguous")
    if m <= 0 or n <= 0:
        raise ValueError(f"M and N must be positive; got M={m}, N={n}")
    if alpha is not None:
        if alpha.device != a.device:
            raise ValueError(
                f"alpha must be on the same device as the GEMM inputs ({a.device}); "
                f"got {alpha.device}"
            )
        if (
            alpha.dtype != torch.float32
            or tuple(alpha.shape) != (1,)
            or not alpha.is_contiguous()
        ):
            raise ValueError(
                "alpha must be a contiguous float32 tensor with shape (1,); "
                f"got dtype={alpha.dtype}, shape={tuple(alpha.shape)}, "
                f"contiguous={alpha.is_contiguous()}"
            )
    return tiled, m, n


def _compute_blackwell_bf16_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
    block_size: int,
    enable_pdl: bool,
    backend: BlackwellBf16Fp4Backend,
) -> torch.Tensor:
    """Validate the prepared ABI, select the route and launch the generated programs."""

    _require_blackwell_source_arch(a.device)
    tiled, m, n = _validate_blackwell_bf16_fp4(
        a, b, b_descale, alpha, out_dtype, out, block_size, backend
    )
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    _launch(
        _plan(a, b, b_descale, alpha, out, tiled=tiled, enable_pdl=bool(enable_pdl))
    )
    return out


__all__: list[str] = []
