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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from .cake_jit import (
    HOST_PLAN_CONSTANTS,
    MODULES,
    load_module,
    program_registered,
    select_module,
)

SUPPORTED_COMPUTE_CAPABILITIES = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
DESCRIPTOR_ALIGN = 128  # bytes; CUtensorMap slots of the caller-owned workspaces
_DTYPE_NAMES = {torch.bfloat16: "bfloat16", torch.float32: "float32"}


# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------


def host_plan_constants() -> dict[str, Any]:
    """The production planner constants delivered with the generated programs."""
    if not HOST_PLAN_CONSTANTS:
        raise NotImplementedError(
            "The generated ragged grouped GEMM programs are not registered in this "
            "checkout yet (empty HOST_PLAN_CONSTANTS in "
            "flashinfer.experimental.cake_moe_grouped_gemm.cake_jit)"
        )
    return HOST_PLAN_CONSTANTS


def arch_for(device: torch.device) -> Optional[str]:
    """Exact architecture name of ``device`` or ``None`` when unsupported."""
    return SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))


def sm_count(device: torch.device) -> int:
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def generated_program_available(
    device: torch.device, op: str = "fwd", out_dtype: Optional[torch.dtype] = None
) -> bool:
    """True when this checkout registers the program of ``op`` for ``device``."""
    arch = arch_for(device)
    if arch is None or not HOST_PLAN_CONSTANTS:
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
    if str(tile_k) in cycles_per_step:
        return int(cycles_per_step[str(tile_k)])
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
    """The production k-tile rule for a weight gradient of shape ``(sum_m, num_groups, k)``."""
    c = constants or host_plan_constants()
    return wgrad_tile_k(
        k,
        arch,
        sum_m / max(num_groups, 1),
        k512_archs=tuple(c["k512_archs"]),
        k512_min_rows_per_group=int(c["k512_min_rows_per_group"]),
    )


def plan_fwd_dgrad(
    op: str,
    *,
    sum_m: int,
    num_groups: int,
    n: int,
    k: int,
    sm_count: int,
    constants: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Launch plan of ``fwd`` / ``dgrad``: one persistent launch."""
    c = constants or host_plan_constants()
    cols = n if op == "fwd" else k
    tiles = max_cluster_tiles_upper_bound(
        sum_m,
        num_groups,
        cols,
        cluster_m=int(c["cluster_m"]),
        block_n=int(c["block_n"]),
    )
    grid_x = persistent_grid(sm_count, tiles, cta_group=int(c["cta_group"]))
    return dict(grid=[grid_x, 1, 1], launches=1, tile_k=None)


def plan_wgrad(
    *,
    sum_m: int,
    num_groups: int,
    n: int,
    k: int,
    tile_k: int,
    sm_count: int,
    reduce_threads: int,
    constants: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Launch plan of ``wgrad`` for the selected ``tile_k``.

    ``reduce_threads`` is the thread count of the tail-reduce program (its
    launch block); it sizes the reduce grid.
    """
    c = constants or host_plan_constants()
    block_n, block_k = int(c["block_n"]), int(c["block_k"])
    cluster_m, cta_group = int(c["cluster_m"]), int(c["cta_group"])
    tiles = num_groups * (n // block_n) * (k // tile_k)
    grid_x = persistent_grid(sm_count, tiles, cta_group=cta_group)
    clusters = grid_x // cta_group
    num_tail = 0
    if int(c["wgrad_tail_split"]):
        num_tail = wgrad_tail_tiles(num_groups, n, k, tile_k, clusters, block_n=block_n)
    splits = 1
    if num_tail:
        tail = c["tail_plan"]
        splits = wgrad_tail_plan(
            num_tail,
            clusters,
            sum_m,
            num_groups,
            tile_k,
            block_k=block_k,
            cluster_m=cluster_m,
            max_splits=int(tail["max_splits"]),
            sm_clock_ghz=float(tail["sm_clock_ghz"]),
            dram_tbps=float(tail["dram_tbps"]),
            launch_us=float(tail["launch_us"]),
            cycles_per_step=tail["cycles_per_step"],
        )
    raster_rows = wgrad_raster_rows(
        n // block_n, k // tile_k, clusters, tile_k, block_n=block_n, block_k=block_k
    )
    reduce_grid = None
    if splits > 1:
        reduce_grid = [
            wgrad_reduce_grid(
                num_tail,
                tile_k,
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
        num_tail=num_tail,
        tail_splits=splits,
        raster_rows=raster_rows,
        tail_reduce_grid=reduce_grid,
        launches=1 + int(splits > 1),
    )


# ---------------------------------------------------------------------------
# Prepared launches
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Launch:
    stage: str
    module: str
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)
    # Descriptor preparation entry of a pointer-ABI stage (same arguments); None otherwise.
    prepare: Optional[Callable[..., Any]] = field(default=None, repr=False)

    def __call__(self) -> None:
        self.entry(*self.arguments)


@dataclass(frozen=True)
class GroupedGemmLaunch:
    """One prepared ragged grouped GEMM.

    ``launch()`` runs the bound program on the current torch stream into the
    caller-owned ``out`` with no CUDA allocation and no host synchronisation
    and returns ``out``.  The kernels read ``offs`` on device at every launch,
    so the same prepared launch (or a CUDA Graph capturing it) stays valid when
    the caller writes new group boundaries into ``offs`` or new values into the
    operands.  Prepare a new launch when a shape, dtype or tensor binding
    changes.  ``plan`` records the host decisions (grid, k tile, tail split,
    raster, launch count); ``stages`` is the stage set of the bound program.
    """

    op: str
    record_name: str
    stages: tuple[str, ...]
    plan: dict[str, Any]
    out: torch.Tensor
    launches: tuple[_Launch, ...] = field(repr=False)
    zero_fill: bool = False
    # Caller-owned buffers of this launch: descriptor workspaces, the per-CTA
    # descriptor slots and the fp32 tail partials.  Kept alive with the launch.
    workspaces: tuple[torch.Tensor, ...] = field(default=(), repr=False)

    def launch(self) -> torch.Tensor:
        if self.zero_fill:
            self.out.zero_()
        with tvm_ffi.use_torch_stream():
            for item in self.launches:
                item.entry(*item.arguments)
        return self.out

    __call__ = launch

    @property
    def stage_launchers(self) -> dict[str, Callable[[], None]]:
        """One zero-argument launch per stage actually launched (for tests and timing)."""
        result: dict[str, Callable[[], None]] = {}
        for item in self.launches:

            def run(item: _Launch = item) -> None:
                with tvm_ffi.use_torch_stream():
                    item.entry(*item.arguments)

            result[item.stage] = run
        return result

    @property
    def stage_modules(self) -> dict[str, str]:
        return {item.stage: item.module for item in self.launches}


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


def _check_offs(offs: torch.Tensor, num_groups: int) -> None:
    if offs.dtype != torch.int32 or offs.ndim != 1 or offs.numel() != num_groups:
        raise ValueError(
            f"offs must be a one-dimensional int32 tensor of {num_groups} cumulative "
            f"end offsets (got {offs.dtype}, shape {tuple(offs.shape)})"
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
    name: str, stage: str, kwargs: dict[str, Any], device: torch.device
) -> tuple[_Launch, list]:
    """Order ``kwargs`` by the generated argument plan of ``stage`` and load its entries.

    Allocates the stage's caller-owned TMA descriptor workspace when the record
    declares one (pointer ABI) and returns it with the launch so the caller
    keeps it alive.
    """
    record = MODULES[name]
    physical = record[stage]
    retained = []
    workspace_bytes = int(physical.get("tma_workspace_bytes", 0))
    if workspace_bytes:
        kwargs = dict(kwargs)
        kwargs["tma_descriptor_workspace"] = _aligned_bytes(
            workspace_bytes, device, f"{name}/{stage} descriptor workspace"
        )
        retained.append(kwargs["tma_descriptor_workspace"])
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), kwargs["grid"], strict=True))
    arguments = []
    for kind, arg in physical["arg_plan"]:
        if kind == "grid":
            arguments.append(int(grid[arg]))
        elif arg in kwargs:
            arguments.append(kwargs[arg])
        else:
            raise KeyError(
                f"generated program {name!r} stage {stage!r} expects argument {arg!r} "
                f"({kind}); the host binding provides {sorted(kwargs)}"
            )
    module = load_module(name, stage)
    prepare_entry = physical.get("tma_prepare_entry")
    launch = _Launch(
        stage=stage,
        module=physical["module"],
        entry=getattr(module, physical["ffi_entry"]),
        arguments=tuple(arguments),
        prepare=getattr(module, prepare_entry) if prepare_entry else None,
    )
    return launch, retained


def _prepare_descriptors(launches: list[_Launch]) -> None:
    """Encode the TMA descriptors of every pointer-ABI stage once (outside graph capture)."""
    with tvm_ffi.use_torch_stream():
        for item in launches:
            if item.prepare is not None:
                item.prepare(*item.arguments)


def _finish(
    op: str,
    name: str,
    plan: dict[str, Any],
    out: torch.Tensor,
    launches: list[_Launch],
    workspaces: list,
    *,
    zero_fill: bool = False,
) -> GroupedGemmLaunch:
    _prepare_descriptors(launches)
    return GroupedGemmLaunch(
        op=op,
        record_name=name,
        stages=tuple(MODULES[name]["stages"]),
        plan=plan,
        out=out,
        launches=tuple(launches),
        zero_fill=zero_fill,
        workspaces=tuple(workspaces),
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
    plan = plan_fwd_dgrad(
        op,
        sum_m=sum_m,
        num_groups=num_groups,
        n=n,
        k=k,
        sm_count=sm_count(device),
        constants=c,
    )
    if sum_m == 0:
        return _finish(op, name, plan, out, [], [])
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
        grid=plan["grid"],
    )
    launch, retained = _bind(name, "main", kwargs, device)
    return _finish(op, name, plan, out, [launch], retained)


def prepare_grouped_gemm_fwd(
    x: torch.Tensor,
    w: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> GroupedGemmLaunch:
    """Prepare ``Y[sum_m, N] = grouped X @ W[e].T`` (bf16 in, bf16 out).

    ``x``: ``[sum_m, K]`` bf16; ``w``: ``[E, N, K]`` bf16; ``offs``: int32
    ``[E]`` device end offsets; ``N % 256 == 0``, ``K % 64 == 0``.  ``out``
    may be row-padded (any row stride, unit column stride).
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
    ``[E]`` device end offsets; ``K % 256 == 0``, ``N % 64 == 0``.
    """
    return _prepare_row_op("dgrad", g, w, offs, out)


def prepare_grouped_gemm_wgrad(
    g: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> GroupedGemmLaunch:
    """Prepare ``dW[E, N, K] = grouped G[e].T @ X[e]`` (bf16 in; bf16 or fp32 out), deterministic.

    ``g``: ``[sum_m, N]`` bf16; ``x``: ``[sum_m, K]`` bf16; ``offs``: int32
    ``[E]`` device end offsets (``E = offs.numel()``); ``N % 256 == 0`` and
    ``K`` a multiple of the selected k tile (256, or 512 when it is chosen).
    Empty groups produce exact zeros.  ``out_dtype`` defaults to the dtype of
    ``out`` (bf16 when neither is given).
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
    num_groups = int(offs.numel())
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
    record = MODULES[name]
    reduce_threads = (
        int(record["tail_reduce"]["launch"]["block"][0])
        if "tail_reduce" in record
        else 0
    )
    plan = plan_wgrad(
        sum_m=sum_m,
        num_groups=num_groups,
        n=n,
        k=k,
        tile_k=tile_k,
        sm_count=sm_count(device),
        reduce_threads=reduce_threads,
        constants=c,
    )
    if sum_m == 0:
        return _finish("wgrad", name, plan, out, [], [], zero_fill=True)
    grid_x, clusters, splits = plan["grid"][0], plan["clusters"], plan["tail_splits"]
    slots = _aligned_bytes(
        grid_x * int(c["tmap_slots_per_cta"]) * int(c["tmap_slot_bytes"]),
        device,
        "per-CTA descriptor slots",
    )
    units = plan["num_tail"] * splits if splits > 1 else 0
    partials = torch.empty(
        max(units, 1) * int(c["cluster_m"]) * tile_k, dtype=torch.float32, device=device
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
        grid=plan["grid"],
    )
    launch, retained = _bind(name, "main", main_kwargs, device)
    launches = [launch]
    workspaces.extend(retained)
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
        reduce_launch, retained = _bind(name, "tail_reduce", reduce_kwargs, device)
        launches.append(reduce_launch)
        workspaces.extend(retained)
    return _finish("wgrad", name, plan, out, launches, workspaces)
