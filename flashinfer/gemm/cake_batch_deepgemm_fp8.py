# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Prepared masked batch FP8 GEMM launches of the generated Cake programs (SM100a, SM103a, SM107a).

The same mathematical contract as :func:`flashinfer.gemm.batch_deepgemm_fp8_nt_groupwise`:
``out[g, :masked_m[g], :] = (a[g] . b[g]^T)`` with FP8 E4M3 operands, per-row
128-wide K-block A scales, 128x128 B block scales, FP32 accumulation with the
block scales applied in DeepGEMM's K order, BF16 output; the contents of rows
at or beyond ``masked_m[g]`` are unspecified (whole tiles may be stored), as on
the DeepGEMM path.

The host side selects the route with the Cake dispatcher's ordered first-match
route table of :mod:`flashinfer.jit.gemm.cake_batch_deepgemm_fp8`, read per
``(architecture, SM count)``, binds the positional argument
plan of each registered program and launches the one or two kernels of the route
back to back on PyTorch's current stream.  Tensor maps are encoded by the binding
and passed to the kernel by value, so a prepared launch owns no descriptor
storage; the only device memory a route may need is the packed-scale workspace
of the swap-AB routes, allocated at preparation (or supplied by the caller) and
never inside ``launch()``.  Every launch, including the first, may be captured
into a CUDA graph.

Selection reads host scalars and tensor metadata only; it never reads a CUDA
tensor's contents.

On a dual-die SM107a device (two locality domains) whose weights ``b`` were
allocated by :func:`localize_weights`, a packed-scale serving problem in a
listed cell runs the serving program's locality-domain twin: one launch over
both dies, each die claiming first the groups of ``b`` homed in its memory,
through a pair of per-die claim counters owned by the prepared launch.  A twin
is bitwise equal to the whole-GPU build of the same source for any cut; it is
equal within the family's tolerance to the whole-GPU serving program shipped
here, which is an older build, so falling back to the whole-GPU program may
change the low bits of the result.  Every other call runs the whole-GPU
program.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from .. import cake_locality
from ..jit.gemm.cake_batch_deepgemm_fp8 import (
    ARG_PLANS,
    DOMAIN_SERVING_CELLS,
    DOMAIN_SERVING_PROGRAMS,
    PROGRAMS,
    SERVING_PROGRAMS,
    SUPPORTED_COMPUTE_CAPABILITIES,
    RoutePlan,
    StagePlan,
    device_arch,
    device_sm_count,
    generated_program_available,
    load_cake_batch_deepgemm_fp8_module,
    module_name,
    route_chain,
)

BLOCK = 128
SUPPORTED_NK = frozenset(
    {
        (128, 512),
        (512, 128),
        (4096, 7168),
        (7168, 2048),
        (6144, 7168),
        (7168, 3072),
        (4096, 4096),
        (4096, 2048),
    }
)
# The geometries an architecture's route chain owns when it owns fewer than
# SUPPORTED_NK: SM107a runs the masked BLOCK_N programs alone.
ARCH_SUPPORTED_NK = {
    "sm_107a": frozenset({(128, 512), (512, 128), (4096, 7168), (7168, 2048)}),
}
# The zero-copy packed-scale serving programs are specialised on (N, K, B), on these
# architectures.
PACKED_NK = frozenset({(4096, 7168), (7168, 2048)})
PACKED_ARCHS = frozenset({"sm_100a", "sm_103a", "sm_107a"})
PACKED_GROUPS = frozenset({32, 64})
PACKED_K_GRANULE = 512
SERVING_ROUTE = "seed_serving_m32_packed"
# The locality-domain twin of the serving route (dual-die SM107a, ``b`` homed by
# groups): its claim counters, PAIR_COUNTER_WORDS int32 words per die, are the
# route's workspaces.
DOMAIN_SERVING_ROUTE = "seed_serving_m32_packed_domain"
PAIR_COUNTER_WORDS = 2
DOMAIN_ARCH = "sm_107a"
DOMAIN_COUNTERS = ("counters0", "counters1")
# Values of expected_m at both sides of every bound of serving_config and of
# expected_m_class: every serving key a weight shape reaches is reached by one.
DOMAIN_EXPECTED_M_PROBES = (0, 1, 2, 3, 4, 7, 8, 15, 16, 17)
WORKSPACE_DTYPES = {
    "sfa_packed": torch.int32,
    "sfb_packed": torch.int32,
    "sfa_m32_packed": torch.uint32,
    "sfb_m32_packed": torch.uint32,
    "counters0": torch.int32,
    "counters1": torch.int32,
}


def supported_nk(arch: str) -> frozenset[tuple[int, int]]:
    """The ``(N, K)`` geometries the route chain of ``arch`` owns."""
    return ARCH_SUPPORTED_NK.get(arch, SUPPORTED_NK)


def serving_config(
    m: int, k: int, expected_m: int, sm_count: int, arch: str
) -> tuple[int, int, str, int, int]:
    """``(num_stages, num_epi_stages, a_cache_hint, cta_reserve, n_l2_group)`` of the serving route.

    Port of the Cake serving seed's configuration rule: the choice depends only on
    caller-visible routing metadata (``K``, ``expected_m``, device), never on a
    trace quantile, so every replay of one profile runs one compiled program.

    ``n_l2_group`` is even in every config: the two CTAs of a cluster take
    consecutive tiles and must share one 32-row M block. ``expected_m`` is only a
    hint, so at ``expected_m=1`` a group can still span several M blocks; with
    ``n_l2_group=1`` the two CTAs of such a group would compute different M
    blocks and the output would be wrong.
    """
    if k == 7168:
        if expected_m <= 1:
            return 6, 1, "evict_normal", 0, 2
        if expected_m <= 3:
            return 5, 2, "evict_normal", 16, 8
        if sm_count == 148:
            cta_reserve = 8 if arch == "sm_103a" else 20
        else:
            cta_reserve = 24
        return 5, 2, "evict_normal", cta_reserve, 8
    if expected_m <= 1:
        return 5, 3, "evict_normal", 0, 8
    if expected_m == 2:
        return 5, 2, "evict_normal", 20, 8
    return 5, 2, "evict_normal", 24, 8


def serving_program_key(
    n: int,
    k: int,
    groups: int,
    num_stages: int,
    num_epi_stages: int,
    a_cache_hint: str,
    n_l2_group: int,
) -> str:
    return f"n{n}_k{k}_g{groups}_s{num_stages}e{num_epi_stages}_{a_cache_hint}_l{n_l2_group}"


def expected_m_class(expected_m: int) -> str:
    """The expected_m class of a locality-domain serving cell."""
    if expected_m <= 1:
        return "em<=1"
    if expected_m < 8:
        return "em2-7"
    if expected_m < 16:
        return "em8-15"
    return "em16+"


def domain_serving_program(
    arch: str,
    sm_count: int,
    key: str,
    expected_m: int,
    counts: tuple[int, int],
) -> Optional[str]:
    """The locality-domain twin of serving configuration ``key`` for a device of
    ``sm_count`` SMs split ``counts`` over its two dies, or ``None`` when the cell
    ``(sm_count, die class, expected_m class)`` is not routed to a twin."""
    qualified = f"{arch}:{key}"
    if arch != DOMAIN_ARCH or qualified not in DOMAIN_SERVING_PROGRAMS:
        return None
    cell = (cake_locality.die_class(counts), expected_m_class(expected_m))
    if qualified not in DOMAIN_SERVING_CELLS.get(sm_count, {}).get(cell, ()):
        return None
    return DOMAIN_SERVING_PROGRAMS[qualified]


def domain_serving_routed(
    arch: str,
    sm_count: int,
    counts: tuple[int, int],
    n: int,
    k: int,
    groups: int,
) -> bool:
    """True when every packed-scale serving problem on weights of ``groups`` x
    ``(n, k)`` runs a locality-domain twin on a device of ``sm_count`` SMs split
    ``counts`` over its dies: whatever its ``expected_m``, the serving key it
    selects has a twin listed in the cell of that ``expected_m``.  Homed weights
    of any other shape could run a whole-GPU program, which is slower on homed
    weights than on ordinary ones, so :func:`localize_weights` refuses them."""
    if (
        arch != DOMAIN_ARCH
        or (n, k) not in PACKED_NK
        or groups not in PACKED_GROUPS
        or sm_count not in DOMAIN_SERVING_CELLS
    ):
        return False
    for expected_m in DOMAIN_EXPECTED_M_PROBES:
        # The serving rule reads K, expected_m and the device, not M.
        stages, epi, hint, _, l2 = serving_config(0, k, expected_m, sm_count, arch)
        key = serving_program_key(n, k, groups, stages, epi, hint, l2)
        if domain_serving_program(arch, sm_count, key, expected_m, counts) is None:
            return False
    return True


def serving_plan(
    arch: str,
    sm_count: int,
    groups: int,
    m: int,
    n: int,
    k: int,
    expected_m: int,
    *,
    counts: Optional[tuple[int, int]] = None,
) -> RoutePlan:
    """The packed-scale serving route: one specialised program, grid from the SM count.

    ``counts`` (the SMs of each die of a dual-die device whose ``b`` is homed by
    groups) selects the locality-domain twin of the program when the cell is
    listed in ``DOMAIN_SERVING_CELLS``; the twin's grid is one CTA per SM and it
    binds a pair of per-die claim counters."""
    num_stages, num_epi_stages, a_cache_hint, cta_reserve, n_l2_group = serving_config(
        m, k, expected_m, sm_count, arch
    )
    key = serving_program_key(
        n, k, groups, num_stages, num_epi_stages, a_cache_hint, n_l2_group
    )
    program = SERVING_PROGRAMS.get(f"{arch}:{key}", SERVING_PROGRAMS.get(key))
    if program is None:
        raise NotImplementedError(
            f"no generated packed-scale serving program for N={n}, K={k}, B={groups}, "
            f"expected_m={expected_m} (configuration {key})"
        )
    twin = (
        None
        if counts is None
        else domain_serving_program(arch, sm_count, key, expected_m, counts)
    )
    if twin is not None:
        return RoutePlan(
            route=DOMAIN_SERVING_ROUTE,
            parent=SERVING_ROUTE,
            stages=(StagePlan(program=twin, grid=(sm_count // 2 * 2, 1, 1), args=()),),
            workspaces={name: (PAIR_COUNTER_WORDS,) for name in DOMAIN_COUNTERS},
        )
    raw_ctas = max(2, sm_count - cta_reserve - ((sm_count - cta_reserve) % 2))
    # The serving kernel binds its arguments by name (see ``_serving_bindings``).
    return RoutePlan(
        route=SERVING_ROUTE,
        parent=SERVING_ROUTE,
        stages=(StagePlan(program=program, grid=(raw_ctas, 1, 1), args=()),),
        workspaces={},
    )


def select_route(
    arch: str, sm_count: int, B: int, M: int, N: int, K: int, expected_m: int
) -> Optional[RoutePlan]:
    """The Cake dispatcher's route for a float32-scale problem (``None`` = named gap)."""
    chain = route_chain(arch, sm_count)
    if chain is None:
        raise NotImplementedError(
            f"no generated Cake batch DeepGEMM FP8 route chain for {arch} devices with {sm_count} SMs"
        )
    return chain(int(B), int(M), int(N), int(K), int(expected_m))


def _device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else int(device.index)


def _require_tensor(
    tensor: Any,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    strides: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if strides is None:
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    elif tuple(tensor.stride()) != strides:
        raise ValueError(
            f"{name} must have strides {strides}, got {tuple(tensor.stride())}"
        )
    if int(tensor.data_ptr()) % 16:
        raise ValueError(f"{name} storage must be 16-byte aligned")
    return tensor


@dataclass(frozen=True)
class _Problem:
    arch: str
    sm_count: int
    device: torch.device
    B: int
    M: int
    N: int
    K: int
    expected_m: int
    packed_scales: bool
    plan: RoutePlan
    # The locality-domain twin's first group of ``b`` homed on die 1 and the SM
    # table it takes by value (``None`` on every other route).
    g_cut: Optional[int] = None
    sm_table: Any = None


def _domain_homing(
    b: torch.Tensor, index: int, groups: int
) -> Optional[tuple[int, Any]]:
    """``(g_cut, sm_table)`` when ``b`` is a :func:`localize_weights` allocation on a
    device with a probed SM table, else ``None``.  ``g_cut`` is the group of ``b``
    nearest to its die cut (the twin's result does not depend on it)."""
    # The registry first: it builds nothing when no split allocation was made.
    homed = cake_locality.homing(b)
    if homed is None or homed.device_index != index:
        return None
    table = cake_locality.sm_table(index)
    if table is None:
        return None
    group_bytes = b.stride(0) * b.element_size()
    cut = homed.cut - (int(b.data_ptr()) - homed.base)
    g_cut = min(groups, max(0, (2 * cut + group_bytes) // (2 * group_bytes)))
    return g_cut, table


def _resolve(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    *,
    out: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
    scale_granularity_mnk: tuple[int, int, int],
) -> _Problem:
    """Validate the problem against the exported band and select its route."""
    if not isinstance(a, torch.Tensor) or a.device.type != "cuda":
        raise ValueError("a must be a CUDA torch.Tensor")
    device = a.device
    index = _device_index(device)
    arch = device_arch(index)
    if arch is None:
        raise ValueError(
            "the Cake batch DeepGEMM FP8 backend requires an SM100a, SM103a or SM107a device "
            f"(compute capability {sorted(SUPPORTED_COMPUTE_CAPABILITIES)}); "
            f"got {torch.cuda.get_device_capability(index)}"
        )
    sm_count = device_sm_count(index)
    # SM107a runs its packed-scale serving programs on every SM count (their grid
    # comes from the SM count); its float32-scale route chain is checked below.
    if arch != DOMAIN_ARCH and route_chain(arch, sm_count) is None:
        raise ValueError(
            f"the Cake batch DeepGEMM FP8 backend is exported for {arch} devices with "
            f"{sorted(_exported_sm_counts(arch))} SMs; this device has {sm_count}"
        )
    if tuple(int(v) for v in scale_granularity_mnk) != (1, BLOCK, BLOCK):
        raise ValueError(
            f"the Cake batch DeepGEMM FP8 backend requires scale_granularity_mnk=(1, {BLOCK}, {BLOCK})"
        )
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("a must have shape (B, M, K) and b must have shape (B, N, K)")
    B, M, K = (int(v) for v in a.shape)
    N = int(b.shape[1])
    if min(B, M, N, K) <= 0 or M % BLOCK or N % BLOCK or K % BLOCK:
        raise ValueError(
            f"the Cake batch DeepGEMM FP8 backend requires positive, {BLOCK}-aligned M, N and K; "
            f"got B={B}, M={M}, N={N}, K={K}"
        )
    if (N, K) not in supported_nk(arch):
        raise ValueError(
            f"the Cake batch DeepGEMM FP8 backend owns (N, K) in {sorted(supported_nk(arch))} "
            f"on {arch}; got ({N}, {K})"
        )
    if (
        isinstance(expected_m, bool)
        or not isinstance(expected_m, int)
        or not 0 <= expected_m <= M
    ):
        raise ValueError(f"expected_m must be an int in [0, M], got {expected_m!r}")
    _require_tensor(a, "a", shape=(B, M, K), dtype=torch.float8_e4m3fn, device=device)
    _require_tensor(b, "b", shape=(B, N, K), dtype=torch.float8_e4m3fn, device=device)
    _require_tensor(masked_m, "masked_m", shape=(B,), dtype=torch.int32, device=device)
    result_dtype = out.dtype if out is not None else (out_dtype or torch.bfloat16)
    if result_dtype != torch.bfloat16:
        raise ValueError("the Cake batch DeepGEMM FP8 backend writes a bfloat16 output")
    if out is not None:
        _require_tensor(
            out, "out", shape=(B, M, N), dtype=torch.bfloat16, device=device
        )
    if not isinstance(a_scale, torch.Tensor) or not isinstance(b_scale, torch.Tensor):
        raise TypeError("a_scale and b_scale must be torch.Tensors")
    packed = a_scale.dtype == torch.int32 or b_scale.dtype == torch.int32
    g_cut: Optional[int] = None
    table: Any = None
    if packed:
        if a_scale.dtype != torch.int32 or b_scale.dtype != torch.int32:
            raise ValueError(
                "packed UE8M0 scales require both a_scale and b_scale to be int32"
            )
        if arch not in PACKED_ARCHS:
            raise ValueError(
                f"packed UE8M0 scales have no Cake batch DeepGEMM FP8 program on {arch}; "
                "pass float32 scales"
            )
        if (N, K) not in PACKED_NK or B not in PACKED_GROUPS or N % 256 or K % 256:
            raise ValueError(
                "packed UE8M0 scales are supported for (N, K) in "
                f"{sorted(PACKED_NK)} with B in {sorted(PACKED_GROUPS)}; got B={B}, N={N}, K={K}"
            )
        cols = K // PACKED_K_GRANULE
        _require_tensor(
            a_scale,
            "a_scale",
            shape=(B, M, cols),
            dtype=torch.int32,
            device=device,
            strides=(M * cols, 1, M),
        )
        _require_tensor(
            b_scale,
            "b_scale",
            shape=(B, N, cols),
            dtype=torch.int32,
            device=device,
            strides=(N * cols, 1, N),
        )
        domain = _domain_homing(b, index, B) if arch == DOMAIN_ARCH else None
        plan = serving_plan(
            arch,
            sm_count,
            B,
            M,
            N,
            K,
            expected_m,
            counts=None if domain is None else domain[1].counts,
        )
        if plan.route == DOMAIN_SERVING_ROUTE:
            assert domain is not None
            g_cut, table = domain
    else:
        if route_chain(arch, sm_count) is None:
            raise ValueError(
                f"the Cake batch DeepGEMM FP8 backend is exported for {arch} devices with "
                f"{sorted(_exported_sm_counts(arch))} SMs (packed UE8M0 scales: any SM "
                f"count of {DOMAIN_ARCH}); this device has {sm_count}"
            )
        _require_tensor(
            a_scale,
            "a_scale",
            shape=(B, M, K // BLOCK),
            dtype=torch.float32,
            device=device,
        )
        _require_tensor(
            b_scale,
            "b_scale",
            shape=(B, N // BLOCK, K // BLOCK),
            dtype=torch.float32,
            device=device,
        )
        plan = select_route(arch, sm_count, B, M, N, K, expected_m)
        if plan is None:
            raise ValueError(
                f"the Cake batch DeepGEMM FP8 dispatcher owns no route for B={B}, M={M}, N={N}, K={K}, "
                f"expected_m={expected_m} on {arch}"
            )
    for stage in plan.stages:
        module_name(arch, stage.program)
    return _Problem(
        arch,
        sm_count,
        device,
        B,
        M,
        N,
        K,
        int(expected_m),
        packed,
        plan,
        g_cut=g_cut,
        sm_table=table,
    )


def _exported_sm_counts(arch: str) -> set[int]:
    from ..jit.gemm.cake_batch_deepgemm_fp8 import DISPATCH_TABLES

    return set(DISPATCH_TABLES.get(arch, {}))


def check_batch_deepgemm_fp8_nt_groupwise_cake(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    *,
    scale_granularity_mnk: tuple[int, int, int] = (1, BLOCK, BLOCK),
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> bool:
    """Admission predicate of ``backend="cake"``: True inside the exported band, raises outside it."""
    _resolve(
        a,
        b,
        a_scale,
        b_scale,
        masked_m,
        expected_m,
        out=out,
        out_dtype=out_dtype,
        scale_granularity_mnk=scale_granularity_mnk,
    )
    return True


def is_batch_deepgemm_fp8_nt_groupwise_cake_available(device: torch.device) -> bool:
    """True when generated programs and a float32-scale route chain are registered
    for ``device``.  The packed-scale serving programs also run on SM107a devices
    without a route chain (any SM count)."""
    return device.type == "cuda" and generated_program_available(device)


def _grid3(grid: tuple[int, ...]) -> tuple[int, int, int]:
    x, y, z = (int(v) for v in grid)
    return (x, y, z)


def bind_arguments(
    arg_plan: list[list[str]],
    positional: tuple[Any, ...],
    grid: tuple[int, int, int],
    *,
    program: str,
) -> tuple[Any, ...]:
    """Interleave the route's positional kernel arguments with the grid slots of the argument plan."""
    grid_by_axis = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = []
    values = iter(positional)
    for kind, name in arg_plan:
        if kind == "grid":
            arguments.append(grid_by_axis[name])
        elif kind in ("tma_buffer", "buffer", "parameter"):
            arguments.append(next(values))
        else:
            raise RuntimeError(
                f"generated program {program} binds {kind} {name!r}, which this host does not declare"
            )
    if next(values, None) is not None:
        raise RuntimeError(
            f"generated program {program} takes fewer arguments than the route binds"
        )
    return tuple(arguments)


def bind_named_arguments(
    arg_plan: list[list[str]],
    bindings: dict[str, Any],
    grid: tuple[int, int, int],
    *,
    program: str,
) -> tuple[Any, ...]:
    grid_by_axis = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = []
    for kind, name in arg_plan:
        if kind == "grid":
            arguments.append(grid_by_axis[name])
        elif (
            kind in ("tma_buffer", "buffer", "parameter", "param_array")
            and name in bindings
        ):
            arguments.append(bindings[name])
        else:
            raise RuntimeError(
                f"generated program {program} binds {kind} {name!r}, which this host does not declare"
            )
    return tuple(arguments)


@dataclass
class PreparedBatchDeepGemmFp8NtGroupwise:
    """One prepared masked batch FP8 GEMM launch.

    Tensor storage, shapes and dtypes are bound at preparation; tensor *contents*
    may change between launches.  ``launch()`` submits the route's kernels (one,
    or scale-pack + GEMM) on PyTorch's current stream for the bound device and
    returns the output.  Every launch, including the first, may be captured into a
    CUDA graph; ``launch()`` allocates nothing.

    The locality-domain twin route binds the claim counters ``workspace["counters0"]``
    and ``workspace["counters1"]``: zero before the first launch, left zero by
    every launch.  Launches of one prepared launch must therefore not overlap
    (successive launches on one stream, or graph replays on one stream, never
    do); prepare one launch per concurrent stream.  A twin launch prepared under
    CUDA graph capture without caller-owned counters zeroes its counters inside
    that capture, so it belongs to that capture: ``launch()`` raises outside
    capture, and it must be launched in the capture it was prepared in.
    """

    route: str
    parent: str
    arch: str
    sm_count: int
    programs: tuple[str, ...]
    grids: tuple[tuple[int, int, int], ...]
    out: torch.Tensor
    workspace: dict[str, torch.Tensor]
    _launches: tuple[tuple[Callable[..., Any], tuple[Any, ...]], ...]
    # The locality-domain twin's first group of ``b`` homed on die 1 (``None`` on
    # every other route).
    g_cut: Optional[int] = None
    # True when the launch's counters were zeroed inside a graph capture.
    capture_only: bool = False

    def launch(self) -> torch.Tensor:
        with torch.cuda.device(self.out.device), tvm_ffi.use_torch_stream():
            if self.capture_only and not torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "this locality-domain launch was prepared under CUDA graph "
                    "capture, which zeroes its claim counters: launch it only in "
                    "that capture, or prepare it outside capture"
                )
            for entry, arguments in self._launches:
                entry(*arguments)
        return self.out

    __call__ = launch

    @property
    def num_kernels(self) -> int:
        return len(self._launches)


def _allocate_workspace(
    plan: RoutePlan, device: torch.device, *, launch_now: bool = False
) -> dict[str, torch.Tensor]:
    """The route's workspace.  ``launch_now`` (the public API's call, launched at
    once on the current stream) takes the twin's counters from
    :func:`flashinfer.cake_locality.launch_counters` (a pair reused per stream,
    or a pre-zeroed pair for a capture) instead of allocating a pair the
    prepared launch owns."""
    workspace = {
        key: torch.empty(shape, dtype=WORKSPACE_DTYPES[key], device=device)
        for key, shape in plan.workspaces.items()
        if key not in DOMAIN_COUNTERS
    }
    if DOMAIN_COUNTERS[0] in plan.workspaces:
        pair: tuple[torch.Tensor, ...]
        if launch_now:
            pair = cake_locality.launch_counters(PAIR_COUNTER_WORDS, device)
        elif torch.cuda.is_current_stream_capturing():
            # Locality-domain memory is not allocated under capture: zeroed
            # counters from the capture's pool (same bits, unhomed claims).
            pair = tuple(
                torch.zeros(
                    plan.workspaces[name], dtype=WORKSPACE_DTYPES[name], device=device
                )
                for name in DOMAIN_COUNTERS
            )
        else:
            pair = cake_locality.counters(PAIR_COUNTER_WORDS, device)
        workspace.update(zip(DOMAIN_COUNTERS, pair, strict=True))
    return workspace


def _check_workspace(
    plan: RoutePlan, workspace: dict[str, torch.Tensor], device: torch.device
) -> None:
    if set(workspace) != set(plan.workspaces):
        raise ValueError(
            f"route {plan.route} binds workspaces {sorted(plan.workspaces)}, got {sorted(workspace)}"
        )
    for key, shape in plan.workspaces.items():
        _require_tensor(
            workspace[key],
            key,
            shape=tuple(shape),
            dtype=WORKSPACE_DTYPES[key],
            device=device,
        )
    if DOMAIN_COUNTERS[0] in plan.workspaces:
        # Each die claims through its own counters: the two must not share a byte.
        spans = sorted(
            (
                int(workspace[name].data_ptr()),
                int(workspace[name].data_ptr())
                + workspace[name].numel() * workspace[name].element_size(),
            )
            for name in DOMAIN_COUNTERS
        )
        if spans[0][1] > spans[1][0]:
            raise ValueError(
                f"workspaces {DOMAIN_COUNTERS[0]} and {DOMAIN_COUNTERS[1]} overlap; "
                "each die needs its own zeroed counters"
            )


def _family_bindings(
    problem: _Problem,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    out: torch.Tensor,
    workspace: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Family tensor / workspace keys of the rendered route chain -> bound tensors."""
    return {
        "a": a.view(torch.uint8),
        "b": b.view(torch.uint8),
        "a_scale": a_scale,
        "b_scale": b_scale,
        "a_scale_bits": a_scale.view(torch.int32),
        "b_scale_bits": b_scale.view(torch.int32),
        "masked_m": masked_m,
        "out": out,
        **workspace,
    }


def _serving_bindings(
    problem: _Problem,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    out: torch.Tensor,
    workspace: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Kernel parameter names of the serving program -> bound values (zero-copy scale views)."""
    cols = problem.K // PACKED_K_GRANULE
    # The logical [B, MN, K/512] int32 scales live in MN-major [B, K/512, MN] storage;
    # reinterpret that storage as the contiguous 2-D surface the scale tensor maps read.
    sfa = a_scale.as_strided((problem.B * cols, problem.M), (problem.M, 1)).view(
        torch.uint32
    )
    sfb = b_scale.as_strided((problem.B * cols, problem.N), (problem.N, 1)).view(
        torch.uint32
    )
    bindings: dict[str, Any] = {
        "A": a.view(torch.uint8),
        "B": b.view(torch.uint8),
        "C_tma": out,
        "SFA_packed": sfa,
        "SFB_packed": sfb,
        "masked_m": masked_m,
        "num_groups": problem.B,
        "shape_m": problem.M,
        "N": problem.N,
        "K": problem.K,
    }
    if problem.plan.route == DOMAIN_SERVING_ROUTE:
        bindings.update(
            g_cut=problem.g_cut,
            counters0=workspace["counters0"],
            counters1=workspace["counters1"],
            sm_table=problem.sm_table.entries,
        )
    return bindings


def prepare_batch_deepgemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    *,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[dict[str, torch.Tensor]] = None,
    scale_granularity_mnk: tuple[int, int, int] = (1, BLOCK, BLOCK),
) -> PreparedBatchDeepGemmFp8NtGroupwise:
    r"""Prepare a masked batch FP8 GEMM launch of the generated Cake programs.

    Parameters
    ----------
    a : torch.Tensor
        Contiguous FP8 E4M3 activations of shape ``(B, M, K)``.
    b : torch.Tensor
        Contiguous FP8 E4M3 expert weights of shape ``(B, N, K)``.
    a_scale, b_scale : torch.Tensor
        Either contiguous float32 scales of shapes ``(B, M, K // 128)`` and
        ``(B, N // 128, K // 128)``, or the native MN-major packed UE8M0 int32
        scales of the serving routes: shapes ``(B, M, K // 512)`` and
        ``(B, N, K // 512)`` with strides ``(M * K // 512, 1, M)`` and
        ``(N * K // 512, 1, N)`` (``(N, K)`` in ``{(4096, 7168), (7168, 2048)}``,
        ``B`` in ``{32, 64}``), consumed without conversion.
    masked_m : torch.Tensor
        Contiguous int32 row counts of shape ``(B,)``; ``0 <= masked_m[g] <= M``.
    expected_m : int
        Host-known expected row count per group in ``[0, M]``; routes depend on it.
    out : Optional[torch.Tensor]
        Contiguous bfloat16 output of shape ``(B, M, N)``.  Allocated if omitted.
    workspace : Optional[dict[str, torch.Tensor]]
        Caller-owned workspace of the selected route (see
        ``RoutePlan.workspaces``): the packed-scale workspace of a swap-AB route,
        or the zeroed int32 claim counters ``counters0`` / ``counters1`` of the
        locality-domain twin route (homed on die 0 / die 1 by
        :func:`flashinfer.cake_locality.counters`).  Allocated at preparation if
        omitted: a prepared launch owns its counters (the public API's call
        instead reuses a counter pair per stream, see
        :func:`run_batch_deepgemm_fp8_nt_groupwise`).

    Returns
    -------
    PreparedBatchDeepGemmFp8NtGroupwise
        Call ``.launch()`` to run the GEMM on the current stream.

    Notes
    -----
    ``M``, ``N`` and ``K`` must be multiples of 128 and ``(N, K)`` one of the eight
    exported geometries; a problem outside the band raises ``ValueError``.
    Preparation performs no device work beyond the optional workspace
    allocation; the device's architecture and SM count are read once per process.
    On a dual-die SM107a device, packed-scale problems whose ``b`` comes from
    :func:`localize_weights` run the locality-domain twin route in the listed
    cells (``route == "seed_serving_m32_packed_domain"``).
    """
    return _prepare(
        a,
        b,
        a_scale,
        b_scale,
        masked_m,
        expected_m,
        out=out,
        workspace=workspace,
        scale_granularity_mnk=scale_granularity_mnk,
        launch_now=False,
    )


def _prepare(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    *,
    out: Optional[torch.Tensor],
    workspace: Optional[dict[str, torch.Tensor]],
    scale_granularity_mnk: tuple[int, int, int],
    launch_now: bool,
) -> PreparedBatchDeepGemmFp8NtGroupwise:
    problem = _resolve(
        a,
        b,
        a_scale,
        b_scale,
        masked_m,
        expected_m,
        out=out,
        out_dtype=None,
        scale_granularity_mnk=scale_granularity_mnk,
    )
    device = problem.device
    if out is None:
        out = torch.empty(
            (problem.B, problem.M, problem.N), dtype=torch.bfloat16, device=device
        )
    plan = problem.plan
    capture_only = False
    if workspace is None:
        workspace = _allocate_workspace(plan, device, launch_now=launch_now)
        capture_only = (
            not launch_now
            and DOMAIN_COUNTERS[0] in plan.workspaces
            and torch.cuda.is_current_stream_capturing()
        )
    else:
        _check_workspace(plan, workspace, device)
    launches = []
    if problem.packed_scales:
        (stage,) = plan.stages
        program = PROGRAMS[stage.program]
        entry = getattr(
            load_cake_batch_deepgemm_fp8_module(
                module_name(problem.arch, stage.program)
            ),
            program["ffi_entry"],
        )
        bindings = _serving_bindings(
            problem, a, b, a_scale, b_scale, masked_m, out, workspace
        )
        launches.append(
            (
                entry,
                bind_named_arguments(
                    ARG_PLANS[program["arg_plan"]],
                    bindings,
                    stage.grid,
                    program=stage.program,
                ),
            )
        )
    else:
        bindings = _family_bindings(
            problem, a, b, a_scale, b_scale, masked_m, out, workspace
        )
        for stage in plan.stages:
            program = PROGRAMS[stage.program]
            entry = getattr(
                load_cake_batch_deepgemm_fp8_module(
                    module_name(problem.arch, stage.program)
                ),
                program["ffi_entry"],
            )
            positional = tuple(
                bindings[item] if isinstance(item, str) else int(item)
                for item in stage.args
            )
            launches.append(
                (
                    entry,
                    bind_arguments(
                        ARG_PLANS[program["arg_plan"]],
                        positional,
                        stage.grid,
                        program=stage.program,
                    ),
                )
            )
    return PreparedBatchDeepGemmFp8NtGroupwise(
        route=plan.route,
        parent=plan.parent,
        arch=problem.arch,
        sm_count=problem.sm_count,
        programs=tuple(stage.program for stage in plan.stages),
        grids=tuple(_grid3(stage.grid) for stage in plan.stages),
        out=out,
        workspace=workspace,
        _launches=tuple(launches),
        g_cut=problem.g_cut,
        capture_only=capture_only,
    )


def localize_weights(b: torch.Tensor, g_cut: Optional[int] = None) -> torch.Tensor:
    """A copy of the expert weights ``b`` (``(B, N, K)``, contiguous) homed by groups
    on the two dies of a dual-die SM107a device: groups ``[0, g_cut)`` in die 0's
    memory, the rest in die 1's, in one virtual address range.

    ``g_cut`` defaults to the SM-proportional share of die 0,
    ``split_units(B, counts)[0]`` over the device's probed SM table.  Passing the
    copy as ``b`` makes every packed-scale serving problem on it run the
    locality-domain twin of its serving program; the twin's result is the same
    bits for any ``g_cut``.

    Returns ``b`` itself, unhomed, when homing could make a call slower: the
    device has no SM table (:func:`flashinfer.cake_locality.sm_table` is
    ``None``), ``b`` is not float8_e4m3fn weights of a packed-scale serving shape
    (``(N, K)`` in ``PACKED_NK``, ``B`` in ``PACKED_GROUPS``), some ``expected_m``
    of that shape selects a serving program without a twin listed in its cell
    (:func:`domain_serving_routed`), or the cut is not a multiple of the
    allocation granularity of locality-domain memory.  Raises ``ValueError`` for
    a ``g_cut`` outside ``[0, B]``.

    With the cells listed for 212-SM devices, only ``B=64`` weights of
    ``(N, K) = (4096, 7168)`` are homed: at every ``expected_m`` their serving
    program has a listed twin, including the ``expected_m >= 16`` cell listed for
    homed weights only (see ``DOMAIN_SERVING_CELLS``).  ``B=32`` weights of
    ``(4096, 7168)`` and ``B=32`` and ``B=64`` weights of ``(7168, 2048)`` are
    returned unhomed: their ``expected_m <= 1`` serving program has no twin.

    Call it outside CUDA graph capture.  The memory is freed with the returned
    tensor; a CUDA graph that reads the tensor after it is freed faults on
    replay, so keep it alive as long as such a graph.
    """
    if not isinstance(b, torch.Tensor) or b.device.type != "cuda" or b.ndim != 3:
        raise ValueError("b must be a CUDA torch.Tensor of shape (B, N, K)")
    if not b.is_contiguous():
        raise ValueError("b must be contiguous")
    groups, n, k = (int(extent) for extent in b.shape)
    if g_cut is not None and not 0 <= int(g_cut) <= groups:
        raise ValueError(f"g_cut must be in [0, {groups}], got {g_cut}")
    if b.dtype != torch.float8_e4m3fn:
        return b
    index = _device_index(b.device)
    arch = device_arch(index)
    if arch != DOMAIN_ARCH:
        return b
    table = cake_locality.sm_table(index)
    if table is None:
        return b
    if not domain_serving_routed(
        arch, device_sm_count(index), table.counts, n, k, groups
    ):
        return b
    if g_cut is None:
        g_cut = cake_locality.split_units(groups, table.counts)[0]
    group_bytes = n * k * b.element_size()
    cut = int(g_cut) * group_bytes
    if 0 < cut < groups * group_bytes and cut % cake_locality.granularity(index):
        return b
    homed = cake_locality.split_empty(tuple(b.shape), b.dtype, cut, index)
    homed.copy_(b)
    return homed


def run_batch_deepgemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    """One masked batch FP8 GEMM into ``out`` (the ``backend="cake"`` path of the public API).

    The route's packed-scale workspace, when it needs one, is a per-call
    ``torch.empty`` released when the call returns, and the call may be captured
    into a CUDA graph.

    The locality-domain twin's claim counters are not allocated per call: every
    launch leaves its pair zero, so the call takes an already zero pair from
    :func:`flashinfer.cake_locality.launch_counters` (die-homed when FlashInfer's
    locality-domain pools exist, ordinary zeroed tensors otherwise).  Eager calls
    on one stream reuse that stream's pair (their launches are ordered); a call
    under CUDA graph capture gets a pre-zeroed pair of its own, used by no eager
    call and no other capture, so the graph holds no memset node and may replay
    on any stream while eager calls run.  The same graph must not replay on two
    streams at once (its launch would share its pair with itself), as for any
    workspace a graph holds; a prepared launch
    (:func:`prepare_batch_deepgemm_fp8_nt_groupwise`) keeps its own pair.
    """
    return _prepare(
        a,
        b,
        a_scale,
        b_scale,
        masked_m,
        expected_m,
        out=out,
        workspace=None,
        scale_granularity_mnk=(1, BLOCK, BLOCK),
        launch_now=True,
    ).launch()


__all__ = [
    "ARCH_SUPPORTED_NK",
    "PACKED_ARCHS",
    "PACKED_GROUPS",
    "PACKED_NK",
    "DOMAIN_SERVING_ROUTE",
    "PAIR_COUNTER_WORDS",
    "PreparedBatchDeepGemmFp8NtGroupwise",
    "SUPPORTED_NK",
    "WORKSPACE_DTYPES",
    "bind_arguments",
    "bind_named_arguments",
    "check_batch_deepgemm_fp8_nt_groupwise_cake",
    "domain_serving_program",
    "domain_serving_routed",
    "expected_m_class",
    "is_batch_deepgemm_fp8_nt_groupwise_cake_available",
    "localize_weights",
    "prepare_batch_deepgemm_fp8_nt_groupwise",
    "run_batch_deepgemm_fp8_nt_groupwise",
    "select_route",
    "serving_config",
    "serving_plan",
    "serving_program_key",
    "supported_nk",
]
