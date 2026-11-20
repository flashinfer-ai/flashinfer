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
through a pair of per-die claim counters: a prepared launch owns its pair, and
the public API's call reuses the pair of its stream (a captured call takes a
pre-zeroed pair of its own; see :func:`run_batch_deepgemm_fp8_nt_groupwise`).  A twin
is bitwise equal to the whole-GPU build of the same source for any cut; it is
equal within the family's tolerance to the whole-GPU serving program shipped
here, which is an older build, so falling back to the whole-GPU program may
change the low bits of the result.  Every other call runs the whole-GPU
program.
"""

from __future__ import annotations

import contextlib
import operator
import threading
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

    ``cta_reserve`` is a measured schedule choice: the grid is the SM count less
    the reserve, in whole CTA pairs. On 212-SM devices the full grid (reserve 0)
    is faster from ``expected_m`` 36 up (by 2-11% at ``expected_m`` 36 to 4096 on
    every measured shape) and slower at ``expected_m`` 32 and below, which keep
    the default reserve.
    """
    if k == 7168:
        if expected_m <= 1:
            return 6, 1, "evict_normal", 0, 2
        if expected_m <= 3:
            return 5, 2, "evict_normal", 16, 8
        if sm_count == 148:
            cta_reserve = 8 if arch == "sm_103a" else 20
        elif sm_count == 212 and expected_m >= 36:
            cta_reserve = 0
        else:
            cta_reserve = 24
        return 5, 2, "evict_normal", cta_reserve, 8
    if expected_m <= 1:
        return 5, 3, "evict_normal", 0, 8
    if expected_m == 2:
        return 5, 2, "evict_normal", 20, 8
    if sm_count == 212 and expected_m >= 36:
        return 5, 2, "evict_normal", 0, 8
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
    return _domain_cell_program(
        arch, sm_count, key, expected_m, cake_locality.die_class(counts)
    )


def _domain_cell_program(
    arch: str, sm_count: int, key: str, expected_m: int, die_class: str
) -> Optional[str]:
    qualified = f"{arch}:{key}"
    if arch != DOMAIN_ARCH or qualified not in DOMAIN_SERVING_PROGRAMS:
        return None
    cell = (die_class, expected_m_class(expected_m))
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
    return _domain_class_routed(
        arch, sm_count, cake_locality.die_class(counts), n, k, groups
    )


def _domain_class_routed(
    arch: str, sm_count: int, die_class: str, n: int, k: int, groups: int
) -> bool:
    """:func:`domain_serving_routed` for the die class ``die_class``."""
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
        if _domain_cell_program(arch, sm_count, key, expected_m, die_class) is None:
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
    # The serving kernel binds its arguments by name (see ``_serving_launch``).
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


# Device index -> (generated-program architecture or None, SM count): fixed for
# the process, so a call does not query the device properties again.
_DEVICES: dict[int, tuple[Optional[str], int]] = {}


def _device_info(index: int) -> tuple[Optional[str], int]:
    """``(device_arch(index), device_sm_count(index))``, queried once per device."""
    info = _DEVICES.get(index)
    if info is None:
        info = _DEVICES.setdefault(index, (device_arch(index), device_sm_count(index)))
    return info


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
    if tensor.shape != shape:
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


# Route decisions already made: (arch, SM count, packed scales, B, M, N, K,
# expected_m, die SM counts of homed ``b`` or None) -> RoutePlan, and ("routed",
# arch, SM count, die SM counts, N, K, B) -> domain_serving_routed(...).  The
# decisions read only these, so a call re-validates its tensors and reuses them.
# At most PLAN_LIMIT entries; the oldest is dropped past it.
PLAN_LIMIT = 4096
_PLANS: dict[tuple[Any, ...], Any] = {}
# (arch, program) -> (FFI entry, argument plan) of a loaded generated program.
_ENTRIES: dict[tuple[str, str], tuple[Callable[..., Any], list[list[str]]]] = {}


def _remember_plan(
    key: tuple[Any, ...], plan: Any, memo: Optional[dict[Any, Any]] = None
) -> None:
    """Store ``plan`` under ``key`` in ``memo`` (default ``_PLANS``), dropping the
    oldest entries past :data:`PLAN_LIMIT`."""
    if memo is None:
        memo = _PLANS
    while len(memo) >= PLAN_LIMIT:
        try:
            del memo[next(iter(memo))]
        except (KeyError, RuntimeError, StopIteration):  # another thread evicted it
            break
    memo[key] = plan


def _program_entry(
    arch: str, program: str
) -> tuple[Callable[..., Any], list[list[str]]]:
    cached = _ENTRIES.get((arch, program))
    if cached is None:
        record = PROGRAMS[program]
        entry = getattr(
            load_cake_batch_deepgemm_fp8_module(module_name(arch, program)),
            record["ffi_entry"],
        )
        cached = _ENTRIES.setdefault(
            (arch, program), (entry, ARG_PLANS[record["arg_plan"]])
        )
    return cached


# Device index -> TVM FFI device, for the FFI stream context of a launch.
_FFI_DEVICES: dict[int, Any] = {}


def _ffi_stream(index: int) -> Any:
    """The FFI stream context on PyTorch's current stream of device ``index``
    (also the capture stream under CUDA graph capture): the raw stream handle and
    a cached FFI device, the same context as ``tvm_ffi.use_torch_stream()``
    without building a stream object per call."""
    ffi_device = _FFI_DEVICES.get(index)
    if ffi_device is None:
        ffi_device = _FFI_DEVICES.setdefault(index, tvm_ffi.device(f"cuda:{index}"))
    return tvm_ffi.use_raw_stream(ffi_device, torch._C._cuda_getCurrentRawStream(index))


def _run_launches(device: torch.device, launches: Any) -> None:
    """Submit ``(FFI entry, arguments)`` launches on the current stream of
    ``device``, making it the current device only when it is not."""
    index = _device_index(device)
    # CUDA is initialized (the operands live on the device): read the current
    # device without torch.cuda.current_device()'s lazy-init wrapper.
    if torch._C._cuda_getDevice() == index:
        with _ffi_stream(index):
            for entry, arguments in launches:
                entry(*arguments)
    else:
        with torch.cuda.device(index), _ffi_stream(index):
            for entry, arguments in launches:
                entry(*arguments)


# The per-call values of a serving program's arguments, in this order; every
# other argument of a serving launch is fixed by its problem and route.
_SERVING_VALUES = (
    "A",
    "B",
    "C_tma",
    "SFA_packed",
    "SFB_packed",
    "masked_m",
    "counters0",
    "counters1",
)
# (device index, arch, program, grid, B, M, N, K, g_cut) -> (FFI entry, the
# fixed arguments, a getter of the argument tuple from the per-call values
# followed by the fixed ones, the SM table tensor bound): a serving launch's
# arguments without a dict per call.  The device is part of the key because the
# twin binds its device's SM table: calls alternating between two devices keep
# one template each.  At most PLAN_LIMIT entries, as _PLANS.
_SERVING_LAUNCHES: dict[tuple[Any, ...], tuple[Any, tuple[Any, ...], Any, Any]] = {}


def _serving_launch(
    problem: _Problem, stage: StagePlan
) -> tuple[Any, tuple[Any, ...], Any]:
    """``(FFI entry, fixed arguments, getter)`` of the serving launch of ``problem``:
    ``getter(values + fixed)`` is the launch's argument tuple, where ``values``
    are the per-call values named by ``_SERVING_VALUES``."""
    table = None if problem.sm_table is None else problem.sm_table.entries
    grid = _grid3(stage.grid)
    key = (
        problem.device.index,
        problem.arch,
        stage.program,
        grid,
        problem.B,
        problem.M,
        problem.N,
        problem.K,
        problem.g_cut,
    )
    cached = _SERVING_LAUNCHES.get(key)
    if cached is not None and cached[3] is table:
        return cached[0], cached[1], cached[2]
    entry, arg_plan = _program_entry(problem.arch, stage.program)
    domain = problem.plan.route == DOMAIN_SERVING_ROUTE
    fixed_values: dict[str, Any] = {
        "num_groups": problem.B,
        "shape_m": problem.M,
        "N": problem.N,
        "K": problem.K,
    }
    if domain:
        fixed_values.update(g_cut=problem.g_cut, sm_table=table)
    per_call = _SERVING_VALUES if domain else _SERVING_VALUES[:6]
    grid_by_axis = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    fixed: list[Any] = []
    picks: list[int] = []
    for kind, name in arg_plan:
        if kind == "grid":
            value = grid_by_axis[name]
        elif kind in ("tma_buffer", "buffer", "parameter", "param_array") and (
            name in per_call
        ):
            picks.append(_SERVING_VALUES.index(name))
            continue
        elif (
            kind in ("tma_buffer", "buffer", "parameter", "param_array")
            and name in fixed_values
        ):
            value = fixed_values[name]
        else:
            raise RuntimeError(
                f"generated program {stage.program} binds {kind} {name!r}, which this host does not declare"
            )
        picks.append(len(_SERVING_VALUES) + len(fixed))
        fixed.append(value)
    # itemgetter of one index returns the item, not a tuple.
    getter: Any = (
        operator.itemgetter(*picks)
        if len(picks) > 1
        else (lambda values, only=picks[0]: (values[only],))
    )
    built = (entry, tuple(fixed), getter, table)
    _remember_plan(key, built, _SERVING_LAUNCHES)
    return entry, built[1], getter


def _domain_homing(
    b: torch.Tensor, index: int, groups: int, arch: str, sm_count: int
) -> Optional[tuple[int, Any]]:
    """``(g_cut, sm_table)`` when ``b`` covers the whole allocation
    :func:`localize_weights` returned (that tensor or a full alias of it: an
    ``nn.Parameter`` over it, ``detach()``, ``.data``, a view of the same shape;
    not a view of part of it, not a copy) on a device with a probed SM table
    whose memory runs at its full clock
    (:func:`flashinfer.cake_locality.memory_at_full_clock`), else ``None``.
    ``g_cut`` is the group of ``b`` nearest to its die cut (the twin's result
    does not depend on it).  Reads the record the allocation's
    storage carries (:func:`flashinfer.cake_locality.tensor_homing`), not the
    allocation registry.  A shape :func:`localize_weights` refuses
    (:func:`domain_serving_routed` is False) plans unhomed even when its ``b`` was
    homed some other way (e.g. by :func:`flashinfer.cake_locality.split_empty`)."""
    homed = cake_locality.tensor_homing(b)
    if homed is None or homed.device_index != index:
        return None
    if b.numel() * b.element_size() != homed.nbytes:
        return None
    table = cake_locality.sm_table(index)
    if table is None or not cake_locality.memory_at_full_clock(index):
        return None
    n, k = int(b.shape[1]), int(b.shape[2])
    key = ("routed", arch, sm_count, table.counts, n, k, groups)
    routed = _PLANS.get(key)
    if routed is None:
        routed = domain_serving_routed(arch, sm_count, table.counts, n, k, groups)
        _remember_plan(key, routed)
    if not routed:
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
    arch, sm_count = _device_info(index)
    if arch is None:
        raise ValueError(
            "the Cake batch DeepGEMM FP8 backend requires an SM100a, SM103a or SM107a device "
            f"(compute capability {sorted(SUPPORTED_COMPUTE_CAPABILITIES)}); "
            f"got {torch.cuda.get_device_capability(index)}"
        )
    # SM107a runs its packed-scale serving programs on every SM count (their grid
    # comes from the SM count); its float32-scale route chain is checked below.
    if arch != DOMAIN_ARCH and route_chain(arch, sm_count) is None:
        raise ValueError(
            f"the Cake batch DeepGEMM FP8 backend is exported for {arch} devices with "
            f"{sorted(_exported_sm_counts(arch))} SMs; this device has {sm_count}"
        )
    if scale_granularity_mnk != (1, BLOCK, BLOCK) and tuple(
        int(v) for v in scale_granularity_mnk
    ) != (1, BLOCK, BLOCK):
        raise ValueError(
            f"the Cake batch DeepGEMM FP8 backend requires scale_granularity_mnk=(1, {BLOCK}, {BLOCK})"
        )
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("a must have shape (B, M, K) and b must have shape (B, N, K)")
    # Plain ints (a SymInt never reaches the _PLANS / _LAYOUTS keys).
    B, M, K = map(int, a.shape)
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
        domain = (
            _domain_homing(b, index, B, arch, sm_count) if arch == DOMAIN_ARCH else None
        )
        counts = None if domain is None else domain[1].counts
        key = (arch, sm_count, True, B, M, N, K, expected_m, counts)
        plan = _PLANS.get(key)
        if plan is None:
            plan = serving_plan(arch, sm_count, B, M, N, K, expected_m, counts=counts)
            for stage in plan.stages:
                module_name(arch, stage.program)
            _remember_plan(key, plan)
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
        key = (arch, sm_count, False, B, M, N, K, expected_m, None)
        cached = _PLANS.get(key)
        if cached is None:
            selected = select_route(arch, sm_count, B, M, N, K, expected_m)
            if selected is None:
                raise ValueError(
                    f"the Cake batch DeepGEMM FP8 dispatcher owns no route for B={B}, M={M}, N={N}, K={K}, "
                    f"expected_m={expected_m} on {arch}"
                )
            for stage in selected.stages:
                module_name(arch, stage.program)
            _remember_plan(key, selected)
            cached = selected
        plan = cached
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


class _PublicCall(threading.local):
    """The public API call running in this thread (see :func:`_open_public_call`):
    ``entry`` is ``_OPEN`` until its admission check runs, then what that check
    left for the run of the same call, else ``None``."""

    entry: Any = None


_PUBLIC = _PublicCall()
# The entry of a public call whose admission check has not run yet.
_OPEN = object()


def _open_public_call() -> None:
    """Start of a public API call with ``backend="cake"`` (called by the public
    wrapper first, before its admission check): drops whatever an earlier call
    left behind and lets this call's check leave its resolved problem for this
    call's run."""
    _PUBLIC.entry = _OPEN


def _operand_addresses(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    out: Optional[torch.Tensor],
) -> tuple[Any, ...]:
    """The address of every operand (of ``out`` when given)."""
    return (
        a.data_ptr(),
        b.data_ptr(),
        a_scale.data_ptr(),
        b_scale.data_ptr(),
        masked_m.data_ptr(),
        None if out is None else out.data_ptr(),
    )


def _operand_layout(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    out: Optional[torch.Tensor],
) -> tuple[Any, ...]:
    """Shape, strides and dtype of every operand (of ``out`` when given)."""
    return (
        a.shape,
        a.stride(),
        a.dtype,
        b.shape,
        b.stride(),
        b.dtype,
        a_scale.shape,
        a_scale.stride(),
        a_scale.dtype,
        b_scale.shape,
        b_scale.stride(),
        b_scale.dtype,
        masked_m.shape,
        masked_m.stride(),
        masked_m.dtype,
        None if out is None else (out.shape, out.stride(), out.dtype),
    )


# (packed scales, out given, B, M, N, K) -> the operand layout (as
# _operand_layout) _resolve admits for that problem with canonical strides.  At
# most PLAN_LIMIT entries, as _PLANS.
_LAYOUTS: dict[tuple[Any, ...], tuple[Any, ...]] = {}


def _admitted_layout(problem: _Problem, out_given: bool) -> tuple[Any, ...]:
    """The layout of operands ``problem`` was resolved for: every operand
    contiguous, the packed scales MN-major, as :func:`_resolve` requires.  (A
    layout it also admits with other strides, such as a size-1 dimension of
    any stride, differs from it and is resolved again.)"""
    key = (problem.packed_scales, out_given, problem.B, problem.M, problem.N, problem.K)
    layout = _LAYOUTS.get(key)
    if layout is None:
        _, _, B, M, N, K = key
        if problem.packed_scales:
            cols = K // PACKED_K_GRANULE
            scales: tuple[Any, ...] = (
                (B, M, cols),
                (M * cols, 1, M),
                torch.int32,
                (B, N, cols),
                (N * cols, 1, N),
                torch.int32,
            )
        else:
            kb, nb = K // BLOCK, N // BLOCK
            scales = (
                (B, M, kb),
                (M * kb, kb, 1),
                torch.float32,
                (B, nb, kb),
                (nb * kb, kb, 1),
                torch.float32,
            )
        layout = (
            (B, M, K),
            (M * K, K, 1),
            torch.float8_e4m3fn,
            (B, N, K),
            (N * K, K, 1),
            torch.float8_e4m3fn,
            *scales,
            (B,),
            (1,),
            torch.int32,
            ((B, M, N), (M * N, N, 1), torch.bfloat16) if out_given else None,
        )
        _remember_plan(key, layout, _LAYOUTS)
    return layout


def _admit_public_call(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    *,
    scale_granularity_mnk: tuple[int, int, int],
    out: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
) -> bool:
    """:func:`check_batch_deepgemm_fp8_nt_groupwise_cake` for the public API.

    Inside a public call (opened by :func:`_open_public_call`) the resolved
    problem is left, once, for the run of the same call
    (:func:`_run_public_call`) with the operands' fingerprint: their addresses
    and the layout (shapes, strides, dtypes) the check admitted.  So the call
    resolves its problem once.  Any other call of the check (outside a public
    call, or a second one in the same call) leaves nothing and drops what was
    left."""
    state = _PUBLIC
    opened = state.entry is _OPEN
    state.entry = None
    problem = _resolve(
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
    if opened:
        state.entry = (
            _operand_addresses(a, b, a_scale, b_scale, masked_m, out),
            _admitted_layout(problem, out is not None),
            expected_m,
            problem,
        )
    return True


def _take_public_admission() -> Optional[tuple[Any, ...]]:
    """What this public call's admission check left (taken once), or ``None``
    (``skip_check=True``, or no check ran in this call); closes the call.  The
    public API takes it before anything else of its run, so nothing is left
    behind when the run then fails.

    An entry remains only when a public call fails after its ``bypass`` hook
    opened it and before its body took it: ``_OPEN`` when the call fails
    before its admission check (argument binding, backend or compute
    capability rejected), or this call's admission when it fails between the
    check and the body.  Either is harmless: only this function reads the
    entry, and only the body of a public ``backend="cake"`` call calls it,
    after that call's hook replaced any entry with ``_OPEN`` (the hook runs
    on every call, ``skip_check=True`` too).  A check outside a public call
    after a leftover ``_OPEN`` leaves an admission the same way, replaced by
    the next public call's hook.  :func:`run_batch_deepgemm_fp8_nt_groupwise`
    never reads the entry."""
    state = _PUBLIC
    entry = state.entry
    state.entry = None
    return None if entry is _OPEN else entry


def _run_public_call(
    admitted: Optional[tuple[Any, ...]],
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    expected_m: int,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    """The run of a public API call: :func:`run_batch_deepgemm_fp8_nt_groupwise`
    reusing the problem ``admitted`` (this call's admission) resolved when every
    operand (``out`` too when the check had it) still has the checked address
    and the shape, strides and dtype the check admitted, and ``expected_m`` is
    the checked int (``out`` the check did not see is validated here);
    otherwise the problem is resolved again, as a fresh call would."""
    problem: Optional[_Problem] = None
    if admitted is not None:
        addresses, layout, checked_m, resolved = admitted
        checked_out = out if addresses[-1] is not None else None
        if (
            type(expected_m) is int
            and expected_m == checked_m
            and addresses
            == _operand_addresses(a, b, a_scale, b_scale, masked_m, checked_out)
            and layout == _operand_layout(a, b, a_scale, b_scale, masked_m, checked_out)
        ):
            if checked_out is None:
                # The check saw no output: the public API allocated this one since.
                _require_tensor(
                    out,
                    "out",
                    shape=(resolved.B, resolved.M, resolved.N),
                    dtype=torch.bfloat16,
                    device=resolved.device,
                )
            problem = resolved
    if problem is None:
        problem = _resolve(
            a,
            b,
            a_scale,
            b_scale,
            masked_m,
            expected_m,
            out=out,
            out_dtype=None,
            scale_granularity_mnk=(1, BLOCK, BLOCK),
        )
    return _run(problem, a, b, a_scale, b_scale, masked_m, out)


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
    do); prepare one launch per concurrent stream.  Counters the launch allocates
    at preparation outside capture are zeroed before preparation returns, so the
    first launch may run on any stream, and are marked used on every stream a
    launch runs on: dropping the prepared launch right after ``launch()`` is
    safe, their memory is reused only after those launches are done.  A twin launch prepared under CUDA graph capture without
    caller-owned counters zeroes its counters inside that capture, so it belongs
    to that capture: ``launch()`` raises outside capture and in any other capture.
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
    # True when the launch's counters were zeroed inside a graph capture, and
    # CUDA's id of that capture (0 when it could not be read).
    capture_only: bool = False
    capture_id: int = 0
    # Raw handles of the streams the launch's own counters are marked used on
    # (``Tensor.record_stream``), so their memory outlives this object's launches
    # on every stream; ``None`` when the launch does not own its counters.
    _counter_streams: Optional[set[int]] = None

    def launch(self) -> torch.Tensor:
        index = _device_index(self.out.device)
        with torch.cuda.device(index), _ffi_stream(index):
            if self.capture_only and (
                not cake_locality._stream_capturing(index)
                or cake_locality._capture_id(index) != self.capture_id
            ):
                raise RuntimeError(
                    "this locality-domain launch was prepared under CUDA graph "
                    "capture, which zeroes its claim counters: launch it only in "
                    "that capture, or prepare it outside capture"
                )
            if self._counter_streams is not None:
                handle = int(torch._C._cuda_getCurrentRawStream(index))
                if handle not in self._counter_streams:
                    stream = torch.cuda.current_stream(index)
                    for name in DOMAIN_COUNTERS:
                        self.workspace[name].record_stream(stream)
                    self._counter_streams.add(handle)
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
            pair = cake_locality.launch_counters(
                PAIR_COUNTER_WORDS, _device_index(device)
            )
        elif cake_locality._stream_capturing(_device_index(device)):
            # Locality-domain memory is not allocated under capture: zeroed
            # counters from the capture's pool (same bits, unhomed claims).
            pair = tuple(
                torch.zeros(
                    plan.workspaces[name], dtype=WORKSPACE_DTYPES[name], device=device
                )
                for name in DOMAIN_COUNTERS
            )
        else:
            # Zeroed on a private stream and waited for: the launch may run on
            # any stream.
            pair = cake_locality.counters(PAIR_COUNTER_WORDS, device, settle=True)
        workspace.update(zip(DOMAIN_COUNTERS, pair, strict=True))
    return workspace


def _byte_extent(tensor: torch.Tensor) -> tuple[int, int]:
    """``[start, end)``: the device bytes a tensor view can address (empty when it
    has no elements), from its shape and strides, not its whole storage."""
    start = int(tensor.data_ptr())
    if tensor.numel() == 0:
        return start, start
    last = sum(
        (size - 1) * stride
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    return start, start + (last + 1) * tensor.element_size()


def _check_workspace(
    plan: RoutePlan,
    workspace: dict[str, torch.Tensor],
    device: torch.device,
    *,
    operands: dict[str, torch.Tensor],
) -> None:
    """Validate caller-owned ``workspace`` tensors: the keys, shapes, dtypes and
    alignment the route binds, two dies' counters that share no byte, and no
    workspace byte inside the exact extent of an ``operands`` tensor (a kernel
    writes its workspaces while it reads the operands, e.g. its row counts from
    ``masked_m``). Disjoint views of one buffer are accepted."""
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
    extents = {name: _byte_extent(tensor) for name, tensor in operands.items()}
    for key in plan.workspaces:
        start, end = _byte_extent(workspace[key])
        for name, (first, last) in extents.items():
            if start < last and first < end:
                raise ValueError(
                    f"workspace {key} overlaps {name}; a workspace needs memory "
                    "of its own"
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


def _serving_arguments(
    problem: _Problem,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    out: torch.Tensor,
    counters0: Optional[torch.Tensor],
    counters1: Optional[torch.Tensor],
) -> tuple[Any, tuple[Any, ...]]:
    """``(FFI entry, arguments)`` of the serving program's launch (zero-copy
    scale views; the twin's claim counters ``counters0`` / ``counters1``)."""
    (stage,) = problem.plan.stages
    entry, fixed, getter = _serving_launch(problem, stage)
    cols = problem.K // PACKED_K_GRANULE
    # The logical [B, MN, K/512] int32 scales live in MN-major [B, K/512, MN] storage;
    # reinterpret that storage as the contiguous 2-D surface the scale tensor maps read.
    sfa = a_scale.as_strided((problem.B * cols, problem.M), (problem.M, 1)).view(
        torch.uint32
    )
    sfb = b_scale.as_strided((problem.B * cols, problem.N), (problem.N, 1)).view(
        torch.uint32
    )
    values = (
        a.view(torch.uint8),
        b.view(torch.uint8),
        out,
        sfa,
        sfb,
        masked_m,
        counters0,
        counters1,
    )
    return entry, getter(values + fixed)


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
        :func:`flashinfer.cake_locality.counters`).  No workspace tensor may share
        a byte with ``a``, ``b``, the scales, ``masked_m`` or ``out``
        (``ValueError``).  Allocated at preparation if omitted: a prepared launch owns its counters (the public API's call
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
    capture_id = 0
    counter_streams: Optional[set[int]] = None
    if workspace is None:
        index = _device_index(device)
        owns_counters = not launch_now and DOMAIN_COUNTERS[0] in plan.workspaces
        capture_only = owns_counters and cake_locality._stream_capturing(index)
        if capture_only:
            read = cake_locality._capture_id(index)
            if not read:
                raise RuntimeError(
                    "cannot read the id of the current CUDA graph capture (CUDA "
                    "Python runtime bindings are missing): pass caller-owned "
                    "counters as workspace, or prepare outside capture"
                )
            capture_id = read
        workspace = _allocate_workspace(plan, device, launch_now=launch_now)
        if owns_counters and not capture_only:
            # counters(settle=True) marked them used on this stream.
            counter_streams = {int(torch._C._cuda_getCurrentRawStream(index))}
    else:
        _check_workspace(
            plan,
            workspace,
            device,
            operands={
                "a": a,
                "b": b,
                "a_scale": a_scale,
                "b_scale": b_scale,
                "masked_m": masked_m,
                "out": out,
            },
        )
    launches = _bind_launches(problem, a, b, a_scale, b_scale, masked_m, out, workspace)
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
        capture_id=capture_id,
        _counter_streams=counter_streams,
    )


def _bind_launches(
    problem: _Problem,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    out: torch.Tensor,
    workspace: dict[str, torch.Tensor],
) -> list[tuple[Callable[..., Any], tuple[Any, ...]]]:
    """``(FFI entry, arguments)`` of each kernel of the problem's route."""
    plan = problem.plan
    launches = []
    if problem.packed_scales:
        launches.append(
            _serving_arguments(
                problem,
                a,
                b,
                a_scale,
                b_scale,
                masked_m,
                out,
                workspace.get(DOMAIN_COUNTERS[0]),
                workspace.get(DOMAIN_COUNTERS[1]),
            )
        )
    else:
        bindings = _family_bindings(
            problem, a, b, a_scale, b_scale, masked_m, out, workspace
        )
        for stage in plan.stages:
            entry, arg_plan = _program_entry(problem.arch, stage.program)
            positional = tuple(
                bindings[item] if isinstance(item, str) else int(item)
                for item in stage.args
            )
            launches.append(
                (
                    entry,
                    bind_arguments(
                        arg_plan, positional, stage.grid, program=stage.program
                    ),
                )
            )
    return launches


def localize_weights(b: torch.Tensor, g_cut: Optional[int] = None) -> torch.Tensor:
    """A copy of the expert weights ``b`` (``(B, N, K)``, contiguous) homed by groups
    on the two dies of a dual-die SM107a device: groups ``[0, g_cut)`` in die 0's
    memory, the rest in die 1's, in one virtual address range.

    ``g_cut`` defaults to the SM-proportional share of die 0,
    ``split_units(B, counts)[0]`` over the device's probed SM table.  Passing the
    copy as ``b`` makes every packed-scale serving problem on it run the
    locality-domain twin of its serving program; the twin's result is the same
    bits for any ``g_cut``.

    Returns ``b`` itself, unhomed, when the call is under CUDA graph capture, or
    when homing could make a call slower: ``b`` is not float8_e4m3fn weights of a
    packed-scale serving shape (``(N, K)`` in ``PACKED_NK``, ``B`` in
    ``PACKED_GROUPS``) on an SM107a device, no die class of the device's SM count
    routes every ``expected_m`` of that shape to a twin, the device has no SM
    table (:func:`flashinfer.cake_locality.sm_table` is ``None``), its memory
    does not run at its full clock
    (:func:`flashinfer.cake_locality.memory_at_full_clock` is ``False``), some
    ``expected_m`` of that shape selects a serving program without a twin listed
    in the cell of the device's dies (:func:`domain_serving_routed`), or the cut
    is not a multiple of the allocation granularity of locality-domain memory.
    The checks that need no device work come first: a refused shape is returned
    before the SM table is probed or anything is built or synchronized.  Raises
    ``ValueError`` for a ``g_cut`` outside ``[0, B]``.

    With the cells listed for 212-SM devices, only ``B=64`` weights of
    ``(N, K) = (4096, 7168)`` are homed: at every ``expected_m`` their serving
    program has a listed twin, including the ``expected_m >= 16`` cell listed for
    homed weights only (see ``DOMAIN_SERVING_CELLS``).  ``B=32`` weights of
    ``(4096, 7168)`` and ``B=32`` and ``B=64`` weights of ``(7168, 2048)`` are
    returned unhomed: their ``expected_m <= 1`` serving program has no twin.

    Homing weights also fills the device's reserve of pre-zeroed claim counter
    pairs for captured public calls
    (:func:`flashinfer.cake_locality.reserve_launch_counters`), so a CUDA graph
    captured before any eager call holds no memset node; a failure there does
    not fail this call.

    Homed weights are meant for ``backend="cake"``.  ``backend="deepgemm"``
    reads them correctly but more slowly: on 212-SM devices it took 1.39-1.50x
    the time of plain weights at ``expected_m`` 2-64.  The twin's gain comes
    from die-local memory reads and was measured with the memory at its full
    clock.  On a device whose memory ran at a lower clock (driver 620.43, 3186
    of 4752 MHz) the twin route took 1.01-1.08x the time of plain weights at
    ``expected_m <= 64`` and 0.92-0.95x at 512 and 4096 with plain weights on
    the reserved grid, and 1.01-1.14x at every measured ``expected_m`` (1 to
    4096) with plain weights on the full grid from ``expected_m`` 36.  So this
    returns ``b`` unhomed on a device whose memory clock NVML reads below
    :data:`flashinfer.cake_locality.MEMORY_CLOCK_FRACTION` of its maximum, or
    cannot read (:func:`flashinfer.cake_locality.memory_at_full_clock`), and
    weights homed there take the whole-GPU route.

    On 212-SM devices, plain weights run their serving program on the full grid
    from ``expected_m`` 36; the homed twin is unchanged.  For ``B=64`` weights of
    ``(4096, 7168)``, plain weights then took 0.95x the time of homed weights at
    ``expected_m`` 64 and 512 and 0.97x at 4096.  Homed weights stay faster at
    smaller ``expected_m``: plain weights took 1.04x their time at 36 and
    1.10-1.35x at 1-32.

    Call it outside CUDA graph capture (under capture it returns ``b``).  The
    memory is freed with the returned tensor; a CUDA graph that reads the tensor
    after it is freed faults on replay, so keep it alive as long as such a graph.
    For an ``nn.Parameter`` ``b`` (or a plain Python subclass of it) the homed
    copy is returned as a parameter of the same class with the same
    ``requires_grad`` and a shallow copy of ``b.__dict__`` (the instance
    attributes a framework may have attached, such as a weight loader); the
    subclass's constructor is not called.  A tensor subclass with its own
    ``__torch_dispatch__`` (a wrapper subclass whose data lives in inner
    tensors), a parameter or not, is returned unhomed.  Every alias of the
    whole returned tensor is homed as well (an ``nn.Parameter`` over it,
    ``detach()``, ``.data``, a view of the same shape, and a tensor given it
    as ``.data``); a view of part of it is not: pass the whole tensor.
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
    if type(b).__torch_dispatch__ is not torch.Tensor.__torch_dispatch__:
        # A __torch_dispatch__ (wrapper) subclass keeps its data in inner
        # tensors, which a homed copy of the outer tensor would not carry.
        return b
    index = _device_index(b.device)
    if cake_locality._stream_capturing(index):
        return b
    arch, sm_count = _device_info(index)
    if arch != DOMAIN_ARCH:
        return b
    # Before any probe, build or synchronization: some die class must route
    # every expected_m of this shape.
    if not any(
        _domain_class_routed(arch, sm_count, die, n, k, groups)
        for die in ("even", "uneven")
    ):
        return b
    table = cake_locality.sm_table(index)
    if table is None or not cake_locality.memory_at_full_clock(index):
        return b
    if not domain_serving_routed(arch, sm_count, table.counts, n, k, groups):
        return b
    if g_cut is None:
        g_cut = cake_locality.split_units(groups, table.counts)[0]
    group_bytes = n * k * b.element_size()
    cut = int(g_cut) * group_bytes
    if 0 < cut < groups * group_bytes and cut % cake_locality.granularity(index):
        return b
    homed = cake_locality.split_empty(tuple(b.shape), b.dtype, cut, index)
    with torch.no_grad():
        homed.copy_(b)
    # A failure here does not fail the call: captured calls then zero their own
    # counters until an eager call refills the reserve.
    with contextlib.suppress(Exception):
        cake_locality.reserve_launch_counters(PAIR_COUNTER_WORDS, index)
    if isinstance(b, torch.nn.Parameter):
        # The class of ``b`` (a subclass too) without calling its constructor,
        # as nn.Parameter itself builds one, and a shallow copy of its instance
        # attributes (e.g. a weight loader a framework attached to it).
        parameter = torch.Tensor._make_subclass(type(b), homed, b.requires_grad)
        parameter.__dict__.update(b.__dict__)
        return parameter
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

    The locality-domain twin's claim counters are not allocated or zeroed per
    call in steady state: every launch leaves its pair zero, so the call takes an
    already zero pair from :func:`flashinfer.cake_locality.launch_counters`
    (die-homed when FlashInfer's locality-domain pools exist, ordinary zeroed
    tensors otherwise).  Eager calls on one stream reuse that stream's pair
    (their launches are ordered), made at the stream's first call; on the
    default stream and the per-thread stream handles the pair is kept per
    thread.  A call under CUDA graph capture takes a pre-zeroed pair of its own
    from a reserve filled by :func:`localize_weights` and topped up by eager
    calls, used by no eager call and no other capture, so the graph holds no
    memset node and may replay on any stream while eager calls run; when the
    reserve is empty, the captured call zeroes its pair inside the graph.  The
    same graph must not replay on two streams at once (its launch would share
    its pair with itself), as for any workspace a graph holds; a stream wrapped
    by ``torch.cuda.ExternalStream`` must not be destroyed while calls on it are
    in flight.  A prepared launch (:func:`prepare_batch_deepgemm_fp8_nt_groupwise`)
    keeps its own pair.

    Every call validates its tensors (the public API's call validates them once,
    in its admission check, and its run reuses that check's problem only for
    operands of the same fingerprint); the route decision itself is remembered
    per problem geometry, device and homing, so a call makes no other per-call
    decision.
    """
    problem = _resolve(
        a,
        b,
        a_scale,
        b_scale,
        masked_m,
        expected_m,
        out=out,
        out_dtype=None,
        scale_granularity_mnk=(1, BLOCK, BLOCK),
    )
    return _run(problem, a, b, a_scale, b_scale, masked_m, out)


def _run(
    problem: _Problem,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Submit the launches of the resolved ``problem`` into ``out``."""
    if problem.packed_scales:
        # One serving launch; the twin takes its stream's reused counter pair.
        counters: tuple[Any, Any] = (None, None)
        if problem.plan.route == DOMAIN_SERVING_ROUTE:
            counters = cake_locality.launch_counters(
                PAIR_COUNTER_WORDS, _device_index(problem.device)
            )
        launches: Any = (
            _serving_arguments(
                problem, a, b, a_scale, b_scale, masked_m, out, *counters
            ),
        )
    else:
        workspace = _allocate_workspace(problem.plan, problem.device, launch_now=True)
        launches = _bind_launches(
            problem, a, b, a_scale, b_scale, masked_m, out, workspace
        )
    _run_launches(problem.device, launches)
    return out


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
