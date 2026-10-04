# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Prepared masked batch FP8 GEMM launches of the generated Cake programs (SM100a, SM103a).

The same mathematical contract as :func:`flashinfer.gemm.batch_deepgemm_fp8_nt_groupwise`:
``out[g, :masked_m[g], :] = (a[g] . b[g]^T)`` with FP8 E4M3 operands, per-row
128-wide K-block A scales, 128x128 B block scales, FP32 accumulation with the
block scales applied in DeepGEMM's K order, BF16 output; rows at or beyond
``masked_m[g]`` are not written.

The host side selects the route with the Cake dispatcher's ordered first-match
chain rendered per ``(architecture, SM count)`` into
:mod:`flashinfer.jit.gemm.cake_batch_deepgemm_fp8`, binds the positional argument
plan of each registered program and launches the one or two kernels of the route
back to back on PyTorch's current stream.  Tensor maps are encoded by the binding
and passed to the kernel by value, so a prepared launch owns no descriptor
storage; the only device memory a route may need is the packed-scale workspace
of the swap-AB routes, allocated at preparation (or supplied by the caller) and
never inside ``launch()``.  Every launch, including the first, may be captured
into a CUDA graph.

Selection reads host scalars and tensor metadata only; it never reads a CUDA
tensor's contents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import tvm_ffi

from ..jit.gemm.cake_batch_deepgemm_fp8 import (
    ARG_PLANS,
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
# The zero-copy packed-scale serving programs are specialised on (N, K, B).
PACKED_NK = frozenset({(4096, 7168), (7168, 2048)})
PACKED_GROUPS = frozenset({32, 64})
PACKED_K_GRANULE = 512
SERVING_ROUTE = "seed_serving_m32_packed"
WORKSPACE_DTYPES = {
    "sfa_packed": torch.int32,
    "sfb_packed": torch.int32,
    "sfa_m32_packed": torch.uint32,
    "sfb_m32_packed": torch.uint32,
}


def serving_config(
    m: int, k: int, expected_m: int, sm_count: int, arch: str
) -> tuple[int, int, str, int, int]:
    """``(num_stages, num_epi_stages, a_cache_hint, cta_reserve, n_l2_group)`` of the serving route.

    Port of the Cake serving seed's configuration rule: the choice depends only on
    caller-visible routing metadata (``K``, ``expected_m``, device), never on a
    trace quantile, so every replay of one profile runs one compiled program.
    """
    if k == 7168:
        if expected_m <= 1:
            return 6, 1, "evict_normal", 0, 1
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


def serving_plan(
    arch: str, sm_count: int, groups: int, m: int, n: int, k: int, expected_m: int
) -> RoutePlan:
    """The packed-scale serving route: one specialised program, grid from the SM count."""
    num_stages, num_epi_stages, a_cache_hint, cta_reserve, n_l2_group = serving_config(
        m, k, expected_m, sm_count, arch
    )
    key = serving_program_key(
        n, k, groups, num_stages, num_epi_stages, a_cache_hint, n_l2_group
    )
    program = SERVING_PROGRAMS.get(key)
    if program is None:
        raise NotImplementedError(
            f"no generated packed-scale serving program for N={n}, K={k}, B={groups}, "
            f"expected_m={expected_m} (configuration {key})"
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
            "the Cake batch DeepGEMM FP8 backend requires an SM100a or SM103a device "
            f"(compute capability {sorted(SUPPORTED_COMPUTE_CAPABILITIES)}); "
            f"got {torch.cuda.get_device_capability(index)}"
        )
    sm_count = device_sm_count(index)
    if route_chain(arch, sm_count) is None:
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
    if (N, K) not in SUPPORTED_NK:
        raise ValueError(
            f"the Cake batch DeepGEMM FP8 backend owns (N, K) in {sorted(SUPPORTED_NK)}; got ({N}, {K})"
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
    if packed:
        if a_scale.dtype != torch.int32 or b_scale.dtype != torch.int32:
            raise ValueError(
                "packed UE8M0 scales require both a_scale and b_scale to be int32"
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
        plan = serving_plan(arch, sm_count, B, M, N, K, expected_m)
    else:
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
    return _Problem(arch, sm_count, device, B, M, N, K, int(expected_m), packed, plan)


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
    """True when generated programs and a route chain are registered for ``device``."""
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
        elif kind in ("tma_buffer", "buffer", "parameter") and name in bindings:
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

    def launch(self) -> torch.Tensor:
        with torch.cuda.device(self.out.device), tvm_ffi.use_torch_stream():
            for entry, arguments in self._launches:
                entry(*arguments)
        return self.out

    __call__ = launch

    @property
    def num_kernels(self) -> int:
        return len(self._launches)


def _allocate_workspace(
    plan: RoutePlan, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: torch.empty(shape, dtype=WORKSPACE_DTYPES[key], device=device)
        for key, shape in plan.workspaces.items()
    }


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
    return {
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
        Caller-owned packed-scale workspace of the selected route (see
        ``RoutePlan.workspaces``); allocated at preparation if omitted.

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
        scale_granularity_mnk=scale_granularity_mnk,
    )
    device = problem.device
    if out is None:
        out = torch.empty(
            (problem.B, problem.M, problem.N), dtype=torch.bfloat16, device=device
        )
    plan = problem.plan
    if workspace is None:
        workspace = _allocate_workspace(plan, device)
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
        bindings = _serving_bindings(problem, a, b, a_scale, b_scale, masked_m, out)
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
    )


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
    ``torch.empty`` released when the call returns: nothing is cached across calls
    and the call may be captured into a CUDA graph.
    """
    return prepare_batch_deepgemm_fp8_nt_groupwise(
        a, b, a_scale, b_scale, masked_m, expected_m, out=out
    ).launch()


__all__ = [
    "PACKED_GROUPS",
    "PACKED_NK",
    "PreparedBatchDeepGemmFp8NtGroupwise",
    "SUPPORTED_NK",
    "WORKSPACE_DTYPES",
    "bind_arguments",
    "bind_named_arguments",
    "check_batch_deepgemm_fp8_nt_groupwise_cake",
    "is_batch_deepgemm_fp8_nt_groupwise_cake_available",
    "prepare_batch_deepgemm_fp8_nt_groupwise",
    "run_batch_deepgemm_fp8_nt_groupwise",
    "select_route",
    "serving_config",
    "serving_plan",
    "serving_program_key",
]
