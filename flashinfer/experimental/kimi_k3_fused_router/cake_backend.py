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

Experimental generated-program backend of the Kimi-K3 fused MoE router.

One launch routes FP32 gate logits for 896 experts (sigmoid gate, top-16
selection on ``sigmoid(logit) + bias``, weights renormalized over the selected
sigmoid scores) and writes the expert-aligned route plan (``sorted_token_ids``,
``expert_ids``, ``num_tokens_post_padded``, per-expert counts / offsets /
scatter offsets) consumed by grouped MoE GEMMs.  The program is a family of
kernels, one dispatch arm per ``(num_tokens, block_m)`` cell of a
per-architecture table measured at the powers of two from 1 to 8192 tokens;
every other token count up to 8192 is served by the arm of the next measured
cell up
(the arms take ``num_tokens`` at runtime; only arm LC, one cluster of exactly
``num_tokens`` CTAs for 1, 2, 4, 8 or 16 tokens, is bound to its row count).
Most arms launch as a cooperative persistent grid bounded by the device's SM
count (arms L and LP -- the same one-join plan-builder kernel family for up to
128 tokens, LP being the variant that loads the bias and the first row's
logits into registers before its prologue barrier -- launch at least their
128 plan-owner CTAs; arms Q4S and Q4SP -- the same 4-CTA-cluster kernel family
at four CTAs per SM, Q4SP being the variant that prefetches the next row's
logits into registers -- are bounded by the driver's co-resident cluster
capacity; arm GW, the warp-per-row two-join kernel for the largest batches,
uses a per-architecture CTAs-per-SM bound); arm LC launches one non-cooperative
cluster of ``num_tokens`` CTAs (the 16-CTA cluster is above the portable
maximum of 8 and the generated program opts into it with the non-portable
cluster-size attribute).  Nothing is planned on the host and nothing is
allocated at launch, so a prepared runner is CUDA Graph safe.  Device facts
(compute capability, SM count, cluster occupancy) are queried once per device
and program.  See ``README.md`` in this package.
"""

from __future__ import annotations

import bisect
import functools
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional

import torch
import tvm_ffi

from .cake_jit import (
    MODULES,
    kernel_key,
    load_kimi_k3_fused_router_module,
    program_for,
    queries_occupancy,
    registered_keys,
)

NUM_EXPERTS = 896
TOP_K = 16
BLOCK_M_VALUES = (8, 16)
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
# Largest token count the kernels serve: every arm reconstructs the route plan
# from per-expert token bitmaps sized for 8192 rows (the largest measured cell).
MAX_NUM_TOKENS = 8192

# Kernel geometry shared by every arm: 224 threads = 7 warps, each thread owns
# four experts; the plan-building arms use 896 / 7 = 128 owner CTAs.
THREADS = 224
NUM_WARPS = THREADS // 32
OWNER_CTAS = NUM_EXPERTS // NUM_WARPS
# Arms L / LP: one-join plan builder; admits at most this many tokens and
# launches at least ARM_L_MIN_GRID CTAs (the plan owners) whatever num_tokens is.
ARM_L_MAX_TOKENS = 128
ARM_L_MIN_GRID = 128
# Arm LP: the L body with the bias and first-row logits loads issued before the
# prologue barrier (a register prefetch), a second kernel of the same family;
# identical outputs, thread count, admission guard and grid rule.
ARM_L_FAMILY = ("L", "LP")
# Arm LC: one kernel per token count, launched as a single cluster of num_tokens
# CTAs (a single CTA for one token).  The 16-token kernel is a 16-CTA cluster,
# above the portable maximum of 8: the generated program sets the non-portable
# cluster-size attribute on that kernel before the launch, and preparation
# requires the driver to admit at least one such cluster on the device.
ARM_LC_TOKENS = (1, 2, 4, 8, 16)
# Arm GW: persistent grid of CTAs-per-SM x SM count (launch bounds of the kernel,
# __launch_bounds__(224, 4) on both architectures).
ARM_GW_CTAS_PER_SM = {(10, 0): 4, (10, 3): 4}
# Arm M: one-join bitmap plan builder.  Its owner scratch is sized per
# architecture: 256 rows on SM100 (the kernel measured at the 256-token cell),
# 2048 rows on SM103.
ARM_M_MAX_TOKENS = {(10, 0): 256, (10, 3): 2048}
# Arm Q4S: 4-CTA clusters, kernel compiled with __launch_bounds__(224, 4); grid =
# min(num_tokens, 4 x SM count, co-resident cluster capacity) in whole clusters.
ARM_Q4S_MAX_TOKENS = 2048
ARM_Q4S_CLUSTER = 4
ARM_Q4S_CTAS_PER_SM = 4
# Arm Q4SP: the Q4S body with a register prefetch of the next row's logits, a
# second kernel of the same family; identical cluster shape, launch bounds,
# admission guards and grid rule.
ARM_Q4S_FAMILY = ("Q4S", "Q4SP")

# Exact keyword set of the generated program's ``run`` entry (bound by the
# export's argument plan); ``grid`` is expanded to ``grid_x/y/z``.
MAIN_KWARGS = (
    "logits",
    "bias",
    "topk_weights",
    "topk_ids",
    "sorted_token_ids",
    "expert_ids",
    "num_tokens_post_padded",
    "expert_counts",
    "expert_offsets",
    "expert_scatter_offsets",
    "M",
    "grid",
)

# Measured dispatch cells, keyed by (num_tokens, block_m): the 28 shapes the
# generated program was benchmarked at on each architecture.  Both tables cover
# the same cells with the same arms; they differ in exactly two cells of the
# L / LP family (see _SM103_SHAPE_ROUTE below).
_SM100_SHAPE_ROUTE: dict[tuple[int, int], str] = {
    (1, 8): "LC",
    (1, 16): "LC",
    (2, 8): "LC",
    (2, 16): "LC",
    (4, 8): "LC",
    (4, 16): "LC",
    (8, 8): "LC",
    (8, 16): "LC",
    (16, 8): "LC",
    (16, 16): "LC",
    (32, 8): "LP",
    (32, 16): "LP",
    (64, 8): "LP",
    (64, 16): "LP",  # sm_103a: L
    (128, 8): "L",  # sm_103a: LP
    (128, 16): "L",
    (256, 8): "M",
    (256, 16): "M",
    (512, 8): "Q4S",
    (512, 16): "Q4S",
    (1024, 8): "Q4S",
    (1024, 16): "Q4S",
    (2048, 8): "Q4SP",
    (2048, 16): "Q4SP",
    (4096, 8): "GW",
    (4096, 16): "GW",
    (8192, 8): "GW",
    (8192, 16): "GW",
}
# sm_103a: the sm_100a table with the two L / LP cells that differ between the
# architectures -- sm_100a keeps L at (128, 8) and routes (64, 16) to LP;
# sm_103a keeps L at (64, 16) and routes (128, 8) to LP.
_SM103_SHAPE_ROUTE: dict[tuple[int, int], str] = {
    **_SM100_SHAPE_ROUTE,
    (64, 16): "L",
    (128, 8): "LP",
}
SHAPE_ROUTES = {"sm_100a": _SM100_SHAPE_ROUTE, "sm_103a": _SM103_SHAPE_ROUTE}
MEASURED_NUM_TOKENS = tuple(sorted({rows for rows, _ in _SM100_SHAPE_ROUTE}))
# Token counts below the smallest shared-kernel cell that are not an LC row
# count take the arm of this cell.
_SMALLEST_SHARED_CELL = 32


# ---------------------------------------------------------------------------
# Route plan ABI
# ---------------------------------------------------------------------------


class KimiK3RoutePlan(NamedTuple):
    """Device-resident routing outputs and the expert-aligned route plan.

    ``topk_weights`` / ``topk_ids`` are ``[num_tokens, 16]`` (ids ascending
    within a token).  ``sorted_token_ids`` stores flattened
    ``token * 16 + route`` pair indices grouped by expert in ascending pair
    order, every expert segment padded to a multiple of ``block_m`` with the
    sentinel ``num_tokens * 16``; ``expert_ids`` names the expert of each
    ``block_m`` block and ``num_tokens_post_padded`` (device int32 scalar) the
    valid extent of both.  ``expert_counts`` are the per-expert pair counts,
    ``expert_offsets`` the padded prefix sums (897 entries) and
    ``expert_scatter_offsets`` equals ``expert_counts`` after the launch.
    Capacity beyond the valid extent is left untouched.
    """

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor
    expert_counts: torch.Tensor
    expert_offsets: torch.Tensor
    expert_scatter_offsets: torch.Tensor


def max_route_blocks(num_tokens: int, block_m: int) -> int:
    """Worst-case number of ``block_m`` route blocks for ``num_tokens``."""
    pairs = num_tokens * TOP_K
    nonempty = min(NUM_EXPERTS, pairs)
    return nonempty + (pairs - nonempty) // block_m


def allocate_kimi_k3_route_plan(
    num_tokens: int, block_m: int, device: torch.device
) -> KimiK3RoutePlan:
    """Allocate a worst-case-capacity :class:`KimiK3RoutePlan` (no launch)."""
    num_tokens, block_m = _validate_problem(num_tokens, block_m)
    blocks = max_route_blocks(num_tokens, block_m)
    return KimiK3RoutePlan(
        topk_weights=torch.empty(
            (num_tokens, TOP_K), dtype=torch.float32, device=device
        ),
        topk_ids=torch.empty((num_tokens, TOP_K), dtype=torch.int32, device=device),
        sorted_token_ids=torch.empty(
            blocks * block_m, dtype=torch.int32, device=device
        ),
        expert_ids=torch.empty(blocks, dtype=torch.int32, device=device),
        num_tokens_post_padded=torch.empty(1, dtype=torch.int32, device=device),
        expert_counts=torch.empty(NUM_EXPERTS, dtype=torch.int32, device=device),
        expert_offsets=torch.empty(NUM_EXPERTS + 1, dtype=torch.int32, device=device),
        expert_scatter_offsets=torch.empty(
            NUM_EXPERTS, dtype=torch.int32, device=device
        ),
    )


# ---------------------------------------------------------------------------
# Dispatch and launch geometry (pure functions; unit-tested without a GPU)
# ---------------------------------------------------------------------------


def route_cell(num_tokens: int) -> Optional[int]:
    """Measured token count whose arm serves ``num_tokens`` (``None`` above
    ``MAX_NUM_TOKENS``).

    A measured count is its own cell; any other count takes the next measured
    count up, except that counts below 32 which are not an LC row count take
    the 32-token cell (the LC kernels are bound to their exact row count).
    """
    rows = int(num_tokens)
    if rows in ARM_LC_TOKENS:
        return rows
    if rows < _SMALLEST_SHARED_CELL:
        return _SMALLEST_SHARED_CELL
    index = bisect.bisect_left(MEASURED_NUM_TOKENS, rows)
    if index == len(MEASURED_NUM_TOKENS):
        return None
    return MEASURED_NUM_TOKENS[index]


def route_arm(arch: str, num_tokens: int, block_m: int) -> str:
    """Dispatch arm for ``(num_tokens, block_m)`` on ``arch``.

    Measured cells map through the architecture's table; other token counts
    take the arm of :func:`route_cell`; counts above ``MAX_NUM_TOKENS`` are
    rejected (no kernel serves them).
    """
    try:
        table = SHAPE_ROUTES[arch]
    except KeyError:
        raise NotImplementedError(
            f"the generated Kimi-K3 fused router is registered for {sorted(SHAPE_ROUTES)}, not {arch}"
        ) from None
    rows, block_m = int(num_tokens), int(block_m)
    if block_m not in BLOCK_M_VALUES:
        raise ValueError(f"block_m must be one of {BLOCK_M_VALUES}, got {block_m}")
    if rows <= 0:
        raise ValueError("num_tokens must be positive")
    if rows > MAX_NUM_TOKENS:
        raise ValueError(f"num_tokens must be at most {MAX_NUM_TOKENS}, got {rows}")
    cell = route_cell(rows)
    assert cell is not None
    return table[(cell, block_m)]


def persistent_grid_cap(compute_capability: tuple[int, int], sm_count: int) -> int:
    """Persistent grid bound: three CTAs per SM on CC 10.0, four from CC 10.3."""
    ctas_per_sm = 4 if tuple(compute_capability) >= (10, 3) else 3
    return ctas_per_sm * int(sm_count)


def launch_grid(
    arm: str,
    num_tokens: int,
    *,
    compute_capability: tuple[int, int],
    sm_count: int,
    max_active_clusters: Optional[int] = None,
) -> int:
    """``grid_x`` of the launch for ``arm`` on the described device.

    ``max_active_clusters`` is the driver's co-resident cluster capacity for the
    Q4S-family kernels (required for arms Q4S and Q4SP only).
    """
    rows = int(num_tokens)
    major, minor = (int(v) for v in compute_capability)
    compute_capability = (major, minor)
    grid_rows = 8 if rows in (2, 4) else rows
    cap = persistent_grid_cap(compute_capability, sm_count)
    grid_x = max(1, min(grid_rows, cap))
    if arm == "LC":
        if rows not in ARM_LC_TOKENS:
            raise RuntimeError(f"arm LC serves exactly num_tokens in {ARM_LC_TOKENS}")
        return rows
    if arm in ARM_L_FAMILY:
        if rows > ARM_L_MAX_TOKENS:
            raise RuntimeError(f"arm {arm} admits at most {ARM_L_MAX_TOKENS} tokens")
        return max(1, min(max(grid_rows, ARM_L_MIN_GRID), cap))
    if arm == "M":
        try:
            max_tokens = ARM_M_MAX_TOKENS[compute_capability]
        except KeyError:
            raise RuntimeError(
                f"arm M has no owner scratch bound for compute capability {compute_capability}"
            ) from None
        if rows > max_tokens or rows < OWNER_CTAS:
            raise RuntimeError(
                f"arm M admits {OWNER_CTAS} <= num_tokens <= {max_tokens} on compute "
                f"capability {compute_capability}"
            )
        return grid_x
    if arm in ARM_Q4S_FAMILY:
        if rows > ARM_Q4S_MAX_TOKENS or rows < OWNER_CTAS:
            raise RuntimeError(
                f"arm {arm} admits {OWNER_CTAS} <= num_tokens <= {ARM_Q4S_MAX_TOKENS}"
            )
        if max_active_clusters is None:
            raise RuntimeError(
                f"arm {arm} needs the driver's co-resident cluster capacity"
            )
        cluster_cap = int(max_active_clusters) * ARM_Q4S_CLUSTER
        grid_x = min(grid_rows, ARM_Q4S_CTAS_PER_SM * int(sm_count), cluster_cap)
        grid_x = (grid_x // ARM_Q4S_CLUSTER) * ARM_Q4S_CLUSTER
        if grid_x < OWNER_CTAS:
            raise RuntimeError(
                f"arm {arm} needs {OWNER_CTAS} co-resident owner CTAs; the driver admits "
                f"{cluster_cap} clustered CTAs"
            )
        return grid_x
    if arm == "GW":
        try:
            ctas_per_sm = ARM_GW_CTAS_PER_SM[compute_capability]
        except KeyError:
            raise RuntimeError(
                f"arm GW has no launch bound for compute capability {compute_capability}"
            ) from None
        return max(1, min(grid_rows, ctas_per_sm * int(sm_count)))
    raise ValueError(f"unknown dispatch arm {arm!r}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _require_plain_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _validate_problem(num_tokens: Any, block_m: Any) -> tuple[int, int]:
    num_tokens = _require_plain_int("num_tokens", num_tokens)
    block_m = _require_plain_int("block_m", block_m)
    if num_tokens <= 0 or num_tokens > MAX_NUM_TOKENS:
        raise ValueError(
            f"num_tokens must be positive and at most {MAX_NUM_TOKENS}, got {num_tokens}"
        )
    if block_m not in BLOCK_M_VALUES:
        raise ValueError(f"block_m must be one of {BLOCK_M_VALUES}, got {block_m}")
    return num_tokens, block_m


def validate_kimi_k3_fused_router_inputs(
    logits: torch.Tensor, bias: torch.Tensor, block_m: int
) -> tuple[int, int]:
    """Shape / dtype validation of the gate inputs; returns ``(num_tokens, block_m)``.

    Device placement and compute capability are checked at preparation so this
    also runs on host tensors.
    """
    if not isinstance(logits, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise TypeError("logits and bias must be torch tensors")
    if logits.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError("logits and bias must be float32")
    if logits.ndim != 2 or logits.shape[1] != NUM_EXPERTS:
        raise ValueError(f"logits must have shape [num_tokens, {NUM_EXPERTS}]")
    if tuple(bias.shape) != (NUM_EXPERTS,):
        raise ValueError(f"bias must have shape [{NUM_EXPERTS}]")
    if not logits.is_contiguous() or not bias.is_contiguous():
        raise ValueError("logits and bias must be contiguous")
    return _validate_problem(int(logits.shape[0]), block_m)


def _validate_plan(
    plan: KimiK3RoutePlan, *, num_tokens: int, block_m: int, device: torch.device
) -> None:
    if not isinstance(plan, KimiK3RoutePlan):
        raise TypeError(f"plan must be a KimiK3RoutePlan, got {type(plan).__name__}")
    blocks = max_route_blocks(num_tokens, block_m)
    expected = {
        "topk_weights": (torch.float32, (num_tokens, TOP_K), None),
        "topk_ids": (torch.int32, (num_tokens, TOP_K), None),
        "sorted_token_ids": (torch.int32, None, blocks * block_m),
        "expert_ids": (torch.int32, None, blocks),
        "num_tokens_post_padded": (torch.int32, (1,), None),
        "expert_counts": (torch.int32, (NUM_EXPERTS,), None),
        "expert_offsets": (torch.int32, (NUM_EXPERTS + 1,), None),
        "expert_scatter_offsets": (torch.int32, (NUM_EXPERTS,), None),
    }
    for name, (dtype, shape, min_numel) in expected.items():
        tensor = getattr(plan, name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"plan.{name} must be a torch.Tensor")
        if tensor.device != device:
            raise ValueError(f"plan.{name} must be on {device}, got {tensor.device}")
        if tensor.dtype != dtype:
            raise TypeError(f"plan.{name} must have dtype {dtype}, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"plan.{name} must be contiguous")
        if shape is not None and tuple(tensor.shape) != shape:
            raise ValueError(
                f"plan.{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
        if shape is None and tensor.ndim != 1:
            raise ValueError(f"plan.{name} must be one-dimensional")
        if min_numel is not None and tensor.numel() < min_numel:
            raise ValueError(
                f"plan.{name} needs capacity {min_numel} for num_tokens={num_tokens}, "
                f"block_m={block_m}; got {tensor.numel()}"
            )


def _require_nonoverlapping(tensors: dict[str, torch.Tensor]) -> None:
    ranges: list[tuple[int, int, str]] = []
    for name, tensor in tensors.items():
        start = int(tensor.data_ptr())
        end = start + int(tensor.numel()) * int(tensor.element_size())
        if start == end:
            continue
        for previous_start, previous_end, previous_name in ranges:
            if start < previous_end and previous_start < end:
                raise ValueError(f"{name} must not overlap {previous_name}")
        ranges.append((start, end, name))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KimiK3FusedRouterRunner:
    """Launch the prepared router.

    Calling the runner or ``launch()`` submits one kernel on the current
    stream with no CUDA allocation and no host synchronization, writes the
    bound :class:`KimiK3RoutePlan` and returns it.  The kernel reads
    ``logits`` / ``bias`` on device at every launch, so the same runner (or a
    CUDA Graph capturing it) stays valid when the caller writes new values into
    those buffers.  Prepare a new runner when shapes or tensor bindings change.
    """

    module_name: str
    arm: str
    grid_x: int
    plan: KimiK3RoutePlan
    entry: Callable[..., Any]
    arguments: tuple

    def launch(self) -> KimiK3RoutePlan:
        with tvm_ffi.use_torch_stream():
            self.entry(*self.arguments)
        return self.plan

    __call__ = launch


# ---------------------------------------------------------------------------
# Device facts (resolved once per device / program)
# ---------------------------------------------------------------------------


@functools.cache
def _device_facts(device_index: int) -> tuple[tuple[int, int], int]:
    """``(compute_capability, sm_count)`` of CUDA device ``device_index``."""
    properties = torch.cuda.get_device_properties(device_index)
    return (int(properties.major), int(properties.minor)), int(
        properties.multi_processor_count
    )


def _device_arch(device: torch.device) -> str:
    capability = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            "the Kimi-K3 fused router requires compute capability 10.0 or 10.3 "
            f"(got {capability[0]}.{capability[1]})"
        )
    return arch


@functools.cache
def _max_active_clusters(name: str, arch: str, block_m: int, device_index: int) -> int:
    """Co-resident cluster capacity of program ``name`` on ``device_index``.

    The query runs once per (program, architecture, block_m, device); the
    answer depends only on the kernel's resource usage and the device.
    """
    launch = MODULES[name]["launch"]
    block = [int(v) for v in launch["block"]] + [1, 1, 1]
    cluster = [int(v) for v in launch["cluster"]] + [1, 1, 1]
    module = load_kimi_k3_fused_router_module(name, arch, block_m)
    with torch.cuda.device(device_index):
        return int(
            module.max_active_clusters(
                int(device_index),
                block[0],
                block[1],
                block[2],
                cluster[0],
                cluster[1],
                cluster[2],
                int(launch["dynamic_smem_bytes"]),
                bool(launch["cooperative"]),
            )
        )


def generated_program_available(
    device: torch.device,
    num_tokens: Optional[int] = None,
    block_m: Optional[int] = None,
) -> bool:
    """True when this checkout registers the program for ``device``.

    With ``num_tokens`` / ``block_m`` the check names the dispatch arm of that
    shape; without them it asks whether every arm of the device's route table
    is registered.
    """
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))
    if arch is None:
        return False
    registered = registered_keys(arch)
    if num_tokens is None and block_m is None:
        needed = {
            kernel_key(arm, rows if arm == "LC" else None)
            for (rows, _), arm in SHAPE_ROUTES[arch].items()
        }
        return bool(MODULES) and needed <= registered
    if num_tokens is None or block_m is None:
        raise ValueError("pass both num_tokens and block_m or neither")
    arm = route_arm(arch, num_tokens, block_m)
    return kernel_key(arm, int(num_tokens) if arm == "LC" else None) in registered


def prepare_kimi_k3_fused_router(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    block_m: int = 8,
    plan: Optional[KimiK3RoutePlan] = None,
) -> KimiK3FusedRouterRunner:
    """Validate the inputs, select the dispatch arm and bind the launch.

    Every allocation happens here (only the optional ``plan``); the returned
    runner launches with none.  The JIT module of the selected arm is built
    and loaded here, so prepare outside CUDA Graph capture.
    """
    num_tokens, block_m = validate_kimi_k3_fused_router_inputs(logits, bias, block_m)
    device = logits.device
    if device.type != "cuda" or bias.device != device:
        raise ValueError("logits and bias must be on one CUDA device")
    arch = _device_arch(device)
    arm = route_arm(arch, num_tokens, block_m)
    if plan is None:
        plan = allocate_kimi_k3_route_plan(num_tokens, block_m, device)
    else:
        _validate_plan(plan, num_tokens=num_tokens, block_m=block_m, device=device)
    _require_nonoverlapping({"logits": logits, "bias": bias, **plan._asdict()})

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    name = program_for(arch, arm, num_tokens if arm == "LC" else None)
    record = MODULES[name]
    compute_capability, sm_count = _device_facts(device_index)
    with torch.cuda.device(device_index):
        module = load_kimi_k3_fused_router_module(name, arch, block_m)
    max_active_clusters = None
    if queries_occupancy(name):
        max_active_clusters = _max_active_clusters(name, arch, block_m, device_index)
        if arm == "LC" and max_active_clusters < 1:
            raise RuntimeError(
                f"the {num_tokens}-token kernel needs one co-resident {num_tokens}-CTA "
                f"cluster; the driver admits {max_active_clusters} on device {device_index}"
            )
    grid_x = launch_grid(
        arm,
        num_tokens,
        compute_capability=compute_capability,
        sm_count=sm_count,
        max_active_clusters=max_active_clusters,
    )
    main_kwargs = dict(
        logits=logits,
        bias=bias,
        topk_weights=plan.topk_weights,
        topk_ids=plan.topk_ids,
        sorted_token_ids=plan.sorted_token_ids,
        expert_ids=plan.expert_ids,
        num_tokens_post_padded=plan.num_tokens_post_padded,
        expert_counts=plan.expert_counts,
        expert_offsets=plan.expert_offsets,
        expert_scatter_offsets=plan.expert_scatter_offsets,
        M=num_tokens,
        grid=(grid_x, 1, 1),
    )
    assert tuple(main_kwargs) == MAIN_KWARGS
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), main_kwargs["grid"], strict=True))
    arguments = tuple(
        grid[kwarg] if kind == "grid" else main_kwargs[kwarg]
        for kind, kwarg in record["arg_plan"]
    )
    entry = getattr(module, record["ffi_entry"])
    return KimiK3FusedRouterRunner(name, arm, grid_x, plan, entry, arguments)


def kimi_k3_fused_router(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    block_m: int = 8,
    plan: Optional[KimiK3RoutePlan] = None,
) -> KimiK3RoutePlan:
    """Route ``logits`` and build the expert-aligned plan in one launch."""
    return prepare_kimi_k3_fused_router(logits, bias, block_m=block_m, plan=plan)()
