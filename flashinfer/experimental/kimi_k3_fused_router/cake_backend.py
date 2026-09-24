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

Generated-program backend: Kimi-K3 fused MoE router (SM100 / SM103).

One launch routes FP32 gate logits for 896 experts (sigmoid gate, top-16
selection on ``sigmoid(logit) + bias``, weights renormalized over the selected
sigmoid scores) and writes the expert-aligned route plan (``sorted_token_ids``,
``expert_ids``, ``num_tokens_post_padded``, per-expert counts / offsets /
scatter offsets) consumed by grouped MoE GEMMs.  The program is a family of
kernels: one dispatch arm per exact ``(num_tokens, block_m)`` shape of the
routed set, selected by a per-architecture table, each launched as a
persistent grid bounded by the device's SM count (cooperative launch; arm Q
additionally uses 4-CTA clusters and is bounded by the driver's co-resident
cluster capacity).  Nothing is planned on the host and nothing is allocated
at launch, so a prepared runner is CUDA Graph safe.  See ``README.md`` in
this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional

import torch
import tvm_ffi

from .cake_jit import (
    MODULES,
    load_kimi_k3_fused_router_module,
    registered_programs,
    select_module,
    uses_cluster_launch,
)

NUM_EXPERTS = 896
TOP_K = 16
BLOCK_M_VALUES = (8, 16)
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
_INT32_MAX = 2**31 - 1

# Kernel geometry shared by every arm: 224 threads = 7 warps, each thread owns
# four experts; the plan-building arms use 896 / 7 = 128 owner CTAs.
THREADS = 224
NUM_WARPS = THREADS // 32
OWNER_CTAS = NUM_EXPERTS // NUM_WARPS
ARM_L_MAX_TOKENS = 512
ARM_M_MAX_TOKENS = 2048
ARM_Q_MAX_TOKENS = 2048
ARM_Q_CLUSTER = 4
ARM_M4S_CTAS_PER_SM = 4
ARM_M4S_OWNER_STRIDE = 4

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

# Per-shape dispatch arm, keyed by (num_tokens, block_m).  Both tables cover
# the same 28 shapes; SM100 (B200) serves num_tokens = 512 with arm M4S where
# SM103 (B300) already fits arm Q.
_SM100_SHAPE_ROUTE: dict[tuple[int, int], str] = {
    (1, 8): "A",
    (1, 16): "A",
    (2, 8): "L",
    (2, 16): "L",
    (4, 8): "L",
    (4, 16): "L",
    (8, 8): "L",
    (8, 16): "L",
    (16, 8): "L",
    (16, 16): "L",
    (32, 8): "L",
    (32, 16): "L",
    (64, 8): "L",
    (64, 16): "L",
    (128, 8): "L",
    (128, 16): "L",
    (256, 8): "M",
    (256, 16): "M",
    (512, 8): "M4S",
    (512, 16): "M4S",
    (1024, 8): "Q",
    (1024, 16): "Q",
    (2048, 8): "Q",
    (2048, 16): "Q",
    (4096, 8): "G",
    (4096, 16): "G",
    (8192, 8): "G",
    (8192, 16): "G",
}
_SM103_SHAPE_ROUTE: dict[tuple[int, int], str] = {
    **{key: arm for key, arm in _SM100_SHAPE_ROUTE.items() if arm != "M4S"},
    (512, 8): "Q",
    (512, 16): "Q",
}
SHAPE_ROUTES = {"sm_100a": _SM100_SHAPE_ROUTE, "sm_103a": _SM103_SHAPE_ROUTE}
SUPPORTED_NUM_TOKENS = tuple(sorted({rows for rows, _ in _SM100_SHAPE_ROUTE}))


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


def route_arm(arch: str, num_tokens: int, block_m: int) -> str:
    """Dispatch arm for an exact ``(num_tokens, block_m)`` shape on ``arch``."""
    try:
        return SHAPE_ROUTES[arch][(int(num_tokens), int(block_m))]
    except KeyError:
        raise NotImplementedError(
            "the generated Kimi-K3 fused router serves exactly num_tokens in "
            f"{SUPPORTED_NUM_TOKENS} with block_m in {BLOCK_M_VALUES}; got "
            f"num_tokens={num_tokens}, block_m={block_m} on {arch}"
        ) from None


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
    """``grid_x`` of the persistent launch for ``arm`` on the described device.

    ``max_active_clusters`` is the driver's co-resident cluster capacity for the
    arm-Q kernel (required for arm Q only).
    """
    rows = int(num_tokens)
    grid_rows = 8 if rows in (2, 4) else rows
    grid_x = max(1, min(grid_rows, persistent_grid_cap(compute_capability, sm_count)))
    if arm == "A":
        if rows != 1:
            raise RuntimeError("arm A serves exactly one token")
        return 1
    if arm == "L":
        if rows > ARM_L_MAX_TOKENS:
            raise RuntimeError(f"arm L admits at most {ARM_L_MAX_TOKENS} tokens")
        return grid_x
    if arm == "M":
        if rows > ARM_M_MAX_TOKENS or rows < OWNER_CTAS:
            raise RuntimeError(
                f"arm M admits {OWNER_CTAS} <= num_tokens <= {ARM_M_MAX_TOKENS}"
            )
        return grid_x
    if arm == "M4S":
        if rows > ARM_M_MAX_TOKENS or rows < ARM_M4S_OWNER_STRIDE * OWNER_CTAS:
            raise RuntimeError(
                f"arm M4S admits {ARM_M4S_OWNER_STRIDE * OWNER_CTAS} <= num_tokens "
                f"<= {ARM_M_MAX_TOKENS}"
            )
        grid_x = max(1, min(rows, ARM_M4S_CTAS_PER_SM * int(sm_count)))
        if grid_x < ARM_M4S_OWNER_STRIDE * OWNER_CTAS:
            raise RuntimeError(
                f"arm M4S needs a grid of at least {ARM_M4S_OWNER_STRIDE * OWNER_CTAS} CTAs"
            )
        return grid_x
    if arm == "Q":
        if rows > ARM_Q_MAX_TOKENS or rows < OWNER_CTAS:
            raise RuntimeError(
                f"arm Q admits {OWNER_CTAS} <= num_tokens <= {ARM_Q_MAX_TOKENS}"
            )
        if max_active_clusters is None:
            raise RuntimeError("arm Q needs the driver's co-resident cluster capacity")
        cluster_cap = int(max_active_clusters) * ARM_Q_CLUSTER
        grid_x = (min(grid_x, cluster_cap) // ARM_Q_CLUSTER) * ARM_Q_CLUSTER
        if grid_x < OWNER_CTAS:
            raise RuntimeError(
                f"arm Q needs {OWNER_CTAS} co-resident owner CTAs; the driver admits "
                f"{cluster_cap} clustered CTAs"
            )
        return grid_x
    if arm == "G":
        return grid_x
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
    if num_tokens <= 0 or num_tokens * TOP_K > _INT32_MAX:
        raise ValueError(
            "num_tokens must be positive and num_tokens * 16 must fit in int32"
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


def _device_arch(device: torch.device) -> str:
    capability = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            "the Kimi-K3 fused router requires compute capability 10.0 or 10.3 "
            f"(got {capability[0]}.{capability[1]})"
        )
    return arch


def generated_program_available(
    device: torch.device,
    num_tokens: Optional[int] = None,
    block_m: Optional[int] = None,
) -> bool:
    """True when this checkout registers the program for ``device``.

    With ``num_tokens`` / ``block_m`` the check names the exact dispatch arm of
    that shape; without them it asks whether every arm of the device's route
    table is registered.
    """
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))
    if arch is None:
        return False
    registered = registered_programs(arch)
    if num_tokens is None and block_m is None:
        needed = {(arm, bm) for (_, bm), arm in SHAPE_ROUTES[arch].items()}
        return bool(MODULES) and needed <= registered
    if num_tokens is None or block_m is None:
        raise ValueError("pass both num_tokens and block_m or neither")
    key = (int(num_tokens), int(block_m))
    arm = SHAPE_ROUTES[arch].get(key)
    return arm is not None and (arm, key[1]) in registered


def _max_active_clusters(module: Any, record: dict[str, Any], device_index: int) -> int:
    launch = record["main"]["launch"]
    block = [int(v) for v in launch["block"]] + [1, 1, 1]
    cluster = [int(v) for v in launch["cluster"]] + [1, 1, 1]
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
    module_name = select_module(arch, arm, block_m)
    record = MODULES[module_name]
    physical = record["main"]
    with torch.cuda.device(device_index):
        module = load_kimi_k3_fused_router_module(module_name, "main")
        properties = torch.cuda.get_device_properties(device_index)
        max_active_clusters = None
        if uses_cluster_launch(record):
            max_active_clusters = _max_active_clusters(module, record, device_index)
        grid_x = launch_grid(
            arm,
            num_tokens,
            compute_capability=(int(properties.major), int(properties.minor)),
            sm_count=int(properties.multi_processor_count),
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
        grid[name] if kind == "grid" else main_kwargs[name]
        for kind, name in physical["arg_plan"]
    )
    entry = getattr(module, physical["ffi_entry"])
    return KimiK3FusedRouterRunner(module_name, arm, grid_x, plan, entry, arguments)


def kimi_k3_fused_router(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    block_m: int = 8,
    plan: Optional[KimiK3RoutePlan] = None,
) -> KimiK3RoutePlan:
    """Route ``logits`` and build the expert-aligned plan in one launch."""
    return prepare_kimi_k3_fused_router(logits, bias, block_m=block_m, plan=plan)()
