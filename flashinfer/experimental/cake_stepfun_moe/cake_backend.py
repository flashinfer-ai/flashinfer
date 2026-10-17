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

Cake backend: the StepFun fused MoE on exact SM100 / SM103 (cake-build/1 form).

The forward is a sequence of exported Cake kernels -- routing (Renormalize
top-k, one or two launches), FC1 with the StepFun epilogue
``clamp(up, -L, L) * min(silu(gate), L)``, the NVFP4 per-token requantization
(per-token family only), FC2 and finalize -- each a program of ``cake_jit``
reached through its generated TVM-FFI binding.  This module owns what the
trtllm-gen C++ stage runners owned: family and unit selection, the shape
legality of a (family, tile), the host plan (``max_num_ctas``, grids, the
routing window, the finalize variant), the buffer set, the keyword-to-kernel
argument binding and the ordered launch sequence.  The bindings own the
tensor checks, the TMA descriptors, the dynamic-smem opt-in and the cluster /
cooperative / PDL launch attributes.

Every host fact is read from the registry the export renders into
``cake_jit`` (``PROGRAMS[name]["unit"]``, ``STAGE_TABLES``,
``HOST_PLAN_CONSTANTS``): which tensors a unit takes, the storage dtype the
binding checks for each of them, the rank of each tensor-map view, block and
cluster shapes, routing windows, split-K geometry and block-scale layouts.
Nothing here defaults a value the registry does not carry; an unservable
request raises a ``ValueError`` / ``NotImplementedError`` naming what is
missing, and nothing falls back to a trtllm-gen kernel.

A prepared launch performs no allocation and no host synchronisation and is
CUDA-graph capturable; the caller runs it once eagerly before capturing
(first-call binding side effects: dynamic-smem opt-in, lazy module load).
Nothing here reads the environment: every switch is a keyword, resolved by
the public fused-MoE facade (``CakeStepFunConfig`` / ``CakeStepFunRunner``).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import tvm_ffi

from . import cake_jit
from .cake_jit import HOST_PLAN_CONSTANTS, PROGRAMS, STAGE_TABLES

SUPPORTED_COMPUTE_CAPABILITIES: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}

# Argument-plan kinds a StepFun binding takes; any other kind is a render
# error and is never bound.
_TENSOR_KINDS = frozenset({"tma_buffer", "buffer"})
_ADMITTED_KINDS = frozenset({"tma_buffer", "buffer", "parameter", "launch", "grid"})
# Block-scale granularity (elements per scale byte) of the block-scaled
# activation dtypes of the FC1 output / FC2 input.
_SF_BLOCK_ELEMENTS = {"E2m1": 16, "MxE4m3": 32}
_ROUTING_LOGITS_DTYPES = {torch.bfloat16: "bfloat16", torch.float32: "float32"}
_FINALIZE_WEIGHT_DTYPES = {torch.bfloat16: "bfloat16", torch.float32: "float32"}


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def _device_index(device: torch.device) -> int:
    index = device.index
    return torch.cuda.current_device() if index is None else int(index)


@functools.cache
def _device_facts(index: int) -> tuple[tuple[int, int], int]:
    """(compute capability, SM count) of CUDA device ``index``: queried once per process."""
    capability = torch.cuda.get_device_capability(index)
    properties = torch.cuda.get_device_properties(index)
    return (int(capability[0]), int(capability[1])), int(
        properties.multi_processor_count
    )


def arch_for(device: torch.device) -> str:
    """Exact architecture name of ``device`` (``sm_100a`` or ``sm_103a``).

    Raises ``NotImplementedError`` naming the compute capability for any other
    device: the StepFun programs are exported for these two targets only and
    nothing falls back.
    """
    capability, _ = _device_facts(_device_index(device))
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise NotImplementedError(
            "the Cake StepFun fused-MoE programs are exported for exact SM100 and "
            f"SM103 only, got SM{capability[0]}{capability[1]} (device {device})"
        )
    return arch


def sm_count(device: torch.device) -> int:
    """Streaming-multiprocessor count of ``device`` (the cooperative routing grid base)."""
    return _device_facts(_device_index(device))[1]


# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------


def _require_registry() -> None:
    if not PROGRAMS or not STAGE_TABLES or not HOST_PLAN_CONSTANTS:
        raise NotImplementedError(
            "The generated StepFun fused-MoE programs are not registered in this "
            "checkout (empty literals in flashinfer.experimental.cake_stepfun_moe."
            "cake_jit); the registry is filled by the Cake generated-program export."
        )


def _constants() -> dict[str, Any]:
    _require_registry()
    return HOST_PLAN_CONSTANTS


def _family_facts(family: str) -> dict[str, Any]:
    families = _constants()["families"]
    if family not in families:
        raise ValueError(
            f"unknown StepFun family {family!r}; exported families: "
            f"{', '.join(families)}"
        )
    return families[family]


def act_per_byte(family: str) -> int:
    """Activation / weight elements per storage byte of ``family`` (2 for NVFP4, else 1)."""
    return int(_family_facts(family)["act_per_byte"])


def _record(program: str) -> dict[str, Any]:
    _require_registry()
    try:
        return PROGRAMS[program]
    except KeyError:
        raise ValueError(f"unknown StepFun program {program!r}") from None


def _unit(program: str) -> dict[str, Any]:
    return _record(program)["unit"]


def _arg_keys(program: str) -> frozenset[str]:
    """Kernel parameter names of ``program`` (tensors and scalars; grid / launch slots excluded)."""
    return frozenset(
        key
        for kind, key in _record(program)["arg_plan"]
        if kind in _TENSOR_KINDS or kind == "parameter"
    )


def _stage_units(stage: str) -> list[tuple[str, dict[str, Any]]]:
    _require_registry()
    if stage not in STAGE_TABLES:
        raise ValueError(
            f"unknown stage {stage!r}; stages: {', '.join(cake_jit.STAGE_ORDER)}"
        )
    return [(program, _unit(program)) for program in STAGE_TABLES[stage]]


def _torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"the registry names an unknown torch dtype {name!r}")
    return dtype


# ---------------------------------------------------------------------------
# Host plan (pure functions over the literals; copies of the production formulas)
# ---------------------------------------------------------------------------


def _positive_int(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def max_num_ctas(
    num_tokens: int, top_k: int, num_experts: int, tile_tokens_dim: int
) -> int:
    """trtllm-gen ``Routing::getMaxNumCtasInBatchDim``: the maximal CTA count of the
    token dimension for ``num_tokens * top_k`` expanded tokens padded to ``tile_tokens_dim``
    per expert (one tile per expert first, the rest greedily poured into full tiles)."""
    num_tokens = _positive_int("num_tokens", num_tokens, minimum=0)
    top_k = _positive_int("top_k", top_k)
    num_experts = _positive_int("num_experts", num_experts)
    tile_tokens_dim = _positive_int("tile_tokens_dim", tile_tokens_dim)
    remaining = num_tokens * top_k
    filled = min(num_experts, remaining)
    ctas = filled
    remaining -= filled
    if remaining > 0:
        ctas += remaining // tile_tokens_dim
    return ctas


def max_padded_tokens(
    num_tokens: int, top_k: int, num_experts: int, tile_tokens_dim: int
) -> int:
    """Row capacity of the permuted buffers: ``max_num_ctas * tile_tokens_dim``."""
    return (
        max_num_ctas(num_tokens, top_k, num_experts, tile_tokens_dim) * tile_tokens_dim
    )


def route_tile(num_tokens: int) -> int:
    """The planner's centre tile for ``num_tokens``: the first exported FI tile that is
    ``>= num_tokens * top_k / num_experts`` (the export's model constants), else the largest."""
    constants = _constants()
    average = num_tokens * constants["top_k"] / constants["num_experts"]
    for tile in constants["fi_tiles"]:
        if tile >= average:
            return int(tile)
    return int(constants["fi_tiles"][-1])


def exported_tiles(stage: str, family: str) -> tuple[int, ...]:
    """Ascending tiles the package exports for ``family`` in ``stage`` (``fc1`` / ``fc2``)."""
    if stage not in ("fc1", "fc2"):
        raise ValueError(f"exported_tiles: stage must be 'fc1' or 'fc2', got {stage!r}")
    _family_facts(family)
    return tuple(
        sorted(
            {
                int(unit["tile_n"])
                for _, unit in _stage_units(stage)
                if unit["family"] == family
            }
        )
    )


def _gemm_program(stage: str, family: str, tile_n: int) -> str:
    _family_facts(family)
    for program, unit in _stage_units(stage):
        if unit["family"] == family and int(unit["tile_n"]) == int(tile_n):
            return program
    tiles = exported_tiles(stage, family)
    raise ValueError(
        f"no exported {stage} program for family {family!r} and tile {tile_n}; "
        f"exported {stage} tiles of the family: {list(tiles) or 'none'}"
    )


def fc1_program(family: str, tile_n: int) -> str:
    """Program name of the FC1 unit of ``(family, tile_n)``."""
    return _gemm_program("fc1", family, tile_n)


def fc2_program(family: str, tile_n: int) -> str:
    """Program name of the FC2 unit of ``(family, tile_n)``."""
    return _gemm_program("fc2", family, tile_n)


def fc2_activation_sf_layout(family: str, tile_n: int) -> str:
    """Block-scale layout the FC2 unit of ``(family, tile_n)`` reads for its activation
    operand (``r8c4`` / ``r128c4`` / ``none``): what FC1 or the requantization must write."""
    return str(_unit(fc2_program(family, tile_n))["sf_layout_a"])


def requant_program(sf_layout: str) -> str:
    """Program name of the NVFP4 per-token requantization unit writing ``sf_layout``."""
    for program, unit in _stage_units("requant"):
        if unit["variant"] == sf_layout:
            return program
    variants = [unit["variant"] for _, unit in _stage_units("requant")]
    raise ValueError(
        f"no exported requantization program writes block scales in {sf_layout!r}; "
        f"exported layouts: {variants}"
    )


def routing_plan(
    num_tokens: int, routing_input: str, logits_dtype: str | None
) -> tuple[str, ...]:
    """Program names of the routing launch(es) serving ``num_tokens`` from
    ``routing_input`` (``scores`` with ``logits_dtype`` ``bfloat16`` / ``float32``, or
    ``topk_ids`` with ``None``): the first dispatcher row of that input kind whose token
    window contains ``num_tokens``, preceded by its histogram pre-unit when it has one."""
    windows = []
    for program, unit in _stage_units("routing"):
        if unit["input"] != routing_input or unit["logits_dtype"] != logits_dtype:
            continue
        low, high = int(unit["min_tokens"]), int(unit["max_tokens"])
        windows.append((low, high))
        if low <= num_tokens <= high:
            return tuple(name for name in (unit["pre"], program) if name)
    raise ValueError(
        f"no exported routing program reads {routing_input} "
        f"({logits_dtype or 'int32'}) for {num_tokens} tokens; exported token windows: "
        f"{windows or 'none'}"
    )


def routing_inputs() -> tuple[str, ...]:
    """Routing input kinds the package exports (``scores`` and / or ``topk_ids``)."""
    return tuple(
        dict.fromkeys(str(unit["input"]) for _, unit in _stage_units("routing"))
    )


def routing_logits_dtypes() -> tuple[str, ...]:
    """Logit dtypes the exported score routers read (``bfloat16`` / ``float32``)."""
    return tuple(
        dict.fromkeys(
            str(unit["logits_dtype"])
            for _, unit in _stage_units("routing")
            if unit["logits_dtype"] is not None
        )
    )


def finalize_variant(hidden_size: int, num_tokens: int) -> str:
    """The finalize dispatch: ``scalar`` below the CTA limit, ``vector`` otherwise."""
    constants = _constants()["finalize"]
    blocks = math.ceil(hidden_size / constants["hidden_per_cta"]) * min(
        int(constants["max_token_blocks"]), num_tokens
    )
    return "scalar" if blocks < int(constants["scalar_cta_limit"]) else "vector"


def finalize_program(kernel_variant: str, expert_weights_dtype: str) -> str:
    """Program name of the finalize unit of ``kernel_variant`` reading ``expert_weights_dtype`` weights."""
    for program, unit in _stage_units("finalize"):
        if (
            unit["kernel_variant"] == kernel_variant
            and unit["expert_weights_dtype"] == expert_weights_dtype
        ):
            return program
    variants = [
        (unit["kernel_variant"], unit["expert_weights_dtype"])
        for _, unit in _stage_units("finalize")
    ]
    raise ValueError(
        f"no exported finalize program of variant {kernel_variant!r} reads "
        f"{expert_weights_dtype} expert weights; exported: {variants}"
    )


def finalize_weight_dtypes() -> tuple[str, ...]:
    """Expert-weight dtypes the exported finalize units read (``bfloat16`` / ``float32``)."""
    return tuple(
        dict.fromkeys(
            str(unit["expert_weights_dtype"]) for _, unit in _stage_units("finalize")
        )
    )


def _fc1_shape_error(
    program: str, hidden_size: int, intermediate_size: int
) -> str | None:
    unit = _unit(program)
    cluster_x = int(_record(program)["launch"]["cluster"][0])
    rows, block_k = int(unit["output_rows_per_cta"]), int(unit["block_k"])
    if hidden_size % block_k:
        return f"hidden_size {hidden_size} is not a multiple of block_k {block_k}"
    if intermediate_size % rows:
        return f"intermediate_size {intermediate_size} is not a multiple of {rows}"
    if (intermediate_size // rows) % cluster_x:
        return (
            f"intermediate_size / {rows} = {intermediate_size // rows} is not a "
            f"multiple of the cluster width {cluster_x}"
        )
    return None


def _fc2_shape_error(
    program: str, hidden_size: int, intermediate_size: int
) -> str | None:
    unit = _unit(program)
    cluster_x = int(_record(program)["launch"]["cluster"][0])
    rows, block_k = int(unit["output_rows_per_cta"]), int(unit["block_k"])
    if intermediate_size % block_k:
        return f"intermediate_size {intermediate_size} is not a multiple of block_k {block_k}"
    if hidden_size % rows:
        return f"hidden_size {hidden_size} is not a multiple of {rows}"
    if (hidden_size // rows) % cluster_x:
        return (
            f"hidden_size / {rows} = {hidden_size // rows} is not a multiple of the "
            f"cluster width {cluster_x}"
        )
    k_tiles = intermediate_size // block_k
    if k_tiles < int(unit["min_k_tiles"]):
        return f"K tiles {k_tiles} < min_k_tiles {unit['min_k_tiles']}"
    if k_tiles % int(unit["k_tiles_multiple"]):
        return f"K tiles {k_tiles} is not a multiple of {unit['k_tiles_multiple']}"
    return None


@dataclass(frozen=True)
class Plan:
    """The exported units serving one ``(family, tile)``: the tactic of the facade."""

    family: str
    tile_n: int
    fc1_program: str
    fc2_program: str
    requant_program: str | None
    sf_layout_a: str


def plans_for(
    family: str, num_tokens: int, hidden_size: int, intermediate_size: int
) -> tuple[Plan, ...]:
    """Every tile (ascending) that has both an FC1 and an FC2 unit of ``family`` and
    passes the shape rules for ``hidden_size`` / ``intermediate_size``.  ``num_tokens``
    is the planner's argument; the rules at this export are shape-only.  Raises
    ``ValueError`` naming the family, the exported tiles and the rule each failed."""
    facts = _family_facts(family)
    _positive_int("num_tokens", num_tokens, minimum=0)
    _positive_int("hidden_size", hidden_size)
    _positive_int("intermediate_size", intermediate_size)
    tiles = sorted(
        set(exported_tiles("fc1", family)) & set(exported_tiles("fc2", family))
    )
    plans: list[Plan] = []
    reasons: list[str] = []
    for tile in tiles:
        fc1, fc2 = fc1_program(family, tile), fc2_program(family, tile)
        error = _fc1_shape_error(
            fc1, hidden_size, intermediate_size
        ) or _fc2_shape_error(fc2, hidden_size, intermediate_size)
        if error is not None:
            reasons.append(f"tile {tile}: {error}")
            continue
        layout = str(_unit(fc2)["sf_layout_a"])
        requant = requant_program(layout) if facts["per_token"] else None
        plans.append(Plan(family, tile, fc1, fc2, requant, layout))
    if not plans:
        detail = f"; {'; '.join(reasons)}" if reasons else ""
        raise ValueError(
            f"family {family!r} has no servable tile for hidden_size {hidden_size} and "
            f"intermediate_size {intermediate_size} (tiles exported in both stages: "
            f"{tiles or 'none'}){detail}"
        )
    return tuple(plans)


def default_plan(
    family: str, num_tokens: int, hidden_size: int, intermediate_size: int
) -> Plan:
    """The planner's centre tile applied to the exported ladder: the smallest servable
    tile ``>= route_tile(num_tokens)``, else the largest servable tile."""
    plans = plans_for(family, num_tokens, hidden_size, intermediate_size)
    centre = route_tile(num_tokens)
    for plan in plans:
        if plan.tile_n >= centre:
            return plan
    return plans[-1]


def _resolve_plan(
    family: str,
    plan: Plan | int | None,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
) -> Plan:
    plans = plans_for(family, num_tokens, hidden_size, intermediate_size)
    if plan is None:
        centre = route_tile(num_tokens)
        return next((p for p in plans if p.tile_n >= centre), plans[-1])
    if isinstance(plan, Plan):
        if plan.family != family:
            raise ValueError(
                f"plan of family {plan.family!r} given for family {family!r}"
            )
        if plan not in plans:
            raise ValueError(
                f"{plan} is not servable for hidden_size {hidden_size} / intermediate_size "
                f"{intermediate_size}; servable tiles: {[p.tile_n for p in plans]}"
            )
        return plan
    if isinstance(plan, bool) or not isinstance(plan, int):
        raise TypeError(
            f"plan must be None, a tile or a Plan, got {type(plan).__name__}"
        )
    for candidate in plans:
        if candidate.tile_n == plan:
            return candidate
    raise ValueError(
        f"tile {plan} is not servable for family {family!r} (hidden_size {hidden_size}, "
        f"intermediate_size {intermediate_size}); servable tiles: {[p.tile_n for p in plans]}"
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _tensor(
    name: str,
    value: Any,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    dtypes: tuple[torch.dtype, ...] | None = None,
    dim: int | None = None,
    shape: tuple[int, ...] | None = None,
    min_numel: int | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor, got device {value.device}")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} must be {dtype}, got {value.dtype}")
    if dtypes is not None and value.dtype not in dtypes:
        raise ValueError(f"{name} must be one of {list(dtypes)}, got {value.dtype}")
    if dim is not None and value.dim() != dim:
        raise ValueError(f"{name} must be {dim}-D, got shape {tuple(value.shape)}")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(value.shape)}"
        )
    if min_numel is not None and value.numel() < min_numel:
        raise ValueError(
            f"{name} must hold at least {min_numel} elements, got {value.numel()}"
        )
    return value


def _optional(name: str, value: Any, required: bool, context: str) -> None:
    if required and value is None:
        raise ValueError(f"{context} requires {name}")
    if not required and value is not None:
        raise ValueError(f"{context} does not take {name}")


def _reshape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    try:
        return tensor.reshape(shape)
    except RuntimeError:
        raise ValueError(
            f"{name} of shape {tuple(tensor.shape)} ({tensor.dtype}) cannot be viewed as "
            f"{shape}"
        ) from None


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.dtype == torch.uint8 else tensor.view(torch.uint8)


def _sf_block_view(
    name: str, sf_flat: torch.Tensor, *, rows: int, cols: int, layout: str
) -> torch.Tensor:
    """The tensor-map view the FC2 units read of ``rows x cols`` block scales in
    ``layout``: ``r8c4`` -> ``(rows/8, cols/4, 32)``, ``r128c4`` -> ``(rows/128, cols/4, 2, 256)``."""
    block_rows = _constants()["requant"]["sf_block_rows"]
    if layout not in block_rows:
        raise ValueError(
            f"{name}: unknown block-scale layout {layout!r}; known: {list(block_rows)}"
        )
    rows_per_block = int(block_rows[layout])
    if rows % rows_per_block or cols % 4:
        raise ValueError(
            f"{name}: {layout} block scales need rows % {rows_per_block} == 0 and "
            f"columns % 4 == 0, got {rows} x {cols}"
        )
    if sf_flat.numel() < rows * cols:
        raise ValueError(
            f"{name} holds {sf_flat.numel()} scale bytes, {rows * cols} needed"
        )
    flat = _bytes(sf_flat).reshape(-1)[: rows * cols]
    if layout == "r8c4":
        return flat.view(rows // 8, cols // 4, 32)
    return flat.view(rows // 128, cols // 4, 2, 256)


def _compute_log2(tile_tokens_dim: int) -> int:
    exponent = tile_tokens_dim.bit_length() - 1
    if tile_tokens_dim <= 0 or (1 << exponent) != tile_tokens_dim:
        raise ValueError(
            f"tile_tokens_dim must be a power of two, got {tile_tokens_dim}"
        )
    return exponent


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Launch:
    stage: str
    program: str
    launcher: Callable[..., None] = field(repr=False)
    run: Callable[..., Any] = field(repr=False)
    values: dict[str, Any] = field(repr=False)
    grid: tuple[int, int, int]

    def __call__(self) -> None:
        self.launcher(self.run, self.values, self.grid)


def _bind(
    program: str,
    values: dict[str, Any],
    grid: tuple[int, int, int],
    *,
    enable_pdl: bool,
    arch: str,
    stage: str,
) -> _Launch:
    """Bind ``values`` (keyed by kernel parameter name) to ``program``: every tensor is
    viewed to the storage dtype the binding checks, tensor-map views are checked against
    the descriptor's minimum source rank (``unit["tma_ranks"]`` is the floor the
    binding's own ``ndim >=`` guard derives from the descriptor's trailing-axis
    expressions, not the view rank), the grid for positivity and cluster divisibility;
    the generated positional launcher of ``cake_launch`` orders the arguments."""
    record = _record(program)
    unit = record["unit"]
    cluster = [int(c) for c in record["launch"]["cluster"]]
    if len(grid) != 3:
        raise ValueError(f"{stage} {program}: grid must have three entries, got {grid}")
    grid = tuple(int(g) for g in grid)
    for axis, (extent, span) in enumerate(zip(grid, cluster, strict=True)):
        if extent <= 0:
            raise ValueError(
                f"{stage} {program}: grid axis {axis} must be > 0, got {grid}"
            )
        if extent % span:
            raise ValueError(
                f"{stage} {program}: grid axis {axis} ({extent}) is not a multiple of the "
                f"cluster extent {span}"
            )
    bound: dict[str, Any] = {"enable_pdl": int(bool(enable_pdl))}
    for kind, key in record["arg_plan"]:
        if kind not in _ADMITTED_KINDS:
            raise ValueError(
                f"{stage} {program}: argument plan kind {kind!r} ({key}) is not one the "
                "StepFun host binds"
            )
        if kind == "grid":
            continue
        if kind == "launch":
            if key != "enable_pdl":
                raise ValueError(f"{stage} {program}: unknown launch slot {key!r}")
            continue
        if key not in values or values[key] is None:
            raise KeyError(
                f"generated program {program!r} ({stage}) expects argument {key!r} ({kind}); "
                f"the host provides {sorted(k for k, v in values.items() if v is not None)}"
            )
        value = values[key]
        if kind == "parameter":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"{stage} {program}: parameter {key!r} must be an int or float, got "
                    f"{type(value).__name__}"
                )
            bound[key] = value
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{stage} {program}: argument {key!r} must be a tensor, got "
                f"{type(value).__name__}"
            )
        storage = _torch_dtype(unit["arg_dtypes"][key])
        tensor = value if value.dtype == storage else value.view(storage)
        if kind == "tma_buffer":
            rank = int(unit["tma_ranks"][key])
            if tensor.dim() < rank:
                raise ValueError(
                    f"{stage} {program}: tensor-map operand {key!r} needs at least {rank} "
                    f"dimensions for the exported descriptor, got shape {tuple(tensor.shape)}"
                )
        if not tensor.is_contiguous():
            raise ValueError(f"{stage} {program}: argument {key!r} must be contiguous")
        bound[key] = tensor
    # ``cake_launch`` is rendered by the export next to ``cake_jit``; imported here so
    # the host plan stays importable (with named errors) before the delivery lands.
    from . import cake_launch

    module = cake_jit.load_module(program, arch)
    return _Launch(
        stage=stage,
        program=program,
        launcher=cake_launch.LAUNCH[program],
        run=getattr(module, record["ffi_entry"]),
        values=bound,
        grid=grid,
    )


def _stage_launch(
    program: str,
    values: dict[str, Any],
    grid: tuple[int, int, int],
    *,
    enable_pdl: bool,
    arch: str,
    stage: str,
) -> "StepFunStageLaunch":
    launch = _bind(program, values, grid, enable_pdl=enable_pdl, arch=arch, stage=stage)
    return StepFunStageLaunch(
        stage=stage, programs=(program,), grids=(launch.grid,), launches=(launch,)
    )


# ---------------------------------------------------------------------------
# Prepared launches
# ---------------------------------------------------------------------------


def _run_launches(launches: tuple[_Launch, ...]) -> None:
    with tvm_ffi.use_torch_stream():
        for item in launches:
            item()


@dataclass(frozen=True)
class StepFunStageLaunch:
    """One prepared stage on caller-owned buffers (one launch, or the routing pair)."""

    stage: str
    programs: tuple[str, ...]
    grids: tuple[tuple[int, int, int], ...]
    launches: tuple[_Launch, ...] = field(repr=False)

    def launch(self) -> None:
        _run_launches(self.launches)

    __call__ = launch


@dataclass(frozen=True)
class StepFunMoeLaunch:
    """One prepared StepFun forward: routing (one or two launches), FC1, the per-token
    requantization (per-token family only), FC2 and, with ``do_finalize``, finalize.

    ``launch()`` runs the bound programs on the current torch stream with no
    allocation and no host synchronisation; the routing kernels read the token
    contents on device at every launch, so the same prepared launch (or a CUDA
    Graph capturing it) stays valid for new values of the same shapes.  Run it
    once eagerly before capturing (binding smem opt-in, lazy module load).
    """

    family: str
    arch: str
    plan: Plan
    do_finalize: bool
    routing_programs: tuple[str, ...]
    finalize_program: str | None
    launches: tuple[_Launch, ...] = field(repr=False)
    buffers: dict[str, torch.Tensor] = field(repr=False)
    expert_weights: torch.Tensor = field(repr=False)

    def launch(self) -> None:
        _run_launches(self.launches)

    __call__ = launch

    def outputs(self) -> list[torch.Tensor]:
        """``[output]`` when finalizing, else ``[gemm2_output, expert_weights,
        expanded_idx_to_permuted_idx]`` in the trtllm-gen layout (permuted bf16 rows,
        the routing-written bf16 weights or the caller's weights, the flat route list)."""
        if self.do_finalize:
            return [self.buffers["output"]]
        return [
            self.buffers["gemm2_output"],
            self.expert_weights,
            self.buffers["expanded_idx_to_permuted_idx"].reshape(-1),
        ]

    @property
    def stage_launchers(self) -> dict[str, Callable[[], None]]:
        """One zero-argument launch per stage (``routing_pre``, ``routing``, ``fc1``,
        ``requant``, ``fc2``, ``finalize``) for tests and timing."""
        result: dict[str, Callable[[], None]] = {}
        for item in self.launches:

            def run(item: _Launch = item) -> None:
                _run_launches((item,))

            result[item.stage] = run
        return result


# ---------------------------------------------------------------------------
# Stage values
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Geometry:
    num_tokens: int
    hidden_size: int
    num_experts: int
    intermediate_size: int
    top_k: int
    tile_tokens_dim: int
    max_ctas: int
    max_padded: int


def _geometry(
    family: str,
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    *,
    top_k: int,
    num_experts_for_ctas: int,
    tile_tokens_dim: int,
) -> _Geometry:
    """Problem size read from the operands: ``T`` and ``H`` from the activations, ``E``
    from the FC1 weight rows, ``I`` from the FC2 weights, cross-checked against FC1."""
    per_byte = act_per_byte(family)
    _tensor("hidden_states", hidden_states, dim=2)
    num_tokens, packed_hidden = (int(s) for s in hidden_states.shape)
    hidden_size = packed_hidden * per_byte
    if gemm1_weights.dim() < 2 or gemm2_weights.dim() < 2:
        raise ValueError(
            "gemm1_weights and gemm2_weights must be at least 2-D [E, ...]"
        )
    num_rows = int(gemm1_weights.shape[0])
    if int(gemm2_weights.shape[0]) != num_rows:
        raise ValueError(
            f"gemm2_weights has {gemm2_weights.shape[0]} expert rows, gemm1_weights {num_rows}"
        )
    if num_rows < 1 or hidden_size < 1:
        raise ValueError(
            "weights need at least one expert row and a positive hidden size"
        )
    fc2_elements = gemm2_weights.numel() * per_byte
    if fc2_elements % (num_rows * hidden_size):
        raise ValueError(
            f"gemm2_weights ({tuple(gemm2_weights.shape)}, {gemm2_weights.dtype}) does not "
            f"hold E x hidden_size x intermediate_size elements for E={num_rows}, "
            f"hidden_size={hidden_size}"
        )
    intermediate_size = fc2_elements // (num_rows * hidden_size)
    if (
        gemm1_weights.numel() * per_byte
        != num_rows * 2 * intermediate_size * hidden_size
    ):
        raise ValueError(
            f"gemm1_weights ({tuple(gemm1_weights.shape)}, {gemm1_weights.dtype}) does not "
            f"hold E x 2*intermediate_size x hidden_size elements for E={num_rows}, "
            f"intermediate_size={intermediate_size}, hidden_size={hidden_size}"
        )
    max_ctas = max_num_ctas(num_tokens, top_k, num_experts_for_ctas, tile_tokens_dim)
    return _Geometry(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        num_experts=num_rows,
        intermediate_size=intermediate_size,
        top_k=top_k,
        tile_tokens_dim=tile_tokens_dim,
        max_ctas=max_ctas,
        max_padded=max_ctas * tile_tokens_dim,
    )


def _routing_tables_check(
    tables: dict[str, Any], *, device: torch.device, max_ctas: int, max_padded: int
) -> None:
    """Extents of the routing tables a GEMM / finalize stage reads (int32, same device)."""
    extents = {
        "permuted_idx_to_token_idx": max_padded,
        "cta_idx_xy_to_batch_idx": max_ctas,
        "cta_idx_xy_to_mn_limit": max_ctas,
        "num_non_exiting_ctas": 1,
        "total_num_padded_tokens": 1,
    }
    for name, min_numel in extents.items():
        if tables.get(name) is not None:
            _tensor(
                name,
                tables[name],
                device=device,
                dtype=torch.int32,
                min_numel=min_numel,
            )


def _fc1_values(
    program: str,
    family: str,
    geometry: _Geometry,
    operands: dict[str, Any],
) -> tuple[dict[str, Any], tuple[int, int, int]]:
    """Kernel values of one FC1 launch (contract: the exporter's operand views)."""
    unit = _unit(program)
    keys = _arg_keys(program)
    g = geometry
    device = operands["hidden_states"].device
    per_token = bool(unit["per_token"])
    context = f"FC1 family {family!r} tile {g.tile_tokens_dim}"
    needs = {
        "hidden_states_scale": "SFB" in keys,
        "gemm1_weights_scale": "SFA" in keys,
        "output1_scale_scalar": "scale_c" in keys,
        "output1_scale_gate_scalar": "scale_gate" in keys,
        "per_token_scale": per_token,
        "gemm1_output_scale": "SFC" in keys and not per_token,
    }
    for name, required in needs.items():
        _optional(name, operands.get(name), required, context)
    rows = int(unit["output_rows_per_cta"])
    block_k = int(unit["block_k"])
    E, H, I, T = g.num_experts, g.hidden_size, g.intermediate_size, g.num_tokens
    weights = _tensor("gemm1_weights", operands["gemm1_weights"], device=device)
    if unit["weight_layout"] == "BlockMajorK":
        weights = _reshape("gemm1_weights", weights, (E, H // 64, 2 * I, 64))
    else:
        weights = _reshape(
            "gemm1_weights", weights, (E, 2 * I, H // act_per_byte(family))
        )
    clamp = _tensor(
        "gemm1_clamp_limit",
        operands["gemm1_clamp_limit"],
        device=device,
        dtype=torch.float32,
        shape=(E,),
    )
    out_dtype = str(unit["dtypes"][2])
    out_cols = I // 2 if out_dtype == "E2m1" else I
    gemm1_output = _tensor(
        "gemm1_output",
        operands["gemm1_output"],
        device=device,
        dtype=_torch_dtype(unit["arg_dtypes"]["C"]),
        shape=(g.max_padded, out_cols),
    )
    values: dict[str, Any] = {
        "A": weights,
        "B": operands["hidden_states"],
        "C": gemm1_output,
        "route_map": operands["permuted_idx_to_token_idx"],
        "tile_expert": operands["cta_idx_xy_to_batch_idx"],
        "tile_mn_limit": operands["cta_idx_xy_to_mn_limit"],
        "num_non_exiting_ctas": operands["num_non_exiting_ctas"],
        "total_tiles": operands["num_non_exiting_ctas"],
        "total_num_padded_tokens": operands.get("total_num_padded_tokens"),
        "clamp_limit": clamp,
        "work_counter": operands.get("work_counter"),
        "M_out": I,
        "K": H,
        "grid_m": I // rows,
        "grid_n": g.max_ctas,
        "K_tiles": H // block_k,
    }
    if needs["gemm1_weights_scale"]:
        act_dtype = str(unit["dtypes"][0])
        block = _SF_BLOCK_ELEMENTS.get(act_dtype)
        if block is None:
            raise ValueError(
                f"{context}: activation dtype {act_dtype} has no block-scale operand"
            )
        sfa = _bytes(
            _tensor(
                "gemm1_weights_scale", operands["gemm1_weights_scale"], device=device
            )
        )
        values["SFA"] = _reshape(
            "gemm1_weights_scale", sfa, (E * I // 64, H // (4 * block), 2, 256)
        )
        sfb = _bytes(
            _tensor(
                "hidden_states_scale", operands["hidden_states_scale"], device=device
            )
        )
        values["SFB"] = _reshape("hidden_states_scale", sfb, (T, H // block))
    if per_token:
        values["SFC"] = _tensor(
            "per_token_scale",
            operands["per_token_scale"],
            device=device,
            dtype=torch.float32,
            min_numel=T,
        )
    elif needs["gemm1_output_scale"]:
        block = _SF_BLOCK_ELEMENTS.get(out_dtype)
        if block is None:
            raise ValueError(
                f"{context}: output dtype {out_dtype} has no block-scale operand"
            )
        values["SFC"] = _tensor(
            "gemm1_output_scale",
            operands["gemm1_output_scale"],
            device=device,
            dtype=torch.uint8,
            min_numel=g.max_padded * I // block,
        )
    if needs["output1_scale_scalar"]:
        values["scale_c"] = _tensor(
            "output1_scale_scalar",
            operands["output1_scale_scalar"],
            device=device,
            dtype=torch.float32,
            shape=(E,),
        )
        values["scale_gate"] = _tensor(
            "output1_scale_gate_scalar",
            operands["output1_scale_gate_scalar"],
            device=device,
            dtype=torch.float32,
            shape=(E,),
        )
    # The step epilogue ignores alpha / beta; the production host aliases them to a
    # valid per-expert vector (no allocation).
    alias = values.get("scale_c", clamp)
    values["act_alpha"] = alias
    values["act_beta"] = alias
    return values, (I // rows, g.max_ctas, 1)


def _fc2_values(
    program: str,
    family: str,
    geometry: _Geometry,
    operands: dict[str, Any],
) -> tuple[dict[str, Any], tuple[int, int, int]]:
    """Kernel values of one FC2 launch (contract: the exporter's operand views)."""
    unit = _unit(program)
    keys = _arg_keys(program)
    g = geometry
    device = operands["gemm2_input"].device
    context = f"FC2 family {family!r} tile {g.tile_tokens_dim}"
    needs = {
        "gemm2_weights_scale": "SFA" in keys,
        "gemm2_input_scale": "SFB" in keys,
        "output2_scale_scalar": "scale_c" in keys,
        "per_token_scale": "token_sf" in keys or "per_token_scale" in keys,
    }
    for name, required in needs.items():
        _optional(name, operands.get(name), required, context)
    rows = int(unit["output_rows_per_cta"])
    block_k = int(unit["block_k"])
    E, H, I = g.num_experts, g.hidden_size, g.intermediate_size
    weights = _tensor("gemm2_weights", operands["gemm2_weights"], device=device)
    if unit["weight_layout"] == "BlockMajorK":
        weights = _reshape("gemm2_weights", weights, (E, I // 64, H, 64))
    else:
        weights = _reshape("gemm2_weights", weights, (E, H, I // act_per_byte(family)))
    act_dtype = str(unit["dtypes"][0])
    in_cols = I // 2 if act_dtype == "E2m1" else I
    gemm2_input = _tensor(
        "gemm2_input",
        operands["gemm2_input"],
        device=device,
        shape=(g.max_padded, in_cols),
    )
    gemm2_output = _tensor(
        "gemm2_output",
        operands["gemm2_output"],
        device=device,
        dtype=torch.bfloat16,
        shape=(g.max_padded, H),
    )
    values: dict[str, Any] = {
        "A": weights,
        "B": gemm2_input,
        "C": gemm2_output,
        "tile_expert": operands["cta_idx_xy_to_batch_idx"],
        "tile_mn_limit": operands["cta_idx_xy_to_mn_limit"],
        "num_non_exiting_ctas": operands["num_non_exiting_ctas"],
        "total_tiles": operands["num_non_exiting_ctas"],
        "total_num_padded_tokens": operands.get("total_num_padded_tokens"),
        "work_counter": operands.get("work_counter"),
        "M_out": H,
        "K": I,
        "grid_m": H // rows,
        "grid_n": g.max_ctas,
        "K_tiles": I // block_k,
    }
    if needs["gemm2_weights_scale"] or needs["gemm2_input_scale"]:
        block = _SF_BLOCK_ELEMENTS.get(act_dtype)
        if block is None:
            raise ValueError(
                f"{context}: activation dtype {act_dtype} has no block-scale operand"
            )
        if needs["gemm2_weights_scale"]:
            sfa = _bytes(
                _tensor(
                    "gemm2_weights_scale",
                    operands["gemm2_weights_scale"],
                    device=device,
                )
            )
            values["SFA"] = _reshape(
                "gemm2_weights_scale", sfa, (E * H // 128, I // (4 * block), 2, 256)
            )
        if needs["gemm2_input_scale"]:
            scales = _tensor(
                "gemm2_input_scale", operands["gemm2_input_scale"], device=device
            )
            values["SFB"] = _sf_block_view(
                "gemm2_input_scale",
                scales,
                rows=g.max_padded,
                cols=I // block,
                layout=str(unit["sf_layout_a"]),
            )
    if needs["output2_scale_scalar"]:
        values["scale_c"] = _tensor(
            "output2_scale_scalar",
            operands["output2_scale_scalar"],
            device=device,
            dtype=torch.float32,
            shape=(E,),
        )
    if needs["per_token_scale"]:
        token_sf = _tensor(
            "per_token_scale",
            operands["per_token_scale"],
            device=device,
            dtype=torch.float32,
            min_numel=g.max_padded,
        )
        values["token_sf"] = token_sf
        values["per_token_scale"] = token_sf
    return values, (H // rows, g.max_ctas, int(unit["split_k"]))


def _routing_input(
    routing_logits: torch.Tensor | None, topk_ids: torch.Tensor | None
) -> tuple[str, str | None]:
    if (routing_logits is None) == (topk_ids is None):
        raise ValueError("exactly one of routing_logits and topk_ids must be given")
    if routing_logits is not None:
        dtype = _ROUTING_LOGITS_DTYPES.get(routing_logits.dtype)
        if dtype is None:
            raise ValueError(
                f"routing_logits must be bfloat16 or float32, got {routing_logits.dtype}"
            )
        return "scores", dtype
    return "topk_ids", None


def _routing_grid(
    program: str,
    *,
    device: torch.device,
    reserved_sms: int,
    num_tokens: int,
    top_k: int,
) -> tuple[int, int, int]:
    unit = _unit(program)
    rule = unit["grid_rule"]
    if rule == "fixed":
        grid = tuple(int(g) for g in unit["grid"])
    elif rule == "coop_sms":
        sms = sm_count(device)
        if isinstance(reserved_sms, bool) or not isinstance(reserved_sms, int):
            raise TypeError(
                f"reserved_sms must be an int, got {type(reserved_sms).__name__}"
            )
        if not 0 <= reserved_sms < sms:
            raise ValueError(
                f"reserved_sms {reserved_sms} must satisfy 0 <= value < SM count {sms}"
            )
        grid = (sms - reserved_sms, 1, 1)
    else:
        raise ValueError(f"routing program {program}: unknown grid rule {rule!r}")
    capacity_per_thread = int(unit["max_expanded_per_thread"])
    if capacity_per_thread > 0:
        block = int(_record(program)["launch"]["block"][0])
        capacity = grid[0] * block * capacity_per_thread
        if num_tokens * top_k > capacity:
            raise ValueError(
                f"routing program {program} ({unit['variant']}) holds {capacity} expanded "
                f"tokens over grid {grid} x {block} threads; {num_tokens} x {top_k} requested"
            )
    return grid


def _routing_launches(
    *,
    routing_logits: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    tables: dict[str, torch.Tensor],
    num_tokens: int,
    num_experts: int,
    top_k: int,
    tile_tokens_dim: int,
    local_expert_offset: int,
    local_num_experts: int,
    device: torch.device,
    arch: str,
    enable_pdl: bool,
    reserved_sms: int,
) -> tuple[tuple[str, ...], list[_Launch]]:
    kind, logits_dtype = _routing_input(routing_logits, topk_ids)
    programs = routing_plan(num_tokens, kind, logits_dtype)
    values: dict[str, Any] = {
        "topk_weights": tables["topk_weights"],
        "topk_packed": tables["topk_packed"],
        "expert_counts": tables["expert_count_histogram"],
        "permuted_idx_size": tables["total_num_padded_tokens"],
        "expanded_idx_to_permuted_idx": tables["expanded_idx_to_permuted_idx"],
        "permuted_idx_to_token_idx": tables["permuted_idx_to_token_idx"],
        "cta_idx_xy_to_batch_idx": tables["cta_idx_xy_to_batch_idx"],
        "cta_idx_xy_to_mn_limit": tables["cta_idx_xy_to_mn_limit"],
        "num_non_exiting_ctas": tables["num_non_exiting_ctas"],
        "num_tokens": num_tokens,
        "num_experts": num_experts,
        "top_k": top_k,
        "padding_log2": _compute_log2(tile_tokens_dim),
        "tile_tokens_dim": tile_tokens_dim,
        "local_experts_start_idx": local_expert_offset,
        "local_experts_stride_log2": int(
            _constants()["routing"]["derived"]["local_experts_stride_log2"]
        ),
        "num_local_experts": local_num_experts,
    }
    if routing_logits is not None:
        values["scores"] = routing_logits
    else:
        values["topk_ids"] = topk_ids
    launches = []
    for index, program in enumerate(programs):
        stage = "routing_pre" if index < len(programs) - 1 else "routing"
        grid = _routing_grid(
            program,
            device=device,
            reserved_sms=reserved_sms,
            num_tokens=num_tokens,
            top_k=top_k,
        )
        launches.append(
            _bind(program, values, grid, enable_pdl=enable_pdl, arch=arch, stage=stage)
        )
    return programs, launches


def _routing_checks(
    *,
    routing_logits: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    num_experts: int,
    top_k: int,
    local_expert_offset: int,
    local_num_experts: int,
    tile_tokens_dim: int,
    device: torch.device,
) -> int:
    """Shared argument checks of the routing stage; returns ``num_tokens``."""
    _positive_int("num_experts", num_experts)
    _positive_int("top_k", top_k)
    _positive_int("local_expert_offset", local_expert_offset, minimum=0)
    _positive_int("local_num_experts", local_num_experts)
    _positive_int("tile_tokens_dim", tile_tokens_dim)
    if top_k > num_experts:
        raise ValueError(f"top_k {top_k} exceeds num_experts {num_experts}")
    if local_expert_offset + local_num_experts > num_experts:
        raise ValueError(
            f"local experts [{local_expert_offset}, {local_expert_offset + local_num_experts}) "
            f"exceed num_experts {num_experts}"
        )
    kind, _ = _routing_input(routing_logits, topk_ids)
    if kind == "scores":
        logits = _tensor("routing_logits", routing_logits, device=device, dim=2)
        if int(logits.shape[1]) != num_experts:
            raise ValueError(
                f"routing_logits must be [num_tokens, {num_experts}], got {tuple(logits.shape)}"
            )
        num_tokens = int(logits.shape[0])
    else:
        ids = _tensor("topk_ids", topk_ids, device=device, dtype=torch.int32, dim=2)
        if int(ids.shape[1]) != top_k:
            raise ValueError(
                f"topk_ids must be [num_tokens, {top_k}], got {tuple(ids.shape)}"
            )
        num_tokens = int(ids.shape[0])
    if num_tokens < 1:
        raise ValueError("the exported routing programs launch for num_tokens >= 1")
    return num_tokens


def _requant_values(
    program: str,
    operands: dict[str, Any],
    *,
    max_padded: int,
    global_scale_inv: float,
) -> tuple[dict[str, Any], tuple[int, int, int]]:
    constants = _constants()["requant"]
    unit = _unit(program)
    expected = float(constants["global_scale_inv"])
    if not math.isclose(float(global_scale_inv), expected, rel_tol=1e-9, abs_tol=0.0):
        raise NotImplementedError(
            f"global_scale_inv {global_scale_inv!r} is not the exported requantization "
            f"unit's 1 / (e4m3_max * 6) = {expected!r} (only e4m3_max = "
            f"{unit['e4m3_max']} units exist; NVFP4 4over6 recipes are not exported)"
        )
    device = operands["gemm1_output"].device
    gemm1_output = _tensor(
        "gemm1_output",
        operands["gemm1_output"],
        device=device,
        dtype=torch.bfloat16,
        dim=2,
    )
    if int(gemm1_output.shape[0]) != max_padded:
        raise ValueError(
            f"gemm1_output must hold max_padded = {max_padded} rows, got {tuple(gemm1_output.shape)}"
        )
    intermediate_size = int(gemm1_output.shape[1])
    expanded = _tensor(
        "expanded_idx_to_permuted_idx",
        operands["expanded_idx_to_permuted_idx"],
        device=device,
        dtype=torch.int32,
    ).reshape(-1)
    layout = str(unit["variant"])
    rows_per_block = int(constants["sf_block_rows"][layout])
    if max_padded % rows_per_block:
        raise ValueError(
            f"the requantized {layout} scales need a padded row count divisible by "
            f"{rows_per_block}, got {max_padded}"
        )
    sf_rows = -(-max_padded // int(constants["sf_rows_multiple"])) * int(
        constants["sf_rows_multiple"]
    )
    values = {
        "input": gemm1_output,
        "expanded_idx_to_permuted_idx": expanded,
        "out_fp4": _tensor(
            "output",
            operands["output"],
            device=device,
            dtype=torch.uint8,
            shape=(max_padded, intermediate_size // 2),
        ),
        "out_sf": _tensor(
            "output_scale",
            operands["output_scale"],
            device=device,
            dtype=torch.uint8,
            min_numel=sf_rows * intermediate_size // 16,
        ),
        "per_token_scale": _tensor(
            "per_token_scale",
            operands["per_token_scale"],
            device=device,
            dtype=torch.float32,
            min_numel=max_padded,
        ),
        "global_scale_inv": float(global_scale_inv),
        "m": int(expanded.numel()),
        "n": intermediate_size,
    }
    return values, (int(expanded.numel()) * int(constants["rows_per_cta"]), 1, 1)


def _finalize_values(
    program: str,
    operands: dict[str, Any],
) -> tuple[dict[str, Any], tuple[int, int, int]]:
    unit = _unit(program)
    device = operands["output"].device
    output = _tensor(
        "output", operands["output"], device=device, dtype=torch.bfloat16, dim=2
    )
    num_tokens, hidden_size = (int(s) for s in output.shape)
    gemm2_output = _tensor(
        "gemm2_output",
        operands["gemm2_output"],
        device=device,
        dtype=torch.bfloat16,
        dim=2,
    )
    if int(gemm2_output.shape[1]) < hidden_size:
        raise ValueError(
            f"gemm2_output width {gemm2_output.shape[1]} is below the output width {hidden_size}"
        )
    weights = _tensor(
        "expert_weights",
        operands["expert_weights"],
        device=device,
        dtype=_torch_dtype(unit["arg_dtypes"]["expert_weights"]),
        dim=2,
    )
    if int(weights.shape[0]) != num_tokens:
        raise ValueError(
            f"expert_weights must be [num_tokens={num_tokens}, top_k], got {tuple(weights.shape)}"
        )
    top_k = int(weights.shape[1])
    expanded = _tensor(
        "expanded_idx_to_permuted_idx",
        operands["expanded_idx_to_permuted_idx"],
        device=device,
        dtype=torch.int32,
    ).reshape(-1)
    if expanded.numel() != num_tokens * top_k:
        raise ValueError(
            f"expanded_idx_to_permuted_idx must hold num_tokens * top_k = {num_tokens * top_k} "
            f"entries, got {expanded.numel()}"
        )
    variant = str(unit["kernel_variant"])
    max_top_k = int(unit["max_top_k"])
    if variant == "scalar" and top_k != max_top_k:
        raise ValueError(
            f"the scalar finalize unit serves top_k == {max_top_k} exactly, got {top_k}"
        )
    if variant == "vector" and (top_k % 4 or top_k > max_top_k or hidden_size % 8):
        raise ValueError(
            f"the vector finalize unit needs top_k % 4 == 0, top_k <= {max_top_k} and "
            f"hidden_size % 8 == 0, got top_k {top_k}, hidden_size {hidden_size}"
        )
    constants = _constants()["finalize"]
    if variant == "scalar":
        grid = (
            math.ceil(hidden_size / int(constants["hidden_per_cta"])),
            min(int(constants["max_token_blocks"]), num_tokens),
            1,
        )
    else:
        grid = (num_tokens, 1, 1)
    values = {
        "in_ptr": gemm2_output,
        "expert_weights": weights,
        "out_ptr": output,
        "expanded_idx_to_permuted_idx": expanded,
        "total_num_padded_tokens": operands.get("total_num_padded_tokens"),
        "hidden_dim": hidden_size,
        "hidden_dim_padded": int(gemm2_output.shape[1]),
        "num_tokens": num_tokens,
        "top_k": top_k,
    }
    return values, grid


def _finalize_program_for(
    hidden_size: int, num_tokens: int, expert_weights: torch.Tensor
) -> str:
    dtype = _FINALIZE_WEIGHT_DTYPES.get(expert_weights.dtype)
    if dtype is None:
        raise ValueError(
            f"expert_weights must be bfloat16 or float32, got {expert_weights.dtype}"
        )
    return finalize_program(finalize_variant(hidden_size, num_tokens), dtype)


# ---------------------------------------------------------------------------
# Whole path
# ---------------------------------------------------------------------------


def prepare_stepfun_moe(
    *,
    family: str,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor | None = None,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor | None = None,
    output1_scale_scalar: torch.Tensor | None = None,
    output1_scale_gate_scalar: torch.Tensor | None = None,
    gemm1_clamp_limit: torch.Tensor,
    per_token_scale: torch.Tensor | None = None,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor | None = None,
    output2_scale_scalar: torch.Tensor | None = None,
    routing_logits: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    num_experts: int,
    top_k: int,
    local_expert_offset: int,
    local_num_experts: int,
    plan: Plan | int | None = None,
    output: torch.Tensor | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = True,
    reserved_sms: int = 8,
    global_scale_inv: float = 1.0 / (448.0 * 6.0),
    poison_routing_tail: bool = False,
) -> StepFunMoeLaunch:
    """Validate, select the units of ``(family, plan)``, allocate the buffer set once
    and bind the launch sequence routing -> FC1 -> (requantization) -> FC2 ->
    (finalize).

    ``family`` is ``nvfp4``, ``nvfp4_bf16tok``, ``bf16``, ``fp8`` or ``mxfp8``; the
    tensors are the trtllm-gen weight / activation views of that family (the
    family-optional scales must be given exactly when its units take them) and
    ``gemm1_clamp_limit`` the per-expert fp32 limits in the units the kernels clamp.
    Exactly one of ``routing_logits`` (bf16 / fp32 ``[T, num_experts]``) or ``topk_ids``
    (int32 ``[T, top_k]``, with ``topk_weights`` bf16 / fp32 ``[T, top_k]``) selects the
    routing input.  ``plan`` is ``None`` (the planner's centre tile), a tile or a
    :class:`Plan`.  ``output`` (bf16 ``[T, hidden_size]``) is allocated when ``None``
    and ``do_finalize``; with ``do_finalize=False`` no finalize launch is bound and
    :meth:`StepFunMoeLaunch.outputs` returns the permuted FC2 output, the expert
    weights and the route list.  ``reserved_sms`` sizes the cooperative routing
    grid (``SM count - reserved_sms``); ``global_scale_inv`` must be the exported
    requantization unit's ``1 / (448 * 6)``; ``poison_routing_tail`` fills the
    never-written tails of the routing tables with the measurement sentinels.
    """
    _require_registry()
    facts = _family_facts(family)
    device = _tensor("hidden_states", hidden_states, dim=2).device
    arch = arch_for(device)
    _positive_int("num_experts", num_experts)
    _positive_int("top_k", top_k)
    _positive_int("local_expert_offset", local_expert_offset, minimum=0)
    _positive_int("local_num_experts", local_num_experts)
    if top_k > num_experts:
        raise ValueError(f"top_k {top_k} exceeds num_experts {num_experts}")
    if local_expert_offset + local_num_experts > num_experts:
        raise ValueError(
            f"local experts [{local_expert_offset}, {local_expert_offset + local_num_experts}) "
            f"exceed num_experts {num_experts}"
        )
    if int(gemm1_weights.shape[0]) != local_num_experts:
        raise ValueError(
            f"gemm1_weights holds {gemm1_weights.shape[0]} expert rows, local_num_experts is "
            f"{local_num_experts}"
        )
    num_tokens = _routing_checks(
        routing_logits=routing_logits,
        topk_ids=topk_ids,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        tile_tokens_dim=1,
        device=device,
    )
    if num_tokens != int(hidden_states.shape[0]):
        raise ValueError(
            f"the routing input describes {num_tokens} tokens, hidden_states "
            f"{hidden_states.shape[0]}"
        )
    per_byte = act_per_byte(family)
    hidden_size = int(hidden_states.shape[1]) * per_byte
    probe = _geometry(
        family,
        hidden_states,
        gemm1_weights,
        gemm2_weights,
        top_k=top_k,
        num_experts_for_ctas=num_experts,
        tile_tokens_dim=1,
    )
    resolved = _resolve_plan(
        family, plan, num_tokens, hidden_size, probe.intermediate_size
    )
    tile = resolved.tile_n
    g = _geometry(
        family,
        hidden_states,
        gemm1_weights,
        gemm2_weights,
        top_k=top_k,
        num_experts_for_ctas=num_experts,
        tile_tokens_dim=tile,
    )
    T, H, I = g.num_tokens, g.hidden_size, g.intermediate_size
    fc1_unit = _unit(resolved.fc1_program)
    routing_constants = _constants()["routing"]
    requant_constants = _constants()["requant"]
    if topk_ids is not None:
        if topk_weights is None:
            raise ValueError("topk_weights is required with topk_ids")
        weights = _tensor("topk_weights", topk_weights, device=device, dim=2)
        if tuple(weights.shape) != (T, top_k):
            raise ValueError(
                f"topk_weights must be [{T}, {top_k}], got {tuple(weights.shape)}"
            )
        if weights.dtype not in _FINALIZE_WEIGHT_DTYPES:
            raise ValueError(
                f"topk_weights must be bfloat16 or float32, got {weights.dtype}"
            )
    elif topk_weights is not None:
        raise ValueError("topk_weights is taken with topk_ids only")

    def table(shape: tuple[int, ...], dtype: torch.dtype = torch.int32) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device=device)

    # Routing tables in the router's sizes (the kernels own their initialisation).
    tables: dict[str, torch.Tensor] = {
        "topk_packed": topk_ids if topk_ids is not None else table((T, top_k)),
        "topk_weights": (
            topk_weights
            if topk_weights is not None and topk_weights.dtype == torch.bfloat16
            else table((T, top_k), torch.bfloat16)
        ),
        "expert_count_histogram": table(
            (max(2 * num_experts, int(routing_constants["histogram_min_entries"])),)
        ),
        "total_num_padded_tokens": table((1,)),
        "expanded_idx_to_permuted_idx": table((T, top_k)),
        "permuted_idx_to_token_idx": table(
            (g.max_padded + int(routing_constants["permuted_idx_slack"]),)
        ),
        "cta_idx_xy_to_batch_idx": table((g.max_ctas,)),
        "cta_idx_xy_to_mn_limit": table((g.max_ctas,)),
        "num_non_exiting_ctas": table((1,)),
    }
    if poison_routing_tail:
        poison = int(routing_constants["tail_poison"])
        tables["cta_idx_xy_to_batch_idx"].fill_(poison)
        tables["cta_idx_xy_to_mn_limit"].copy_(
            (torch.arange(g.max_ctas, device=device, dtype=torch.int32) + 1) * tile
        )
        tables["permuted_idx_to_token_idx"].fill_(poison)
        tables["expanded_idx_to_permuted_idx"].fill_(-1)
    expert_weights = (
        topk_weights if topk_weights is not None else tables["topk_weights"]
    )

    buffers: dict[str, torch.Tensor] = dict(tables)
    fc1_out_dtype = str(fc1_unit["dtypes"][2])
    buffers["gemm1_output"] = torch.empty(
        (g.max_padded, I // 2 if fc1_out_dtype == "E2m1" else I),
        dtype=_torch_dtype(fc1_unit["arg_dtypes"]["C"]),
        device=device,
    )
    fc1_keys = _arg_keys(resolved.fc1_program)
    fc2_keys = _arg_keys(resolved.fc2_program)
    gemm1_output_scale = None
    if "SFC" in fc1_keys and not fc1_unit["per_token"]:
        block = _SF_BLOCK_ELEMENTS.get(fc1_out_dtype)
        if block is None:
            raise ValueError(
                f"FC1 unit {resolved.fc1_program} writes block scales for output dtype "
                f"{fc1_out_dtype}, which has no block-scale granularity"
            )
        rows_per_block = int(requant_constants["sf_block_rows"][resolved.sf_layout_a])
        if g.max_padded % rows_per_block:
            raise ValueError(
                f"the {resolved.sf_layout_a} block scales of tile {tile} need a padded row "
                f"count divisible by {rows_per_block}, got {g.max_padded} for {T} tokens"
            )
        gemm1_output_scale = torch.empty(
            (g.max_padded * I // block,), dtype=torch.uint8, device=device
        )
        buffers["gemm1_output_scale"] = gemm1_output_scale
    if "work_counter" in fc1_keys or "work_counter" in fc2_keys:
        buffers["work_counter"] = torch.zeros((1,), dtype=torch.int32, device=device)
    buffers["gemm2_output"] = torch.empty(
        (g.max_padded, H), dtype=torch.bfloat16, device=device
    )
    if do_finalize:
        if output is None:
            output = torch.empty((T, H), dtype=torch.bfloat16, device=device)
        buffers["output"] = _tensor(
            "output", output, device=device, dtype=torch.bfloat16, shape=(T, H)
        )
    elif output is not None:
        raise ValueError("output is taken with do_finalize=True only")

    launches: list[_Launch] = []
    routing_programs, routing_launches = _routing_launches(
        routing_logits=routing_logits,
        topk_ids=topk_ids,
        tables=tables,
        num_tokens=T,
        num_experts=num_experts,
        top_k=top_k,
        tile_tokens_dim=tile,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        device=device,
        arch=arch,
        enable_pdl=enable_pdl,
        reserved_sms=reserved_sms,
    )
    launches.extend(routing_launches)

    fc1_operands = dict(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        output1_scale_scalar=output1_scale_scalar,
        output1_scale_gate_scalar=output1_scale_gate_scalar,
        gemm1_clamp_limit=gemm1_clamp_limit,
        per_token_scale=per_token_scale,
        permuted_idx_to_token_idx=tables["permuted_idx_to_token_idx"],
        cta_idx_xy_to_batch_idx=tables["cta_idx_xy_to_batch_idx"],
        cta_idx_xy_to_mn_limit=tables["cta_idx_xy_to_mn_limit"],
        num_non_exiting_ctas=tables["num_non_exiting_ctas"],
        total_num_padded_tokens=tables["total_num_padded_tokens"],
        gemm1_output=buffers["gemm1_output"],
        gemm1_output_scale=gemm1_output_scale,
        work_counter=buffers.get("work_counter"),
    )
    values, grid = _fc1_values(resolved.fc1_program, family, g, fc1_operands)
    launches.append(
        _bind(
            resolved.fc1_program,
            values,
            grid,
            enable_pdl=enable_pdl,
            arch=arch,
            stage="fc1",
        )
    )

    gemm2_input = buffers["gemm1_output"]
    gemm2_input_scale = gemm1_output_scale
    fc2_per_token_scale = None
    if facts["per_token"]:
        if resolved.requant_program is None:
            raise ValueError(
                f"family {family!r} is per-token but the plan has no requant unit"
            )
        sf_multiple = int(requant_constants["sf_rows_multiple"])
        sf_rows = -(-g.max_padded // sf_multiple) * sf_multiple
        buffers["requant_output"] = torch.empty(
            (g.max_padded, I // 2), dtype=torch.uint8, device=device
        )
        buffers["requant_output_scale"] = torch.empty(
            (sf_rows * I // 16,), dtype=torch.uint8, device=device
        )
        buffers["requant_per_token_scale"] = torch.empty(
            (g.max_padded,), dtype=torch.float32, device=device
        )
        values, grid = _requant_values(
            resolved.requant_program,
            dict(
                gemm1_output=buffers["gemm1_output"],
                expanded_idx_to_permuted_idx=tables["expanded_idx_to_permuted_idx"],
                output=buffers["requant_output"],
                output_scale=buffers["requant_output_scale"],
                per_token_scale=buffers["requant_per_token_scale"],
            ),
            max_padded=g.max_padded,
            global_scale_inv=global_scale_inv,
        )
        launches.append(
            _bind(
                resolved.requant_program,
                values,
                grid,
                enable_pdl=enable_pdl,
                arch=arch,
                stage="requant",
            )
        )
        gemm2_input = buffers["requant_output"]
        gemm2_input_scale = buffers["requant_output_scale"]
        fc2_per_token_scale = buffers["requant_per_token_scale"]

    fc2_operands = dict(
        gemm2_input=gemm2_input,
        gemm2_input_scale=gemm2_input_scale if "SFB" in fc2_keys else None,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        output2_scale_scalar=output2_scale_scalar,
        per_token_scale=fc2_per_token_scale,
        cta_idx_xy_to_batch_idx=tables["cta_idx_xy_to_batch_idx"],
        cta_idx_xy_to_mn_limit=tables["cta_idx_xy_to_mn_limit"],
        num_non_exiting_ctas=tables["num_non_exiting_ctas"],
        total_num_padded_tokens=tables["total_num_padded_tokens"],
        gemm2_output=buffers["gemm2_output"],
        work_counter=buffers.get("work_counter"),
    )
    values, grid = _fc2_values(resolved.fc2_program, family, g, fc2_operands)
    launches.append(
        _bind(
            resolved.fc2_program,
            values,
            grid,
            enable_pdl=enable_pdl,
            arch=arch,
            stage="fc2",
        )
    )

    finalize_name = None
    if do_finalize:
        finalize_name = _finalize_program_for(H, T, expert_weights)
        values, grid = _finalize_values(
            finalize_name,
            dict(
                gemm2_output=buffers["gemm2_output"],
                expert_weights=expert_weights,
                expanded_idx_to_permuted_idx=tables["expanded_idx_to_permuted_idx"],
                total_num_padded_tokens=tables["total_num_padded_tokens"],
                output=buffers["output"],
            ),
        )
        launches.append(
            _bind(
                finalize_name,
                values,
                grid,
                enable_pdl=enable_pdl,
                arch=arch,
                stage="finalize",
            )
        )
    return StepFunMoeLaunch(
        family=family,
        arch=arch,
        plan=resolved,
        do_finalize=do_finalize,
        routing_programs=routing_programs,
        finalize_program=finalize_name,
        launches=tuple(launches),
        buffers=buffers,
        expert_weights=expert_weights,
    )


# ---------------------------------------------------------------------------
# Stage entries (caller-owned buffers; the operand lists of the former FFI operations)
# ---------------------------------------------------------------------------


def prepare_routing(
    *,
    routing_logits: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    topk_packed: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_count_histogram: torch.Tensor,
    total_num_padded_tokens: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    permuted_idx_to_token_idx: torch.Tensor,
    cta_idx_xy_to_batch_idx: torch.Tensor,
    cta_idx_xy_to_mn_limit: torch.Tensor,
    num_non_exiting_ctas: torch.Tensor,
    num_experts: int,
    top_k: int,
    local_expert_offset: int,
    local_num_experts: int,
    tile_tokens_dim: int,
    enable_pdl: bool = True,
    reserved_sms: int = 8,
) -> StepFunStageLaunch:
    """Bind the Renormalize routing stage (one launch, or histogram + cooperative pair)
    on caller-owned tables sized for ``tile_tokens_dim``."""
    _require_registry()
    source = routing_logits if routing_logits is not None else topk_ids
    if not isinstance(source, torch.Tensor):
        raise ValueError("exactly one of routing_logits and topk_ids must be given")
    device = source.device
    num_tokens = _routing_checks(
        routing_logits=routing_logits,
        topk_ids=topk_ids,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        tile_tokens_dim=tile_tokens_dim,
        device=device,
    )
    arch = arch_for(device)
    max_ctas = max_num_ctas(num_tokens, top_k, num_experts, tile_tokens_dim)
    max_padded = max_ctas * tile_tokens_dim
    tables = {
        "topk_packed": _tensor(
            "topk_packed",
            topk_packed,
            device=device,
            dtype=torch.int32,
            shape=(num_tokens, top_k),
        ),
        "topk_weights": _tensor(
            "topk_weights",
            topk_weights,
            device=device,
            dtype=torch.bfloat16,
            shape=(num_tokens, top_k),
        ),
        "expert_count_histogram": _tensor(
            "expert_count_histogram",
            expert_count_histogram,
            device=device,
            dtype=torch.int32,
            min_numel=2 * num_experts,
        ),
        "total_num_padded_tokens": total_num_padded_tokens,
        "expanded_idx_to_permuted_idx": _tensor(
            "expanded_idx_to_permuted_idx",
            expanded_idx_to_permuted_idx,
            device=device,
            dtype=torch.int32,
            min_numel=num_tokens * top_k,
        ),
        "permuted_idx_to_token_idx": permuted_idx_to_token_idx,
        "cta_idx_xy_to_batch_idx": cta_idx_xy_to_batch_idx,
        "cta_idx_xy_to_mn_limit": cta_idx_xy_to_mn_limit,
        "num_non_exiting_ctas": num_non_exiting_ctas,
    }
    _routing_tables_check(
        tables, device=device, max_ctas=max_ctas, max_padded=max_padded
    )
    programs, launches = _routing_launches(
        routing_logits=routing_logits,
        topk_ids=topk_ids,
        tables=tables,
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        tile_tokens_dim=tile_tokens_dim,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        device=device,
        arch=arch,
        enable_pdl=enable_pdl,
        reserved_sms=reserved_sms,
    )
    return StepFunStageLaunch(
        stage="routing",
        programs=programs,
        grids=tuple(item.grid for item in launches),
        launches=tuple(launches),
    )


def prepare_fc1(
    *,
    family: str,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor | None = None,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor | None = None,
    output1_scale_scalar: torch.Tensor | None = None,
    output1_scale_gate_scalar: torch.Tensor | None = None,
    gemm1_clamp_limit: torch.Tensor,
    per_token_scale: torch.Tensor | None = None,
    permuted_idx_to_token_idx: torch.Tensor,
    cta_idx_xy_to_batch_idx: torch.Tensor,
    cta_idx_xy_to_mn_limit: torch.Tensor,
    num_non_exiting_ctas: torch.Tensor,
    total_num_padded_tokens: torch.Tensor,
    gemm1_output: torch.Tensor,
    gemm1_output_scale: torch.Tensor | None = None,
    top_k: int,
    tile_tokens_dim: int,
    enable_pdl: bool = True,
) -> StepFunStageLaunch:
    """Bind the FC1 (StepFun epilogue) unit of ``(family, tile_tokens_dim)`` on
    caller-owned routing tables and output buffers.  The CTA capacity follows the
    weight rows (``gemm1_weights.shape[0]`` experts), as the former operation did."""
    _require_registry()
    device = _tensor("hidden_states", hidden_states, dim=2).device
    arch = arch_for(device)
    _positive_int("top_k", top_k)
    _positive_int("tile_tokens_dim", tile_tokens_dim)
    program = fc1_program(family, tile_tokens_dim)
    g = _geometry(
        family,
        hidden_states,
        gemm1_weights,
        _fc2_probe(family, gemm1_weights, hidden_states),
        top_k=top_k,
        num_experts_for_ctas=int(gemm1_weights.shape[0]),
        tile_tokens_dim=tile_tokens_dim,
    )
    error = _fc1_shape_error(program, g.hidden_size, g.intermediate_size)
    if error is not None:
        raise ValueError(f"FC1 family {family!r} tile {tile_tokens_dim}: {error}")
    tables = dict(
        permuted_idx_to_token_idx=permuted_idx_to_token_idx,
        cta_idx_xy_to_batch_idx=cta_idx_xy_to_batch_idx,
        cta_idx_xy_to_mn_limit=cta_idx_xy_to_mn_limit,
        num_non_exiting_ctas=num_non_exiting_ctas,
        total_num_padded_tokens=total_num_padded_tokens,
    )
    _routing_tables_check(
        tables, device=device, max_ctas=g.max_ctas, max_padded=g.max_padded
    )
    work_counter = (
        torch.zeros((1,), dtype=torch.int32, device=device)
        if "work_counter" in _arg_keys(program)
        else None
    )
    operands = dict(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        output1_scale_scalar=output1_scale_scalar,
        output1_scale_gate_scalar=output1_scale_gate_scalar,
        gemm1_clamp_limit=gemm1_clamp_limit,
        per_token_scale=per_token_scale,
        gemm1_output=gemm1_output,
        gemm1_output_scale=gemm1_output_scale,
        work_counter=work_counter,
        **tables,
    )
    values, grid = _fc1_values(program, family, g, operands)
    return _stage_launch(
        program, values, grid, enable_pdl=enable_pdl, arch=arch, stage="fc1"
    )


def _fc2_probe(
    family: str, gemm1_weights: torch.Tensor, hidden_states: torch.Tensor
) -> torch.Tensor:
    """A zero-copy stand-in for ``gemm2_weights`` with the FC2 element count the FC1
    weights imply (``E x hidden_size x intermediate_size``), so ``_geometry`` reads the
    intermediate size of a standalone FC1 call from the FC1 weights alone."""
    per_byte = act_per_byte(family)
    num_rows = int(gemm1_weights.shape[0])
    hidden_size = int(hidden_states.shape[1]) * per_byte
    total = gemm1_weights.numel() * per_byte
    if num_rows < 1 or total % (2 * num_rows * hidden_size):
        raise ValueError(
            f"gemm1_weights ({tuple(gemm1_weights.shape)}, {gemm1_weights.dtype}) does not "
            f"hold E x 2*intermediate_size x hidden_size elements for hidden_size "
            f"{hidden_size}"
        )
    intermediate_size = total // (2 * num_rows * hidden_size)
    flat = gemm1_weights.reshape(-1)
    return flat[: num_rows * hidden_size * intermediate_size // per_byte].reshape(
        num_rows, hidden_size, intermediate_size // per_byte
    )


def prepare_requant(
    *,
    gemm1_output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    per_token_scale: torch.Tensor,
    sf_layout: str,
    enable_pdl: bool = True,
    global_scale_inv: float = 1.0 / (448.0 * 6.0),
) -> StepFunStageLaunch:
    """Bind the NVFP4 per-token requantization of the bf16 FC1 output into
    ``output`` (E2m1), ``output_scale`` (block scales in ``sf_layout``) and
    ``per_token_scale``; one CTA per expanded token."""
    _require_registry()
    device = _tensor("gemm1_output", gemm1_output, dim=2).device
    arch = arch_for(device)
    program = requant_program(sf_layout)
    values, grid = _requant_values(
        program,
        dict(
            gemm1_output=gemm1_output,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
            output=output,
            output_scale=output_scale,
            per_token_scale=per_token_scale,
        ),
        max_padded=int(gemm1_output.shape[0]),
        global_scale_inv=global_scale_inv,
    )
    return _stage_launch(
        program, values, grid, enable_pdl=enable_pdl, arch=arch, stage="requant"
    )


def prepare_fc2(
    *,
    family: str,
    gemm2_input: torch.Tensor,
    gemm2_input_scale: torch.Tensor | None = None,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor | None = None,
    output2_scale_scalar: torch.Tensor | None = None,
    per_token_scale: torch.Tensor | None = None,
    cta_idx_xy_to_batch_idx: torch.Tensor,
    cta_idx_xy_to_mn_limit: torch.Tensor,
    num_non_exiting_ctas: torch.Tensor,
    total_num_padded_tokens: torch.Tensor,
    gemm2_output: torch.Tensor,
    num_tokens: int,
    top_k: int,
    tile_tokens_dim: int,
    enable_pdl: bool = True,
) -> StepFunStageLaunch:
    """Bind the FC2 unit of ``(family, tile_tokens_dim)`` on the permuted FC1 (or
    requantized) output and caller-owned routing tables.  As the former operation did,
    ``hidden_size`` is the ``gemm2_output`` column count and ``intermediate_size`` the
    ``gemm2_input`` column count times the elements per byte; ``gemm2_weights`` gives
    the expert rows (the CTA capacity) and is cross-checked by element count, so both
    the MajorK ``[E, H, I/act_per_byte]`` view and the bf16 BlockMajorK
    ``[E, I/64, H, 64]`` view are accepted."""
    _require_registry()
    device = _tensor("gemm2_input", gemm2_input, dim=2).device
    arch = arch_for(device)
    _positive_int("num_tokens", num_tokens)
    _positive_int("top_k", top_k)
    _positive_int("tile_tokens_dim", tile_tokens_dim)
    program = fc2_program(family, tile_tokens_dim)
    per_byte = act_per_byte(family)
    output = _tensor(
        "gemm2_output", gemm2_output, device=device, dtype=torch.bfloat16, dim=2
    )
    hidden_size = int(output.shape[1])
    intermediate_size = int(gemm2_input.shape[1]) * per_byte
    weights = _tensor("gemm2_weights", gemm2_weights, device=device)
    if weights.dim() < 2:
        raise ValueError(f"gemm2_weights must be [E, ...], got {tuple(weights.shape)}")
    num_rows = int(weights.shape[0])
    if (
        num_rows < 1
        or weights.numel() * per_byte != num_rows * hidden_size * intermediate_size
    ):
        raise ValueError(
            f"gemm2_weights ({tuple(weights.shape)}, {weights.dtype}) does not hold "
            f"E x hidden_size x intermediate_size elements for E={num_rows}, "
            f"hidden_size={hidden_size} (gemm2_output columns), "
            f"intermediate_size={intermediate_size} (gemm2_input columns x {per_byte})"
        )
    error = _fc2_shape_error(program, hidden_size, intermediate_size)
    if error is not None:
        raise ValueError(f"FC2 family {family!r} tile {tile_tokens_dim}: {error}")
    max_ctas = max_num_ctas(num_tokens, top_k, num_rows, tile_tokens_dim)
    g = _Geometry(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        num_experts=num_rows,
        intermediate_size=intermediate_size,
        top_k=top_k,
        tile_tokens_dim=tile_tokens_dim,
        max_ctas=max_ctas,
        max_padded=max_ctas * tile_tokens_dim,
    )
    tables = dict(
        cta_idx_xy_to_batch_idx=cta_idx_xy_to_batch_idx,
        cta_idx_xy_to_mn_limit=cta_idx_xy_to_mn_limit,
        num_non_exiting_ctas=num_non_exiting_ctas,
        total_num_padded_tokens=total_num_padded_tokens,
    )
    _routing_tables_check(
        tables, device=device, max_ctas=max_ctas, max_padded=g.max_padded
    )
    work_counter = (
        torch.zeros((1,), dtype=torch.int32, device=device)
        if "work_counter" in _arg_keys(program)
        else None
    )
    operands = dict(
        gemm2_input=gemm2_input,
        gemm2_input_scale=gemm2_input_scale,
        gemm2_weights=weights,
        gemm2_weights_scale=gemm2_weights_scale,
        output2_scale_scalar=output2_scale_scalar,
        per_token_scale=per_token_scale,
        gemm2_output=gemm2_output,
        work_counter=work_counter,
        **tables,
    )
    values, grid = _fc2_values(program, family, g, operands)
    return _stage_launch(
        program, values, grid, enable_pdl=enable_pdl, arch=arch, stage="fc2"
    )


def prepare_finalize(
    *,
    gemm2_output: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    total_num_padded_tokens: torch.Tensor,
    output: torch.Tensor,
    num_experts: int,
    enable_pdl: bool = True,
) -> StepFunStageLaunch:
    """Bind the finalize unit (scalar / vector by the dispatch rule, expert-weight
    dtype by ``expert_weights``) reducing the permuted FC2 rows into ``output``."""
    _require_registry()
    device = _tensor("output", output, dim=2).device
    arch = arch_for(device)
    _positive_int("num_experts", num_experts)
    _tensor(
        "total_num_padded_tokens",
        total_num_padded_tokens,
        device=device,
        dtype=torch.int32,
        min_numel=1,
    )
    weights = _tensor("expert_weights", expert_weights, device=device, dim=2)
    program = _finalize_program_for(int(output.shape[1]), int(output.shape[0]), weights)
    values, grid = _finalize_values(
        program,
        dict(
            gemm2_output=gemm2_output,
            expert_weights=weights,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
            total_num_padded_tokens=total_num_padded_tokens,
            output=output,
        ),
    )
    return _stage_launch(
        program, values, grid, enable_pdl=enable_pdl, arch=arch, stage="finalize"
    )


def routing(**kwargs: Any) -> None:
    """``prepare_routing(**kwargs).launch()``."""
    prepare_routing(**kwargs).launch()


def fc1(**kwargs: Any) -> None:
    """``prepare_fc1(**kwargs).launch()``."""
    prepare_fc1(**kwargs).launch()


def requant(**kwargs: Any) -> None:
    """``prepare_requant(**kwargs).launch()``."""
    prepare_requant(**kwargs).launch()


def fc2(**kwargs: Any) -> None:
    """``prepare_fc2(**kwargs).launch()``."""
    prepare_fc2(**kwargs).launch()


def finalize(**kwargs: Any) -> None:
    """``prepare_finalize(**kwargs).launch()``."""
    prepare_finalize(**kwargs).launch()
