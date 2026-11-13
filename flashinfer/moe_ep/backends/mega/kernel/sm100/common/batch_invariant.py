# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Numerical policy for the SM100/SM103/SM107 BF16 MegaMoE path.

The policy version identifies admitted implementations and tuning records.
The separate arithmetic fingerprint identifies the computation, independent
of scheduling, geometry and token count.
Scheduling may move complete output tiles between workers, but every tile
uses the same MMA sequence, BF16 handoff and ordered FP32 top-k reduction.
New numerical implementations must be validated before extending this policy.

This module is shared by public configs and the low-level shim. Keep it free
of kernel imports so validating a config does not bootstrap a CuTe source tree.
"""

from __future__ import annotations

import dataclasses
from typing import Any

# v3 admits shape-compatible tile variants; the v2 arithmetic fingerprint is
# unchanged. Invalidate old single-geometry tuning records to permit a sweep
# of the expanded implementation space.
BF16_BATCH_INVARIANT_VERSION = 3

# Keep this allowlist separate from the general tuning space: adding a new
# knob must not silently opt it into the batch-invariant numerical contract.
_FIXED_FIELDS = {
    "force_static_sched": True,
    "clc_bundle_size": None,
    "num_sched_stages": None,
    "enable_in_kernel_fc2_reduce": False,
    "in_kernel_fc2_reduce": False,
    "use_custom_finalize": False,
    "enable_iket": False,
    "group_hint": None,
    "flag_batch": 1,
    "epi_flag_batch": (1, 1),
}
_SCHEDULING_VALUES = {
    "load_balance_mode": ("static", "atomic_counter"),
    "token_back_mode": ("epi_warps", "standalone_warps", "reuse_dispatch_warps"),
}
_GEOMETRY_FIELDS = frozenset({"mma_tiler_mnk", "cluster_shape_mnk", "use_2cta_instrs"})
_TUNABLE_FIELDS = (
    frozenset(_FIXED_FIELDS) | frozenset(_SCHEDULING_VALUES) | _GEOMETRY_FIELDS
)
_CONTRACT_FIELDS = frozenset(
    {
        "rank",
        "world_size",
        "num_tokens_per_rank",
        "num_topk",
        "num_total_experts",
        "hidden",
        "intermediate",
        "gate_up_clamp",
        "apply_routing_weights_before_fc2",
        "batch_invariant",
        "intermediate_size",
        "top_k",
        "kernel_name",
        "activation_clamp",
        "fast_math",
        "knobs",
        "finalize_mode",
    }
)


def bf16_plan_config(config: Any = None, knobs: dict | None = None):
    """Resolve correlated geometry exactly as applying a knob override does.

    A tile override derives its CTA count and cluster unless those fields are
    explicitly supplied. An explicit incompatible combination is rejected.
    Config dataclasses themselves contain complete geometries and are never
    silently repaired.
    """
    from flashinfer.moe_ep.batch_invariant.spec import ALLOWED_PLAN_CONFIGS, PlanConfig

    defaults = PlanConfig()
    knobs = {} if knobs is None else knobs
    values = {
        name: knobs.get(name, getattr(config, name, getattr(defaults, name)))
        for name in _GEOMETRY_FIELDS
    }
    if "mma_tiler_mnk" in knobs:
        tile = knobs["mma_tiler_mnk"]
        if type(tile) is not tuple or len(tile) != 3:
            raise ValueError("BF16 MegaMoE mma_tiler_mnk must be an (M, N, K) tuple.")
        use_2cta = tile[0] == 256
        if "use_2cta_instrs" not in knobs:
            values["use_2cta_instrs"] = use_2cta
        if "cluster_shape_mnk" not in knobs:
            values["cluster_shape_mnk"] = (2 if use_2cta else 1, 1, 1)
    geometry = PlanConfig(**values)
    if geometry not in ALLOWED_PLAN_CONFIGS:
        raise ValueError(f"unvalidated BF16 MegaMoE implementation: {geometry}")
    return geometry


def bf16_plan_smem_bytes(
    hidden: int,
    num_experts: int,
    geometry: Any,
    token_back_mode: str = "epi_warps",
) -> int:
    """Minimum per-CTA shared memory for the validated BF16 layouts.

    This includes one A/B pipeline stage, the D epilogue, dispatcher state,
    and the extra standalone token-back buffer. Larger pipelines may use
    remaining memory, but at least one stage must fit within the target's
    capacity before the kernel can be instantiated.
    """
    if type(hidden) is not int or hidden <= 0:
        raise ValueError("hidden must be a positive integer")
    if type(num_experts) is not int or num_experts <= 0:
        raise ValueError("num_experts must be a positive integer")
    if token_back_mode not in _SCHEDULING_VALUES["token_back_mode"]:
        raise ValueError(f"unsupported token_back_mode={token_back_mode!r}")
    geometry = bf16_plan_config(geometry)
    _, n, k = geometry.mma_tiler_mnk
    a = 128 * k * 2
    b = (n // (2 if geometry.use_2cta_instrs else 1)) * k * 2
    d = 128 * n
    misc = 1024 + 32 + ((4 * num_experts + 15) // 16 * 16)
    misc += (8 * hidden + 127) // 128 * 128
    if token_back_mode == "standalone_warps":
        misc += 8224
    return a + b + d + misc


def bf16_batch_invariant_smem_capacity(arch: str | None = None) -> int:
    """Per-CTA capacity for native compilation on an admitted architecture."""
    if arch is None:
        arch = bf16_batch_invariant_arch()
    capacities = {"sm_100": 232448, "sm_103": 232448, "sm_107": 334848}
    if not isinstance(arch, str) or arch not in capacities:
        raise ValueError(
            f"unsupported batch_invariant BF16 MegaMoE architecture: {arch!r}"
        )
    return capacities[arch]


def validate_bf16_batch_invariant_device() -> None:
    """Limit the numerical contract to admitted native device families."""
    import torch

    if not torch.cuda.is_available():
        return
    capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    if capability not in ((10, 0), (10, 3), (10, 7)):
        raise ValueError(
            "batch_invariant BF16 MegaMoE requires SM100, SM103 or SM107; "
            f"got compute capability {capability}."
        )


def validate_bf16_batch_invariant_compile_target() -> None:
    """SM107 uses its native target, including its larger SMEM capacity.

    Query the DSL's captured enum: changing the environment after importing
    CuTe cannot retarget compilation. Legacy SM100/SM103 target behavior is
    unaffected by this additional SM107 check.
    """
    import torch

    if not torch.cuda.is_available():
        return
    if torch.cuda.get_device_capability(torch.cuda.current_device()) != (10, 7):
        return
    import os
    import sys

    # The DSL otherwise detects visible device 0 at its first CuTe import.
    # Select the current SM107 device's native default before that capture,
    # while preserving explicit targets and never resetting an existing DSL.
    if "cutlass.cute" not in sys.modules:
        os.environ.setdefault("CUTE_DSL_ARCH", "sm_107a")
    try:
        from cutlass.base_dsl import Arch
        from cutlass.cutlass_dsl import CuTeDSL

        native = Arch["sm_107a"]
        captured = CuTeDSL._get_dsl().get_arch_enum()
    except (ImportError, KeyError, AttributeError) as exc:
        raise RuntimeError(
            "batch_invariant BF16 MegaMoE on SM107 requires a CuTe DSL build "
            "with native sm_107a support."
        ) from exc
    if captured is not native:
        raise ValueError(
            "batch_invariant BF16 MegaMoE on SM107 requires the captured "
            f"CuTe DSL target sm_107a; got {captured!s}. Restart Python with "
            "CUTE_DSL_ARCH=sm_107a set before importing CuTe DSL."
        )


def validate_bf16_batch_invariant(
    config: Any,
    knobs: dict | None = None,
    *,
    hidden: int | None = None,
    num_experts: int | None = None,
    arch: str | None = None,
) -> None:
    """Reject unsupported effective knobs when the session enables invariance."""
    if not getattr(config, "batch_invariant", False):
        return
    knobs = {} if knobs is None else knobs
    unknown = set(knobs) - _TUNABLE_FIELDS
    if unknown:
        raise ValueError(
            "batch_invariant BF16 MegaMoE does not allow these tuning knobs: "
            f"{sorted(unknown)}."
        )
    fields = {field.name for field in dataclasses.fields(config)}
    unknown_fields = fields - _TUNABLE_FIELDS - _CONTRACT_FIELDS
    if unknown_fields:
        raise ValueError(
            "batch_invariant BF16 MegaMoE has unvalidated config fields: "
            f"{sorted(unknown_fields)}."
        )
    try:
        geometry = bf16_plan_config(config, knobs)
    except ValueError as exc:
        raise ValueError(f"batch_invariant BF16 MegaMoE geometry: {exc}") from exc
    hidden = hidden if hidden is not None else getattr(config, "hidden", None)
    intermediate = getattr(
        config, "intermediate", getattr(config, "intermediate_size", None)
    )
    if hidden is not None and intermediate is not None:
        from flashinfer.moe_ep.batch_invariant.spec import plan_config_compatible

        if not plan_config_compatible(hidden, intermediate, geometry):
            raise ValueError(
                "batch_invariant BF16 MegaMoE tile K must divide both hidden and "
                "intermediate after rounding each to 64; "
                f"got hidden={hidden}, intermediate={intermediate}, "
                f"tile K={geometry.mma_tiler_mnk[2]}."
            )
    num_experts = (
        num_experts
        if num_experts is not None
        else getattr(config, "num_total_experts", None)
    )
    if hidden is not None and num_experts is not None:
        token_back_mode = knobs.get(
            "token_back_mode", getattr(config, "token_back_mode", "epi_warps")
        )
        required = bf16_plan_smem_bytes(hidden, num_experts, geometry, token_back_mode)
        capacity = bf16_batch_invariant_smem_capacity(arch)
        if required > capacity:
            raise ValueError(
                "batch_invariant BF16 MegaMoE plan needs at least "
                f"{required} shared-memory bytes per CTA; the target allows {capacity}."
            )
    for name, expected in _FIXED_FIELDS.items():
        value = knobs.get(name, getattr(config, name, expected))
        if type(value) is not type(expected) or value != expected:
            raise ValueError(
                f"batch_invariant BF16 MegaMoE requires {name}={expected!r}; "
                f"got {value!r}."
            )
    for name, allowed in _SCHEDULING_VALUES.items():
        value = knobs.get(name, getattr(config, name, allowed[0]))
        if not any(type(value) is type(item) and value == item for item in allowed):
            raise ValueError(
                f"batch_invariant BF16 MegaMoE does not support {name}={value!r}."
            )


def bf16_batch_invariant_signature(config: Any) -> tuple:
    """Arithmetic identity shared by every admitted plan and batch shape."""
    validate_bf16_batch_invariant(config)
    return (
        "sm100_bf16",
        BF16_BATCH_INVARIANT_VERSION,
        getattr(config, "world_size", None),
        getattr(config, "num_total_experts", None),
        getattr(config, "num_topk", getattr(config, "top_k", None)),
        getattr(config, "hidden", None),
        getattr(config, "intermediate", getattr(config, "intermediate_size", None)),
        (
            config.gate_up_clamp
            if getattr(config, "gate_up_clamp", None) is not None
            else getattr(config, "activation_clamp", None)
        ),
        getattr(config, "apply_routing_weights_before_fc2", True),
        "bf16_handoff_ordered_fp32_topk",
        bf16_batch_invariant_fingerprint(config),
    )


def bf16_batch_invariant_arch() -> str:
    """The active device family, or SM100 for device-free cache bookkeeping."""
    import torch

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        return f"sm_{major}{minor}"
    return "sm_100"


def bf16_batch_invariant_fingerprint(
    config: Any,
    *,
    hidden: int | None = None,
    num_experts: int | None = None,
    arch: str | None = None,
) -> str:
    """Bind a complete session's numerical settings to the arithmetic spec.

    Public configs obtain the missing dimensions from FleetParams. Host-only
    admission tests use the SM100 class; a public receipt without a CUDA
    device must supply its architecture explicitly through the public API.
    """
    from flashinfer.moe_ep.batch_invariant.spec import spec_fingerprint

    if arch is None:
        arch = bf16_batch_invariant_arch()
    pinned_knobs = getattr(config, "knobs", None)
    validate_bf16_batch_invariant(
        config,
        pinned_knobs if isinstance(pinned_knobs, dict) else None,
        hidden=hidden,
        num_experts=num_experts,
        arch=arch,
    )
    return spec_fingerprint(
        hidden if hidden is not None else getattr(config, "hidden", None),
        getattr(config, "intermediate", getattr(config, "intermediate_size", None)),
        getattr(config, "num_topk", getattr(config, "top_k", None)),
        num_experts
        if num_experts is not None
        else getattr(config, "num_total_experts", None),
        apply_routing_weights_before_fc2=config.apply_routing_weights_before_fc2,
        gate_up_clamp=(
            config.gate_up_clamp
            if config.gate_up_clamp is not None
            else getattr(config, "activation_clamp", None)
        ),
        arch=arch,
    )
