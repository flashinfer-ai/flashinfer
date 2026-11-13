# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Admission and arithmetic identity for the specified NVFP4 MegaMoE paths.

Keep this module CPU safe. Numerical implementations are admitted explicitly;
adding a general tuning knob must never silently extend the BI plan space.
"""

from __future__ import annotations

import math
from typing import Any

NVFP4_BATCH_INVARIANT_VERSION = 2


def nvfp4_batch_invariant_metadata(config: Any) -> dict:
    """Metadata passed by public APIs, whose three expert scalars resolve to tensors."""
    if not getattr(config, "batch_invariant", False):
        return {}
    return dict(
        batch_invariant=True,
        input_norm_const=config.input_norm_const,
        has_fc1_alpha=True,
        has_fc2_alpha=True,
        has_fc1_norm_const=True,
    )


_LEGACY_DEFAULTS = {
    "mma_tiler_mnk": (256, 128, 256),
    "cluster_shape_mnk": (2, 1, 1),
    "use_2cta_instrs": True,
    "force_static_sched": True,
    "clc_bundle_size": None,
    "num_sched_stages": None,
    "enable_iket": False,
    "group_hint": None,
    "flag_batch": 1,
    "epi_flag_batch": (1, 1),
    "non_ubulk_fc2_store": True,
    "load_balance_mode": "static",
    "token_back_mode": "epi_warps",
    "in_kernel_fc2_reduce": False,
}
_STANDARD_DEFAULTS = {
    "mma_tiler_mnk": (256, 128, 128),
    "cluster_shape_mn": (2, 1),
    "fallback_cluster_shape_mn": None,
    "schedule_policy": ("grouped", None),
    "work_id_mode": "grid_stride",
    "fc2_use_bulk": False,
    "fc2_tma_stages": None,
    "epi_flag_batches": (4, 2),
    "token_in_flag_batch": 1,
    "token_back_mode": "epi_warps",
    "reduce_topk_in_kernel": False,
}


def nvfp4_batch_invariant_implementation(config: Any) -> str:
    """Resolve the arithmetic implementation without consulting batch size."""
    if getattr(config, "kernel_variant", "inference") == "genphase":
        return "genphase"
    name = getattr(config, "kernel_name", "")
    if name in ("sm100_nvfp4_nvfp4_bf16_cutedsl", "nvfp4_cutedsl"):
        return "legacy"
    if config.__class__.__name__ == "MegaMoENvfp4Config":
        return "legacy"
    return "standard"


def nvfp4_batch_invariant_arch() -> str:
    """Bind receipts to the current GPU architecture; CPU metadata defaults to SM100."""
    from .batch_invariant import bf16_batch_invariant_arch

    return bf16_batch_invariant_arch()


def nvfp4_batch_invariant_default_knobs(
    implementation: str, *, arch: str | None = None
) -> dict:
    """Return a batch-independent, fixed geometry for one numerical implementation."""
    if implementation == "legacy":
        return dict(_LEGACY_DEFAULTS)
    if implementation not in ("standard", "genphase"):
        raise ValueError(f"unknown NVFP4 BI implementation: {implementation!r}")
    knobs = dict(_STANDARD_DEFAULTS)
    if arch == "sm_107" or implementation == "genphase":
        knobs["mma_tiler_mnk"] = (256, 128, 256)
    if implementation == "genphase":
        knobs.update(cluster_shape_mn=(4, 1), fc2_use_bulk=True)
    return knobs


def _same(value, expected) -> bool:
    if isinstance(expected, tuple):
        return (
            type(value) is tuple
            and len(value) == len(expected)
            and all(_same(item, ref) for item, ref in zip(value, expected, strict=True))
        )
    return type(value) is type(expected) and value == expected


def nvfp4_batch_invariant_knobs_valid(
    knobs: dict, *, implementation: str, arch: str | None = None
) -> bool:
    """Validate persisted plans without importing or allocating a kernel workspace."""
    from types import SimpleNamespace

    try:
        validate_nvfp4_batch_invariant(
            SimpleNamespace(batch_invariant=True),
            knobs,
            implementation=implementation,
            arch=arch,
        )
    except (TypeError, ValueError):
        return False
    return True


def validate_nvfp4_batch_invariant(
    config: Any,
    knobs: dict | None = None,
    *,
    implementation: str | None = None,
    arch: str | None = None,
) -> None:
    """Reject an unsupported numerical mode or plan before allocation/compilation."""
    if type(getattr(config, "batch_invariant", False)) is not bool:
        raise ValueError("batch_invariant must be a bool")
    if not getattr(config, "batch_invariant", False):
        return
    implementation = implementation or nvfp4_batch_invariant_implementation(config)
    arch = arch or nvfp4_batch_invariant_arch()
    if arch not in ("sm_100", "sm_103", "sm_107"):
        raise ValueError(f"unsupported NVFP4 BI architecture: {arch!r}")
    if implementation == "legacy" and arch == "sm_107":
        raise ValueError("NVFP4 BI legacy implementation requires SM100 or SM103")
    if getattr(config, "quant_kind", "nvfp4") != "nvfp4":
        raise ValueError("batch_invariant NVFP4 policy requires quant_kind='nvfp4'")
    if getattr(config, "kernel_variant", "inference") == "auto":
        raise ValueError("batch_invariant NVFP4 does not support kernel_variant='auto'")
    top_k = getattr(config, "top_k", getattr(config, "num_topk", None))
    if top_k is not None and (type(top_k) is not int or not 1 <= top_k <= 32):
        raise ValueError("batch_invariant NVFP4 top_k must be an integer in [1, 32]")
    intermediate = getattr(
        config, "intermediate_size", getattr(config, "intermediate", None)
    )
    alignment = (
        128 if implementation == "legacy" and hasattr(config, "intermediate") else 64
    )
    if intermediate is not None and (
        type(intermediate) is not int or intermediate <= 0 or intermediate % alignment
    ):
        raise ValueError(
            "batch_invariant NVFP4 post-activation intermediate must be a positive multiple of 64"
        )
    hidden = getattr(config, "hidden", None)
    if hidden is not None and (type(hidden) is not int or hidden <= 0 or hidden % 64):
        raise ValueError(
            "batch_invariant NVFP4 hidden must be a positive multiple of 64"
        )
    if type(getattr(config, "apply_routing_weights_before_fc2", True)) is not bool:
        raise ValueError(
            "batch_invariant NVFP4 apply_routing_weights_before_fc2 must be a bool"
        )
    from .activation_batch_invariant import (
        batch_invariant_activation,
        batch_invariant_variant_kwargs,
        block_scaled_activation_family,
    )

    batch_invariant_activation(
        config, family=block_scaled_activation_family(implementation, arch, nvfp4=True)
    )
    variant = batch_invariant_variant_kwargs(config)
    if variant.get("per_token_scale") and not (
        implementation == "legacy"
        or (implementation == "standard" and arch in ("sm_100", "sm_103"))
    ):
        raise ValueError(
            "per-token FC1 scaling BI is only implemented by legacy and Blackwell standard NVFP4"
        )
    fixed = {
        "enable_in_kernel_fc2_reduce": False,
        "in_kernel_fc2_reduce": False,
        "reduce_topk_in_kernel": False,
    }
    for name, expected in fixed.items():
        if not _same(getattr(config, name, expected), expected):
            raise ValueError(f"batch_invariant NVFP4 requires {name}={expected!r}")
    for name in ("has_fc1_alpha", "has_fc2_alpha", "has_fc1_norm_const"):
        if not _same(getattr(config, name, True), True):
            raise ValueError(
                f"batch_invariant NVFP4 requires {name}=True (use explicit ones tensors)"
            )
    from .activation_batch_invariant import validate_batch_invariant_output

    validate_batch_invariant_output(
        config, implementation=implementation, arch=arch, kind="nvfp4"
    )
    norm = getattr(config, "input_norm_const", 1.0)
    if (
        isinstance(norm, bool)
        or not isinstance(norm, (int, float))
        or not math.isfinite(norm)
        or norm <= 0
    ):
        raise ValueError(
            "batch_invariant NVFP4 input_norm_const must be positive and finite"
        )
    from flashinfer.moe_ep.batch_invariant.spec import f32_bits

    norm_bits = int(f32_bits(norm), 16)
    if norm_bits == 0 or norm_bits & 0x7F800000 == 0x7F800000:
        raise ValueError(
            "batch_invariant NVFP4 input_norm_const must round to positive finite FP32"
        )
    defaults = nvfp4_batch_invariant_default_knobs(implementation, arch=arch)
    supplied = dict(knobs or {})
    if knobs is None and isinstance(getattr(config, "knobs", None), dict):
        supplied = dict(config.knobs)
    admitted_knobs = set(defaults)
    if implementation == "genphase" and arch in ("sm_100", "sm_103"):
        admitted_knobs = {"mma_tiler_mnk", "cluster_shape_mn", "schedule_policy"}
    if unknown := set(supplied) - admitted_knobs:
        raise ValueError(
            f"batch_invariant NVFP4 has unvalidated tuning knobs: {sorted(unknown)}"
        )
    scheduling = {
        "token_back_mode": ("epi_warps", "standalone_warps", "reuse_dispatch_warps"),
        "load_balance_mode": ("static", "atomic_counter"),
        "work_id_mode": ("grid_stride", "atomic_counter"),
        "schedule_policy": (("grouped", None), ("phase_interleave", None)),
    }
    if implementation == "genphase":
        scheduling = {"work_id_mode": ("grid_stride", "atomic_counter")}
    if implementation == "standard":
        from .quantized_batch_invariant_tiles import (
            validate_quantized_batch_invariant_tile,
        )

        validate_quantized_batch_invariant_tile(
            config, kind="nvfp4", arch=arch, knobs=supplied
        )
    for name, expected in defaults.items():
        value = supplied.get(name, getattr(config, name, expected))
        if value is None and name in (
            "mma_tiler_mnk",
            "cluster_shape_mn",
            "schedule_policy",
        ):
            value = expected
        if implementation == "standard" and name == "mma_tiler_mnk":
            continue
        allowed = scheduling.get(name, (expected,))
        if not any(_same(value, item) for item in allowed):
            raise ValueError(
                f"batch_invariant NVFP4 does not admit {name}={value!r} for {arch}/{implementation}"
            )
    if implementation == "standard":
        schedule = supplied.get(
            "schedule_policy",
            getattr(config, "schedule_policy", defaults["schedule_policy"]),
        )
        work = supplied.get(
            "work_id_mode", getattr(config, "work_id_mode", defaults["work_id_mode"])
        )
        if schedule == ("phase_interleave", None) and work != "atomic_counter":
            raise ValueError(
                "NVFP4 phase_interleave requires work_id_mode='atomic_counter'"
            )


def nvfp4_batch_invariant_fingerprint(
    config: Any,
    *,
    hidden: int | None = None,
    num_experts: int | None = None,
    implementation: str | None = None,
    arch: str | None = None,
    public_inputs: bool = False,
) -> str:
    """Describe the actual provider chain; scale tensor payloads remain operator inputs."""
    from .activation_batch_invariant import batch_invariant_fingerprint_kwargs

    from flashinfer.moe_ep.batch_invariant.nvfp4_spec import spec_fingerprint

    implementation = implementation or nvfp4_batch_invariant_implementation(config)
    arch = arch or nvfp4_batch_invariant_arch()
    validate_nvfp4_batch_invariant(config, implementation=implementation, arch=arch)
    clamp = getattr(config, "gate_up_clamp", None)
    if clamp is None:
        clamp = getattr(config, "activation_clamp", None)
    intermediate = getattr(
        config, "intermediate", getattr(config, "intermediate_size", None)
    )
    if implementation == "legacy" and hasattr(config, "intermediate"):
        intermediate //= 2
    if implementation == "standard":
        from .quantized_batch_invariant_tiles import (
            validate_quantized_batch_invariant_tile,
        )

        validate_quantized_batch_invariant_tile(
            config,
            kind="nvfp4",
            arch=arch,
            knobs=getattr(config, "knobs", None)
            if isinstance(getattr(config, "knobs", None), dict)
            else None,
            hidden=hidden,
        )
    return spec_fingerprint(
        hidden=hidden if hidden is not None else config.hidden,
        intermediate=intermediate,
        top_k=getattr(config, "num_topk", getattr(config, "top_k", None)),
        num_experts=num_experts
        if num_experts is not None
        else config.num_total_experts,
        arch=arch,
        implementation=implementation,
        apply_routing_weights_before_fc2=config.apply_routing_weights_before_fc2,
        gate_up_clamp=clamp,
        input_norm_const=getattr(config, "input_norm_const", 1.0),
        **{
            f"has_{name}": public_inputs
            or getattr(config, f"has_{name}", getattr(config, name, None) is not None)
            for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const")
        },
        **batch_invariant_fingerprint_kwargs(config),
    )
