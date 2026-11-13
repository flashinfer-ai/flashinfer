# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CPU-safe admission for the independently versioned W8A8 arithmetic."""

from __future__ import annotations

from typing import Any

from .nvfp4_batch_invariant import _same, nvfp4_batch_invariant_arch

MXFP8_BATCH_INVARIANT_VERSION = 2
MXFP8_KINDS = ("mxfp8_e4m3", "mxfp8_e5m2")


def mxfp8_quant_kind(config):
    return getattr(config, "quant_kind", getattr(config, "kind", None))


def mxfp8_batch_invariant_implementation(config):
    if getattr(config, "kernel_variant", "inference") == "genphase":
        return "genphase"
    if (
        type(config).__name__ == "MegaMoEMxfp8Config"
        or getattr(config, "kernel_name", None) == "sm100_mxfp8_mxfp8_bf16_cutedsl"
    ):
        return "legacy"
    return "standard"


def mxfp8_batch_invariant_default_knobs(
    implementation, *, quant_kind="mxfp8_e4m3", arch=None
):
    if quant_kind not in MXFP8_KINDS:
        raise ValueError("MXFP8 BI requires mxfp8_e4m3 or mxfp8_e5m2")
    if implementation == "legacy":
        return dict(
            mma_tiler_mnk=(256, 256, 128),
            cluster_shape_mnk=(2, 1, 1),
            use_2cta_instrs=True,
            load_balance_mode="static",
            group_hint=None,
            force_static_sched=True,
            clc_bundle_size=None,
            num_sched_stages=None,
            flag_batch=4,
            epi_flag_batch=(1, 1),
            token_back_mode="epi_warps",
            in_kernel_fc2_reduce=False,
            enable_iket=False,
        )
    if implementation not in ("standard", "genphase"):
        raise ValueError("MXFP8 BI requires legacy, standard or genphase")
    return dict(
        mma_tiler_mnk=(256, 128, 128),
        cluster_shape_mn=(4, 1) if implementation == "genphase" else (2, 1),
        fallback_cluster_shape_mn=None,
        schedule_policy=("grouped", None),
        work_id_mode="grid_stride",
        fc2_use_bulk=implementation == "genphase",
        fc2_tma_stages=None,
        epi_flag_batches=(4, 2),
        token_in_flag_batch=1,
        token_back_mode="epi_warps",
        reduce_topk_in_kernel=False,
    )


def mxfp8_batch_invariant_metadata(config):
    if not getattr(config, "batch_invariant", False):
        return {}
    return dict(
        batch_invariant=True,
        input_norm_const=1.0,
        has_fc1_alpha=False,
        has_fc2_alpha=False,
        has_fc1_norm_const=False,
    )


def validate_mxfp8_batch_invariant(
    config: Any, knobs=None, *, implementation=None, arch=None
):
    if type(getattr(config, "batch_invariant", False)) is not bool:
        raise ValueError("batch_invariant must be a bool")
    if not getattr(config, "batch_invariant", False):
        return
    arch = arch or nvfp4_batch_invariant_arch()
    kind = mxfp8_quant_kind(config)
    implementation = implementation or mxfp8_batch_invariant_implementation(config)
    if (
        kind not in MXFP8_KINDS
        or arch not in ("sm_100", "sm_103", "sm_107")
        or (implementation == "legacy" and arch == "sm_107")
    ):
        raise ValueError(
            f"unsupported MXFP8 BI numerical class: {arch}/{kind}/{implementation}"
        )
    if getattr(config, "kernel_variant", "inference") not in ("inference", "genphase"):
        raise ValueError("batch_invariant MXFP8 does not support cross-variant auto")
    for name in ("top_k", "num_topk"):
        if hasattr(config, name) and (
            type(getattr(config, name)) is not int
            or not 1 <= getattr(config, name) <= 32
        ):
            raise ValueError(
                "batch_invariant MXFP8 top_k must be an integer in [1, 32]"
            )
    for name in ("hidden", "intermediate", "intermediate_size"):
        if hasattr(config, name) and (
            type(getattr(config, name)) is not int
            or getattr(config, name) <= 0
            or getattr(config, name) % 128
        ):
            raise ValueError(
                f"batch_invariant MXFP8 {name} must be a positive multiple of 128"
            )
    early = getattr(config, "apply_routing_weights_before_fc2", True)
    if type(early) is not bool:
        raise ValueError(
            "batch_invariant MXFP8 routing-weight placement must be a bool"
        )
    if implementation == "legacy" and not early:
        raise ValueError("legacy batch_invariant MXFP8 requires early routing weights")
    from .activation_batch_invariant import (
        batch_invariant_activation,
        block_scaled_activation_family,
    )

    batch_invariant_activation(
        config, family=block_scaled_activation_family(implementation, arch)
    )
    fixed = dict(
        enable_in_kernel_fc2_reduce=False,
        in_kernel_fc2_reduce=False,
        reduce_topk_in_kernel=False,
        enable_fc1_activation_per_token_scale=False,
        has_fc1_alpha=False,
        has_fc2_alpha=False,
        has_fc1_norm_const=False,
        has_3_global_scales=False,
        input_norm_const=1.0,
        normal_scales=True,
    )
    for name, expected in fixed.items():
        if not _same(getattr(config, name, expected), expected):
            raise ValueError(f"batch_invariant MXFP8 requires {name}={expected!r}")
    for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const"):
        if getattr(config, name, None) is not None:
            raise ValueError(f"batch_invariant MXFP8 does not accept {name}")
    from .activation_batch_invariant import validate_batch_invariant_output

    validate_batch_invariant_output(
        config, implementation=implementation, arch=arch, kind=kind
    )
    defaults = mxfp8_batch_invariant_default_knobs(
        implementation, quant_kind=kind, arch=arch
    )
    supplied = dict(knobs or {})
    if knobs is None and isinstance(getattr(config, "knobs", None), dict):
        supplied = dict(config.knobs)
    keys = set(defaults)
    if implementation == "genphase" and arch in ("sm_100", "sm_103"):
        keys = {"mma_tiler_mnk", "cluster_shape_mn", "schedule_policy"}
    if unknown := set(supplied) - keys:
        raise ValueError(
            f"batch_invariant MXFP8 has unvalidated tuning knobs: {sorted(unknown)}"
        )
    scheduling: dict[str, tuple[Any, ...]] = (
        dict(
            load_balance_mode=("static", "atomic_counter"),
            token_back_mode=("epi_warps", "reuse_dispatch_warps"),
        )
        if implementation == "legacy"
        else (
            dict(
                token_back_mode=(
                    "epi_warps",
                    "standalone_warps",
                    "reuse_dispatch_warps",
                ),
                work_id_mode=("grid_stride", "atomic_counter"),
                schedule_policy=(("grouped", None), ("phase_interleave", None)),
            )
            if implementation == "standard"
            else dict(work_id_mode=("grid_stride", "atomic_counter"))
        )
    )
    if implementation == "standard":
        from .quantized_batch_invariant_tiles import (
            validate_quantized_batch_invariant_tile,
        )

        validate_quantized_batch_invariant_tile(
            config, kind=kind, arch=arch, knobs=supplied
        )
    for name, expected in defaults.items():
        configured = getattr(config, name, expected)
        if implementation == "legacy" and name == "token_back_mode":
            by_dispatch = getattr(config, "token_back_by_dispatch", False)
            if type(by_dispatch) is not bool:
                raise ValueError("token_back_by_dispatch must be a bool")
            configured = "reuse_dispatch_warps" if by_dispatch else "epi_warps"
        value = supplied.get(name, configured)
        if value is None and name in (
            "mma_tiler_mnk",
            "cluster_shape_mn",
            "schedule_policy",
        ):
            value = expected
        if implementation == "standard" and name == "mma_tiler_mnk":
            continue
        if not any(_same(value, v) for v in scheduling.get(name, (expected,))):
            raise ValueError(
                f"batch_invariant MXFP8 does not admit {name}={value!r} for {arch}/{kind}/{implementation}"
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
                "MXFP8 phase_interleave requires work_id_mode='atomic_counter'"
            )


def mxfp8_batch_invariant_fingerprint(
    config, *, hidden=None, num_experts=None, implementation=None, arch=None
):
    from .activation_batch_invariant import batch_invariant_fingerprint_kwargs

    from flashinfer.moe_ep.batch_invariant.mxfp8_spec import spec_fingerprint

    arch = arch or nvfp4_batch_invariant_arch()
    implementation = implementation or mxfp8_batch_invariant_implementation(config)
    validate_mxfp8_batch_invariant(config, implementation=implementation, arch=arch)
    clamp = getattr(config, "gate_up_clamp", None)
    if clamp is None:
        clamp = getattr(config, "activation_clamp", None)
    if implementation == "standard":
        from .quantized_batch_invariant_tiles import (
            validate_quantized_batch_invariant_tile,
        )

        validate_quantized_batch_invariant_tile(
            config,
            kind=mxfp8_quant_kind(config),
            arch=arch,
            knobs=getattr(config, "knobs", None)
            if isinstance(getattr(config, "knobs", None), dict)
            else None,
            hidden=hidden,
        )
    return spec_fingerprint(
        hidden=hidden if hidden is not None else config.hidden,
        intermediate=getattr(
            config, "intermediate", getattr(config, "intermediate_size", None)
        ),
        top_k=getattr(config, "num_topk", getattr(config, "top_k", None)),
        num_experts=num_experts
        if num_experts is not None
        else config.num_total_experts,
        arch=arch,
        implementation=implementation,
        quant_kind=mxfp8_quant_kind(config),
        apply_routing_weights_before_fc2=getattr(
            config, "apply_routing_weights_before_fc2", True
        ),
        gate_up_clamp=clamp,
        **batch_invariant_fingerprint_kwargs(config),
    )
