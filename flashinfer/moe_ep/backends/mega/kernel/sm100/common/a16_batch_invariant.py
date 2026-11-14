"""Admission and receipts for BF16 activations with online-decoded weights."""

from __future__ import annotations

from .activation_batch_invariant import batch_invariant_activation
from .batch_invariant import bf16_batch_invariant_arch

A16_BATCH_INVARIANT_VERSION = 1


def a16_batch_invariant_default_knobs(kind="bf16_nvfp4"):
    w4 = kind in ("bf16_nvfp4", "w4a16")
    values = dict(
        mma_tiler_mnk=(256, 128, 256 if w4 else 128),
        cluster_shape_mnk=(2, 1, 1),
        use_2cta_instrs=True,
        load_balance_mode="static",
        group_hint=None,
        num_sched_stages=None if w4 else 2,
        flag_batch=1,
        epi_flag_batch=(1, 1),
        token_back_mode="epi_warps",
        in_kernel_fc2_reduce=False,
        enable_iket=False,
    )
    if w4:
        values["force_static_sched"] = True
    else:
        values.update(
            transform_buffer="tmem",
            accumulator_overlap=False,
            transform_k_tile=128,
            clc_bundle_size=None,
        )
    return values


def a16_kind(config):
    kind = getattr(config, "kind", "bf16_nvfp4")
    try:
        return {
            "bf16_nvfp4": "w4a16",
            "bf16_mxfp8_e4m3": "w8a16_e4m3",
            "bf16_mxfp8_e5m2": "w8a16_e5m2",
        }[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported A16 batch-invariant kind {kind!r}") from exc


def validate_a16_batch_invariant(config, knobs=None):
    if type(getattr(config, "batch_invariant", False)) is not bool:
        raise ValueError("batch_invariant must be a bool")
    if not getattr(config, "batch_invariant", False):
        return
    kind = a16_kind(config)
    if type(getattr(config, "has_fc1_norm_const", False)) is not bool:
        raise ValueError("has_fc1_norm_const must be a bool")
    values = dict(knobs or {})
    numerical = (
        "batch_invariant",
        "gate_up_clamp",
        "activation",
        "swiglu_alpha",
        "swiglu_beta",
        "situ_beta",
        "situ_linear_beta",
        "apply_routing_weights_before_fc2",
        "enable_in_kernel_fc2_reduce",
        "use_custom_finalize",
        "has_fc1_norm_const",
    )
    allowed = set(numerical) | {
        "mma_tiler_mnk",
        "cluster_shape_mnk",
        "use_2cta_instrs",
        "load_balance_mode",
        "group_hint",
        "force_static_sched",
        "num_sched_stages",
        "flag_batch",
        "epi_flag_batch",
        "token_back_mode",
        "in_kernel_fc2_reduce",
        "enable_iket",
        "transform_buffer",
        "accumulator_overlap",
        "transform_k_tile",
        "clc_bundle_size",
    }
    unknown = values.keys() - allowed
    if unknown:
        raise ValueError(f"unvalidated A16 BI knobs: {sorted(unknown)}")
    for name in numerical:
        if name in values and values[name] != getattr(
            config,
            name,
            False
            if name
            in (
                "enable_in_kernel_fc2_reduce",
                "use_custom_finalize",
                "apply_routing_weights_before_fc2",
                "has_fc1_norm_const",
            )
            else ("swiglu" if name == "activation" else None),
        ):
            raise ValueError(f"A16 BI tuning cannot change numerical field {name}")
    get = lambda name, default=None: values.get(name, getattr(config, name, default))
    for name in (
        "has_fc1_norm_const",
        "enable_in_kernel_fc2_reduce",
        "in_kernel_fc2_reduce",
        "use_custom_finalize",
        "enable_iket",
        "use_2cta_instrs",
    ):
        if type(get(name, name == "use_2cta_instrs")) is not bool:
            raise ValueError(f"{name} must be a bool")
    batch_invariant_activation(config, family="w4a16" if kind == "w4a16" else "bf16")
    for name in (
        "enable_in_kernel_fc2_reduce",
        "in_kernel_fc2_reduce",
        "accumulator_overlap",
        "enable_iket",
    ):
        if get(name, False):
            raise ValueError(f"batch_invariant A16 MegaMoE requires {name}=False")
    tile = (256, 128, 256 if kind == "w4a16" else 128)
    required = {
        "mma_tiler_mnk": tile,
        "cluster_shape_mnk": (2, 1, 1),
        "use_2cta_instrs": True,
        "flag_batch": 1,
        "epi_flag_batch": (1, 1),
        "group_hint": None,
        "num_sched_stages": None if kind == "w4a16" else 2,
    }
    if kind != "w4a16":
        required.update(transform_buffer="tmem", transform_k_tile=128)
    else:
        required["force_static_sched"] = True
    for name, expected in required.items():
        actual = get(name, expected)
        if isinstance(expected, tuple):
            actual = tuple(actual)
        if actual != expected:
            raise ValueError(
                f"batch_invariant A16 requires {name}={expected!r}; got {actual!r}"
            )
    if get("load_balance_mode", "static") not in ("static", "atomic_counter"):
        raise ValueError("unsupported A16 batch-invariant schedule")
    if get("token_back_mode", "epi_warps") not in ("epi_warps", "reuse_dispatch_warps"):
        raise ValueError("unsupported A16 batch-invariant return carrier")


def a16_batch_invariant_fingerprint(
    config, *, hidden=None, num_experts=None, arch=None
):
    from flashinfer.moe_ep.batch_invariant.a16_spec import spec_fingerprint

    validate_a16_batch_invariant(config)
    clamp = getattr(config, "gate_up_clamp", None)
    if clamp is None:
        clamp = getattr(config, "activation_clamp", None)
    kind = a16_kind(config)
    deferred = getattr(config, "use_custom_finalize", False)
    if hasattr(config, "finalize_mode"):
        from flashinfer.moe_ep.config import FinalizeMode

        deferred = config.finalize_mode is FinalizeMode.NONE
    return spec_fingerprint(
        hidden if hidden is not None else config.hidden,
        getattr(config, "intermediate_size", getattr(config, "intermediate", None)),
        getattr(config, "top_k", getattr(config, "num_topk", None)),
        num_experts if num_experts is not None else config.num_total_experts,
        kind=kind,
        apply_routing_weights_before_fc2=(
            True
            if kind != "w4a16"
            else getattr(config, "apply_routing_weights_before_fc2", False)
        ),
        gate_up_clamp=clamp,
        arch=arch or bf16_batch_invariant_arch(),
        activation=getattr(config, "activation", "swiglu"),
        swiglu_alpha=getattr(config, "swiglu_alpha", None),
        swiglu_beta=getattr(config, "swiglu_beta", None),
        situ_beta=getattr(config, "situ_beta", None),
        situ_linear_beta=getattr(config, "situ_linear_beta", None),
        handoff_norm=getattr(
            config,
            "has_fc1_norm_const",
            getattr(config, "fc1_norm_const", None) is not None,
        ),
        output_mode="deferred" if deferred else "finalized",
    )
