# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared host metadata for explicitly specified activation variants."""

from flashinfer.moe_ep.batch_invariant.activation_variants import ActivationVariant

_FIELDS = ("activation", "swiglu_alpha", "swiglu_beta", "situ_beta", "situ_linear_beta")


def batch_invariant_activation_fields(config):
    """Emit only new recipes, preserving the original default ABI and receipts."""
    values = {
        name: getattr(config, name, "swiglu" if name == "activation" else None)
        for name in _FIELDS
    }
    if values["activation"] == "swiglu" and all(
        values[name] is None for name in _FIELDS[1:]
    ):
        return {}
    return values


def batch_invariant_activation(config, *, family, nondefault_only=False):
    fields = batch_invariant_activation_fields(config)
    if nondefault_only and not fields:
        return None
    clamp = getattr(config, "gate_up_clamp", None)
    if clamp is None:
        clamp = getattr(config, "activation_clamp", None)
    return ActivationVariant(family=family, gate_up_clamp=clamp, **fields)


def block_scaled_activation_family(implementation, arch, *, nvfp4=False):
    if implementation == "legacy":
        return "nvfp4_legacy" if nvfp4 else "mxfp8_legacy"
    return "next107" if arch == "sm_107" else "next100"


def batch_invariant_variant_kwargs(config):
    """Keywords understood by every arithmetic fingerprint implementation."""
    values = batch_invariant_activation_fields(config)
    per_token = getattr(config, "enable_fc1_activation_per_token_scale", False)
    if type(per_token) is not bool:
        raise ValueError("enable_fc1_activation_per_token_scale must be a bool")
    if per_token:
        values["per_token_scale"] = True
    return values


def batch_invariant_fingerprint_kwargs(config):
    values = batch_invariant_variant_kwargs(config)
    wire = getattr(config, "combine_dtype", "bf16")
    if wire != "bf16":
        values["combine_dtype"] = wire
    if batch_invariant_output_mode(config) == "deferred":
        values["output_mode"] = "deferred"
    return values


def batch_invariant_output_mode(config):
    from flashinfer.moe_ep.config import FinalizeMode

    if hasattr(config, "finalize_mode"):
        return "deferred" if config.finalize_mode is FinalizeMode.NONE else "finalized"
    return (
        "deferred"
        if getattr(config, "use_custom_finalize", False)
        and not getattr(config, "batch_invariant_cake_finalize", False)
        else "finalized"
    )


def validate_batch_invariant_output(config, *, implementation, arch, kind):
    """Admit only output ABIs implemented by the selected provider."""
    from flashinfer.moe_ep.config import FinalizeMode

    wire = getattr(config, "combine_dtype", "bf16")
    allowed_wires = ("bf16", "nvfp4", "mxfp8")
    if implementation == "genphase" or kind == "bf16":
        allowed_wires = ("bf16",)
    elif implementation == "legacy" and kind != "nvfp4":
        allowed_wires = ("bf16", "mxfp8")
    if wire not in allowed_wires:
        raise ValueError(
            f"batch_invariant {implementation}/{kind} does not support combine_dtype={wire!r}"
        )
    deferred = batch_invariant_output_mode(config) == "deferred"
    if deferred and (
        wire != "bf16"
        or implementation == "genphase"
        or arch == "sm_107"
        and kind != "bf16"
    ):
        raise ValueError(
            "batch_invariant deferred output requires a Blackwell legacy/standard BF16 combine ABI"
        )
    if type(getattr(config, "use_custom_finalize", False)) is not bool:
        raise ValueError("use_custom_finalize must be a bool")
    cake = getattr(config, "batch_invariant_cake_finalize", False)
    if type(cake) is not bool:
        raise ValueError("batch_invariant_cake_finalize must be a bool")
    mode = getattr(config, "finalize_mode", None)
    cake = cake or mode is FinalizeMode.CAKE
    if cake and not (
        implementation == "standard"
        and arch in ("sm_100", "sm_103")
        and kind == "nvfp4"
        and wire == "bf16"
        and getattr(config, "apply_routing_weights_before_fc2", True)
    ):
        raise ValueError(
            "batch_invariant CAKE requires Blackwell standard NVFP4 early routing and BF16 combine"
        )
    if getattr(config, "batch_invariant_cake_finalize", False) and not getattr(
        config, "use_custom_finalize", False
    ):
        raise ValueError(
            "batch_invariant_cake_finalize requires use_custom_finalize=True"
        )
    if mode is not None:
        allowed = (FinalizeMode.AUTO, FinalizeMode.DEFAULT)
        if implementation != "genphase":
            allowed += (FinalizeMode.PERSISTENT, FinalizeMode.NONE)
        if (
            implementation == "standard"
            and kind == "nvfp4"
            and arch in ("sm_100", "sm_103")
        ):
            allowed += (FinalizeMode.CAKE,)
        if mode not in allowed:
            raise ValueError("unsupported batch_invariant finalization mode")


def batch_invariant_numerical_metadata(config):
    if not getattr(config, "batch_invariant", False):
        return {}
    fields = batch_invariant_activation_fields(config)
    if getattr(config, "enable_fc1_activation_per_token_scale", False):
        fields["enable_fc1_activation_per_token_scale"] = True
    return fields
