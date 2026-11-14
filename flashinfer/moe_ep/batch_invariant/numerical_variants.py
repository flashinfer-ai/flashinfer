# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Additional ordered activation classes; default receipts remain independent.

The coefficient signature records the constants actually emitted by a provider,
including expressions evaluated in host double precision before conversion to
FP32. Runtime per-token scales are operands, rather than static constants.
Quantized/deferred combine is admitted here only after its complete wire recipe
has a device interpretation and independent conformance evidence.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib
import json

from .activation_variants import (
    ActivationVariant,
    activation_accumulator,
    dequant_scale,
)

VERSION = 1
FIELDS = (
    "activation",
    "swiglu_alpha",
    "swiglu_beta",
    "situ_beta",
    "situ_linear_beta",
    "per_token_scale",
    "combine_dtype",
    "output_mode",
)
_MODULES = {
    "bf16": "spec",
    "nvfp4": "nvfp4_spec",
    "mxfp4": "mxfp4_spec",
    "mxfp8": "mxfp8_spec",
}


def is_default(
    *,
    activation="swiglu",
    swiglu_alpha=None,
    swiglu_beta=None,
    situ_beta=None,
    situ_linear_beta=None,
    per_token_scale=False,
    combine_dtype="bf16",
    output_mode="finalized",
):
    if type(per_token_scale) is not bool:
        raise ValueError("per_token_scale must be a bool")
    return (
        activation == "swiglu"
        and swiglu_alpha is None
        and swiglu_beta is None
        and situ_beta is None
        and situ_linear_beta is None
        and not per_token_scale
        and combine_dtype == "bf16"
        and output_mode == "finalized"
    )


def activation_family(family, *, arch, implementation="standard"):
    if family == "bf16":
        return "bf16"
    if implementation == "legacy":
        return "nvfp4_legacy" if family == "nvfp4" else "mxfp8_legacy"
    return "next107" if arch == "sm_107" else "next100"


def recipe(
    family,
    base,
    *,
    activation="swiglu",
    swiglu_alpha=None,
    swiglu_beta=None,
    situ_beta=None,
    situ_linear_beta=None,
    per_token_scale=False,
    combine_dtype="bf16",
    output_mode="finalized",
):
    if family not in _MODULES:
        raise ValueError(f"unknown MegaMoE numerical family: {family}")
    if type(per_token_scale) is not bool:
        raise ValueError("per_token_scale must be a bool")
    if combine_dtype not in ("bf16", "nvfp4", "mxfp8") or output_mode not in (
        "finalized",
        "deferred",
    ):
        raise ValueError("unknown combine dtype or output mode")
    if combine_dtype != "bf16" and (
        family == "bf16"
        or output_mode == "deferred"
        or (
            family == "mxfp8"
            and base.get("implementation") == "legacy"
            and combine_dtype != "mxfp8"
        )
    ):
        raise ValueError(
            "this provider/output mode does not admit that quantized combine wire"
        )
    if per_token_scale and not (
        family == "nvfp4"
        and base["arch"] in ("sm_100", "sm_103")
        and base.get("implementation") in ("legacy", "standard")
    ):
        raise ValueError(
            "per-token FC1 scaling requires legacy/standard Blackwell NVFP4"
        )
    return ActivationVariant(
        family=activation_family(
            family,
            arch=base["arch"],
            implementation=base.get("implementation", "standard"),
        ),
        activation=activation,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        gate_up_clamp=base.get("gate_up_clamp"),
    )


def activation_term(
    fp, variant, *, family, early, has_fc1_alpha=False, per_token_scale=False
):
    alpha = fp.op("input.f32", "fc1_alpha") if has_fc1_alpha else None
    token_scale = (
        fp.op("input.f32", "fc1_activation_per_token_scale")
        if per_token_scale
        else None
    )
    value = activation_accumulator(
        fp,
        fp.op("ref.f32", "fc1_gate"),
        fp.op("ref.f32", "fc1_up"),
        variant=variant,
        alpha=dequant_scale(fp, alpha=alpha, token_scale=token_scale),
        weight=fp.op("input.f32", "route_weight") if early else None,
    )
    return fp.cvt_out(value, "bf16") if family == "bf16" else value


def component_terms(family, shape, base, mode, *, fp=None):
    module = importlib.import_module("." + _MODULES[family], __package__)
    variant = recipe(family, base, **mode)
    combine = mode.get("combine_dtype", "bf16")
    fp = fp or (_wire_fp() if combine != "bf16" else module.OrderedTermAlgebra())
    terms = module.component_terms(*shape, **base, fp=fp)
    terms["activation"] = activation_term(
        fp,
        variant,
        family=family,
        early=base.get("apply_routing_weights_before_fc2", True),
        has_fc1_alpha=base.get("has_fc1_alpha", False),
        per_token_scale=mode.get("per_token_scale", False),
    )
    terms.update(
        _extensions(
            fp,
            family,
            shape,
            base,
            combine,
            mode.get("output_mode", "finalized"),
            fc2=terms["fc2"],
        )
    )
    if mode.get("output_mode") == "deferred":
        terms.pop("finalize", None)
    return terms


def specification(family, shape, base, mode):
    """Replace the activation component, preserving every MMA/quantizer atom."""
    variant = recipe(family, base, **mode)
    return _specification(
        family,
        shape,
        base,
        variant,
        mode.get("per_token_scale", False),
        mode.get("combine_dtype", "bf16"),
        mode.get("output_mode", "finalized"),
    )


def _wire_fp(*, private=False):
    wire = importlib.import_module(".wire_spec", __package__)
    fp = wire.OrderedTermAlgebra()
    if private:
        fp._table = {}
    return fp


def _extensions(fp, family, shape, base, combine, output, *, fc2=None):
    wire = importlib.import_module(".wire_spec", __package__)
    result = {}
    legacy = family == "mxfp8" and base.get("implementation") == "legacy"
    if combine != "bf16":
        if legacy:
            if fc2 is not None:
                assert fc2.op == "op:cvt.rn.bf16.f32"
                result["fc2"] = fc2.args[0]
            else:
                module = importlib.import_module(".mxfp8_spec", __package__)
                result["fc2"] = module.mma_fold(
                    fp,
                    shape[1],
                    "handoff_w2",
                    arch=base["arch"],
                    implementation="legacy",
                    quant_kind=base.get("quant_kind", "mxfp8_e4m3"),
                )
        result.update(
            wire.component_terms(fp, combine_dtype=combine, legacy_mxfp8=legacy)
        )
        result["finalize"] = wire.finalizer_term(
            fp,
            shape[2],
            shape[3],
            base.get("apply_routing_weights_before_fc2", True),
            valid_routes=base.get("valid_routes"),
        )
    if output == "deferred":
        result["partials"] = wire.partials_term(
            fp, shape[2], shape[3], valid_routes=base.get("valid_routes")
        )
        result["deferred_weights"] = wire.deferred_weights_term(
            fp,
            shape[2],
            shape[3],
            base.get("apply_routing_weights_before_fc2", True),
            valid_routes=base.get("valid_routes"),
        )
    return result


def _specification(family, shape, base, variant, per_token_scale, combine, output):
    module = importlib.import_module("." + _MODULES[family], __package__)
    result = module.specification(*shape, **base)
    fp = (
        _wire_fp(private=True)
        if combine != "bf16"
        else module._FingerprintTermAlgebra()
    )
    term = activation_term(
        fp,
        variant,
        family=family,
        early=base.get("apply_routing_weights_before_fc2", True),
        has_fc1_alpha=base.get("has_fc1_alpha", False),
        per_token_scale=per_token_scale,
    )
    result["components"]["activation"] = module._fingerprint_component(term)
    result["numerical_variants_version"] = VERSION
    result["activation_variant"] = variant.signature
    result["per_token_scale"] = per_token_scale
    result["combine_dtype"] = combine
    result["output_mode"] = output
    if per_token_scale:
        result["composition"] = tuple(
            (
                (*binding, "fc1_activation_per_token_scale")
                if binding[0] == "activation"
                else binding
            )
            for binding in result["composition"]
        )
    wire = importlib.import_module(".wire_spec", __package__)
    for name, value in _extensions(fp, family, shape, base, combine, output).items():
        result["components"][name] = (
            module._fingerprint_component(value)
            if name == "fc2"
            else wire.fingerprint_component(value)
        )
    if combine != "bf16":
        result["combine_source"] = (
            "fc2.fp32"
            if family == "mxfp8" and base.get("implementation") == "legacy"
            else "fc2.bf16"
        )
        scale_bindings = (
            (
                ("combine_scale", "combine_amax"),
                ("combine_scale_value", "combine_scale"),
            )
            if combine == "nvfp4"
            else (
                ("combine_scale_value", "combine_amax"),
                ("combine_scale", "combine_scale_value"),
            )
        )
        result["composition"] = (
            tuple(
                (
                    (
                        "ref.f32:combine_decode:slot",
                        "combine_decode",
                        "original_route_slot",
                    )
                    if binding[0] == "ref.bf16:fc2:slot"
                    else binding
                )
                for binding in result["composition"]
            )
            + (("combine_amax", result["combine_source"], "logical_hidden_order"),)
            + scale_bindings
            + (
                ("combine_encoding_scale", "combine_scale_value"),
                ("combine_data", result["combine_source"], "combine_encoding_scale"),
                ("combine_decode", "combine_data", "combine_scale"),
            )
        )
    if output == "deferred":
        result["components"].pop("finalize", None)
        result["deferred_invalid_slots"] = "canonical_bf16_positive_zero"
        result["deferred_invalid_weights"] = "canonical_f32_positive_zero"
        result["composition"] = tuple(
            binding for binding in result["composition"] if binding[0] != "finalize"
        ) + (
            ("partials", "fc2", "original_route_slots", "canonical_invalid_zero"),
            (
                "deferred_weights",
                "route_weight",
                "original_route_slots",
                "canonical_invalid_zero",
            ),
        )
    return result


@lru_cache(maxsize=256)
def _cached_fingerprint(family, shape, base, variant, per_token_scale, combine, output):
    result = _specification(
        family, shape, dict(base), variant, per_token_scale, combine, output
    )
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    return f"megamoe-{family}-" + hashlib.sha256(payload).hexdigest()[:16]


def spec_fingerprint(family, shape, base, mode):
    variant = recipe(family, base, **mode)
    # The raw host arguments are bounded cache keys only. Numerical identity
    # hashes the emitted coefficient bits and ordered terms, never those raw
    # arguments in place of the actual arithmetic.
    return _cached_fingerprint(
        family,
        tuple(shape),
        tuple(sorted(base.items())),
        variant,
        mode.get("per_token_scale", False),
        mode.get("combine_dtype", "bf16"),
        mode.get("output_mode", "finalized"),
    )
