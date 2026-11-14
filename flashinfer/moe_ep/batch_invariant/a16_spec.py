# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""BF16-activation MegaMoE with weights decoded before dense K16 MMA.

These are separate numerical classes from native block-scaled MMA. Weight
decode, its BF16 rounding, expert epilogue scales, activation association and
the optional normalization are explicit components. The instruction atom is
pinned to M256/N128/two-CTA; schedules and return carriers only place work.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

from ...batch_invariant.fp_interface.term import Term
from .activation_variants import ActivationVariant, activation_accumulator
from .spec import (
    OrderedTermAlgebra,
    _FingerprintTermAlgebra,
    _validate_spec,
    finalizer_term,
)

KINDS = ("w4a16", "w8a16_e4m3", "w8a16_e5m2")
PLANS = tuple(
    (schedule, carrier)
    for schedule in ("static", "atomic_counter")
    for carrier in ("epi_warps", "reuse_dispatch_warps")
)


def _digest(term: Term) -> str:
    """Ordered, length-delimited serialization without recursive expansion."""
    digest = hashlib.sha256()
    pending = [term]
    while pending:
        node = pending.pop()
        if isinstance(node, Term):
            digest.update(json.dumps([node.op, len(node.args)]).encode())
            pending.extend(reversed(node.args))
        else:
            data = json.dumps(node, ensure_ascii=True, separators=(",", ":")).encode()
            digest.update(str(len(data)).encode() + b":" + data)
    return digest.hexdigest()[:16]


def weight_decode(fp, kind: str):
    """One logical weight and its block scale, before entering the MMA atom."""
    if kind not in KINDS:
        raise ValueError(f"unsupported A16 weight kind: {kind!r}")
    fmt = "e2m1" if kind == "w4a16" else kind.removeprefix("w8a16_")
    scale_fmt = "e4m3" if kind == "w4a16" else "ue8m0"
    weight = fp.op("decode." + fmt + ".bf16", fp.op("input.weight_bits"))
    scale = fp.op("decode." + scale_fmt + ".bf16", fp.op("input.weight_scale_bits"))
    # The native BF16 multiply includes its NaN and denormal interpretation;
    # no equivalence to a widened FP32 multiply is assumed by the model.
    return fp.op("mul.rn.bf16", weight, scale)


def mma_fold(fp, k: int, operand: str, *, kind: str, blocked: bool = False):
    if type(k) is not int or k < 0 or kind not in KINDS:
        raise ValueError("invalid A16 MMA dimension or weight kind")
    stage = 256 if kind == "w4a16" else 128
    count = (k + stage - 1) // stage
    acc = fp.const(0.0)
    # Independent physical-stage lowering and flat specification.
    steps = (
        (s * (stage // 16) + i for s in range(count) for i in range(stage // 16))
        if blocked
        else range(count * (stage // 16))
    )
    for step in steps:
        operands = fp.op("operand.k16", operand, step, min(16, max(0, k - step * 16)))
        atom = "mma.bf16.k16.m256n128.cta2"
        acc = (
            fp.op(atom + ".init", operands)
            if step == 0
            else fp.op(atom + ".acc", acc, operands)
        )
    return acc


def component_terms(
    hidden: int,
    intermediate: int,
    top_k: int,
    num_experts: int,
    *,
    kind: str = "w4a16",
    apply_routing_weights_before_fc2: bool = False,
    gate_up_clamp: float | None = None,
    arch: str = "sm_100",
    activation: str = "swiglu",
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    situ_beta: float | None = None,
    situ_linear_beta: float | None = None,
    handoff_norm: bool = False,
    output_mode: str = "finalized",
    blocked: bool = False,
    fp=None,
    _variant: ActivationVariant | None = None,
) -> dict[str, Term]:
    _validate_spec(
        hidden, intermediate, top_k, num_experts, apply_routing_weights_before_fc2, arch
    )
    if kind not in KINDS or type(handoff_norm) is not bool:
        raise ValueError("invalid A16 kind or handoff_norm")
    if kind != "w4a16" and (not apply_routing_weights_before_fc2 or handoff_norm):
        raise ValueError("W8A16 requires early weighting without handoff normalization")
    if output_mode not in ("finalized", "deferred"):
        raise ValueError("output_mode must be 'finalized' or 'deferred'")
    variant = _variant or ActivationVariant(
        "w4a16" if kind == "w4a16" else "bf16",
        activation,
        swiglu_alpha,
        swiglu_beta,
        situ_beta,
        situ_linear_beta,
        gate_up_clamp,
    )
    fp = fp or OrderedTermAlgebra()
    terms = {"weight_decode": weight_decode(fp, kind)}
    for branch in ("gate", "up"):
        value = mma_fold(fp, hidden, "fc1." + branch, kind=kind, blocked=blocked)
        if kind == "w4a16":
            value = fp.mul(value, fp.op("input.f32", "fc1_alpha"))
        terms["fc1_" + branch] = value
    value = activation_accumulator(
        fp,
        fp.op("ref.f32", "fc1_gate"),
        fp.op("ref.f32", "fc1_up"),
        variant=variant,
        weight=(
            fp.op("input.f32", "route_weight")
            if apply_routing_weights_before_fc2
            else None
        ),
        handoff_norm=(fp.op("input.f32", "handoff_norm") if handoff_norm else None),
    )
    terms["activation"] = fp.cvt_out(value, "bf16")
    value = mma_fold(
        fp, intermediate, "fc2.handoff_decoded_w2", kind=kind, blocked=blocked
    )
    if kind == "w4a16":
        value = fp.mul(value, fp.op("input.f32", "fc2_alpha"))
    terms["fc2"] = fp.cvt_out(value, "bf16")
    if output_mode == "finalized":
        terms["finalize"] = finalizer_term(
            fp, top_k, num_experts, apply_routing_weights_before_fc2
        )
    else:
        terms["partials"] = fp.op(
            "select.lazy",
            fp.op("route.valid", fp.op("input.expert_id"), num_experts),
            fp.op("ref.bf16", "fc2"),
            fp.op("bf16_bits", "0000"),
        )
        terms["deferred_weight"] = fp.op(
            "select.lazy",
            fp.op("route.valid", fp.op("input.expert_id"), num_experts),
            fp.const(1.0)
            if apply_routing_weights_before_fc2
            else fp.op("input.f32", "route_weight"),
            fp.const(0.0),
        )
    return terms


def plan_component_terms(*args, plan: tuple[str, str], **kwargs):
    if plan not in PLANS:
        raise ValueError(f"unsupported A16 BI plan: {plan!r}")
    return component_terms(*args, **kwargs, blocked=True)


def specification(*args, **kwargs) -> dict[str, Any]:
    terms = component_terms(*args, **kwargs, fp=_FingerprintTermAlgebra())
    # Shape and architecture are part of the meaning of otherwise opaque atoms.
    # Runtime operands, batch capacity, scheduling and EP placement are excluded.
    return {
        "class": "megamoe.a16",
        "version": 1,
        "dimensions": list(args),
        "arch": kwargs.get("arch", "sm_100"),
        "kind": kwargs.get("kind", "w4a16"),
        "composition": (
            ("weight_decode", "expert.weight_bits", "expert.block_scale_bits"),
            ("fc1_gate", "mma.A:decoded.w13_gate", "mma.B:input_row"),
            ("fc1_up", "mma.A:decoded.w13_up", "mma.B:input_row"),
            ("activation", "fc1_gate", "fc1_up", "route_weight", "handoff_norm"),
            ("fc2.handoff_decoded_w2", "mma.A:decoded.w2", "mma.B:activation"),
            ("ref.bf16:fc2:slot", "fc2", "original_route_slot"),
        ),
        "components": {name: _digest(term) for name, term in terms.items()},
    }


@lru_cache(maxsize=256)
def _cached_fingerprint(
    hidden,
    intermediate,
    top_k,
    num_experts,
    kind,
    early,
    arch,
    handoff_norm,
    output_mode,
    variant,
):
    spec = specification(
        hidden,
        intermediate,
        top_k,
        num_experts,
        kind=kind,
        apply_routing_weights_before_fc2=early,
        arch=arch,
        handoff_norm=handoff_norm,
        output_mode=output_mode,
        _variant=variant,
    )
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return "megamoe-" + kind + "-" + hashlib.sha256(encoded).hexdigest()[:16]


def spec_fingerprint(
    hidden,
    intermediate,
    top_k,
    num_experts,
    *,
    kind="w4a16",
    apply_routing_weights_before_fc2=False,
    gate_up_clamp=None,
    arch="sm_100",
    activation="swiglu",
    swiglu_alpha=None,
    swiglu_beta=None,
    situ_beta=None,
    situ_linear_beta=None,
    handoff_norm=False,
    output_mode="finalized",
) -> str:
    _validate_spec(
        hidden, intermediate, top_k, num_experts, apply_routing_weights_before_fc2, arch
    )
    if (
        kind not in KINDS
        or type(handoff_norm) is not bool
        or output_mode not in ("finalized", "deferred")
    ):
        raise ValueError("invalid A16 numerical contract")
    variant = ActivationVariant(
        "w4a16" if kind == "w4a16" else "bf16",
        activation,
        swiglu_alpha,
        swiglu_beta,
        situ_beta,
        situ_linear_beta,
        gate_up_clamp,
    )
    # The variant includes actual emitted coefficient bits. Raw float keys
    # alone would alias signed zeros, and f32-rounded raw parameters can miss
    # differences in host-folded constants.
    return _cached_fingerprint(
        hidden,
        intermediate,
        top_k,
        num_experts,
        kind,
        apply_routing_weights_before_fc2,
        arch,
        handoff_norm,
        output_mode,
        variant,
    )
