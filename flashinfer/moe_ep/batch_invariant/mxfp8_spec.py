# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Ordered W8A8 MXFP8 MegaMoE arithmetic, scoped by architecture/provider.

All BI input quantizers use normal UE8M0 scales. Legacy Blackwell retains its
NaN-propagating clamp, ``(sigmoid * gate) * up`` association and fixed-one
normalization multiplications. Next uses the existing paired-half MX recipe.
MMA K32 (Blackwell) and K64 (Rubin), initialization and physical K128 padding
remain explicit. UE8M0 padding byte zero means 2^-127, not numerical zero.

The main boundary is packed/scaled operands; input components additionally
describe BF16 staging. Opaque MMA/conversion/approximation atoms and inverse
publication assumptions do not establish CUDA lowering or communication
correctness. Existing NVFP4/MXFP4 classes and receipts are independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import struct
from typing import Any

from ...batch_invariant.fp_interface.term import Term
from .mxfp4_spec import OrderedTermAlgebra as _MxTermAlgebra
from .mxfp4_spec import block_amax, combine_half_amax, input_half_amax
from .mxfp4_spec import half_amax as half_amax
from .mxfp4_spec import quantization_encoding_scale as _next_encoding_scale
from .mxfp4_spec import _COMPONENT_OPS as _MX_COMPONENT_OPS
from .nvfp4_spec import fc1_activation as next_fc1_activation
from .nvfp4_spec import fc2_output
from .spec import f32_bits, finalizer_term
from . import numerical_variants as _numerical

SPEC_VERSION = 1
KINDS = ("mxfp8_e4m3", "mxfp8_e5m2")
BLOCK = 32
MIN_NORMAL = 2.0**-126
FP32_MAX = 3.4028234663852886e38


def data_format(quant_kind):
    if quant_kind not in KINDS:
        raise ValueError("quant_kind must be mxfp8_e4m3 or mxfp8_e5m2")
    return "e4m3" if quant_kind == "mxfp8_e4m3" else "e5m2"


class OrderedTermAlgebra(_MxTermAlgebra):
    def cvt_out(self, value, fmt_name):
        if fmt_name == "e5m2":
            return self.op("cvt.rn.satfinite.e5m2.f32", value)
        return super().cvt_out(value, fmt_name)


class _FingerprintTermAlgebra(OrderedTermAlgebra):
    def __init__(self):
        self._table = {}


def legacy_fc1_activation(fp, gate, up, *, weight=None, clamp=None):
    if clamp is not None:
        limit = abs(clamp)
        gate = fp.fmin_nan(gate, fp.const(limit))
        up = fp.fmax_nan(fp.fmin_nan(up, fp.const(limit)), fp.const(-limit))
    exponent = fp.mul(gate, fp.const(-1.4426950408889634))
    sigmoid = fp.rcp(fp.add(fp.ex2(exponent), fp.const(1.0)))
    result = fp.mul(fp.mul(sigmoid, gate), up)
    return result if weight is None else fp.mul(result, weight)


def legacy_amax(fp, values):
    """Pin the formerly compiler-selected MAXIMUMF tree, including +0 seed."""
    if len(values) != BLOCK:
        raise ValueError("legacy MXFP8 quantization requires 32 values")
    level = tuple(fp.abs_bits(v) for v in values)
    while len(level) > 1:
        level = tuple(
            fp.fmax_nan(level[i], level[i + 1]) for i in range(0, len(level), 2)
        )
    return fp.fmax_nan(fp.const(0.0), level[0])


def quantization_scale_value(fp, amax, quant_kind, *, legacy=False, staging=False):
    maximum = 448.0 if data_format(quant_kind) == "e4m3" else 57344.0
    candidate = fp.mul(amax, fp.const(1.0 / maximum))
    if staging:
        candidate = fp.fmax(candidate, fp.const(MIN_NORMAL))
    elif legacy:
        candidate = fp.mul(candidate, fp.const(1.0))
    return fp.ue8m0_roundtrip(candidate)


def quantization_encoding_scale(fp, scale_value, *, legacy=False, staging=False):
    if legacy and not staging:
        return fp.fmin_nan(
            fp.mul(fp.const(1.0), fp.rcp(scale_value)), fp.const(FP32_MAX)
        )
    return _next_encoding_scale(fp, scale_value, staging=staging)


def legacy_quantize_block_precast(fp, values, *, quant_kind):
    scale = quantization_scale_value(
        fp, legacy_amax(fp, values), quant_kind, legacy=True
    )
    encoding = quantization_encoding_scale(fp, scale, legacy=True)
    # The existing legacy epilogue holds FP32 scales and performs its explicit
    # RP UE8M0 store later; the next helper below returns the stored byte.
    return tuple(fp.mul(v, encoding) for v in values), scale


def quantize_half_precast(fp, values, low_amax, high_amax, *, quant_kind):
    if len(values) != 16:
        raise ValueError("MXFP8 next quantization requires 16 values per half")
    scale = quantization_scale_value(
        fp, combine_half_amax(fp, low_amax, high_amax), quant_kind
    )
    encoding = quantization_encoding_scale(fp, scale)
    return tuple(fp.mul(v, encoding) for v in values), fp.cvt_out(scale, "ue8m0")


def input_quantize_amax_precast(fp, values, amax, quant_kind, *, normal_scales=True):
    if normal_scales is not True or len(values) not in (16, 32):
        raise ValueError("MXFP8 BI staging requires normal scales and 16 or 32 values")
    scale = quantization_scale_value(fp, amax, quant_kind, staging=True)
    encoding = quantization_encoding_scale(fp, scale, staging=True)
    return tuple(fp.mul(v, encoding) for v in values), fp.cvt_out(scale, "ue8m0")


@dataclass(frozen=True)
class NumericalClass:
    arch: str
    quant_kind: str
    implementation: str
    instruction_k: int
    stage_k: int = 128
    tile_m: int = 256
    tile_n: int = 128
    cta_group: int = 2


def numerical_class(arch, quant_kind="mxfp8_e4m3", implementation="standard"):
    data_format(quant_kind)
    if arch not in ("sm_100", "sm_103", "sm_107"):
        raise ValueError("MXFP8 MegaMoE requires sm_100, sm_103 or sm_107")
    if implementation not in ("legacy", "standard", "genphase"):
        raise ValueError("implementation must be legacy, standard or genphase")
    if arch == "sm_107" and implementation == "legacy":
        raise ValueError("legacy MXFP8 MegaMoE requires Blackwell")
    return NumericalClass(
        arch,
        quant_kind,
        implementation,
        64 if arch == "sm_107" else 32,
        tile_n=256 if implementation == "legacy" else 128,
    )


def _mma_suffix(cls):
    return f"{cls.quant_kind}.mxf8f6f4.k{cls.instruction_k}.m256n{cls.tile_n}.cta2"


def mma_fold(
    fp,
    k,
    operand,
    *,
    arch,
    quant_kind="mxfp8_e4m3",
    implementation="standard",
    blocked=False,
    stage_k=None,
):
    if type(k) is not int or k < 0:
        raise ValueError("MMA dimension must be nonnegative")
    cls = numerical_class(arch, quant_kind, implementation)
    stage_k = cls.stage_k if stage_k is None else stage_k
    if type(stage_k) is not int or stage_k <= 0 or stage_k % cls.instruction_k:
        raise ValueError("stage_k must be a positive multiple of the native MMA K")
    stages = (k + stage_k - 1) // stage_k
    per_stage = stage_k // cls.instruction_k
    indices = (
        (s * per_stage + inner for s in range(stages) for inner in range(per_stage))
        if blocked
        else range(stages * per_stage)
    )
    order = (
        "A.activation_B.weight"
        if implementation == "legacy" and operand.startswith("input_w13")
        else "A.weight_B.activation"
    )
    accumulator = fp.const(0.0)
    for step in indices:
        operands = fp.op(
            f"operand.{quant_kind}.k{cls.instruction_k}",
            operand,
            step,
            min(cls.instruction_k, max(0, k - step * cls.instruction_k)),
            order,
            f"{data_format(quant_kind)}.ue8m0.block32",
        )
        accumulator = (
            fp.op("mma.init." + _mma_suffix(cls), operands)
            if step == 0
            else fp.op("mma.acc." + _mma_suffix(cls), accumulator, operands)
        )
    return accumulator


def _validate(hidden, intermediate, top_k, num_experts, early, implementation):
    if any(type(k) is not int or k <= 0 or k % 128 for k in (hidden, intermediate)):
        raise ValueError("hidden and intermediate must be positive multiples of 128")
    if (
        type(top_k) is not int
        or not 1 <= top_k <= 32
        or type(num_experts) is not int
        or num_experts <= 0
    ):
        raise ValueError("top_k must lie in [1, 32] and num_experts must be positive")
    if type(early) is not bool or (implementation == "legacy" and not early):
        raise ValueError(
            "MXFP8 routing placement must be bool; legacy requires early weights"
        )


def _quant_components(
    fp, name, values, amax, quant_kind, *, legacy=False, staging=False
):
    scale_ref = fp.op("ref.f32", name + "_scale_value")
    fmt = data_format(quant_kind)
    return {
        name + "_amax": amax,
        name + "_scale_value": quantization_scale_value(
            fp,
            fp.op("ref.f32", name + "_amax"),
            quant_kind,
            legacy=legacy,
            staging=staging,
        ),
        name + "_scale": fp.cvt_out(scale_ref, "ue8m0"),
        name + "_encoding_scale": quantization_encoding_scale(
            fp, scale_ref, legacy=legacy, staging=staging
        ),
        name + "_data": fp.op(
            "block." + fmt,
            *(
                fp.cvt_out(fp.mul(v, fp.op("ref.f32", name + "_encoding_scale")), fmt)
                for v in values
            ),
        ),
    }


def component_terms(
    hidden,
    intermediate,
    top_k,
    num_experts,
    *,
    arch,
    quant_kind="mxfp8_e4m3",
    implementation="standard",
    apply_routing_weights_before_fc2=True,
    gate_up_clamp=None,
    blocked=False,
    valid_routes=None,
    fp=None,
    **numerical_kwargs,
):
    _validate(
        hidden,
        intermediate,
        top_k,
        num_experts,
        apply_routing_weights_before_fc2,
        implementation,
    )
    numerical_class(arch, quant_kind, implementation)
    if not _numerical.is_default(**numerical_kwargs):
        return _numerical.component_terms(
            "mxfp8",
            (hidden, intermediate, top_k, num_experts),
            dict(
                arch=arch,
                quant_kind=quant_kind,
                implementation=implementation,
                apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
                gate_up_clamp=gate_up_clamp,
                blocked=blocked,
                valid_routes=valid_routes,
            ),
            numerical_kwargs,
            fp=fp,
        )
    fp = fp or OrderedTermAlgebra()
    inputs = tuple(fp.widen(fp.op("input.bf16", "row.block32", i)) for i in range(32))
    input_amax = combine_half_amax(
        fp, input_half_amax(fp, inputs[:16]), input_half_amax(fp, inputs[16:])
    )
    terms = _quant_components(fp, "input", inputs, input_amax, quant_kind, staging=True)
    kwargs = dict(
        arch=arch, quant_kind=quant_kind, implementation=implementation, blocked=blocked
    )
    activation = (
        legacy_fc1_activation if implementation == "legacy" else next_fc1_activation
    )
    terms.update(
        fc1_gate=mma_fold(fp, hidden, "input_w13.gate", **kwargs),
        fc1_up=mma_fold(fp, hidden, "input_w13.up", **kwargs),
        activation=activation(
            fp,
            fp.op("ref.f32", "fc1_gate"),
            fp.op("ref.f32", "fc1_up"),
            weight=(
                fp.op("input.f32", "route_weight")
                if apply_routing_weights_before_fc2
                else None
            ),
            clamp=gate_up_clamp,
        ),
    )
    values = tuple(fp.op("ref.f32", "activation", i) for i in range(32))
    legacy = implementation == "legacy"
    terms.update(
        _quant_components(
            fp,
            "handoff",
            values,
            legacy_amax(fp, values) if legacy else block_amax(fp, values),
            quant_kind,
            legacy=legacy,
        )
    )
    terms["fc2"] = fc2_output(fp, mma_fold(fp, intermediate, "handoff_w2", **kwargs))
    terms["finalize"] = finalizer_term(
        fp,
        top_k,
        num_experts,
        apply_routing_weights_before_fc2,
        valid_routes=valid_routes,
    )
    return terms


def plan_component_terms(
    hidden, intermediate, top_k, num_experts, *, stage_k, **kwargs
):
    """Lower an admitted K-stage plan to the unchanged canonical math trace."""
    import sys
    from .stage_partition import plan_component_terms as lower_plan

    return lower_plan(
        sys.modules[__name__],
        (hidden, intermediate, top_k, num_experts),
        stage_k,
        kwargs,
    )


COMPOSITION = (
    ("input_amax", "input.bf16.row.block32.low16", "input.bf16.row.block32.high16"),
    ("input_scale_value", "input_amax", "normal_scale_floor_all_architectures"),
    ("input_scale", "input_scale_value"),
    ("input_encoding_scale", "input_scale_value"),
    ("input_data", "input.bf16.row.block32", "input_encoding_scale"),
    ("boundary", "input.data_ue8m0", "staging.output_or_prequantized_input"),
    ("fc1_gate", "input.data_ue8m0", "route_expert.w13_gate"),
    ("fc1_up", "input.data_ue8m0", "route_expert.w13_up"),
    ("activation", "fc1_gate", "fc1_up", "route_weight"),
    ("handoff_amax", "activation.block32"),
    ("handoff_scale_value", "handoff_amax"),
    ("handoff_scale", "handoff_scale_value"),
    ("handoff_encoding_scale", "handoff_scale_value"),
    ("handoff_data", "activation.block32", "handoff_encoding_scale"),
    ("fc2", "handoff_data", "handoff_scale", "route_expert.w2"),
    ("ref.bf16:fc2:slot", "fc2", "original_route_slot"),
    ("finalize", "ordered_route_slots", "output_row"),
)


_COMPONENT_OPS = (
    _MX_COMPONENT_OPS
    | frozenset(
        "op:" + name
        for name in (
            "max.NaN.f32",
            "min.NaN.f32",
            "cvt.rn.satfinite.e5m2.f32",
            "block.e5m2",
        )
    )
    | frozenset(
        "op:" + name
        for kind in KINDS
        for cls in (
            numerical_class("sm_100", kind, "legacy"),
            numerical_class("sm_100", kind),
            numerical_class("sm_107", kind),
        )
        for name in (
            "mma.init." + _mma_suffix(cls),
            "mma.acc." + _mma_suffix(cls),
            f"operand.{kind}.k{cls.instruction_k}",
        )
    )
)


def _fingerprint_component(term):
    digest = hashlib.sha256()
    stack: list[Any] = [term]
    while stack:
        item = stack.pop()
        if not isinstance(item, Term):
            digest.update(str(item).encode())
            continue
        if item.op not in _COMPONENT_OPS:
            raise ValueError("MXFP8 MegaMoE requires generic ordered primitives")
        digest.update(("(" + item.op + " ").encode())
        stack.append(")")
        for index, arg in enumerate(reversed(item.args)):
            if index:
                stack.append(" ")
            stack.append(arg)
    return digest.hexdigest()[:16]


def specification(
    hidden,
    intermediate,
    top_k,
    num_experts,
    *,
    arch,
    quant_kind="mxfp8_e4m3",
    implementation="standard",
    apply_routing_weights_before_fc2=True,
    gate_up_clamp=None,
    **numerical_kwargs,
):
    if not _numerical.is_default(**numerical_kwargs):
        return _numerical.specification(
            "mxfp8",
            (hidden, intermediate, top_k, num_experts),
            dict(
                arch=arch,
                quant_kind=quant_kind,
                implementation=implementation,
                apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
                gate_up_clamp=gate_up_clamp,
            ),
            numerical_kwargs,
        )
    cls = numerical_class(arch, quant_kind, implementation)
    terms = component_terms(
        hidden,
        intermediate,
        top_k,
        num_experts,
        arch=arch,
        quant_kind=quant_kind,
        implementation=implementation,
        apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
        gate_up_clamp=gate_up_clamp,
        fp=_FingerprintTermAlgebra(),
    )
    return {
        "class": "megamoe.mxfp8",
        "version": SPEC_VERSION,
        "arch": arch,
        "quant_kind": quant_kind,
        "implementation": implementation,
        "hidden": hidden,
        "intermediate": intermediate,
        "top_k": top_k,
        "num_experts": num_experts,
        "instruction_k": cls.instruction_k,
        "stage_k": 128,
        "instruction_mn_cta": (256, cls.tile_n, 2),
        "global_scalars": "absent",
        "input_normal_scales": True,
        "apply_routing_weights_before_fc2": apply_routing_weights_before_fc2,
        "clamp_bits": None if gate_up_clamp is None else f32_bits(abs(gate_up_clamp)),
        "composition": COMPOSITION,
        "components": {name: _fingerprint_component(t) for name, t in terms.items()},
    }


@lru_cache(maxsize=256)
def _cached_spec_fingerprint(
    hidden,
    intermediate,
    top_k,
    num_experts,
    arch,
    quant_kind,
    implementation,
    early,
    clamp_bits,
):
    clamp = (
        None
        if clamp_bits is None
        else struct.unpack(">f", bytes.fromhex(clamp_bits))[0]
    )
    receipt = specification(
        hidden,
        intermediate,
        top_k,
        num_experts,
        arch=arch,
        quant_kind=quant_kind,
        implementation=implementation,
        apply_routing_weights_before_fc2=early,
        gate_up_clamp=clamp,
    )
    return (
        "megamoe-mxfp8-"
        + hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
    )


def spec_fingerprint(
    hidden,
    intermediate,
    top_k,
    num_experts,
    *,
    arch,
    quant_kind="mxfp8_e4m3",
    implementation="standard",
    apply_routing_weights_before_fc2=True,
    gate_up_clamp=None,
    **numerical_kwargs,
):
    _validate(
        hidden,
        intermediate,
        top_k,
        num_experts,
        apply_routing_weights_before_fc2,
        implementation,
    )
    if not _numerical.is_default(**numerical_kwargs):
        return _numerical.spec_fingerprint(
            "mxfp8",
            (hidden, intermediate, top_k, num_experts),
            dict(
                arch=arch,
                quant_kind=quant_kind,
                implementation=implementation,
                apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
                gate_up_clamp=gate_up_clamp,
            ),
            numerical_kwargs,
        )
    return _cached_spec_fingerprint(
        hidden,
        intermediate,
        top_k,
        num_experts,
        arch,
        quant_kind,
        implementation,
        apply_routing_weights_before_fc2,
        None if gate_up_clamp is None else f32_bits(abs(gate_up_clamp)),
    )
