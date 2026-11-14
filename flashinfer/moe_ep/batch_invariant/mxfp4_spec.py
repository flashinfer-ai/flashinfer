# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Ordered, architecture-scoped MXFP4 MegaMoE arithmetic.

``mxfp4_mxfp8`` has E2M1 weights, E4M3 activations and UE8M0 scales per
32 elements. Its native MMA atom is K32 on Blackwell and K64 on Rubin.
Rubin's pure ``mxfp4`` uses E2M1 operands and K128 atoms. The actual Rubin
emitter spells that unified instruction ``kind::mxf4nvf4.block32``.

The main boundary is packed/scaled operands. Independent input-quantization
components describe BF16 staging into that boundary. Prequantized calls
supply the boundary directly. Approximate operations and MMA are opaque,
architecture-scoped atoms; trace proofs do not verify lowering or publication.
Padded slices denote zero-filled storage bytes, including UE8M0 byte zero
(2^-127, not a numerical zero scale). No padded MMA update is removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import struct
from typing import Any, Sequence

from ...batch_invariant.fp_interface.term import Term
from .nvfp4_spec import OrderedTermAlgebra as _Nvfp4TermAlgebra
from .nvfp4_spec import fc1_activation, fc2_output
from .spec import f32_bits, finalizer_term
from . import numerical_variants as _numerical

SPEC_VERSION = 1
BLOCK = 32
HALF = 16
FP32_MAX = 3.4028234663852886e38
MIN_NORMAL = 2.0**-126


class OrderedTermAlgebra(_Nvfp4TermAlgebra):
    def cvt_out(self, value, fmt_name):
        if fmt_name == "ue8m0":
            return self.op("cvt.rp.satfinite.ue8m0.f32", value)
        return super().cvt_out(value, fmt_name)

    def ue8m0_roundtrip(self, value):
        scale = self.cvt_out(value, "ue8m0")
        return self.widen(self.op("cvt.rn.bf16.ue8m0", scale))


class _FingerprintTermAlgebra(OrderedTermAlgebra):
    def __init__(self):
        self._table = {}


def data_format(quant_kind):
    if quant_kind not in ("mxfp4", "mxfp4_mxfp8"):
        raise ValueError("quant_kind must be mxfp4 or mxfp4_mxfp8")
    return "e2m1" if quant_kind == "mxfp4" else "e4m3"


def half_amax(fp, values):
    """FC1: a left fold of xorsign-abs maxima over one logical half-block."""
    if len(values) != HALF:
        raise ValueError("MXFP4 FC1 quantization requires 16 values per half")
    maximum = values[0]
    for value in values[1:]:
        maximum = fp.max_xorsign_abs(maximum, value)
    return fp.abs_bits(maximum)


def input_half_amax(fp, values):
    """Input staging: number maxima from +0, retaining its actual association."""
    if len(values) not in (HALF, BLOCK):
        raise ValueError("MXFP4 input quantization requires 16 or 32 values")
    maximum = fp.const(0.0)
    for value in values:
        maximum = fp.fmax(maximum, fp.fmax(value, fp.neg(value)))
    return maximum


def combine_half_amax(fp, low, high):
    """Both physical owners use logical low-half first, high-half second."""
    return fp.fmax(low, high)


def block_amax(fp, values):
    if len(values) != BLOCK:
        raise ValueError("MXFP4 quantization requires exactly 32 values")
    return combine_half_amax(
        fp, half_amax(fp, values[:HALF]), half_amax(fp, values[HALF:])
    )


def input_block_amax(fp, values, quant_kind):
    if len(values) != BLOCK:
        raise ValueError("MXFP4 input quantization requires exactly 32 values")
    data_format(quant_kind)
    if quant_kind == "mxfp4":
        return input_half_amax(fp, values)
    return combine_half_amax(
        fp, input_half_amax(fp, values[:HALF]), input_half_amax(fp, values[HALF:])
    )


def quantization_scale_value(fp, amax, quant_kind, *, staging=False):
    limit = 6.0 if data_format(quant_kind) == "e2m1" else 448.0
    candidate = fp.mul(amax, fp.const(1.0 / limit))
    if staging:
        candidate = fp.fmax(candidate, fp.const(MIN_NORMAL))
    # The real epilogue rounds to UE8M0, converts to BF16, then widens FP32.
    # In particular byte zero is 2^-127, rather than a floating-point zero.
    return fp.ue8m0_roundtrip(candidate)


def quantization_encoding_scale(fp, scale_value, *, staging=False):
    encoding = fp.fmin(fp.rcp(scale_value), fp.const(FP32_MAX))
    if not staging:
        mask = fp.fmin(fp.mul(scale_value, fp.const(1e30)), fp.const(1.0))
        encoding = fp.mul(encoding, mask)
    return encoding


def quantize_amax_precast(fp, values, amax, quant_kind, *, staging=False):
    """Return scaled FP32 values and a stored UE8M0 byte, before vector cvt.

    Input staging clamps the candidate scale to the minimum normal FP32 value
    and omits FC1's soft mask. They intentionally name different chains.
    """
    if len(values) not in (HALF, BLOCK):
        raise ValueError("MXFP4 quantization requires 16 or 32 values")
    scale_value = quantization_scale_value(fp, amax, quant_kind, staging=staging)
    stored_scale = fp.cvt_out(scale_value, "ue8m0")
    encoding = quantization_encoding_scale(fp, scale_value, staging=staging)
    return tuple(fp.mul(value, encoding) for value in values), stored_scale


def quantize_half_precast(fp, values, low_amax, high_amax, *, quant_kind):
    if len(values) != HALF:
        raise ValueError("MXFP4 FC1 quantization requires 16 values per half")
    return quantize_amax_precast(
        fp, values, combine_half_amax(fp, low_amax, high_amax), quant_kind
    )


def quantize_block_precast(fp, values, quant_kind):
    return quantize_amax_precast(fp, values, block_amax(fp, values), quant_kind)


def input_quantize_block_precast(fp, values, quant_kind):
    return quantize_amax_precast(
        fp, values, input_block_amax(fp, values, quant_kind), quant_kind, staging=True
    )


@dataclass(frozen=True)
class NumericalClass:
    arch: str
    implementation: str
    quant_kind: str
    instruction_k: int
    stage_k: int
    instruction_kind: str
    tile_m: int = 256
    tile_n: int = 128
    cta_group: int = 2


def numerical_class(arch, quant_kind="mxfp4_mxfp8", implementation="standard"):
    data_format(quant_kind)
    if arch not in ("sm_100", "sm_103", "sm_107"):
        raise ValueError("MXFP4 MegaMoE requires sm_100, sm_103 or sm_107")
    if implementation not in ("standard", "genphase"):
        raise ValueError("MXFP4 MegaMoE implementation must be standard or genphase")
    if quant_kind == "mxfp4":
        if arch != "sm_107":
            raise ValueError("pure MXFP4 MegaMoE requires sm_107")
        return NumericalClass(
            arch, implementation, quant_kind, 128, 256, "mxf4nvf4.block32"
        )
    instruction_k = 64 if arch == "sm_107" else 32
    return NumericalClass(
        arch, implementation, quant_kind, instruction_k, 128, "mxf8f6f4"
    )


def _mma_suffix(cls):
    return f"{cls.quant_kind}.{cls.instruction_kind}.k{cls.instruction_k}.m256n128.cta2"


def mma_fold(
    fp,
    k,
    operand,
    *,
    arch,
    quant_kind="mxfp4_mxfp8",
    implementation="standard",
    blocked=False,
    stage_k=None,
):
    if type(k) is not int or k < 0:
        raise ValueError("MMA dimension must be a nonnegative integer")
    cls = numerical_class(arch, quant_kind, implementation)
    stage_k = cls.stage_k if stage_k is None else stage_k
    if type(stage_k) is not int or stage_k <= 0 or stage_k % cls.instruction_k:
        raise ValueError("stage_k must be a positive multiple of the native MMA K")
    stages = (k + stage_k - 1) // stage_k
    per_stage = stage_k // cls.instruction_k
    indices = (
        (
            stage * per_stage + inner
            for stage in range(stages)
            for inner in range(per_stage)
        )
        if blocked
        else range(stages * per_stage)
    )
    accumulator = fp.const(0.0)
    for step in indices:
        operands = fp.op(
            f"operand.{cls.quant_kind}.k{cls.instruction_k}",
            operand,
            step,
            min(cls.instruction_k, max(0, k - step * cls.instruction_k)),
            "weight.e2m1.ue8m0.block32",
            f"activation.{data_format(quant_kind)}.ue8m0.block32",
        )
        accumulator = (
            fp.op("mma.init." + _mma_suffix(cls), operands)
            if step == 0
            else fp.op("mma.acc." + _mma_suffix(cls), accumulator, operands)
        )
    return accumulator


def _validate(hidden, intermediate, top_k, num_experts, early):
    if any(type(k) is not int or k <= 0 or k % 128 for k in (hidden, intermediate)):
        raise ValueError("hidden and intermediate must be positive multiples of 128")
    if type(top_k) is not int or not 1 <= top_k <= 32:
        raise ValueError("top_k must lie in [1, 32]")
    if type(num_experts) is not int or num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if type(early) is not bool:
        raise ValueError("apply_routing_weights_before_fc2 must be bool")


def _quant_components(fp, name, values, amax, quant_kind, *, staging):
    scale_value = quantization_scale_value(
        fp, fp.op("ref.f32", name + "_amax"), quant_kind, staging=staging
    )
    scale_ref = fp.op("ref.f32", name + "_scale_value")
    data = fp.op(
        "block." + data_format(quant_kind),
        *(
            fp.cvt_out(
                fp.mul(value, fp.op("ref.f32", name + "_encoding_scale")),
                data_format(quant_kind),
            )
            for value in values
        ),
    )
    return {
        name + "_amax": amax,
        name + "_scale_value": scale_value,
        name + "_scale": fp.cvt_out(scale_ref, "ue8m0"),
        name + "_encoding_scale": quantization_encoding_scale(
            fp, scale_ref, staging=staging
        ),
        name + "_data": data,
    }


def component_terms(
    hidden,
    intermediate,
    top_k,
    num_experts,
    *,
    arch,
    quant_kind="mxfp4_mxfp8",
    implementation="standard",
    apply_routing_weights_before_fc2=True,
    gate_up_clamp=None,
    blocked=False,
    valid_routes: Sequence[bool] | None = None,
    fp=None,
    **numerical_kwargs,
):
    _validate(
        hidden, intermediate, top_k, num_experts, apply_routing_weights_before_fc2
    )
    numerical_class(arch, quant_kind, implementation)
    if not _numerical.is_default(**numerical_kwargs):
        return _numerical.component_terms(
            "mxfp4",
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
    inputs = tuple(
        fp.widen(fp.op("input.bf16", "row.block32", i)) for i in range(BLOCK)
    )
    terms = _quant_components(
        fp,
        "input",
        inputs,
        input_block_amax(fp, inputs, quant_kind),
        quant_kind,
        staging=True,
    )
    terms.update(
        {
            "fc1_gate": mma_fold(
                fp,
                hidden,
                "input_w13.gate",
                arch=arch,
                quant_kind=quant_kind,
                implementation=implementation,
                blocked=blocked,
            ),
            "fc1_up": mma_fold(
                fp,
                hidden,
                "input_w13.up",
                arch=arch,
                quant_kind=quant_kind,
                implementation=implementation,
                blocked=blocked,
            ),
            "activation": fc1_activation(
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
        }
    )
    values = tuple(fp.op("ref.f32", "activation", i) for i in range(BLOCK))
    terms.update(
        _quant_components(
            fp, "handoff", values, block_amax(fp, values), quant_kind, staging=False
        )
    )
    terms["fc2"] = fc2_output(
        fp,
        mma_fold(
            fp,
            intermediate,
            "handoff_w2",
            arch=arch,
            quant_kind=quant_kind,
            implementation=implementation,
            blocked=blocked,
        ),
    )
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
    ("input_amax", "input.bf16.row.block32"),
    ("input_scale_value", "input_amax", "normal_scale_floor"),
    ("input_scale", "input_scale_value"),
    ("input_encoding_scale", "input_scale_value"),
    ("input_data", "input.bf16.row.block32", "input_encoding_scale"),
    ("boundary", "input.data_ue8m0", "staging.output_or_prequantized_input"),
    ("fc1_gate", "input.data_ue8m0", "route_expert.w13_gate"),
    ("fc1_up", "input.data_ue8m0", "route_expert.w13_up"),
    ("activation", "fc1_gate", "fc1_up", "route_weight"),
    ("handoff_amax", "activation.block32.low16", "activation.block32.high16"),
    ("handoff_scale_value", "handoff_amax"),
    ("handoff_scale", "handoff_scale_value"),
    ("handoff_encoding_scale", "handoff_scale_value"),
    ("handoff_data", "activation.block32", "handoff_encoding_scale"),
    ("fc2", "handoff_data", "handoff_scale", "route_expert.w2"),
    ("ref.bf16:fc2:slot", "fc2", "original_route_slot"),
    ("finalize", "ordered_route_slots", "output_row"),
)


_COMPONENT_OPS = frozenset(
    "op:" + name
    for name in (
        "f32_bits",
        "bf16_bits",
        "input.bf16",
        "input.f32",
        "input.expert_id",
        "ref.f32",
        "ref.bf16",
        "route.valid",
        "select.lazy",
        "add.rn.f32",
        "mul.rn.f32",
        "fma.rn.f32",
        "neg.f32",
        "abs.bits.f32",
        "min.f32",
        "max.f32",
        "min.xorsign.abs.f32",
        "max.xorsign.abs.f32",
        "ex2.approx.ftz.f32",
        "rcp.approx.ftz.f32",
        "cvt.rn.bf16.f32",
        "widen.bf16.f32",
        "cvt.rp.satfinite.ue8m0.f32",
        "cvt.rn.bf16.ue8m0",
        "cvt.rn.satfinite.e2m1.f32",
        "cvt.rn.satfinite.e4m3.f32",
        "block.e2m1",
        "block.e4m3",
    )
) | frozenset(
    "op:" + name
    for cls in (
        numerical_class("sm_100"),
        numerical_class("sm_107"),
        numerical_class("sm_107", "mxfp4"),
    )
    for name in (
        "mma.init." + _mma_suffix(cls),
        "mma.acc." + _mma_suffix(cls),
        f"operand.{cls.quant_kind}.k{cls.instruction_k}",
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
            raise ValueError("MXFP4 MegaMoE components require ordered primitives")
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
    quant_kind="mxfp4_mxfp8",
    implementation="standard",
    apply_routing_weights_before_fc2=True,
    gate_up_clamp=None,
    **numerical_kwargs,
):
    if not _numerical.is_default(**numerical_kwargs):
        return _numerical.specification(
            "mxfp4",
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
    return dict(
        **{"class": "megamoe.mxfp4"},
        version=SPEC_VERSION,
        arch=arch,
        quant_kind=quant_kind,
        implementation=implementation,
        hidden=hidden,
        intermediate=intermediate,
        top_k=top_k,
        num_experts=num_experts,
        instruction_k=cls.instruction_k,
        stage_k=cls.stage_k,
        instruction_kind=cls.instruction_kind,
        instruction_mn_cta=(256, 128, 2),
        apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
        clamp_bits=None if gate_up_clamp is None else f32_bits(abs(gate_up_clamp)),
        global_scalars="absent",
        composition=COMPOSITION,
        components={name: _fingerprint_component(term) for name, term in terms.items()},
    )


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
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    return "megamoe-mxfp4-" + hashlib.sha256(payload).hexdigest()[:16]


def spec_fingerprint(
    hidden,
    intermediate,
    top_k,
    num_experts,
    *,
    arch,
    quant_kind="mxfp4_mxfp8",
    implementation="standard",
    apply_routing_weights_before_fc2=True,
    gate_up_clamp=None,
    **numerical_kwargs,
):
    _validate(
        hidden, intermediate, top_k, num_experts, apply_routing_weights_before_fc2
    )
    if not _numerical.is_default(**numerical_kwargs):
        return _numerical.spec_fingerprint(
            "mxfp4",
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
