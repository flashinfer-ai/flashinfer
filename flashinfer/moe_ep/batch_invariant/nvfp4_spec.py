# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Ordered, architecture-scoped NVFP4 MegaMoE arithmetic.

The block-scaled MMA primitives consume E2M1 values and E4M3 scales per 16
elements. Legacy Blackwell uses K64 instructions; Rubin next uses K128.
Neither is a fold of BF16 K16 instructions. Initialization is a distinct
primitive, and full physical K stages, including zero-filled instructions,
remain in the trace. Geometry is fixed separately for each provider.

Approximate instructions, conversions and MMA are opaque hardware atoms.
Trace equality proves neither CUDA lowering nor remote publication. Receipt
equality is scoped to one architecture and implementation, not across them.

The main computation starts at packed NVFP4 values plus block scales. The
input quantization components additionally specify the BF16 staging recipe
that produces this boundary. Prequantized calls supply the boundary directly;
they do not imply equality with an arbitrary BF16 input row.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

from ...batch_invariant.fp_interface.term import Term
from .spec import OrderedTermAlgebra as _Bf16TermAlgebra
from .spec import f32_bits, finalizer_term
from . import numerical_variants as _numerical

SPEC_VERSION = 1
SF_VEC = 16
LOG2_E = 1.4426950408889634
FP32_MAX = 3.4028234663852886e38


class OrderedTermAlgebra(_Bf16TermAlgebra):
    """No commutative rewrites, including NaN operands and scale factors."""

    def fmin(self, a, b):
        return self.op("min.f32", a, b)

    def fmax(self, a, b):
        return self.op("max.f32", a, b)

    def neg(self, a):
        return self.op("neg.f32", a)

    def min_xorsign_abs(self, a, b):
        return self.op("min.xorsign.abs.f32", a, b)

    def max_xorsign_abs(self, a, b):
        return self.op("max.xorsign.abs.f32", a, b)

    def abs_bits(self, a):
        return self.op("abs.bits.f32", a)

    def cvt_out(self, a, fmt_name):
        if fmt_name not in ("bf16", "e4m3", "e2m1"):
            raise ValueError(f"unsupported NVFP4 MegaMoE conversion: {fmt_name}")
        instruction = (
            "cvt.rn.bf16.f32"
            if fmt_name == "bf16"
            else f"cvt.rn.satfinite.{fmt_name}.f32"
        )
        return self.op(instruction, a)

    def widen_e4m3(self, a):
        return self.op("widen.e4m3.f32", a)


class _FingerprintTermAlgebra(OrderedTermAlgebra):
    def __init__(self):
        # Runtime receipts must not pin every shape in the global interner.
        self._table = {}


def fc1_activation(fp, gate, up, *, alpha=None, weight=None, clamp=None):
    """FP32 dequantization, ordinary SwiGLU and optional early route weight.

    Clamp atoms intentionally differ from the BF16 class: the NVFP4 kernel
    uses ``min.f32`` and ``min.xorsign.abs.f32``, without ``.NaN``.
    """
    if alpha is not None:
        gate = fp.mul(gate, alpha)
        up = fp.mul(up, alpha)
    if clamp is not None:
        limit = fp.const(abs(clamp))
        gate = fp.fmin(gate, limit)
        up = fp.min_xorsign_abs(up, limit)
    exponent = fp.mul(gate, fp.const(-LOG2_E))
    sigmoid = fp.rcp(fp.add(fp.ex2(exponent), fp.const(1.0)))
    result = fp.mul(fp.mul(up, gate), sigmoid)
    if weight is not None:
        result = fp.mul(result, weight)
    return result


fc1_activation_accumulator = fc1_activation


def quantization_scale(fp, amax, *, norm=None):
    candidate = fp.mul(amax, fp.const(1.0 / 6.0))
    if norm is not None:
        candidate = fp.mul(candidate, norm)
    return fp.cvt_out(candidate, "e4m3")


def quantization_encoding_scale(fp, scale, *, norm=None):
    scale_rt = fp.widen_e4m3(scale)
    reciprocal = fp.rcp(scale_rt)
    if norm is not None:
        reciprocal = fp.mul(norm, reciprocal)
    reciprocal = fp.fmin(reciprocal, fp.const(FP32_MAX))
    mask = fp.fmin(fp.mul(scale_rt, fp.const(1e30)), fp.const(1.0))
    return fp.mul(reciprocal, mask)


def _quantize_amax_precast(fp, values, amax, norm):
    scale = quantization_scale(fp, amax, norm=norm)
    encoding_scale = quantization_encoding_scale(fp, scale, norm=norm)
    return tuple(fp.mul(v, encoding_scale) for v in values), scale


def block_amax(fp, values):
    if len(values) != SF_VEC:
        raise ValueError("NVFP4 MegaMoE quantization requires exactly 16 values")
    amax = values[0]
    for value in values[1:]:
        amax = fp.max_xorsign_abs(amax, value)
    return fp.abs_bits(amax)


def quantize_block_precast(fp, values, *, norm=None):
    """FC1 block16: return scaled FP32 values and the E4M3 scale.

    The amax is the actual left fold of xorsign-abs maxima, with its sign
    cleared once afterwards. The capped, softly masked approximate reciprocal
    is deliberately not replaced by a correctly rounded division.

    The device caller converts the returned FP32 vector to packed E2M1 using
    the DSL's vector conversion. Scalar sub-byte Numeric values cannot be
    materialized by every supported DSL version.
    """
    return _quantize_amax_precast(fp, values, block_amax(fp, values), norm)


def quantize_block(fp, values, *, norm=None):
    """The full FC1 block16 quantization, including E2M1 RN conversion."""
    scaled, scale = quantize_block_precast(fp, values, norm=norm)
    return tuple(fp.cvt_out(v, "e2m1") for v in scaled), scale


def input_block_amax(fp, values):
    if len(values) != SF_VEC:
        raise ValueError("NVFP4 MegaMoE input quantization requires 16 values")
    amax = fp.const(0.0)
    for value in values:
        amax = fp.fmax(amax, fp.fmax(value, fp.neg(value)))
    return amax


def input_quantize_block_precast(fp, values, *, norm):
    """The fixed input-stage recipe; its amax differs from FC1 quantization."""
    return _quantize_amax_precast(fp, values, input_block_amax(fp, values), norm)


def input_quantize_block(fp, values, *, norm):
    scaled, scale = input_quantize_block_precast(fp, values, norm=norm)
    return tuple(fp.cvt_out(v, "e2m1") for v in scaled), scale


def fc2_output(fp, accumulator, *, alpha=None):
    """Optional per-expert FP32 dequantization followed by BF16 wire storage."""
    if alpha is not None:
        accumulator = fp.mul(accumulator, alpha)
    return fp.cvt_out(accumulator, "bf16")


@dataclass(frozen=True)
class NumericalClass:
    arch: str
    implementation: str
    instruction_k: int
    stage_k: int
    tile_m: int = 256
    tile_n: int = 128
    cta_group: int = 2


def numerical_class(arch: str, implementation: str) -> NumericalClass:
    if arch not in ("sm_100", "sm_103", "sm_107"):
        raise ValueError("NVFP4 MegaMoE requires sm_100, sm_103 or sm_107")
    if implementation not in ("legacy", "standard", "genphase"):
        raise ValueError("implementation must be legacy, standard or genphase")
    if implementation == "legacy":
        if arch == "sm_107":
            raise ValueError("legacy NVFP4 MegaMoE is not admitted on sm_107")
        return NumericalClass(arch, implementation, 64, 256)
    instruction_k = 128 if arch == "sm_107" else 64
    stage_k = 256 if arch == "sm_107" or implementation == "genphase" else 128
    return NumericalClass(arch, implementation, instruction_k, stage_k)


def mma_fold(
    fp,
    k: int,
    operand: str,
    *,
    arch: str,
    implementation: str,
    blocked=False,
    stage_k=None,
):
    """A persistent block-scaled accumulator, including stage padding."""
    if type(k) is not int or k < 0:
        raise ValueError("MMA dimension must be a nonnegative integer")
    cls = numerical_class(arch, implementation)
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
    accumulator = fp.const(0.0)
    suffix = f"nvfp4.k{cls.instruction_k}.m{cls.tile_m}n{cls.tile_n}.cta{cls.cta_group}"
    for step in indices:
        valid = min(cls.instruction_k, max(0, k - step * cls.instruction_k))
        operands = fp.op(
            f"operand.nvfp4.k{cls.instruction_k}",
            operand,
            step,
            valid,
            "e2m1.e4m3.block16",
        )
        accumulator = (
            fp.op(f"mma.init.{suffix}", operands)
            if step == 0
            else fp.op(f"mma.acc.{suffix}", accumulator, operands)
        )
    return accumulator


def _validate(hidden, intermediate, top_k, num_experts, early, presence):
    if type(hidden) is not int or hidden <= 0 or hidden % 64:
        raise ValueError("hidden must be a positive multiple of 64")
    if type(intermediate) is not int or intermediate <= 0 or intermediate % 64:
        raise ValueError("intermediate must be a positive multiple of 64")
    if type(top_k) is not int or not 1 <= top_k <= 32:
        raise ValueError("top_k must lie in [1, 32]")
    if type(num_experts) is not int or num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if type(early) is not bool or any(type(v) is not bool for v in presence):
        raise ValueError("routing-weight placement and scalar presence must be bool")


def _quant_components(fp, name, values, amax, norm):
    """Bind the block's common subexpressions once in the compositional spec."""
    scale = quantization_scale(fp, fp.op("ref.f32", name + "_amax"), norm=norm)
    encoding = quantization_encoding_scale(
        fp, fp.op("ref.e4m3", name + "_scale"), norm=norm
    )
    data = fp.op(
        "block.e2m1",
        *(
            fp.cvt_out(fp.mul(v, fp.op("ref.f32", name + "_encoding_scale")), "e2m1")
            for v in values
        ),
    )
    return {
        name + "_amax": amax,
        name + "_scale": scale,
        name + "_encoding_scale": encoding,
        name + "_data": data,
    }


def component_terms(
    hidden: int,
    intermediate: int,
    top_k: int,
    num_experts: int,
    *,
    arch: str,
    implementation: str,
    apply_routing_weights_before_fc2: bool = True,
    gate_up_clamp: float | None = None,
    input_norm_const: float = 1.0,
    has_fc1_alpha: bool = False,
    has_fc2_alpha: bool = False,
    has_fc1_norm_const: bool = False,
    valid_routes: Sequence[bool] | None = None,
    blocked: bool = False,
    fp: OrderedTermAlgebra | None = None,
    **numerical_kwargs,
) -> dict[str, Term]:
    _validate(
        hidden,
        intermediate,
        top_k,
        num_experts,
        apply_routing_weights_before_fc2,
        (has_fc1_alpha, has_fc2_alpha, has_fc1_norm_const),
    )
    numerical_class(arch, implementation)
    if not _numerical.is_default(**numerical_kwargs):
        return _numerical.component_terms(
            "nvfp4",
            (hidden, intermediate, top_k, num_experts),
            dict(
                arch=arch,
                implementation=implementation,
                apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
                gate_up_clamp=gate_up_clamp,
                input_norm_const=input_norm_const,
                has_fc1_alpha=has_fc1_alpha,
                has_fc2_alpha=has_fc2_alpha,
                has_fc1_norm_const=has_fc1_norm_const,
                blocked=blocked,
                valid_routes=valid_routes,
            ),
            numerical_kwargs,
            fp=fp,
        )
    if not math.isfinite(input_norm_const) or input_norm_const <= 0:
        raise ValueError("input_norm_const must be positive and finite")
    fp = fp or OrderedTermAlgebra()
    inputs = tuple(
        fp.widen(fp.op("input.bf16", "row.block16", i)) for i in range(SF_VEC)
    )
    terms = _quant_components(
        fp, "input", inputs, input_block_amax(fp, inputs), fp.const(input_norm_const)
    )
    terms.update(
        {
            "fc1_gate": mma_fold(
                fp,
                hidden,
                "input_w13.gate",
                arch=arch,
                implementation=implementation,
                blocked=blocked,
            ),
            "fc1_up": mma_fold(
                fp,
                hidden,
                "input_w13.up",
                arch=arch,
                implementation=implementation,
                blocked=blocked,
            ),
            "activation": fc1_activation(
                fp,
                fp.op("ref.f32", "fc1_gate"),
                fp.op("ref.f32", "fc1_up"),
                alpha=fp.op("input.f32", "fc1_alpha") if has_fc1_alpha else None,
                weight=(
                    fp.op("input.f32", "route_weight")
                    if apply_routing_weights_before_fc2
                    else None
                ),
                clamp=gate_up_clamp,
            ),
        }
    )
    values = tuple(fp.op("ref.f32", "activation", i) for i in range(SF_VEC))
    terms.update(
        _quant_components(
            fp,
            "handoff",
            values,
            block_amax(fp, values),
            fp.op("input.f32", "fc1_norm_const") if has_fc1_norm_const else None,
        )
    )
    terms["fc2"] = fc2_output(
        fp,
        mma_fold(
            fp,
            intermediate,
            "handoff_w2",
            arch=arch,
            implementation=implementation,
            blocked=blocked,
        ),
        alpha=fp.op("input.f32", "fc2_alpha") if has_fc2_alpha else None,
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
    ("input_amax", "input.bf16.row.block16"),
    ("input_scale", "input_amax", "input_norm_const"),
    ("input_encoding_scale", "input_scale", "input_norm_const"),
    ("input_data", "input.bf16.row.block16", "input_encoding_scale"),
    ("boundary", "input.e2m1_e4m3", "staging.output_or_prequantized_input"),
    ("fc1_gate", "input.e2m1_e4m3", "route_expert.w13_gate"),
    ("fc1_up", "input.e2m1_e4m3", "route_expert.w13_up"),
    ("activation", "fc1_gate", "fc1_up", "fc1_alpha", "route_weight"),
    ("handoff_amax", "activation.block16"),
    ("handoff_scale", "handoff_amax", "fc1_norm_const"),
    ("handoff_encoding_scale", "handoff_scale", "fc1_norm_const"),
    ("handoff_data", "activation.block16", "handoff_encoding_scale"),
    ("fc2", "handoff_data", "handoff_scale", "route_expert.w2", "fc2_alpha"),
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
        "ref.e4m3",
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
        "cvt.rn.satfinite.e4m3.f32",
        "cvt.rn.satfinite.e2m1.f32",
        "widen.bf16.f32",
        "widen.e4m3.f32",
        "block.e2m1",
        "operand.nvfp4.k64",
        "operand.nvfp4.k128",
        "mma.init.nvfp4.k64.m256n128.cta2",
        "mma.acc.nvfp4.k64.m256n128.cta2",
        "mma.init.nvfp4.k128.m256n128.cta2",
        "mma.acc.nvfp4.k128.m256n128.cta2",
    )
)


def _fingerprint_component(term: Term) -> str:
    """Incrementally hash ordered S-expressions, without cached prefixes."""
    digest = hashlib.sha256()
    stack: list[Any] = [term]
    while stack:
        value = stack.pop()
        if not isinstance(value, Term):
            digest.update(str(value).encode())
            continue
        if value.op not in _COMPONENT_OPS:
            raise ValueError("NVFP4 MegaMoE components require ordered primitives")
        digest.update(("(" + value.op + " ").encode())
        stack.append(")")
        for index, arg in enumerate(reversed(value.args)):
            if index:
                stack.append(" ")
            stack.append(arg)
    return digest.hexdigest()[:16]


def specification(
    hidden, intermediate, top_k, num_experts, *, arch, implementation, **kwargs
):
    numerical_kwargs = {
        name: kwargs.pop(name) for name in _numerical.FIELDS if name in kwargs
    }
    if not _numerical.is_default(**numerical_kwargs):
        return _numerical.specification(
            "nvfp4",
            (hidden, intermediate, top_k, num_experts),
            dict(arch=arch, implementation=implementation, **kwargs),
            numerical_kwargs,
        )
    cls = numerical_class(arch, implementation)
    terms = component_terms(
        hidden,
        intermediate,
        top_k,
        num_experts,
        arch=arch,
        implementation=implementation,
        fp=_FingerprintTermAlgebra(),
        **kwargs,
    )
    clamp = kwargs.get("gate_up_clamp")
    return {
        "class": "megamoe.nvfp4",
        "version": SPEC_VERSION,
        "arch": arch,
        "implementation": implementation,
        "hidden": hidden,
        "intermediate": intermediate,
        "top_k": top_k,
        "num_experts": num_experts,
        "instruction_k": cls.instruction_k,
        "stage_k": cls.stage_k,
        "instruction_mn_cta": (cls.tile_m, cls.tile_n, cls.cta_group),
        "apply_routing_weights_before_fc2": kwargs.get(
            "apply_routing_weights_before_fc2", True
        ),
        "clamp_bits": None if clamp is None else f32_bits(abs(clamp)),
        "input_norm_bits": f32_bits(kwargs.get("input_norm_const", 1.0)),
        "scalar_presence": tuple(
            kwargs.get(name, False)
            for name in ("has_fc1_alpha", "has_fc2_alpha", "has_fc1_norm_const")
        ),
        "composition": COMPOSITION,
        "components": {
            name: _fingerprint_component(term) for name, term in terms.items()
        },
    }


@lru_cache(maxsize=256)
def _cached_spec_fingerprint(
    hidden,
    intermediate,
    top_k,
    num_experts,
    arch,
    implementation,
    early,
    clamp_bits,
    norm_bits,
    presence,
):
    clamp = (
        None
        if clamp_bits is None
        else struct.unpack(">f", bytes.fromhex(clamp_bits))[0]
    )
    norm = struct.unpack(">f", bytes.fromhex(norm_bits))[0]
    receipt = specification(
        hidden,
        intermediate,
        top_k,
        num_experts,
        arch=arch,
        implementation=implementation,
        apply_routing_weights_before_fc2=early,
        gate_up_clamp=clamp,
        input_norm_const=norm,
        **dict(
            zip(
                ("has_fc1_alpha", "has_fc2_alpha", "has_fc1_norm_const"),
                presence,
                strict=True,
            )
        ),
    )
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    return "megamoe-nvfp4-" + hashlib.sha256(payload).hexdigest()[:16]


def spec_fingerprint(
    hidden: int,
    intermediate: int,
    top_k: int,
    num_experts: int,
    *,
    arch: str,
    implementation: str,
    apply_routing_weights_before_fc2: bool = True,
    gate_up_clamp: float | None = None,
    input_norm_const: float = 1.0,
    has_fc1_alpha: bool = False,
    has_fc2_alpha: bool = False,
    has_fc1_norm_const: bool = False,
    **numerical_kwargs,
) -> str:
    """One receipt per arithmetic class; batch, capacity and scheduling excluded."""
    _validate(
        hidden,
        intermediate,
        top_k,
        num_experts,
        apply_routing_weights_before_fc2,
        (has_fc1_alpha, has_fc2_alpha, has_fc1_norm_const),
    )
    if not _numerical.is_default(**numerical_kwargs):
        return _numerical.spec_fingerprint(
            "nvfp4",
            (hidden, intermediate, top_k, num_experts),
            dict(
                arch=arch,
                implementation=implementation,
                apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
                gate_up_clamp=gate_up_clamp,
                input_norm_const=input_norm_const,
                has_fc1_alpha=has_fc1_alpha,
                has_fc2_alpha=has_fc2_alpha,
                has_fc1_norm_const=has_fc1_norm_const,
            ),
            numerical_kwargs,
        )
    return _cached_spec_fingerprint(
        hidden,
        intermediate,
        top_k,
        num_experts,
        arch,
        implementation,
        apply_routing_weights_before_fc2,
        None if gate_up_clamp is None else f32_bits(abs(gate_up_clamp)),
        f32_bits(input_norm_const),
        (has_fc1_alpha, has_fc2_alpha, has_fc1_norm_const),
    )
