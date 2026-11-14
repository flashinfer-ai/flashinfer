# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Canonical quantized FC2 combine wires.

This wire is distinct from operand/intermediate quantization. NVFP4 stores
the BF16 amax itself and decodes with an additional FP32 multiply by 1/6;
MXFP8 stores a UE8M0 scale and always uses E4M3 data, including W8A8 E5M2.
Standard and legacy NVFP4 producers start after BF16 rounding. Legacy MXFP8
starts from the FP32 accumulator and uses its own amax and scale recipe.
"""

import hashlib

from ...batch_invariant.fp_interface.term import Term
from . import mxfp8_spec
from .mxfp8_spec import OrderedTermAlgebra as _OrderedTermAlgebra

FP32_MAX = 3.4028234663852886e38


class OrderedTermAlgebra(_OrderedTermAlgebra):
    def widen_ue8m0(self, scale):
        return self.widen(self.op("cvt.rn.bf16.ue8m0", scale))


def combine_amax(fp, values):
    if len(values) not in (16, 32):
        raise ValueError("combine quantization requires 16 or 32 values")
    maximum = values[0]
    for value in values[1:]:
        maximum = fp.max_xorsign_abs(maximum, value)
    return fp.abs_bits(maximum)


def combine_decode_scale(fp, scale, *, combine_dtype):
    if combine_dtype == "nvfp4":
        return fp.mul(fp.widen(scale), fp.const(1.0 / 6.0))
    if combine_dtype == "mxfp8":
        if hasattr(fp, "widen_ue8m0"):
            return fp.widen_ue8m0(scale)
        return fp.widen(fp.op("cvt.rn.bf16.ue8m0", scale))
    raise ValueError("quantized combine dtype must be nvfp4 or mxfp8")


def combine_decode_value(fp, decoded_value, scale, *, combine_dtype):
    """The caller supplies the exact FP32 widening of an E2M1/E4M3 code."""
    return fp.mul(
        decoded_value, combine_decode_scale(fp, scale, combine_dtype=combine_dtype)
    )


def combine_quantize_block_precast(fp, values, *, combine_dtype):
    """Logical-hidden-order amax and explicit RN encoding; caller packs data.

    Values are FP32 widenings of FC2's already rounded BF16 wire. The helper
    returns FP32 values for the existing vector RN conversion to E2M1/E4M3
    and the SF Numeric stored by both return carriers.
    """
    size = 16 if combine_dtype == "nvfp4" else 32 if combine_dtype == "mxfp8" else 0
    if len(values) != size:
        raise ValueError("combine block must have 16 NVFP4 or 32 MXFP8 values")
    maximum = combine_amax(fp, values)
    return combine_quantize_amax_precast(
        fp, values, maximum, combine_dtype=combine_dtype
    )


def combine_quantize_amax_precast(fp, values, maximum, *, combine_dtype):
    """Scale lane-local values after the canonical shared block amax."""
    if combine_dtype not in ("nvfp4", "mxfp8"):
        raise ValueError("quantized combine dtype must be nvfp4 or mxfp8")
    if combine_dtype == "nvfp4":
        scale = fp.cvt_out(maximum, "bf16")
        decode_scale = combine_decode_scale(fp, scale, combine_dtype=combine_dtype)
    else:
        decode_scale = fp.ue8m0_roundtrip(fp.mul(maximum, fp.const(1.0 / 448.0)))
        scale = fp.cvt_out(decode_scale, "ue8m0")
    encoding = fp.fmin(fp.rcp(decode_scale), fp.const(FP32_MAX))
    mask = fp.fmin(fp.mul(decode_scale, fp.const(1e30)), fp.const(1.0))
    encoding = fp.mul(encoding, mask)
    return tuple(fp.mul(value, encoding) for value in values), scale


def component_terms(fp, *, combine_dtype, legacy_mxfp8=False):
    """Compositional encoding and decoding of one logical hidden block."""
    if combine_dtype not in ("nvfp4", "mxfp8") or (
        legacy_mxfp8 and combine_dtype != "mxfp8"
    ):
        raise ValueError("invalid quantized combine numerical class")
    size = 16 if combine_dtype == "nvfp4" else 32
    values = tuple(
        (
            fp.op("ref.f32", "fc2", i)
            if legacy_mxfp8
            else fp.widen(fp.op("ref.bf16", "fc2", i))
        )
        for i in range(size)
    )
    maximum = (
        mxfp8_spec.legacy_amax(fp, values) if legacy_mxfp8 else combine_amax(fp, values)
    )
    maximum_ref = fp.op("ref.f32", "combine_amax")
    scale_value = fp.op("ref.f32", "combine_scale_value")
    if combine_dtype == "nvfp4":
        scale = fp.cvt_out(maximum_ref, "bf16")
        value = combine_decode_scale(
            fp, fp.op("ref.bf16", "combine_scale"), combine_dtype=combine_dtype
        )
        stored_ref = fp.op("ref.bf16", "combine_scale")
    else:
        value = mxfp8_spec.quantization_scale_value(
            fp, maximum_ref, "mxfp8_e4m3", legacy=legacy_mxfp8
        )
        scale = fp.cvt_out(scale_value, "ue8m0")
        stored_ref = fp.op("ref.ue8m0", "combine_scale")
    encoding = mxfp8_spec.quantization_encoding_scale(
        fp, scale_value, legacy=legacy_mxfp8
    )
    fmt = "e2m1" if combine_dtype == "nvfp4" else "e4m3"
    return dict(
        combine_amax=maximum,
        combine_scale=scale,
        combine_scale_value=value,
        combine_encoding_scale=encoding,
        combine_data=fp.op(
            "block." + fmt,
            *(
                fp.cvt_out(fp.mul(v, fp.op("ref.f32", "combine_encoding_scale")), fmt)
                for v in values
            ),
        ),
        combine_decode=combine_decode_value(
            fp,
            fp.op("widen." + fmt + ".f32", fp.op("ref." + fmt, "combine_data")),
            stored_ref,
            combine_dtype=combine_dtype,
        ),
    )


def finalizer_term(fp, slots, experts, early, *, valid_routes=None):
    zero = fp.const(0.0)
    acc = zero
    for slot in range(slots):
        value = fp.op("ref.f32", "combine_decode", slot)
        score = fp.op("input.f32", "route_weight", slot)
        if valid_routes is None:
            valid = fp.op("route.valid", fp.op("input.expert_id", slot), experts)
            value, score = (
                fp.op("select.lazy", valid, value, zero),
                fp.op("select.lazy", valid, score, zero),
            )
        elif not valid_routes[slot]:
            value, score = zero, zero
        acc = fp.add(acc, value) if early else fp.fma(value, score, acc)
    return fp.cvt_out(acc, "bf16")


def partials_term(fp, slots, experts, *, valid_routes=None):
    zero = fp.op("bf16_bits", "0000")
    values = []
    for slot in range(slots):
        value = fp.op("ref.bf16", "fc2", slot)
        if valid_routes is None:
            valid = fp.op("route.valid", fp.op("input.expert_id", slot), experts)
            value = fp.op("select.lazy", valid, value, zero)
        elif not valid_routes[slot]:
            value = zero
        values.append(value)
    return fp.op("block.bf16", *values)


def deferred_weights_term(fp, slots, experts, early, *, valid_routes=None):
    """Canonical returned scores, separate from an external reducer's math."""
    zero = fp.const(0.0)
    values = []
    for slot in range(slots):
        value = fp.const(1.0) if early else fp.op("input.f32", "route_weight", slot)
        if valid_routes is None:
            valid = fp.op("route.valid", fp.op("input.expert_id", slot), experts)
            value = fp.op("select.lazy", valid, value, zero)
        elif not valid_routes[slot]:
            value = zero
        values.append(value)
    return fp.op("block.f32", *values)


_COMPONENT_OPS = mxfp8_spec._COMPONENT_OPS | frozenset(
    "op:" + name
    for name in (
        "ref.ue8m0",
        "ref.e2m1",
        "ref.e4m3",
        "widen.e2m1.f32",
        "widen.e4m3.f32",
        "block.bf16",
        "block.f32",
        "bf16_bits",
    )
)


def fingerprint_component(term):
    """Hash ordered wire atoms without retaining expanded prefix strings."""
    digest = hashlib.sha256()
    pending = [term]
    while pending:
        value = pending.pop()
        if not isinstance(value, Term):
            digest.update(str(value).encode())
            continue
        if value.op not in _COMPONENT_OPS:
            raise ValueError(f"unknown combine primitive: {value.op}")
        digest.update(("(" + value.op + " ").encode())
        pending.append(")")
        for index, arg in enumerate(reversed(value.args)):
            if index:
                pending.append(" ")
            pending.append(arg)
    return digest.hexdigest()[:16]
