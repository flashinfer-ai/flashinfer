# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Ordered activation recipes, independent of kernel admission and device code.

The family selects instruction association and host-constant lowering, not a
hardware support promise. Host policies decide which combinations a provider
exposes. Existing default helpers and receipts deliberately do not call this
module yet. In particular, an explicitly supplied alpha/beta pair is not
simplified to the default recipe, even when its values are one and zero.

``coefficients`` contains the actual emitted FP32 constants. Host expressions
such as ``-alpha * LOG2_E`` and ``2 / beta`` are evaluated before conversion,
matching the existing providers. Recording only the input parameter's FP32
bits would miss cases where those derived constants differ.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

LOG2_E = 1.4426950408889634
FAMILIES = (
    "bf16",
    "nvfp4_legacy",
    "next100",
    "next107",
    "w4a16",
    "mxfp8_legacy",
)


def _bits(value: float) -> str:
    try:
        return struct.pack(">f", value).hex()
    except OverflowError:
        return struct.pack(">f", math.copysign(math.inf, value)).hex()


def _number(name: str, value: float | None, *, positive: bool = False) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (positive and value <= 0)
    ):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be a {qualifier} Python number")


@dataclass(frozen=True)
class ActivationVariant:
    """One scalar recipe and its effective constant-bit signature.

    All families permit mathematically defined custom/SiTU combinations here;
    kernel availability is checked separately. ``bf16``, ``mxfp8_legacy`` and
    ``w4a16`` use NaN-propagating min/max clamp atoms. NVFP4 and next families
    use the original number-preferring min and sign-preserving magnitude min.
    Clamp arguments denote a magnitude, as in the existing BI specifications.
    """

    family: str
    activation: str = "swiglu"
    swiglu_alpha: float | None = None
    swiglu_beta: float | None = None
    situ_beta: float | None = None
    situ_linear_beta: float | None = None
    gate_up_clamp: float | None = None
    coefficients: tuple[tuple[str, str], ...] = field(init=False)

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown activation family: {self.family!r}")
        if self.activation not in ("swiglu", "situ"):
            raise ValueError("activation must be 'swiglu' or 'situ'")
        for name in ("swiglu_alpha", "swiglu_beta"):
            _number(name, getattr(self, name))
        for name in ("situ_beta", "situ_linear_beta"):
            _number(name, getattr(self, name), positive=True)
        if (self.swiglu_alpha is None) != (self.swiglu_beta is None):
            raise ValueError("swiglu_alpha and swiglu_beta must be supplied together")
        if self.activation == "swiglu":
            if self.situ_beta is not None or self.situ_linear_beta is not None:
                raise ValueError("SiTU beta parameters require activation='situ'")
        elif self.situ_beta is None or self.swiglu_alpha is not None:
            raise ValueError("SiTU requires situ_beta and excludes custom SwiGLU")
        if self.gate_up_clamp is not None and (
            isinstance(self.gate_up_clamp, bool)
            or not isinstance(self.gate_up_clamp, (int, float))
        ):
            raise ValueError("gate_up_clamp must be a Python number or None")

        values: list[tuple[str, float]] = [("one", 1.0)]
        if self.gate_up_clamp is not None:
            limit = abs(float(self.gate_up_clamp))
            values.extend((("clamp", limit), ("negative_clamp", -limit)))
        factored_alpha = self.family in ("nvfp4_legacy", "mxfp8_legacy")
        if self.activation == "swiglu" and self.swiglu_alpha is not None:
            if factored_alpha:
                values.extend(
                    (
                        ("swiglu_alpha", float(self.swiglu_alpha)),
                        ("negative_log2e", -LOG2_E),
                    )
                )
            else:
                values.append(("negative_alpha_log2e", -self.swiglu_alpha * LOG2_E))
            values.append(("swiglu_beta", float(self.swiglu_beta)))
        else:
            values.append(("negative_log2e", -LOG2_E))
        if self.activation == "situ":
            beta = self.situ_beta
            assert beta is not None
            if self.family == "w4a16":
                values.extend(
                    (
                        ("negative_two_log2e", -2.8853900817779268),
                        ("two", 2.0),
                        ("negative_one", -1.0),
                        ("situ_beta", float(beta)),
                        ("inverse_beta", 1.0 / beta),
                    )
                )
                if self.situ_linear_beta is not None:
                    linear = self.situ_linear_beta
                    values.extend(
                        (
                            ("linear_beta", float(linear)),
                            ("inverse_linear_beta", 1.0 / linear),
                        )
                    )
            else:
                values.extend(
                    (
                        ("two_over_beta", 2.0 / beta),
                        ("two_beta", 2.0 * beta),
                        ("negative_beta", -beta),
                    )
                )
                if self.situ_linear_beta is not None:
                    linear = self.situ_linear_beta
                    values.extend(
                        (
                            ("two_over_linear_beta", 2.0 / linear),
                            ("two_linear_beta", 2.0 * linear),
                            ("negative_linear_beta", -linear),
                        )
                    )
        object.__setattr__(
            self, "coefficients", tuple((name, _bits(value)) for name, value in values)
        )

    def coefficient(self, name: str) -> float:
        """Decode an already normalized emitted constant for an FP backend."""
        for key, bits in self.coefficients:
            if key == name:
                return struct.unpack(">f", bytes.fromhex(bits))[0]
        raise KeyError(name)

    @property
    def signature(self) -> tuple:
        """Hashable numerical identity; runtime operands are supplied separately."""
        return (
            "activation-variants-v1",
            self.family,
            self.activation,
            self.swiglu_alpha is not None,
            self.situ_linear_beta is not None,
            self.gate_up_clamp is not None,
            self.coefficients,
        )


def dequant_scale(fp, *, alpha=None, token_scale=None):
    """The existing per-token dequant fold: ``RNmul(token_scale, alpha)``."""
    if token_scale is None:
        return alpha
    return token_scale if alpha is None else fp.mul(token_scale, alpha)


def _sigmoid(fp, value, negative_log2e, one):
    return fp.rcp(fp.add(fp.ex2(fp.mul(value, negative_log2e)), one))


def activation_accumulator(
    fp,
    gate,
    up,
    *,
    variant: ActivationVariant,
    alpha=None,
    weight=None,
    handoff_norm=None,
):
    """Interpret dequant, clamp, activation, route weight and optional handoff norm.

    ``fp`` provides ordered scalar or elementwise-packed FP operations. Pass the
    result of :func:`dequant_scale` as ``alpha`` for per-token scaling. Runtime
    alpha/weight/norm presence is part of the enclosing operator's receipt;
    ``variant.signature`` describes only static activation attributes.
    """

    def const(name):
        return fp.const(variant.coefficient(name))

    if alpha is not None:
        gate, up = fp.mul(gate, alpha), fp.mul(up, alpha)
    if variant.gate_up_clamp is not None:
        limit = const("clamp")
        if variant.family in ("bf16", "mxfp8_legacy", "w4a16"):
            gate = fp.fmin_nan(gate, limit)
            up = fp.fmax_nan(fp.fmin_nan(up, limit), const("negative_clamp"))
        else:
            gate, up = fp.fmin(gate, limit), fp.min_xorsign_abs(up, limit)
    one = const("one")
    if variant.activation == "swiglu":
        sigmoid_input = gate
        if variant.swiglu_alpha is not None:
            if variant.family in ("nvfp4_legacy", "mxfp8_legacy"):
                sigmoid_input = fp.mul(gate, const("swiglu_alpha"))
                coefficient = const("negative_log2e")
            else:
                coefficient = const("negative_alpha_log2e")
            up = fp.add(up, const("swiglu_beta"))
        else:
            coefficient = const("negative_log2e")
        sigmoid = _sigmoid(fp, sigmoid_input, coefficient, one)
        if variant.family == "mxfp8_legacy":
            result = fp.mul(fp.mul(sigmoid, gate), up)
        elif variant.family == "w4a16" and variant.swiglu_alpha is None:
            result = fp.mul(up, fp.mul(gate, sigmoid))
        else:
            result = fp.mul(fp.mul(up, gate), sigmoid)
    else:
        negative_log2e = const("negative_log2e")
        sigmoid = _sigmoid(fp, gate, negative_log2e, one)
        if variant.family == "w4a16":

            def tanh(value):
                denominator = fp.add(
                    fp.ex2(fp.mul(value, const("negative_two_log2e"))), one
                )
                return fp.add(
                    fp.mul(const("two"), fp.rcp(denominator)), const("negative_one")
                )

            bounded_gate = fp.mul(
                const("situ_beta"), tanh(fp.mul(gate, const("inverse_beta")))
            )
            activated_gate = fp.mul(bounded_gate, sigmoid)
            if variant.situ_linear_beta is not None:
                up = fp.mul(
                    const("linear_beta"), tanh(fp.mul(up, const("inverse_linear_beta")))
                )
            result = fp.mul(up, activated_gate)
        else:
            gate_sigmoid = _sigmoid(
                fp, fp.mul(gate, const("two_over_beta")), negative_log2e, one
            )
            bounded_gate = fp.add(
                fp.mul(gate_sigmoid, const("two_beta")), const("negative_beta")
            )
            if variant.situ_linear_beta is not None:
                up_sigmoid = _sigmoid(
                    fp, fp.mul(up, const("two_over_linear_beta")), negative_log2e, one
                )
                up = fp.add(
                    fp.mul(up_sigmoid, const("two_linear_beta")),
                    const("negative_linear_beta"),
                )
            result = fp.mul(fp.mul(bounded_gate, sigmoid), up)
    if weight is not None:
        result = fp.mul(result, weight)
    if handoff_norm is not None:
        result = fp.mul(result, handoff_norm)
    return result


__all__ = ["ActivationVariant", "FAMILIES", "activation_accumulator", "dequant_scale"]
