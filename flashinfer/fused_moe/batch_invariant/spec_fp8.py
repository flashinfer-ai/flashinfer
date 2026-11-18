# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The fp8 per-tensor MoE classes ``moe.e4m3.<act>`` (W8A8): the declared chains around the fp8 GEMM folds.

The routing, the activations (:func:`.spec.glu_pre` / :func:`.spec.ungated_term`: every typed activation of the layer)
and the combine are the bf16 classes' (:mod:`.spec`); this module declares what the quantized class adds:

* the scales. FlashInfer's per-tensor scales are multipliers (``q = cvt(mul(x, m))``: ``hidden_states_scale_global``,
  ``intermediate_scale_global`` and the per-expert weight multipliers of the preparation). The class's dequantization
  scales are their correctly rounded reciprocals ``inv_scale(m) = div(1, m)``, formed once at weight preparation
  (``s_x``, ``s_inter``, ``s_w1[e]``, ``s_w2[e]``); ``w8a8_alpha(s_a, s_w[e]) = mul(s_a, s_w[e])`` is formed on the device per
  (tile, expert), as the dense fp8 class forms its alpha;
* the input quantization ``x_quant_e4m3_elem(x, m_x) = cvt.rn.satfinite.e4m3(mul(x, m_x))``;
* GEMM1's fused quantization: the gated chain ``glu_quant_e4m3_elem = cvt_e4m3(mul(glu_pre(mul(g, alpha), mul(u, alpha)),
  m_inter))`` (the activation on the alpha-scaled fp32 accumulators, no bf16 rounding, one ``mul.rn`` by the intermediate
  multiplier as given, one saturating RNE conversion to e4m3) and the ungated chain ``act_quant_e4m3_elem`` on the single
  accumulator; the activation's scalars (``spec.ACT_PARAMS``) apply to the alpha-scaled, i.e. dequantized, accumulators;
* GEMM2's chain ``scaled_out_elem(acc, alpha2) = cvt_bf16(mul(acc, alpha2))`` with ``alpha2 = mul(s_inter, s_w2[e])``.

The GEMM folds are the dense fp8 class's (:mod:`flashinfer.gemm.batch_invariant.spec`, the opaque ``mma32`` K-step) at
``K = H`` and ``K = I``; the grouped GEMM is a placement of that class. Every S-expression this module builds is the
reference model's (the Lean cases compare them textually, ``lean/BIKernels/MoEFp8.lean``).
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional, Sequence

from ...gemm.batch_invariant.spec import spec_fingerprint as gemm_fingerprint
from .mterm import Term, TermAlgebra, fingerprint
from .spec import (
    ACTS,
    SOFTMAX_RENORM,
    UNGATED_ACTS,
    MoESpec,
    RoutingMethod,
    glu_pre,
    ungated_term,
)

IN_FMT = "e4m3"
E4M3_MAX = 448.0


def w8a8_alpha(fp: Any, s_a: Any, s_w: Any) -> Any:
    """``alpha_e = mul(s_a, s_w[e])``: the fp8 dense class's alpha, formed on the device per (tile, expert)."""
    return fp.mul(s_a, s_w)


def inv_scale(fp: Any, m: Any) -> Any:
    """``div(1, m)``: the correctly rounded reciprocal of a per-tensor multiplier, the class's dequantization scale."""
    return fp.div(fp.const(1.0), m)


def x_quant_e4m3_elem(fp: Any, x: Any, m: Any) -> Any:
    """The input quantization ``cvt.rn.satfinite.e4m3(mul(x, m_x))`` (a caller who hands the backend e4m3 activations has
    done this, or any other, quantization: the class then starts at the bytes)."""
    return fp.cvt_out(fp.mul(x, m), "e4m3")


def glu_quant_pre(
    fp: Any,
    g: Any,
    u: Any,
    alpha: Any,
    act: str = "silu",
    params: Sequence[Optional[float]] = (),
) -> Any:
    """``glu_pre(mul(g, alpha), mul(u, alpha))``: the gated activation of the alpha-scaled accumulator pair in fp32."""
    return glu_pre(fp, fp.mul(g, alpha), fp.mul(u, alpha), act, params)


def glu_quant_e4m3_elem(
    fp: Any,
    g: Any,
    u: Any,
    alpha: Any,
    inv: Any,
    act: str = "silu",
    params: Sequence[Optional[float]] = (),
) -> Any:
    """GEMM1 of the gated W8A8 classes: ``cvt_e4m3(mul(glu_pre(mul(g, alpha), mul(u, alpha)), inv))`` -- the activation on the
    alpha-scaled accumulators in fp32 (no bf16 rounding), one ``mul.rn`` by the intermediate multiplier, one saturating RNE
    conversion to e4m3."""
    return fp.cvt_out(fp.mul(glu_quant_pre(fp, g, u, alpha, act, params), inv), "e4m3")


def act_quant_pre(
    fp: Any,
    acc: Any,
    alpha: Any,
    act: str = "relu2",
    params: Sequence[Optional[float]] = (),
) -> Any:
    """``act(mul(acc, alpha))``: the ungated activation of the alpha-scaled single accumulator in fp32."""
    assert not tuple(params), (act, params)
    return ungated_term(fp, act, fp.mul(acc, alpha))


def act_quant_e4m3_elem(
    fp: Any,
    acc: Any,
    alpha: Any,
    inv: Any,
    act: str = "relu2",
    params: Sequence[Optional[float]] = (),
) -> Any:
    """GEMM1 of the ungated W8A8 classes: ``cvt_e4m3(mul(act(mul(acc, alpha)), inv))`` on the single accumulator."""
    return fp.cvt_out(fp.mul(act_quant_pre(fp, acc, alpha, act, params), inv), "e4m3")


def scaled_out_elem(fp: Any, acc: Any, s: Any, out_fmt: str = "bf16") -> Any:
    """GEMM2 of the W8A8 classes: ``cvt(mul(acc, alpha2))``."""
    return fp.cvt_out(fp.mul(acc, s), out_fmt)


class Fp8MoESpec(MoESpec):
    """Symbolic view of the class ``moe.e4m3.<act>[H, I, k, E, method]`` for one token: the routing and the combine of
    :class:`.spec.MoESpec`, the GEMM1 chain on the raw fp8 accumulators with the class's scales (the multipliers ``m_x``,
    ``m_inter`` as symbols, the per-expert weight multipliers ``mw1`` / ``mw2`` as leaves), the input quantization and GEMM2's
    scaled chain."""

    def __init__(
        self,
        H: int,
        I: int,
        k: int,
        E: int,
        method: RoutingMethod = SOFTMAX_RENORM,
        act: str = "silu",
        params: Sequence[Optional[float]] = (),
        gated: Optional[bool] = None,
        alg: Optional[TermAlgebra] = None,
    ):
        super().__init__(H, I, k, E, method, "bf16", alg, act, params, gated)

    def m_x(self) -> Term:
        return self.alg.sym("m_x")

    def m_inter(self) -> Term:
        return self.alg.sym("m_inter")

    def m_w(self, which: int, r: int) -> Term:
        """The per-expert weight multiplier of GEMM ``which`` (1 / 2) for slot ``r``: a per-expert leaf."""
        return self.alg.leaf(f"mw{which}", r)

    def gemm1_scale(self, r: int = 0) -> Term:
        """``alpha1 = mul(div(1, m_x), div(1, mw1[e]))``."""
        return w8a8_alpha(
            self.alg,
            inv_scale(self.alg, self.m_x()),
            inv_scale(self.alg, self.m_w(1, r)),
        )

    def gemm2_scale(self, r: int = 0) -> Term:
        """``alpha2 = mul(div(1, m_inter), div(1, mw2[e]))``."""
        return w8a8_alpha(
            self.alg,
            inv_scale(self.alg, self.m_inter()),
            inv_scale(self.alg, self.m_w(2, r)),
        )

    def gemm1(self, r: int = 0, j: int = 0) -> Term:
        """The GEMM1 output element (e4m3): the gated chain on the (gate, up) pair or the ungated chain on ``u``."""
        g, u = self.acc("g", r, j), self.acc("u", r, j)
        if not self.gated:
            return act_quant_e4m3_elem(
                self.alg, u, self.gemm1_scale(r), self.m_inter(), self.act, self.params
            )
        return glu_quant_e4m3_elem(
            self.alg, g, u, self.gemm1_scale(r), self.m_inter(), self.act, self.params
        )

    def x_quant(self) -> Term:
        return x_quant_e4m3_elem(self.alg, self.alg.leaf("x", 0), self.m_x())

    def gemm2(self, r: int = 0, h: int = 0) -> Term:
        return scaled_out_elem(
            self.alg, self.alg.leaf("acc2", r, h), self.gemm2_scale(r), "bf16"
        )

    @property
    def name(self) -> str:
        return "moe.e4m3." + (
            "swiglu"
            if self.gated and self.act == "silu" and not self.params
            else self.act
        )


def spec_fingerprint_fp8(
    H: int,
    I: int,
    k: int,
    E: int,
    method: RoutingMethod = SOFTMAX_RENORM,
    act: str = "silu",
    params: Sequence[Optional[float]] = (),
    gated: Optional[bool] = None,
) -> str:
    """Identifier of the class: the routing weights of slot 0 and slot k-1, the GEMM1 chain (its activation, gating and
    binary32 parameters), the fp8 GEMM classes at ``K = H`` and ``K = I``, the input quantization, GEMM2's chain and the
    combine. The scales' values, like the operands, are inputs."""
    s = Fp8MoESpec(H, I, k, E, method, act, params, gated)
    parts = [
        "moe.e4m3",
        f"H={H}",
        f"I={I}",
        f"k={k}",
        f"E={E}",
        method.name,
        f"act={'glu' if s.gated else 'act'}:{act}",
        f"params={s.params!r}",
        fingerprint(s.weight(0)),
        fingerprint(s.weight(k - 1)),
        fingerprint(s.gemm1()),
        gemm_fingerprint(H, "f32", 1, IN_FMT),
        gemm_fingerprint(I, "f32", 1, IN_FMT),
        fingerprint(s.x_quant()),
        fingerprint(s.gemm2()),
        fingerprint(s.combine()),
    ]
    return "moe-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


__all__ = [
    "ACTS",
    "E4M3_MAX",
    "Fp8MoESpec",
    "IN_FMT",
    "UNGATED_ACTS",
    "act_quant_e4m3_elem",
    "act_quant_pre",
    "glu_quant_e4m3_elem",
    "glu_quant_pre",
    "inv_scale",
    "scaled_out_elem",
    "spec_fingerprint_fp8",
    "w8a8_alpha",
    "x_quant_e4m3_elem",
]
