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
from ...batch_invariant.canon import Digit, Reduction
from .spec import (
    ACTS,
    SOFTMAX_RENORM,
    UNGATED_ACTS,
    MoESpec,
    RoutingMethod,
    act_elem,
    glu_elem,
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


# -- the per-token class ``moe.e4m3.<act>.pertoken``: e4m3 experts with one fp32 scale per output channel, e4m3 activations with
# one dynamic fp32 scale per token (and per intermediate row); the fold is the W8A8 class's. Every scale is a function of the
# token's own row or of the fixed weights; the epilogue scale of an element is the row's scale times the column's channel scale.

PT_ELEM, PT_LANES = (
    8,
    32,
)  # the row amax's digits: 8 consecutive elements per leaf vector, 32 lanes, the rest of the row above


def pertoken_leaf(fp: Any, x: Any) -> Any:
    """The row-amax leaf: ``fabs(x)`` of the exactly widened element."""
    return fp.fabs(x)


def pertoken_row_reduction(K: int, leaf_x: Any) -> Reduction:
    """The row amax of the per-token class, ``fmaxc`` over ``fabs(x_i)``, ``i < K``: digits ``elem`` (8 consecutive elements), ``lane``
    (32 lanes), ``rest`` (the row above 256 elements, a power of two, the absent leaves past ``K`` the identity ``fmaxc(v, +0) = v``).
    ``fmaxc`` is the commutativity-normalised max of non-negative values: every association of the leaves is one term, so the
    quantizer's register tree, shuffles and warp exchange compute it whatever their order."""
    assert K % PT_ELEM == 0, K
    blocks = max(1, -(-K // (PT_ELEM * PT_LANES)))
    rest = 1 << (blocks - 1).bit_length()
    combine = lambda fp, a, b: fp.fmaxc(a, b)  # noqa: E731
    note = (
        "fmaxc(v, +0) == v for every v >= +0; every leaf is fabs(x) >= +0 and never -0"
    )

    def leaf(fp, idx):
        i = idx["rest"] * (PT_ELEM * PT_LANES) + idx["lane"] * PT_ELEM + idx["elem"]
        return pertoken_leaf(fp, leaf_x(fp, i)) if i < K else None

    return Reduction(
        f"e4m3pt.row_amax[K={K}]",
        (
            Digit("elem", PT_ELEM, "pairwise", combine=combine, identity_note=note),
            Digit("lane", PT_LANES, "pairwise", combine=combine, identity_note=note),
            Digit("rest", rest, "pairwise", combine=combine, identity_note=note),
        ),
        leaf,
    )


def pertoken_mult(fp: Any, amax: Any) -> Any:
    """``m = div(448, nzor(amax, 448))``: the row's e4m3 multiplier (1 for a zero row: the select keeps the division finite), the
    per-tensor class's convention -- the multiplier first, the correctly rounded reciprocal as the scale."""
    return fp.div(fp.const(E4M3_MAX), fp.nzor(amax, fp.const(E4M3_MAX)))


def pertoken_quant_elem(fp: Any, x: Any, m: Any) -> Any:
    """``cvt.rn.satfinite.e4m3(mul(x, m))``: the class's quantization on the row's multiplier (``x_quant_e4m3_elem`` with the dynamic
    multiplier in the static one's place)."""
    return fp.cvt_out(fp.mul(x, m), "e4m3")


def pertoken_scale(fp: Any, m: Any) -> Any:
    """``s = div(1, m)``: the row's dequantization scale the GEMM epilogue reads."""
    return inv_scale(fp, m)


def pertoken_elem_scale(fp: Any, s_row: Any, s_col: Any) -> Any:
    """``mul(s_row, s_col)``: the epilogue scale of one output element -- the A row's (token's / intermediate row's) scale times the
    expert's channel scale of the output column, formed by the owning thread per element."""
    return fp.mul(s_row, s_col)


def glu_scaled2_elem(
    fp: Any,
    g: Any,
    u: Any,
    sg: Any,
    su: Any,
    act: str = "silu",
    out_fmt: str = "bf16",
    params: Sequence[Optional[float]] = (),
) -> Any:
    """GEMM1 of the per-token class: ``cvt_bf16(mul(act(mul(g, sg)), mul(u, su)))`` -- the gated chain on the two accumulators under
    their OWN scales (the gate and the up columns are different channels of the expert weight)."""
    return glu_elem(fp, fp.mul(g, sg), fp.mul(u, su), act, out_fmt, params)


def act_scaled_elem(
    fp: Any,
    acc: Any,
    s: Any,
    act: str = "relu2",
    out_fmt: str = "bf16",
    params: Sequence[Optional[float]] = (),
) -> Any:
    """The ungated GEMM1 chain of the per-token class: ``cvt_bf16(act(mul(acc, s)))`` (the scale enters once, before the activation)."""
    return act_elem(fp, fp.mul(acc, s), act, out_fmt, params)


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


class Fp8PerTokenMoESpec(MoESpec):
    """Symbolic view of the per-token class ``moe.e4m3.<act>.pertoken[H, I, k, E, method]`` for one token: the routing and the combine
    of :class:`.spec.MoESpec`; the per-token input quantization (the row amax at ``K = H`` inside the multiplier), the token's scale,
    GEMM1 under the (token x channel) scales with bf16 out, the per-row requantization of the intermediate (``K = I``), GEMM2 under
    the (row x channel) scales."""

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

    def channel_scale(self, which: int, r: int, n: int) -> Term:
        """The expert's per-channel scale of output column ``n`` of GEMM ``which`` (1 / 2) for slot ``r``: a per-expert, per-column leaf."""
        return self.alg.leaf(f"sw{which}", r, n)

    def pts(self, r: int) -> Term:
        """The intermediate row's scale for slot ``r`` (GEMM2's A row)."""
        return self.alg.leaf("pts2", r)

    def mult(self) -> Term:
        """The input quantization's multiplier: ``div(448, nzor(amax, 448))`` of the token row's amax (the declared reduction at ``K = H``)."""
        red = pertoken_row_reduction(self.H, lambda fp, i: fp.leaf("x", i))
        return pertoken_mult(self.alg, red.canonical(self.alg))

    def requant_mult(self, r: int = 0) -> Term:
        """The intermediate requantization's multiplier: the same term on the GEMM1 output row of slot ``r`` (``K = I``)."""
        red = pertoken_row_reduction(self.I, lambda fp, j: fp.leaf("h", r, j))
        return pertoken_mult(self.alg, red.canonical(self.alg))

    def scale1(self, r: int = 0, n: int = 0) -> Term:
        return pertoken_elem_scale(
            self.alg, self.alg.sym("s_x"), self.channel_scale(1, r, n)
        )

    def scale2(self, r: int = 0, h: int = 0) -> Term:
        return pertoken_elem_scale(self.alg, self.pts(r), self.channel_scale(2, r, h))

    def gemm1(self, r: int = 0, j: int = 0) -> Term:
        """The GEMM1 output element (bf16): the gated chain on the (gate, up) pair under their own (token x channel) scales, or the
        ungated chain on ``u`` under its column's."""
        g, u = self.acc("g", r, j), self.acc("u", r, j)
        if not self.gated:
            return act_scaled_elem(
                self.alg, u, self.scale1(r, j), self.act, "bf16", self.params
            )
        return glu_scaled2_elem(
            self.alg,
            g,
            u,
            self.scale1(r, 2 * j + 1),
            self.scale1(r, 2 * j),
            self.act,
            "bf16",
            self.params,
        )

    def x_quant(self) -> Term:
        return pertoken_quant_elem(self.alg, self.alg.leaf("x", 0), self.mult())

    def x_scale(self) -> Term:
        return pertoken_scale(self.alg, self.mult())

    def requant(self, r: int = 0, j: int = 0) -> Term:
        return pertoken_quant_elem(
            self.alg, self.alg.leaf("h", r, j), self.requant_mult(r)
        )

    def gemm2(self, r: int = 0, h: int = 0) -> Term:
        return scaled_out_elem(
            self.alg, self.alg.leaf("acc2", r, h), self.scale2(r, h), "bf16"
        )

    @property
    def name(self) -> str:
        return (
            "moe.e4m3."
            + (
                "swiglu"
                if self.gated and self.act == "silu" and not self.params
                else self.act
            )
            + ".pertoken"
        )


def spec_fingerprint_fp8_pertoken(
    H: int,
    I: int,
    k: int,
    E: int,
    method: RoutingMethod = SOFTMAX_RENORM,
    act: str = "silu",
    params: Sequence[Optional[float]] = (),
    gated: Optional[bool] = None,
) -> str:
    """Identifier of the per-token class: the routing weights, the input quantization (the row amax at ``K = H`` inside) and the
    token scale, GEMM1's two-scale chain, the fp8 GEMM folds at ``K = H`` and ``K = I``, the requantization (``K = I``), GEMM2's
    chain and the combine."""
    s = Fp8PerTokenMoESpec(H, I, k, E, method, act, params, gated)
    parts = [
        "moe.e4m3.pertoken",
        f"H={H}",
        f"I={I}",
        f"k={k}",
        f"E={E}",
        method.name,
        f"act={'glu' if s.gated else 'act'}:{act}",
        f"params={s.params!r}",
        fingerprint(s.weight(0)),
        fingerprint(s.weight(k - 1)),
        fingerprint(s.x_quant()),
        fingerprint(s.x_scale()),
        fingerprint(s.gemm1()),
        gemm_fingerprint(H, "f32", 1, IN_FMT),
        gemm_fingerprint(I, "f32", 1, IN_FMT),
        fingerprint(s.requant()),
        fingerprint(s.gemm2()),
        fingerprint(s.combine()),
    ]
    return "moe-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


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
    "Fp8PerTokenMoESpec",
    "PT_ELEM",
    "PT_LANES",
    "act_scaled_elem",
    "glu_scaled2_elem",
    "pertoken_elem_scale",
    "pertoken_leaf",
    "pertoken_mult",
    "pertoken_quant_elem",
    "pertoken_row_reduction",
    "pertoken_scale",
    "spec_fingerprint_fp8_pertoken",
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
