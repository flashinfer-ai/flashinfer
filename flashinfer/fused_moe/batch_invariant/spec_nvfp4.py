# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Canonical dataflow of the NVFP4 MoE classes ``moe.nvfp4.<act>`` (W4A4: NVFP4 experts with one per-tensor global scale per
expert and GEMM, per-token NVFP4 activations), on the bf16 classes' routing, permutation and combine (:mod:`.spec`).

Per token ``t``, slot ``r`` with expert ``e`` and the class's per-token quantizer ``Q`` (``nvfp4.quant.pertoken``, the
TransformerEngine recipe of :mod:`flashinfer.batch_invariant.quantization`):

* ``(x4, xsf, pts1) = Q(x_t)``: the input row quantized per token (the row's global scale derived from its amax; ``pts1``
  the returned per-token scale, a multiplier);
* ``g_raw, u_raw``: the raw fp32 accumulators of the dense NVFP4 class on the expert's GEMM1 weight (``gemm.nvfp4.kfold`` /
  ``kfold128``: the block scales applied inside the instruction, no alpha);
* ``s1 = mul(alpha1[e], pts1)`` (:func:`w4a4_row_scale`): the expert's alpha (``1 / global_scale``, as given, the multiplier
  of ``mm_fp4``) times the token's per-token scale, one ``mul.rn`` per (row, expert);
* ``a = cvt_bf16(glu_pre(mul(g_raw, s1), mul(u_raw, s1)))`` (:func:`glu_scaled_elem`): the gated chain of the bf16 class on
  the scaled accumulators, or ``cvt_bf16(act(mul(acc, s1)))`` (:func:`act_scaled_elem`) for the ungated chains;
* ``(a4, asf, pts2) = Q(a)``: the materialised bf16 intermediate quantized per token by the same class (two passes: the
  per-token rule's row amax spans the whole intermediate width, which no GEMM1 tile sees);
* ``y = cvt_bf16(mul(acc2, mul(alpha2[e], pts2)))`` (:func:`scaled_out_elem`) on GEMM2's raw accumulators;
* the bf16 class's slot-order combine.

The W4A4 class therefore *contains* the per-token quantization class at row lengths ``H`` and ``I`` (its fingerprints are
part of this class's) and the dense NVFP4 GEMM classes of the two expert weight shapes (the instruction class per weight
shape by the dense rule, :func:`flashinfer.gemm.batch_invariant.schedule.fp4_class_for`, applied to the two weight shapes by
:func:`.api_nvfp4.expert_classes`: a function of the shape and the architecture family only, never of the batch; the rule is
a dispatcher function, so it lives outside this module, which the kernels' cache key includes). Every S-expression here is
the Lean model's (``MoENvfp4.lean``).
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...batch_invariant.fp_interface import exact as fp32
from ...gemm.batch_invariant.spec import NVFP4_FMTS
from ...gemm.batch_invariant.spec import spec_fingerprint as gemm_fingerprint
from . import spec as S
from .mterm import Term, TermAlgebra, fingerprint

CLASS = "moe.nvfp4"
#: The two intermediate quantizations of the W4A4 classes: ``per_token`` (``moe.nvfp4.<act>``: the per-token class on the
#: materialised bf16 intermediate) and ``static`` (``moe.nvfp4.<act>.static``: GEMM1's epilogue quantizes the fp32 gated value per
#: 16-column block with the expert's static global scale).
INTERMEDIATES = ("per_token", "static")
#: Elements per e4m3 block scale of the NVFP4 formats.
NVFP4_BLOCK = 16
#: TransformerEngine's rounded reciprocal of the e2m1 maximum (the per-tensor quantizer's ``C_SIXTH``).
NVFP4_C_SIXTH = fp32.to_f32(1.0 / 6.0)
#: TransformerEngine's inverse base scale of the per-token NVFP4 recipe (``1 / (e4m3 max * e2m1 max)``): the per-token
#: quantizer's input for the activations and the intermediate.
TE_GS_INV = 1.0 / (448.0 * 6.0)
#: ``448 * 6``: the per-tensor global scale of a weight is this constant over the weight's amax (FlashInfer's convention).
E4M3_MAX, E2M1_MAX = 448.0, 6.0


def w4a4_row_scale(fp: Any, alpha_e: Any, pts: Any) -> Any:
    """``s = mul(alpha_e, pts_row)``: the expert's alpha (the caller's ``1 / global_scale[e]``, as given, like ``mm_fp4``'s)
    times the row's per-token scale; one ``mul.rn`` per (row, expert), then one ``mul.rn`` per accumulator element."""
    return fp.mul(alpha_e, pts)


def glu_scaled_elem(
    fp: Any,
    g: Any,
    u: Any,
    s: Any,
    act: str = "silu",
    out_fmt: str = "bf16",
    params: Sequence[Optional[float]] = (),
    hw: bool = False,
) -> Any:
    """GEMM1 of the gated W4A4 classes: ``cvt_bf16(glu_pre(mul(g, s), mul(u, s)))`` from the two raw accumulators (the
    bf16 class's gated chain ``act`` with its parameters on the scaled pair; ``hw``: the hardware arithmetic)."""
    return S.glu_elem(fp, fp.mul(g, s), fp.mul(u, s), act, out_fmt, params, hw)


def act_scaled_elem(
    fp: Any,
    acc: Any,
    s: Any,
    act: str = "relu2",
    out_fmt: str = "bf16",
    params: Sequence[Optional[float]] = (),
    hw: bool = False,
) -> Any:
    """GEMM1 of the ungated W4A4 classes: ``cvt_bf16(act(mul(acc, s)))`` from the single raw accumulator (the row scale
    first, then the bf16 class's ungated chain; ``hw``: the hardware arithmetic)."""
    return S.act_elem(fp, fp.mul(acc, s), act, out_fmt, params, hw)


def scaled_out_elem(fp: Any, acc: Any, s: Any, out_fmt: str = "bf16") -> Any:
    """GEMM2 of the W4A4 classes: ``cvt_bf16(mul(acc, s))`` with ``s`` the row scale of the intermediate row."""
    return fp.cvt_out(fp.mul(acc, s), out_fmt)


# -- the static intermediate class ``moe.nvfp4.<act>.static``: the per-tensor NVFP4 quantization class's block chain
# (:mod:`flashinfer.batch_invariant.quantization.nvfp4_spec`: ``sf_value`` / ``sf_quant`` / ``inv_scale`` / ``quant_elem`` and the
# canonical block amax tree), restated here on the fp32 gated values with the expert's static scale; the device-free tests pin the
# restatement to the quantizer's terms S-expression for S-expression


def glu_scaled_pre(
    fp: Any,
    g: Any,
    u: Any,
    s: Any,
    act: str = "silu",
    params: Sequence[Optional[float]] = (),
    hw: bool = False,
) -> Any:
    """The fp32 gated value of the W4A4 chain before any rounding: ``glu_pre(mul(g, s), mul(u, s))``."""
    return S.glu_pre(fp, fp.mul(g, s), fp.mul(u, s), act, params, hw)


def act_scaled_pre(
    fp: Any,
    acc: Any,
    s: Any,
    act: str = "relu2",
    params: Sequence[Optional[float]] = (),
    hw: bool = False,
) -> Any:
    """The fp32 ungated value of the W4A4 chain before any rounding: ``act(mul(acc, s))``."""
    assert not tuple(params), (act, params)
    return S.ungated_term(fp, act, fp.mul(acc, s), hw)


def nvfp4_block_amax(fp: Any, xs: Sequence[Any]) -> Any:
    """The block amax of 16 values: the adjacent pairwise ``fmaxc`` tree over ``fabs(x)`` (the quantizer's canonical block
    reduction; exact in every association: the max of non-negative values)."""
    vals = [fp.fabs(x) for x in xs]
    assert len(vals) == NVFP4_BLOCK, len(vals)
    while len(vals) > 1:
        vals = [fp.fmaxc(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
    return vals[0]


def nvfp4_sf_value(fp: Any, amax: Any, gs: Any) -> Any:
    """``sf_f = mul(amax, mul(gs, 1/6))``."""
    return fp.mul(amax, fp.mul(gs, fp.const(NVFP4_C_SIXTH)))


def nvfp4_inv_scale(fp: Any, sf_d: Any, gs: Any) -> Any:
    """``inv = fmin(div(1, mul(sf_d, div(1, gs))), FLT_MAX)``: the quantizer's encode scale, correctly rounded."""
    one = fp.const(1.0)
    return fp.fmin(fp.div(one, fp.mul(sf_d, fp.div(one, gs))), fp.const(fp32.F32_MAX))


def nvfp4_block_chain(fp: Any, amax: Any, gs: Any) -> Tuple[Any, Any]:
    """``(sf_d, inv)`` of a block: the e4m3 scale (as its widened value) and the encode scale."""
    sf_d = fp.cvt_out(nvfp4_sf_value(fp, amax, gs), "e4m3")
    return sf_d, nvfp4_inv_scale(fp, sf_d, gs)


def nvfp4_quant_elem(fp: Any, x: Any, inv: Any) -> Any:
    """``cvt_e2m1(mul(x, inv))``."""
    return fp.cvt_out(fp.mul(x, inv), "e2m1")


def static_block(fp: Any, vals: Sequence[Any], gs: Any) -> Tuple[Any, List[Any]]:
    """One block of the static class's GEMM1 output from its 16 fp32 values and the expert's static scale: ``(sf_d, codes)`` --
    the stored e4m3 scale and the 16 e2m1 values: the per-tensor quantizer's block chain and element chain on the fp32 gated
    values (no bf16 rounding, no row amax)."""
    sf_d, inv = nvfp4_block_chain(fp, nvfp4_block_amax(fp, vals), gs)
    return sf_d, [nvfp4_quant_elem(fp, v, inv) for v in vals]


def static_inv2(fp: Any, gs: Any) -> Any:
    """GEMM2's static factor ``inv2[e] = div(1, gs2[e])``: the quantizer's ``div``, formed once per expert on the host and read by
    the kernel as given."""
    return fp.div(fp.const(1.0), gs)


class MoENvfp4Spec(S.MoESpec):
    """Symbolic view of ``moe.nvfp4.<act>[H, I, k, E, method]`` for one token: the routing and the combine of
    :class:`.spec.MoESpec`, the GEMM1 chain on the raw accumulators with the row scale, GEMM2's scaled chain; the two
    per-token quantizations and the GEMM folds are opaque members named by their own fingerprints."""

    def __init__(
        self,
        H: int,
        I: int,
        k: int,
        E: int,
        method: S.RoutingMethod = S.SOFTMAX_RENORM,
        act: str = "silu",
        params: Sequence[Optional[float]] = (),
        gated: Optional[bool] = None,
        fp4_class: Tuple[str, str] = ("nvfp4", "nvfp4"),
        alg: Optional[TermAlgebra] = None,
        intermediate: str = "per_token",
        arith: str = "exact",
    ):
        super().__init__(H, I, k, E, method, "bf16", alg, act, params, gated)
        assert all(c in NVFP4_FMTS for c in fp4_class), fp4_class
        assert intermediate in INTERMEDIATES, intermediate
        assert arith in S.ARITHS, arith
        self.fp4_class = fp4_class
        self.intermediate = intermediate
        # the hardware-arithmetic class: the chain's exponential and divisions on the part's SFU functions (a class of its own
        # where the chain has them; the other chains are the exact class whatever the request)
        self.hw = arith == "hw" and S.hw_arith_applies(self.act, self.gated)

    def alpha(self, which: int, r: int = 0) -> Term:
        """The expert's alpha of GEMM ``which`` (1 / 2) for slot ``r``: a per-expert leaf."""
        return self.alg.leaf(f"alpha{which}", r)

    def pts(self, which: int, r: int = 0) -> Term:
        """The per-token scale of the GEMM's A row (GEMM1: the token's; GEMM2: the intermediate row of slot ``r``)."""
        return self.alg.leaf("pts2", r) if which == 2 else self.alg.leaf("pts1")

    def gs2(self, r: int = 0) -> Term:
        """The static class's per-expert global scale of the intermediate (a per-expert leaf, as the alphas)."""
        return self.alg.leaf("gs2", r)

    def row_scale(self, which: int, r: int = 0) -> Term:
        if which == 2 and self.intermediate == "static":
            return w4a4_row_scale(
                self.alg, self.alpha(2, r), static_inv2(self.alg, self.gs2(r))
            )
        return w4a4_row_scale(self.alg, self.alpha(which, r), self.pts(which, r))

    def gemm1_pre(self, r: int = 0, j: int = 0) -> Term:
        """The fp32 GEMM1 value of column ``j`` before any rounding (the static class quantizes it in place)."""
        s = self.row_scale(1, r)
        if not self.gated:
            return act_scaled_pre(
                self.alg, self.acc("u", r, j), s, self.act, self.params, self.hw
            )
        return glu_scaled_pre(
            self.alg,
            self.acc("g", r, j),
            self.acc("u", r, j),
            s,
            self.act,
            self.params,
            self.hw,
        )

    def gemm1_block(self, r: int = 0, b: int = 0) -> Dict[str, Any]:
        """The static class's GEMM1 output block ``b`` of slot ``r``: ``{"sf": the e4m3 scale term, "codes": the 16 e2m1 terms}``."""
        assert self.intermediate == "static"
        vals = [self.gemm1_pre(r, NVFP4_BLOCK * b + i) for i in range(NVFP4_BLOCK)]
        sf_d, codes = static_block(self.alg, vals, self.gs2(r))
        return {"sf": sf_d, "codes": codes}

    def gemm1(self, r: int = 0, j: int = 0) -> Term:
        """The GEMM1 epilogue output element (bf16; the per-token class then quantizes the row)."""
        s = self.row_scale(1, r)
        if not self.gated:
            return act_scaled_elem(
                self.alg, self.acc("u", r, j), s, self.act, "bf16", self.params, self.hw
            )
        return glu_scaled_elem(
            self.alg,
            self.acc("g", r, j),
            self.acc("u", r, j),
            s,
            self.act,
            "bf16",
            self.params,
            self.hw,
        )

    def gemm2(self, r: int = 0, h: int = 0) -> Term:
        return scaled_out_elem(
            self.alg, self.alg.leaf("acc2", r, h), self.row_scale(2, r), "bf16"
        )

    @property
    def name(self) -> str:
        return (
            f"{CLASS}.{self.act}"
            + (".static" if self.intermediate == "static" else "")
            + (".hw" if self.hw else "")
        )


def spec_fingerprint_nvfp4(
    H: int,
    I: int,
    k: int,
    E: int,
    method: S.RoutingMethod = S.SOFTMAX_RENORM,
    act: str = "silu",
    params: Sequence[Optional[float]] = (),
    gated: Optional[bool] = None,
    fp4_class: Tuple[str, str] = ("nvfp4", "nvfp4"),
    intermediate: str = "per_token",
    arith: str = "exact",
) -> str:
    """Identifier of the class: the routing weights of slots 0 and k-1, the GEMM1 chain, the dense NVFP4 GEMM classes of
    both expert weights (named per GEMM: the instruction class is a dimension), the input quantization class at row length
    ``H`` and the intermediate's at row length ``I`` (the per-token class; the static class: the per-tensor class, with GEMM1's
    term the block scale and the first / last code of block 0 on the fp32 gated values), GEMM2's chain and the combine. The
    activation and its binary32 parameters name the class."""
    # absolute on purpose: a fingerprint-only dependency, outside the kernel cache closures
    from flashinfer.batch_invariant.quantization.nvfp4_spec import PER_TENSOR, PER_TOKEN
    from flashinfer.batch_invariant.quantization.nvfp4_spec import (
        spec_fingerprint as quant_fingerprint,
    )

    s = MoENvfp4Spec(
        H,
        I,
        k,
        E,
        method,
        act,
        params,
        gated,
        fp4_class,
        intermediate=intermediate,
        arith=arith,
    )
    if intermediate == "static":
        blk = s.gemm1_block(0, 0)
        quant = [
            quant_fingerprint(H, PER_TOKEN),
            quant_fingerprint(I, PER_TENSOR),
            "static",
            fingerprint(blk["sf"]),
            fingerprint(blk["codes"][0]),
            fingerprint(blk["codes"][NVFP4_BLOCK - 1]),
        ]
    else:
        quant = [quant_fingerprint(H, PER_TOKEN), quant_fingerprint(I, PER_TOKEN)]
    parts = [
        s.name,
        f"H={H}",
        f"I={I}",
        f"k={k}",
        f"E={E}",
        method.name,
        f"act={'glu' if s.gated else 'act'}:{act}",
        f"params={s.params!r}",
        *fp4_class,
        fingerprint(s.weight(0)),
        fingerprint(s.weight(k - 1)),
        fingerprint(s.gemm1()),
        gemm_fingerprint(H, "f32", 1, fp4_class[0]),
        gemm_fingerprint(I, "f32", 1, fp4_class[1]),
        *quant,
        fingerprint(s.gemm2()),
        fingerprint(s.combine()),
    ]
    return "moe-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
