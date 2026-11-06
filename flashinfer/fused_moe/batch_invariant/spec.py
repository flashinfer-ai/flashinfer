# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Canonical dataflow of the bf16 MoE FFN classes ``moe.bf16.swiglu`` and ``moe.bf16.relu2`` (docs/moe_scoping.md,
section 3).

Everything numeric of the op that is not a tensor-core K-step lives here once, parameterised by an ``fp`` backend
(:class:`.mterm.TermAlgebra` symbolic, :class:`.reference.ExactF32` exact CPU, :class:`.reference.TorchF32`
vectorised CPU, the CuTe-DSL primitives inside the kernels):

* ``exp_term``: the *declared* exponential, a fixed sequence of ``.rn`` ops (clamp, round to the nearest integer with
  the 1.5 * 2^23 shift, Cody-Waite range reduction, a Horner polynomial with separate ``mul`` / ``add``, an exact
  power-of-two scale);
* ``silu_term`` / ``swiglu_elem``: the GEMM1 epilogue chain on the two fp32 accumulators; ``gelu_tanh_term`` /
  ``act_term`` / ``glu_elem``: the gated activation as a class parameter (``act`` in ``ACTS``); ``relu2_term`` /
  ``act_elem``: the ungated chain on one accumulator (``act`` in ``UNGATED_ACTS``);
* ``sigmoid_term``, ``softmax_reduction``, ``routing_weights``: the router;
* ``combine_reduction`` / ``combine_elem``: the weighted fold over the k slots.

The per-element GEMM folds are the dense bf16 class's (:mod:`flashinfer.gemm.batch_invariant.spec`, the opaque
``mma16`` K-step); the grouped GEMM is a placement of that class, not a new one. The quantized classes
(docs/moe_quant_scoping.md) are outside this module.

Every S-expression this module builds is the reference model's (the Lean cases compare them textually). The class
identifier :func:`spec_fingerprint` is not: it hashes the dense GEMM class's identifiers at ``K = H`` and ``K = I``
(:func:`flashinfer.gemm.batch_invariant.spec.spec_fingerprint`), whose hash parts are FlashInfer's.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ...batch_invariant.canon import Digit, Reduction
from ...batch_invariant.fp_interface import exact as fp32
from ...gemm.batch_invariant.spec import spec_fingerprint as gemm_fingerprint
from ...tllm_enums import RoutingMethodType
from .mterm import Term, TermAlgebra, fingerprint

# ----------------------------------------------------------------------------- the declared exp

EXP_MAX = 88.0  # exp(88) = 1.65e38 < FLT_MAX; k <= 127
EXP_MIN = -87.0  # exp(-87) = 1.6e-38 > the smallest normal; k >= -126
LOG2E = 1.44269504088896341
SHIFT = 12582912.0  # 1.5 * 2^23: add(v, SHIFT) - SHIFT rounds v to the nearest even integer for |v| < 2^22
LN2_HI = 0.693359375  # 355 / 512: 10 significant bits, so k * LN2_HI is exact for |k| <= 2^14
LN2_LO = -2.12194440e-4
# e^r on |r| <= ln2 / 2 as 1 + r + r^2 (c2 + r (c3 + ... )): Cephes expf's minimax coefficients,
# written as one Horner chain, highest degree first (binary32 values).
EXP_COEFFS = tuple(
    fp32.to_f32(c)
    for c in (
        1.9875691500e-4,
        1.3981999507e-3,
        8.3334519073e-3,
        4.1665795894e-2,
        1.6666665459e-1,
        5.0000001201e-1,
        1.0,
        1.0,
    )
)


def exp_term(fp: Any, x: Any) -> Any:
    """``exp(x)`` as the fixed op sequence of docs/moe_scoping.md section 3.4. Every op is a single correctly rounded
    binary32 operation or exact, so the CPU reproduces the device bit for bit."""
    x = fp.fmax(fp.fmin(x, fp.const(EXP_MAX)), fp.const(EXP_MIN))
    t = fp.add(fp.mul(x, fp.const(LOG2E)), fp.const(SHIFT))
    kf = fp.add(t, fp.const(-SHIFT))
    r = fp.add(x, fp.mul(kf, fp.const(-LN2_HI)))
    r = fp.add(r, fp.mul(kf, fp.const(-LN2_LO)))
    p = fp.const(EXP_COEFFS[0])
    for c in EXP_COEFFS[1:]:
        p = fp.add(fp.mul(p, r), fp.const(c))
    return fp.mul(p, fp.exp2i(kf))


def sigmoid_term(fp: Any, x: Any) -> Any:
    """``1 / (1 + exp(-x))`` with the declared exp and one correctly rounded division."""
    return fp.div(fp.const(1.0), fp.add(fp.const(1.0), exp_term(fp, fp.neg(x))))


def silu_term(fp: Any, g: Any) -> Any:
    """``g / (1 + exp(-g))``: one division, no separate sigmoid multiply."""
    return fp.div(g, fp.add(fp.const(1.0), exp_term(fp, fp.neg(g))))


def swiglu_elem(fp: Any, g: Any, u: Any, out_fmt: str = "bf16") -> Any:
    """The GEMM1 epilogue chain: ``cvt(silu(g) * u)`` from the two fp32 accumulators."""
    return fp.cvt_out(fp.mul(silu_term(fp, g), u), out_fmt)


# ----------------------------------------------------------------------------- routing


@dataclass(frozen=True)
class RoutingMethod:
    """``score``: ``softmax`` (selection on the logits, weights = probabilities) or ``sigmoid``
    (selection on ``sigmoid(l) + bias``, weights = the unbiased sigmoid). ``renormalize`` divides
    the k weights by their slot-order sum; ``bias`` says a per-expert bias enters the selection
    score; ``scaled`` multiplies the weights by a runtime scalar (DeepSeek's routed_scaling_factor)."""

    score: str = "softmax"
    renormalize: bool = True
    bias: bool = False
    scaled: bool = False

    def __post_init__(self):
        assert self.score in ("softmax", "sigmoid"), self.score
        assert not (self.bias and self.score == "softmax"), (
            "a selection bias is a sigmoid-routing feature"
        )

    @property
    def name(self) -> str:
        return (
            self.score
            + ("_renorm" if self.renormalize else "")
            + ("_bias" if self.bias else "")
            + ("_scaled" if self.scaled else "")
        )


SOFTMAX = RoutingMethod("softmax", renormalize=False)  # FlashInfer Default
SOFTMAX_RENORM = RoutingMethod(
    "softmax", renormalize=True
)  # FlashInfer Renormalize (torch reference form)
SIGMOID = RoutingMethod("sigmoid", renormalize=False)
SIGMOID_RENORM = RoutingMethod("sigmoid", renormalize=True)  # FlashInfer SigmoidRenorm
SIGMOID_BIAS_RENORM_SCALED = RoutingMethod(
    "sigmoid", True, True, True
)  # DeepSeek-V3 (n_group <= 1), noaux
METHODS: Dict[str, RoutingMethod] = {
    m.name: m
    for m in (
        SOFTMAX,
        SOFTMAX_RENORM,
        SIGMOID,
        SIGMOID_RENORM,
        SIGMOID_BIAS_RENORM_SCALED,
        RoutingMethod("sigmoid", True, True, False),
        RoutingMethod("sigmoid", False, True, False),
    )
}

#: The :class:`~flashinfer.tllm_enums.RoutingMethodType` values whose semantics (selection order, weight formula) are
#: one of the methods above; the method's dataflow is this module's (docs/moe_scoping.md section 3.1). ``Default`` is
#: softmax then top-k; ``Renormalize`` (FlashInfer's top-k then softmax over the survivors) and ``RenormalizeNaive``
#: (softmax, top-k, divide by the top-k sum) are both the value softmax -> top-k -> renormalise; ``SigmoidRenorm`` and
#: ``Sigmoid`` select on the sigmoid and weight by it, with and without the renormalisation; ``DeepSeekV3`` is the
#: ``n_group <= 1`` case only (a grouped selection is another method, which the API refuses). The values absent here
#: have no method in this module.
METHOD_OF_FLASHINFER: Dict[RoutingMethodType, RoutingMethod] = {
    RoutingMethodType.Default: SOFTMAX,
    RoutingMethodType.Renormalize: SOFTMAX_RENORM,
    RoutingMethodType.RenormalizeNaive: SOFTMAX_RENORM,
    RoutingMethodType.SigmoidRenorm: SIGMOID_RENORM,
    RoutingMethodType.Sigmoid: SIGMOID,
    RoutingMethodType.DeepSeekV3: SIGMOID_BIAS_RENORM_SCALED,
}


def pow2ceil(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def softmax_reduction(E: int, leaf_z: Callable[[Any, int], Any]) -> Reduction:
    """``S = sum_e z_e`` over the ``expert`` digit (radix ``pow2ceil(E)``, pairwise ``add.rn``,
    experts ``>= E`` absent). ``leaf_z(fp, e)`` returns ``z_e = exp(l_e - m)``. A kernel may pad
    absent experts with ``+0``: every ``z_e`` is ``> +0`` (``exp`` is clamped below at ``e^-87``), so
    ``add(v, +0) = v`` holds for every partial."""
    expert = Digit(
        "expert",
        pow2ceil(E),
        "pairwise",
        combine=lambda fp, a, b: fp.add(a, b),
        identity_note="add(v, +0) == v for v > +0; every z_e = exp(.) >= e^-87 > +0",
    )

    def leaf(fp, idx):
        e = idx["expert"]
        return leaf_z(fp, e) if e < E else None

    return Reduction(f"moe.softmax_sum[E={E}]", (expert,), leaf)


def softmax_terms(fp: Any, logits: Sequence[Any], E: int) -> Tuple[List[Any], Any]:
    """``m``, ``z_e`` and ``S`` for one token: ``z_e = exp(add(l_e, neg(m)))``, ``S`` the canonical
    pairwise tree. Returns ``(z, S)`` with ``z`` a list over experts."""
    m = fp.rowmax(*logits)
    z = [exp_term(fp, fp.add(l, fp.neg(m))) for l in logits]
    S = softmax_reduction(E, lambda fp_, e: z[e]).canonical(fp)
    return z, S


def per_expert_scores(
    fp: Any,
    logits: Sequence[Any],
    method: RoutingMethod,
    bias: Optional[Sequence[Any]] = None,
) -> Tuple[List[Any], List[Any]]:
    """``(selection_score_e, weight_e)`` for every expert. softmax: selection on the logit itself,
    weight ``p_e = div(z_e, S)``; sigmoid: ``s_e = sigmoid(l_e)``, selection on ``add(s_e, bias_e)``
    (or ``s_e``), weight ``s_e``."""
    E = len(logits)
    if method.score == "softmax":
        z, S = softmax_terms(fp, logits, E)
        p = [fp.div(z_e, S) for z_e in z]
        return list(logits), p
    s = [sigmoid_term(fp, l) for l in logits]
    if method.bias:
        assert bias is not None and len(bias) == E
        q = [fp.add(s_e, b_e) for s_e, b_e in zip(s, bias, strict=True)]
    else:
        q = list(s)
    return q, s


def select_topk(scores: Sequence[float], k: int) -> List[int]:
    """The selection: experts in the strict total order ``(score desc, index asc)``; slot ``r``
    holds the expert of rank ``r``. Pure Python on exact binary32 values; ``-0 == +0`` is a tie
    broken by the index, as the device's ``setp`` sees it."""
    assert 1 <= k <= len(scores)
    order = sorted(range(len(scores)), key=lambda e: (-scores[e], e))
    return order[:k]


def slot_weights(
    fp: Any,
    weight_of: Sequence[Any],
    selected: Sequence[int],
    method: RoutingMethod,
    scale: Any = None,
) -> List[Any]:
    """The k routing weights in slot order: ``w_r = weight_of[e_r]``; with renormalisation
    ``S_k = fold_r add(acc, w_r)`` from ``+0`` and ``w_r = div(w_r, S_k)``; with scaling
    ``w_r = mul(w_r, scale)``."""
    ws = [weight_of[e] for e in selected]
    if method.renormalize:
        acc = fp.const(0.0)
        for w in ws:
            acc = fp.add(acc, w)
        ws = [fp.div(w, acc) for w in ws]
    if method.scaled:
        assert scale is not None
        ws = [fp.mul(w, scale) for w in ws]
    return ws


def routing_weights(
    fp: Any,
    logits: Sequence[Any],
    k: int,
    method: RoutingMethod,
    bias: Optional[Sequence[Any]] = None,
    scale: Any = None,
    selected: Optional[Sequence[int]] = None,
) -> Tuple[List[int], List[Any]]:
    """One token's routing: returns ``(selected, weights)``. ``selected`` may be given (symbolic
    backends cannot compare terms); otherwise the scores are selected with :func:`select_topk`."""
    q, wgt = per_expert_scores(fp, logits, method, bias)
    if selected is None:
        selected = select_topk(q, k)
    return list(selected), slot_weights(fp, wgt, selected, method, scale)


# ----------------------------------------------------------------------------- combine


def combine_reduction(k: int, leaf_yw: Callable[[Any, int], Any]) -> Reduction:
    """``acc_{r+1} = add(acc_r, yw_r)`` from ``+0`` over the ``slot`` digit (radix ``k``, a fold in
    slot order). ``leaf_yw(fp, r)`` returns ``mul(f32(y_r), w_r)``."""
    slot = Digit(
        "slot",
        k,
        "fold",
        init=lambda fp: fp.const(0.0),
        step=lambda fp, acc, yw: fp.add(acc, yw),
        identity_note="the fold starts at +0: add(+0, P) == P except -0 -> +0, which every config performs alike",
    )
    return Reduction(
        f"moe.combine[k={k}]", (slot,), lambda fp, idx: leaf_yw(fp, idx["slot"])
    )


def combine_elem(
    fp: Any, ys: Sequence[Any], ws: Sequence[Any], out_fmt: str = "bf16"
) -> Any:
    """``out = cvt( fold_r add(acc, mul(y_r, w_r)) )``; ``ys`` are the (exactly widened) bf16 expert
    outputs of the token's k slots, ``ws`` the routing weights, both in slot order."""
    k = len(ys)
    red = combine_reduction(k, lambda fp_, r: fp_.mul(ys[r], ws[r]))
    return fp.cvt_out(red.canonical(fp), out_fmt)


# ----------------------------------------------------------------------------- the activations as class parameters

# gelu_tanh(x) = 0.5 x (1 + tanh(y)) with y = sqrt(2/pi) (x + 0.044715 x^3), written as x * sigmoid(2y) (the identity
# 0.5 (1 + tanh(y)) = 1 / (1 + e^{-2y}): one declared sigmoid, no cancellation in the negative tail, where the
# (e^{2y} - 1) / (e^{2y} + 1) form loses every bit of 1 + tanh).
GELU_C1 = 0.7978845608028654  # sqrt(2 / pi)
GELU_C2 = 0.044715
ACTS = ("silu", "gelu_tanh")
UNGATED_ACTS = (
    "relu2",
)  # the ungated chains: one accumulator per output (no gate / up pairing)


def gelu_tanh_term(fp: Any, x: Any) -> Any:
    """``mul(x, sigmoid(mul(2, mul(C1, add(x, mul(C2, mul(x, mul(x, x))))))))`` with the declared sigmoid (hence the
    declared exp, clamped at [-87, 88]: the tail saturates at ``x / (1 + e^88)``, below any output format's
    resolution)."""
    x3 = fp.mul(x, fp.mul(x, x))
    y = fp.mul(fp.const(GELU_C1), fp.add(x, fp.mul(fp.const(GELU_C2), x3)))
    return fp.mul(x, sigmoid_term(fp, fp.mul(fp.const(2.0), y)))


def act_term(fp: Any, act: str, g: Any) -> Any:
    assert act in ACTS, act
    return silu_term(fp, g) if act == "silu" else gelu_tanh_term(fp, g)


def relu2_term(fp: Any, x: Any) -> Any:
    """``mul(r, r)`` with ``r = fmax(x, 0)``: ReLU^2 of one fp32 accumulator (FlashInfer ``ActivationType.Relu2``);
    the clamp is the order-preserving ``fmax`` of the routing."""
    r = fp.fmax(x, fp.const(0.0))
    return fp.mul(r, r)


def act_elem(fp: Any, acc: Any, act: str = "relu2", out_fmt: str = "bf16") -> Any:
    """The ungated GEMM1 epilogue chain: ``cvt(act(acc))`` from one fp32 accumulator, ``act`` in ``UNGATED_ACTS``."""
    assert act in UNGATED_ACTS, act
    return fp.cvt_out(relu2_term(fp, acc), out_fmt)


def glu_pre(fp: Any, g: Any, u: Any, act: str = "silu") -> Any:
    """``p = mul(act(g), u)`` in fp32: the gated activation before any rounding."""
    return fp.mul(act_term(fp, act, g), u)


def glu_elem(fp: Any, g: Any, u: Any, act: str = "silu", out_fmt: str = "bf16") -> Any:
    """``cvt(mul(act(g), u))``: ``swiglu_elem`` for ``act = "silu"`` (the same term), GeGLU otherwise."""
    return fp.cvt_out(glu_pre(fp, g, u, act), out_fmt)


# ----------------------------------------------------------------------------- the class


class MoESpec:
    """Symbolic view of the class ``moe.bf16.swiglu[H, I, k, E, method]`` for one token (``act`` in ``UNGATED_ACTS``:
    the ungated class ``moe.bf16.<act>``, GEMM1 on the single accumulator)."""

    def __init__(
        self,
        H: int,
        I: int,
        k: int,
        E: int,
        method: RoutingMethod = SOFTMAX_RENORM,
        out_fmt: str = "bf16",
        alg: Optional[TermAlgebra] = None,
        act: str = "silu",
    ):
        self.H, self.I, self.k, self.E, self.method, self.out_fmt, self.act = (
            H,
            I,
            k,
            E,
            method,
            out_fmt,
            act,
        )
        self.alg = alg or TermAlgebra()

    # leaves
    def logit(self, e: int) -> Term:
        return self.alg.leaf("l", e)

    def bias(self, e: int) -> Term:
        return self.alg.leaf("b", e)

    def acc(self, which: str, r: int, j: int) -> Term:
        """The fp32 GEMM1 accumulator (``g`` gate / ``u`` up) of slot ``r``, column ``j``: an
        opaque-leaf view of ``gemm.bf16.kfold[K=H]`` (the GEMM class has its own spec)."""
        return self.alg.leaf(which, r, j)

    def y(self, r: int, h: int) -> Term:
        return self.alg.leaf("y", r, h)

    # terms
    def routing(
        self, selected: Optional[Sequence[int]] = None
    ) -> Tuple[List[int], List[Term]]:
        a = self.alg
        sel = list(selected) if selected is not None else list(range(self.k))
        bias = [self.bias(e) for e in range(self.E)] if self.method.bias else None
        scale = a.sym("scale") if self.method.scaled else None
        return routing_weights(
            a,
            [self.logit(e) for e in range(self.E)],
            self.k,
            self.method,
            bias,
            scale,
            selected=sel,
        )

    def swiglu(self, r: int = 0, j: int = 0) -> Term:
        return swiglu_elem(
            self.alg, self.acc("g", r, j), self.acc("u", r, j), self.out_fmt
        )

    def gemm1(self, r: int = 0, j: int = 0) -> Term:
        """The GEMM1 epilogue term of the class: the gated SwiGLU, or the ungated ``act_elem`` on the single
        accumulator."""
        if self.act in UNGATED_ACTS:
            return act_elem(self.alg, self.acc("u", r, j), self.act, self.out_fmt)
        return self.swiglu(r, j)

    def combine(self, h: int = 0, selected: Optional[Sequence[int]] = None) -> Term:
        _, ws = self.routing(selected)
        return combine_elem(
            self.alg, [self.y(r, h) for r in range(self.k)], ws, self.out_fmt
        )

    def weight(self, r: int = 0) -> Term:
        return self.routing()[1][r]


def spec_fingerprint(
    H: int,
    I: int,
    k: int,
    E: int,
    method: RoutingMethod = SOFTMAX_RENORM,
    out_fmt: str = "bf16",
    act: str = "silu",
) -> str:
    """Identifier of the class: the routing weight of slot 0 and slot k-1, the GEMM1 chain term (swiglu, or the
    ungated ``act``: a distinct class ``moe.bf16.<act>``), the two GEMM classes (``gemm.bf16.kfold[K=H]``,
    ``[K=I]``) and the combine term."""
    s = MoESpec(H, I, k, E, method, out_fmt, act=act)
    parts = (
        [f"H={H}", f"I={I}", f"k={k}", f"E={E}", method.name, out_fmt]
        + ([f"act={act}"] if act != "silu" else [])
        + [
            fingerprint(s.weight(0)),
            fingerprint(s.weight(k - 1)),
            fingerprint(s.gemm1()),
            gemm_fingerprint(H),
            gemm_fingerprint(I),
            fingerprint(s.combine()),
        ]
    )
    return "moe-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
