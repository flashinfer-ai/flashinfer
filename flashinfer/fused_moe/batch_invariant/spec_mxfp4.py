# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Canonical dataflow of the MXFP4 MoE classes ``moe.mxfp4.<act>`` (MXFP4 experts x MXFP4 activations: e2m1 elements with one
UE8M0 block scale per 32 elements on both operands, no global scale), on the bf16 classes' routing, permutation and combine
(:mod:`.spec`).

Per token ``t``, slot ``r`` with expert ``e`` and the class's quantizer ``Q`` (``mxfp4.quant``, the MX recipe of
:mod:`flashinfer.batch_invariant.quantization.mx_spec`: one UE8M0 scale per 32-element block, no per-token scale):

* ``(x4, xsf) = Q(x_t)``: the input row quantized per block;
* ``g_raw, u_raw``: the raw fp32 accumulators of the dense MXFP4 class on the expert's GEMM1 weight (``gemm.mxfp4.kfold`` /
  ``kfold128``: the block scales of both operands applied inside the instruction, no alpha);
* ``a = cvt_bf16(glu_pre(g_raw, u_raw))`` (the bf16 class's gated chain :func:`.spec.glu_elem`, or the ungated
  :func:`.spec.act_elem`): the accumulators carry the MX scales already, so the chain is the bf16 class's as it is;
* ``(a4, asf) = Q(a)``: the materialised bf16 intermediate quantized per block by the same class;
* ``y = cvt_bf16(acc2)`` on GEMM2's raw accumulators (the dense MX class's chain at a unit alpha);
* the bf16 class's slot-order combine.

The class therefore *contains* the ``mxfp4.quant`` class (its fingerprint is part of this class's) and the dense MXFP4 GEMM
classes of the two expert weight shapes (the instruction class per part, :func:`flashinfer.gemm.batch_invariant.schedule.mx_class_for`:
a function of the architecture family only, never of the batch). A chain with an exponential or a division computes those
steps with the part's SFU, as the NVFP4 pair's (:func:`.spec.hw_arith_applies`). Every S-expression here is the Lean model's
(``MoEMxfp4.lean``).
"""

from __future__ import annotations

import hashlib
from typing import Optional, Sequence, Tuple

from ...batch_invariant.quantization.mx_spec import MXFP4
from ...batch_invariant.quantization.mx_spec import (
    spec_fingerprint as mx_quant_fingerprint,
)
from ...gemm.batch_invariant.schedule import mx_class_for
from ...gemm.batch_invariant.spec import MXFP4_FMTS
from ...gemm.batch_invariant.spec import spec_fingerprint as gemm_fingerprint
from . import spec as S
from .mterm import Term, TermAlgebra, fingerprint

CLASS = "moe.mxfp4"
#: Elements per UE8M0 block scale of the MX formats.
MX_BLOCK = 32


def mx_out_elem(fp, acc, out_fmt: str = "bf16"):
    """GEMM2 of the MXFP4 classes: ``cvt_bf16(acc)`` (the dense MX class's conversion; the scales were applied by the
    instruction)."""
    return fp.cvt_out(acc, out_fmt)


def expert_classes_mx(family: Optional[str]) -> Tuple[str, str]:
    """The MXFP4 instruction class of both expert weights: the part's (``mx_class_for``), the same for GEMM1 and GEMM2."""
    c = mx_class_for("mxfp4", family)
    return c, c


class MoEMxfp4Spec(S.MoESpec):
    """Symbolic view of ``moe.mxfp4.<act>[H, I, k, E, method]`` for one token: the routing, the GEMM1 chain on the raw
    accumulators, GEMM2's conversion and the combine of :class:`.spec.MoESpec`; the two block quantizations and the GEMM folds
    are opaque members named by their own fingerprints."""

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
        mx_class: Tuple[str, str] = ("mxfp4", "mxfp4"),
        alg: Optional[TermAlgebra] = None,
    ):
        super().__init__(H, I, k, E, method, "bf16", alg, act, params, gated)
        assert all(c in MXFP4_FMTS for c in mx_class), mx_class
        self.mx_class = mx_class
        self.hw = S.hw_arith_applies(self.act, self.gated)

    def gemm2(self, r: int = 0, h: int = 0) -> Term:
        """GEMM2's epilogue term: the conversion of the raw accumulator."""
        return mx_out_elem(self.alg, self.alg.leaf("acc2", r, h), "bf16")

    @property
    def name(self) -> str:
        return f"{CLASS}.{self.act}"


def spec_fingerprint_mxfp4(
    H: int,
    I: int,
    k: int,
    E: int,
    method: S.RoutingMethod = S.SOFTMAX_RENORM,
    act: str = "silu",
    params: Sequence[Optional[float]] = (),
    gated: Optional[bool] = None,
    mx_class: Tuple[str, str] = ("mxfp4", "mxfp4"),
) -> str:
    """Identifier of the class: the routing weights of slots 0 and k-1, the GEMM1 chain, the dense MXFP4 GEMM classes of both
    expert weights (the instruction class is a dimension), the ``mxfp4.quant`` class (the input and the intermediate are both its
    outputs; its identifier is independent of the row length), GEMM2's conversion and the combine. The activation and its
    binary32 parameters name the class."""
    s = MoEMxfp4Spec(H, I, k, E, method, act, params, gated, mx_class)
    parts = [
        s.name,
        f"H={H}",
        f"I={I}",
        f"k={k}",
        f"E={E}",
        method.name,
        f"act={'glu' if s.gated else 'act'}:{act}",
        f"params={s.params!r}",
        *mx_class,
        fingerprint(s.weight(0)),
        fingerprint(s.weight(k - 1)),
        fingerprint(s.gemm1()),
        gemm_fingerprint(H, "f32", 1, mx_class[0]),
        gemm_fingerprint(I, "f32", 1, mx_class[1]),
        mx_quant_fingerprint(MXFP4),
        fingerprint(s.gemm2()),
        fingerprint(s.combine()),
    ]
    return "moe-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
