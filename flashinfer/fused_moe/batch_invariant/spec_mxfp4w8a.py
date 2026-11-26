# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Canonical dataflow of the mixed MXFP4-weight x MXFP8-activation MoE FFN classes ``moe.mxfp4w8a.<act>``.

The class quantizes the expert weights with the ``mxfp4.quant`` class (e2m1 values, two per byte, one UE8M0 scale per
32-element block) and the activations with the ``mxfp8.quant`` class (e4m3 values, one UE8M0 scale per 32), both of
:mod:`flashinfer.batch_invariant.quantization.mx_spec`, and runs the two GEMMs as grouped placements of the dense MIXED
class of the part (:mod:`flashinfer.gemm.batch_invariant.spec` at ``in_fmt = "mxfp4w8a"`` on SM100 / SM103,
``"mxfp4w8a_k64"`` on SM107: the mixed-format block-scaled step with both operands' block scales inside it). Because the
scales live inside the step, the class has no software scale: GEMM1's epilogue is the bf16 classes' chain on the
dequantised fp32 accumulators (every typed activation of the layer, :func:`.spec.glu_elem` / :func:`.spec.act_elem`),
the intermediate is quantized by the ``mxfp8.quant`` class on the bf16-rounded chain value (fused in the epilogue, or the
standalone pass on the materialised row: the same bytes), GEMM2's epilogue is the plain ``cvt_bf16`` of the accumulator,
and the routing and the slot-order combine are the bf16 classes' (:mod:`.spec`). The chain terms are therefore the MXFP8
MoE classes' word for word; what distinguishes the class is the step (the mixed instruction folds its products in its
own order: the e4m3 x e4m3 class on the e2m1 values re-coded to e4m3 gives different bits, so neither is a placement of
the other) and the experts' quantizer. The fingerprint names both.

The instruction class is a dimension of the specification: the SM107 step folds 64 elements where the SM100 / SM103 step
folds 32 (`gemm.mxfp4w8a.kfold64` vs `gemm.mxfp4w8a.kfold`), so the fingerprint carries the class of each GEMM by name, as
``fingerprint("mm_mxfp8_mxfp4", ...)`` does for the dense class.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Sequence

from ...batch_invariant.quantization import mx_spec
from ...gemm.batch_invariant.spec import k_granule
from ...gemm.batch_invariant.spec import spec_fingerprint as gemm_fingerprint
from .mterm import Term, TermAlgebra, fingerprint
from .spec import SOFTMAX_RENORM, MoESpec, RoutingMethod

CLASS = "moe.mxfp4w8a"
#: The instruction classes of the dense mixed GEMM (`flashinfer.gemm.batch_invariant.schedule.mx_class_for("mxfp4w8a")`).
MX_CLASSES = ("mxfp4w8a", "mxfp4w8a_k64")
#: Elements per UE8M0 block scale (the OCP MX block; both quantizers' and the instruction's).
BLOCK = mx_spec.BLOCK


def granule_of(in_fmt: str) -> int:
    """The K granule of the class's operand rows: 32 elements on the SM107 class, 128 on the SM100 / SM103 class (the
    TMA's unpacking format of the e2m1 operand reads whole 64-byte runs: the dense mixed class's rule)."""
    assert in_fmt in MX_CLASSES, in_fmt
    return max(BLOCK, k_granule(in_fmt))


class MoEMxfp4w8aSpec(MoESpec):
    """Symbolic view of ``moe.mxfp4w8a.<act>[H, I, k, E, method, mx_class]`` for one token: the bf16 class's routing,
    GEMM1 chain and combine (inherited), GEMM2 as the plain conversion of the accumulator leaf (both operands' block
    scales are inside the opaque mixed step, so no scale term appears anywhere)."""

    def __init__(
        self,
        H: int,
        I: int,
        k: int,
        E: int,
        method: RoutingMethod = SOFTMAX_RENORM,
        alg: Optional[TermAlgebra] = None,
        act: str = "silu",
        params: Sequence[Optional[float]] = (),
        gated: Optional[bool] = None,
        mx_class: str = "mxfp4w8a",
    ):
        super().__init__(H, I, k, E, method, "bf16", alg, act, params, gated)
        assert mx_class in MX_CLASSES, mx_class
        self.mx_class = mx_class

    def gemm2(self, r: int = 0, h: int = 0) -> Term:
        """The GEMM2 epilogue term: ``cvt_bf16(acc2)`` (the dense mixed class's fold, dequantised by the step)."""
        return self.alg.cvt_out(self.alg.leaf("acc2", r, h), "bf16")

    @property
    def name(self) -> str:
        return f"{CLASS}.{self.act}"


def spec_fingerprint_mxfp4w8a(
    H: int,
    I: int,
    k: int,
    E: int,
    method: RoutingMethod = SOFTMAX_RENORM,
    act: str = "silu",
    params: Sequence[Optional[float]] = (),
    gated: Optional[bool] = None,
    mx_class: str = "mxfp4w8a",
) -> str:
    """Identifier of the class: the routing weights of slot 0 and slot k-1, the GEMM1 chain (the chain's gating, name
    and binary32 parameters are dimensions), the dense mixed class of each expert weight at ``K = H`` and ``K = I`` (the
    fold of the part's mixed instruction, named by ``mx_class``), the experts' quantizer class (``mxfp4.quant``), the
    activations' quantizer class twice (``mxfp8.quant``: one class, two placements, the activations and the intermediate),
    GEMM2's conversion and the combine."""
    s = MoEMxfp4w8aSpec(
        H, I, k, E, method, act=act, params=params, gated=gated, mx_class=mx_class
    )
    q8 = mx_spec.spec_fingerprint(mx_spec.MXFP8)
    q4 = mx_spec.spec_fingerprint(mx_spec.MXFP4)
    parts = [
        s.name,
        f"H={H}",
        f"I={I}",
        f"k={k}",
        f"E={E}",
        method.name,
        f"act={'glu' if s.gated else 'act'}:{s.act}",
        f"params={s.params!r}",
        f"mx={mx_class}",
        fingerprint(s.weight(0)),
        fingerprint(s.weight(k - 1)),
        fingerprint(s.gemm1()),
        gemm_fingerprint(H, "f32", 1, mx_class),
        gemm_fingerprint(I, "f32", 1, mx_class),
        q4,
        q8,
        q8,
        fingerprint(s.gemm2()),
        fingerprint(s.combine()),
    ]
    return "moe-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
