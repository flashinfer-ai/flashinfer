# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Canonical dataflow of the MXFP8 x MXFP8 MoE FFN classes ``moe.mxfp8.<act>``.

The class quantizes the activations and the expert weights with the ``mxfp8.quant`` class
(:mod:`flashinfer.batch_invariant.quantization.mx_spec`: e4m3 values, one UE8M0 scale per 32-element block) and runs the
two GEMMs as grouped placements of the dense MXFP8 class of the part (:mod:`flashinfer.gemm.batch_invariant.spec` at
``in_fmt = "mxfp8"`` on SM100 / SM103, ``"mxfp8_k64"`` on SM107: the block scales are operands of the block-scaled
step). Because the scales live inside the step, the class has no software scale: GEMM1's epilogue is the bf16 classes'
chain on the dequantised fp32 accumulators (every typed activation of the layer, :func:`.spec.glu_elem` /
:func:`.spec.act_elem`), the intermediate is quantized by the same quantizer class on the materialised bf16 row (its
declared second pass), GEMM2's epilogue is the plain ``cvt_bf16`` of the accumulator, and the routing and the
slot-order combine are the bf16 classes' (:mod:`.spec`). Everything numeric is therefore already specified once; this
module names the composition and its identifier. The arithmetic is exact: every step of the chain is correctly rounded
and no SFU table enters the bits (the fp4 classes' chains take the part's SFU; these do not), so the bits are a property
of the instruction class alone.

The instruction class is a dimension of the specification: the SM107 step folds 64 elements where the SM100 / SM103 step
folds 32 (`gemm.mxfp8.kfold64` vs `gemm.mxfp8.kfold`), so the fingerprint carries the class of each GEMM by name, as
``fingerprint("mm_mxfp8", ...)`` does for the dense class.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Sequence

from ...batch_invariant.quantization import mx_spec
from ...gemm.batch_invariant.spec import spec_fingerprint as gemm_fingerprint
from .mterm import Term, TermAlgebra, fingerprint
from .spec import SOFTMAX_RENORM, MoESpec, RoutingMethod

CLASS = "moe.mxfp8"
#: The instruction classes of the dense MXFP8 GEMM (`flashinfer.gemm.batch_invariant.schedule.mx_class_for`).
MX_CLASSES = ("mxfp8", "mxfp8_k64")
#: Elements per UE8M0 block scale (the OCP MX block; the quantizer's and the instruction's).
BLOCK = mx_spec.BLOCK


class MoEMxfp8Spec(MoESpec):
    """Symbolic view of ``moe.mxfp8.<act>[H, I, k, E, method, mx_class]`` for one token: the bf16 class's routing,
    GEMM1 chain and combine (inherited), GEMM2 as the plain conversion of the accumulator leaf (the block scales are
    inside the opaque step, so no scale term appears anywhere)."""

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
        mx_class: str = "mxfp8",
    ):
        super().__init__(H, I, k, E, method, "bf16", alg, act, params, gated)
        assert mx_class in MX_CLASSES, mx_class
        self.mx_class = mx_class

    def gemm2(self, r: int = 0, h: int = 0) -> Term:
        """The GEMM2 epilogue term: ``cvt_bf16(acc2)`` (the dense MX class's fold, dequantised by the step)."""
        return self.alg.cvt_out(self.alg.leaf("acc2", r, h), "bf16")

    @property
    def name(self) -> str:
        return f"{CLASS}.{self.act}"


def spec_fingerprint_mxfp8(
    H: int,
    I: int,
    k: int,
    E: int,
    method: RoutingMethod = SOFTMAX_RENORM,
    act: str = "silu",
    params: Sequence[Optional[float]] = (),
    gated: Optional[bool] = None,
    mx_class: str = "mxfp8",
) -> str:
    """Identifier of the class: the routing weights of slot 0 and slot k-1, the GEMM1 chain (the chain's gating, name
    and binary32 parameters are dimensions), the dense MX class of each expert weight at ``K = H`` and ``K = I`` (the
    fold of the part's instruction, named by ``mx_class``), the quantizer class's identifier twice (one class, two
    placements: the activations and the intermediate), GEMM2's conversion and the combine."""
    s = MoEMxfp8Spec(
        H, I, k, E, method, act=act, params=params, gated=gated, mx_class=mx_class
    )
    q = mx_spec.spec_fingerprint(mx_spec.MXFP8)
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
        q,
        q,
        fingerprint(s.gemm2()),
        fingerprint(s.combine()),
    ]
    return "moe-" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
