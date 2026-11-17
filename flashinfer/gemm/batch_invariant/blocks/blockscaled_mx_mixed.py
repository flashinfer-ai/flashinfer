# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The mixed MX operand plans: e2m1 weights (B) against e4m3 activations (A) on the block-scaled body (``blockscaled.py``),
one UE8M0 scale per 32 elements on both operands.

Classes ``gemm.mxfp4w8a.kfold`` / ``kchunk[C]`` (``in_fmt = "mxfp4w8a"``: the sm_100 / sm_103 ``kind::mxf8f6f4`` instruction
with mixed operand formats, 32 elements per step, one scale per row and step) and ``gemm.mxfp4w8a.kfold64`` / ``kchunk64[C]``
(``"mxfp4w8a_k64"``: the sm_107 instruction of the same kind, 64 per step, two scales per row and step): the dense K fold with
a mixed-operand opaque step and the caller's ``alpha`` in the chain. The e2m1 x e4m3 product is exact in binary32, so the
step's per-element function is the exact product; the fold over a step's products is the instruction's own and differs from
the e4m3 x e4m3 step's on the same values (the tests pin it as an observation), which is why each class is its own fold.

What differs from the one-format plans, all placements:

* the operand containers: the sm_100 instruction reads a sub-byte operand of ``kind::mxf8f6f4`` from an 8-bit container
  per element (CUTLASS's `float_e2m1_unpacksmem_t`); the TMA's unpacking tensor-map format fills it from the packed e2m1
  weight (``B_SMEM_DT`` / ``B_TMA_INTERNAL``, and the transaction count stays the packed bytes), on every part that has the
  instruction, and the unpacking runs over whole 64-byte runs of the global row, so K is a multiple of 128 for this class
  (``spec.k_granule``). The sm_107 instruction reads the packed e2m1 operand directly (FlashInfer's sm107 block-scaled
  kernel's mixed path), so the K = 64 plan keeps the packed container and the 32-element granule;
* the swapped decode records (``Knobs.swap_ab``: the weight as the kernel's A, the tokens as its B) exchange the operand
  formats, so they run on a twin plan with A = e2m1 / B = e4m3 (:func:`plan_for`): the instruction variant with the formats
  exchanged joins the hardware hypothesis the swapped-record test checks bit for bit against the dense records;
* the A operand ring bound and the TMEM budget are the one-format plans'.
"""

from __future__ import annotations

import cutlass

from ..generator import Knobs
from ..spec import SF_VEC
from .blockscaled import E8M0_DTYPE, BlockScaledPlan, BlockScaledPlanSM107

E4M3 = cutlass.Float8E4M3FN
E2M1 = cutlass.Float4E2M1FN
#: The sm_100 instruction's 8-bit smem container of a sub-byte operand (``kind::mxf8f6f4``).
CONTAINER = cutlass.Int8


class BlockScaledPlanMXW48(BlockScaledPlan):
    """e4m3 activations x e2m1 weights, UE8M0 scales per 32 on both: the sm_100 / sm_103 ``kind::mxf8f6f4`` instruction at
    32 elements per step; the e2m1 weight unpacked by the TMA into the 8-bit container the instruction reads (one element
    per byte: the instruction's sub-byte operand format, :data:`CONTAINER`)."""

    IN_FMT = "mxfp4w8a"
    A_DT = E4M3
    B_DT = E2M1
    SF_DT = E8M0_DTYPE
    SF_VEC = SF_VEC["mxfp4w8a"]
    B_SMEM_DT = CONTAINER
    B_TMA_INTERNAL = CONTAINER


class BlockScaledPlanMXW48Swapped(BlockScaledPlanMXW48):
    """The swapped decode records' plan: the e2m1 weight is the kernel's A (unpacked), the e4m3 token tile its B."""

    A_DT = E2M1
    B_DT = E4M3
    A_SMEM_DT = CONTAINER
    A_TMA_INTERNAL = CONTAINER
    B_SMEM_DT = None
    B_TMA_INTERNAL = None


class BlockScaledPlanMXW48K64(BlockScaledPlanSM107):
    """e4m3 activations x e2m1 weights, UE8M0 scales per 32: the sm_107 ``kind::mxf8f6f4`` instruction at 64 elements per step
    (two scales per row and step); the packed e2m1 operand read directly (the sm_107 plan's TMA box and smem layout at
    4 bits per element)."""

    IN_FMT = "mxfp4w8a_k64"
    INST_K = 64
    A_DT = E4M3
    B_DT = E2M1
    SF_DT = E8M0_DTYPE
    SF_VEC = SF_VEC["mxfp4w8a_k64"]


class BlockScaledPlanMXW48K64Swapped(BlockScaledPlanMXW48K64):
    """The sm_107 class's swapped decode records: the e2m1 weight as A, the e4m3 tokens as B."""

    A_DT = E2M1
    B_DT = E4M3


PLANS = {"mxfp4w8a": BlockScaledPlanMXW48, "mxfp4w8a_k64": BlockScaledPlanMXW48K64}
SWAPPED_PLANS = {
    "mxfp4w8a": BlockScaledPlanMXW48Swapped,
    "mxfp4w8a_k64": BlockScaledPlanMXW48K64Swapped,
}


def plan_for(in_fmt: str, knobs: Knobs):
    """The plan class of a record of a mixed class: the swapped decode records run the twin with the operand formats
    exchanged (the kernel's A is the weight), every other record the class's plan."""
    return (SWAPPED_PLANS if knobs.swap_ab else PLANS)[in_fmt]
