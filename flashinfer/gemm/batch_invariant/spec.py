# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Canonical dataflow of the dense GEMM classes (bf16; e4m3 / e5m2 with a per-tensor scale).

Per output element (m, n): a single ``fold`` digit over the K-steps of one ``tcgen05.mma`` (16 bf16 elements
per step on ``kind::f16``, 32 fp8 elements on ``kind::f8f6f4``), ``D_{t+1} = MMA(D_t, A[m, step t], B[step t, n])``
from ``D_0 = +0``, then the epilogue chain: for the fp8 classes ``mul.rn(D, alpha)`` with ``alpha = mul.rn(a_scale,
b_scale)``, the two per-tensor fp32 scales of :func:`flashinfer.bmm_fp8` multiplied once on the device, then exactly
one ``cvt.rn`` to the output dtype. ``MMA`` is the opaque hardware K-step; a 32-wide fp8 step is a different
primitive from the 16-wide bf16 step, so the fp8 classes are different terms from the bf16 class at every K. Tile
shapes, 1-/2-CTA MMA, clusters, raster order, swizzle, pipeline depth and the store path are placements that leave
this sequence untouched (verified bitwise on hardware, since no CPU model of the MMA step exists).

Two consistency classes per format:

* ``gemm.<fmt>.kfold``: one fold over all K-steps from +0;
* ``gemm.<fmt>.kchunk[C]``: K cut into ``C`` equal chunks (``C`` a function of the weight shape and the format only,
  never of M: :func:`flashinfer.gemm.batch_invariant.schedule.chunks_for`), each chunk an MMA fold from +0, the
  chunk partials folded in chunk order with ``add.rn`` from +0. A split kernel computes the chunks on different
  CTAs and folds them in a fixup pass; the in-CTA chunk fold computes the same boundaries inside one CTA.

The fp8 classes carry the suffix ``.scaled`` (``gemm.e4m3.kfold.scaled``): the chain multiplies by ``alpha`` once,
after the fold (the fixup applies it for the workspace split; the in-CTA fold after its last chunk).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

from ...batch_invariant.canon import Digit, Reduction
from ...batch_invariant.fp_interface.term import Term, TermAlgebra, fingerprint

#: The operand formats of the dense classes and their opaque K-step: bf16 on ``kind::f16`` (16 elements per step),
#: the two fp8 formats on ``kind::f8f6f4`` (32 elements per step).
IN_FMTS = ("bf16", "e4m3", "e5m2")
FP8_FMTS = ("e4m3", "e5m2")
MMA_K_OF = {"bf16": 16, "e4m3": 32, "e5m2": 32}
MMA_OP_OF = {"bf16": "mma_bf16_k16", "e4m3": "mma_fp8_k32", "e5m2": "mma_fp8_k32"}
ELEM_BITS_OF = {"bf16": 16, "e4m3": 8, "e5m2": 8}
MMA_K = MMA_K_OF["bf16"]  # the bf16 K step (the bf16 class's constant)
MMA_OP = MMA_OP_OF["bf16"]
IN_FMT = "bf16"
ELEM_BITS = ELEM_BITS_OF["bf16"]


def is_scaled(in_fmt: str) -> bool:
    """The fp8 classes carry the per-tensor scale chain; the bf16 class does not."""
    return in_fmt in FP8_FMTS


def class_name(in_fmt: str, chunked: bool) -> str:
    """``gemm.<fmt>.kfold`` / ``gemm.<fmt>.kchunk``, with ``.scaled`` for the fp8 formats."""
    name = f"gemm.{in_fmt}." + ("kchunk" if chunked else "kfold")
    return name + (".scaled" if is_scaled(in_fmt) else "")


def k_steps(K: int, in_fmt: str = "bf16") -> int:
    return -(-K // MMA_K_OF[in_fmt])


def identity_note(in_fmt: str) -> str:
    mk = MMA_K_OF[in_fmt]
    return (
        f"MMA{mk}(D, 0, 0) == D: the zero-filled K remainder (whole padded steps of the last K-block, TMA zero fill of "
        "both operands) adds products that are all +0; every configuration issues the same padded steps (the inner radix "
        "is fixed), so this corner cannot separate two configurations (R3); the fingerprint hashes the raw K so a padded "
        "and an unpadded K are distinct specifications. The kernel's first K step runs the instruction without an accumulator "
        "input (D = the step's products) where the specification writes MMA(+0, ...): the two can differ only in the sign of "
        "an all-zero result (every configuration does the same, so R3 holds; the chunked class's add(+0, P) normalises it)"
    )


IDENTITY_NOTE = identity_note("bf16")


def kslice_leaf(m: int, n: int):
    """The K-slice leaf of step ``t``: the (A, B) slices of row ``m`` and column ``n``."""

    def leaf(fp, idx):
        t = idx["kstep"]
        return fp.op("ab_kslice", fp.op("a_rows", m, t), fp.op("b_cols", t, n))

    return leaf


def reduction(K: int, m: int, n: int, in_fmt: str = "bf16") -> Reduction:
    """The K-fold for output element (m, n) at the format's step. Leaves are the (A, B) K-slices; the K remainder
    (TMA zero-fill) is an identity step and is therefore dropped here."""
    op = MMA_OP_OF[in_fmt]
    kstep = Digit(
        "kstep",
        k_steps(K, in_fmt),
        "fold",
        init=lambda fp: fp.const(0.0),
        step=lambda fp, acc, tile: fp.op(op, acc, tile),
        identity_note=identity_note(in_fmt),
    )
    return Reduction(f"{class_name(in_fmt, False)}[K={K}]", (kstep,), kslice_leaf(m, n))


def reduction_chunked(
    K: int, C: int, m: int, n: int, in_fmt: str = "bf16"
) -> Reduction:
    """The fixed-chunk class ``gemm.<fmt>.kchunk[C]``: K is cut into ``C`` equal chunks; each chunk is an MMA fold from
    +0; the chunk partials are folded in chunk order with ``add.rn`` from +0."""
    op, mk = MMA_OP_OF[in_fmt], MMA_K_OF[in_fmt]
    assert K % C == 0 and (K // C) % mk == 0, (K, C, in_fmt)
    Kc = K // C
    kstep = Digit(
        "kstep",
        Kc // mk,
        "fold",
        init=lambda fp: fp.const(0.0),
        step=lambda fp, acc, tile: fp.op(op, acc, tile),
        identity_note=identity_note(in_fmt),
    )
    chunk = Digit(
        "chunk",
        C,
        "fold",
        init=lambda fp: fp.const(0.0),
        step=lambda fp, acc, partial: fp.add(acc, partial),
        identity_note="add(+0, P) == P except -0 -> +0, which every config performs alike",
    )
    slice_leaf = kslice_leaf(m, n)

    def leaf(fp, idx):
        return slice_leaf(fp, {"kstep": idx["chunk"] * (Kc // mk) + idx["kstep"]})

    return Reduction(f"{class_name(in_fmt, True)}[K={K},C={C}]", (kstep, chunk), leaf)


def scale_term(fp):
    """``alpha = mul.rn(a_scale, b_scale)``: the two per-tensor fp8 scales are runtime scalars (``sym``) and their product
    is formed on the device with one correctly rounded multiply (never on the host, whose double product would round
    differently). How the scales are formed is part of the input, like the operands themselves."""
    return fp.mul(fp.sym("a_scale"), fp.sym("b_scale"))


@dataclass(frozen=True)
class EpilogueChain:
    """The elementwise ops after the fold, in order: ``mul.rn alpha`` for the scaled (fp8) classes, then exactly one
    ``cvt.rn <out>``. One object, one ``apply``: the spec term (:func:`output_elem`, ``fp`` = the term algebra), the
    kernel (``blocks/epilogue.py``: ``fp`` = ``CuteF32`` for the ops before the conversion, the vector ``.to(out dtype)``
    for the conversion itself) and the fixup all run this code."""

    ops: Tuple[str, ...]

    @staticmethod
    def for_class(out_fmt: str, scaled: bool) -> "EpilogueChain":
        return EpilogueChain(
            (("mul.rn alpha",) if scaled else ()) + (f"cvt.rn {out_fmt}",)
        )

    @property
    def out_fmt(self) -> str:
        assert self.ops and self.ops[-1].startswith("cvt.rn "), self.ops
        return self.ops[-1].split()[1]

    @property
    def pre(self) -> Tuple[str, ...]:
        """The ops before the conversion (all in fp32)."""
        return self.ops[:-1]

    @property
    def scaled(self) -> bool:
        return "mul.rn alpha" in self.pre

    def apply_pre(self, fp, x, alpha=None):
        for op in self.pre:
            if op == "mul.rn alpha":
                assert alpha is not None, "the scaled chain needs alpha"
                x = fp.mul(x, alpha)
            else:  # a new chain op must be added here, nowhere else
                raise ValueError(f"unknown epilogue op {op!r}")
        return x

    def apply(self, fp, x, alpha=None):
        return fp.cvt_out(self.apply_pre(fp, x, alpha), self.out_fmt)

    def describe(self) -> str:
        return " -> ".join(self.ops)


def output_elem(fp, acc, out_fmt: str = "bf16", alpha=None):
    """The epilogue chain of the class: ``cvt_rn(acc)`` or, with ``alpha``, ``cvt_rn(mul.rn(acc, alpha))``."""
    return EpilogueChain.for_class(out_fmt, alpha is not None).apply(fp, acc, alpha)


class GemmSpec:
    def __init__(
        self,
        K: int,
        out_fmt: str = "bf16",
        alg: Optional[TermAlgebra] = None,
        chunks: int = 1,
        in_fmt: str = "bf16",
    ):
        self.K, self.out_fmt, self.chunks, self.in_fmt = K, out_fmt, chunks, in_fmt
        self.scaled = is_scaled(in_fmt)
        self.alg = alg or TermAlgebra()

    def acc(self, m: int, n: int) -> Term:
        if self.chunks == 1:
            return reduction(self.K, m, n, self.in_fmt).canonical(self.alg)
        return reduction_chunked(self.K, self.chunks, m, n, self.in_fmt).canonical(
            self.alg
        )

    def out(self, m: int, n: int) -> Term:
        alpha = scale_term(self.alg) if self.scaled else None
        return output_elem(self.alg, self.acc(m, n), self.out_fmt, alpha)


def spec_fingerprint(
    K: int, out_fmt: str = "bf16", chunks: int = 1, in_fmt: str = "bf16"
) -> str:
    """The identifier of the class for one weight: the term encodes ceil(K / step) opaque steps; the raw K (hence the
    zero-filled tail of the last step), the chunk count, the formats and the chain are part of the specification too,
    so they enter the identifier."""
    s = GemmSpec(K, out_fmt, chunks=chunks, in_fmt=in_fmt)
    tag = f"gemm-{in_fmt}-" + ("kfold" if chunks == 1 else f"kchunk{chunks}")
    tag += "-scaled-" if s.scaled else "-"
    parts = [f"K={K}", f"chunks={chunks}", f"out={out_fmt}"]
    if in_fmt != "bf16":
        parts += [f"in={in_fmt}", f"scaled={int(s.scaled)}"]
    parts += [fingerprint(s.out(0, 0)), fingerprint(s.out(1, 1))]
    return tag + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
