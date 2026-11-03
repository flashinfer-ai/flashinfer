# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Canonical dataflow of the dense bf16 GEMM.

Per output element (m, n): a single ``fold`` digit over the K-steps of one ``tcgen05.mma`` (16 bf16 elements
per step), ``D_{t+1} = MMA(D_t, A[m, step t], B[step t, n])`` from ``D_0 = +0``, then ``cvt.rn`` to the output
dtype. ``MMA`` is the opaque hardware K-step. Tile shapes, 1-/2-CTA MMA, clusters, raster order, swizzle,
pipeline depth and the store path are placements that leave this sequence untouched (verified bitwise on
hardware, since no CPU model of the MMA step exists).

Two consistency classes:

* ``gemm.bf16.kfold``: one fold over all K-steps from +0;
* ``gemm.bf16.kchunk[C]``: K cut into ``C`` equal chunks (``C`` a function of the weight shape only, never of
  M: :func:`flashinfer.gemm.batch_invariant.schedule.chunks_for`), each chunk an MMA fold from +0, the chunk
  partials folded in chunk order with ``add.rn`` from +0. A split kernel computes the chunks on different CTAs
  and folds them in a fixup pass; the in-CTA chunk fold computes the same boundaries inside one CTA.
"""

from __future__ import annotations

import hashlib

from ...batch_invariant.canon import Digit, Reduction
from ...batch_invariant.fp_interface.term import Term, TermAlgebra, fingerprint

MMA_K = 16  # bf16 K extent of one tcgen05.mma
MMA_OP = "mma_bf16_k16"
IN_FMT = "bf16"
ELEM_BITS = 16


def class_name(chunked: bool) -> str:
    return "gemm.bf16.kchunk" if chunked else "gemm.bf16.kfold"


def k_steps(K: int) -> int:
    return -(-K // MMA_K)


IDENTITY_NOTE = (
    f"MMA{MMA_K}(D, 0, 0) == D: the zero-filled K remainder (whole padded steps of the last K-block, TMA zero fill of "
    "both operands) adds products that are all +0; every configuration issues the same padded steps (the inner radix "
    "is fixed), so this corner cannot separate two configurations (R3); the fingerprint hashes the raw K so a padded "
    "and an unpadded K are distinct specifications. The kernel's first K step runs the instruction without an accumulator "
    "input (D = the step's products) where the specification writes MMA(+0, ...): the two can differ only in the sign of "
    "an all-zero result (every configuration does the same, so R3 holds; the chunked class's add(+0, P) normalises it)"
)


def kslice_leaf(m: int, n: int):
    """The K-slice leaf of step ``t``: the (A, B) slices of row ``m`` and column ``n``."""

    def leaf(fp, idx):
        t = idx["kstep"]
        return fp.op("ab_kslice", fp.op("a_rows", m, t), fp.op("b_cols", t, n))

    return leaf


def reduction(K: int, m: int, n: int) -> Reduction:
    """The K-fold for output element (m, n). Leaves are the (A, B) K-slices; the K remainder (TMA zero-fill) is an
    identity step and is therefore dropped here."""
    kstep = Digit(
        "kstep",
        k_steps(K),
        "fold",
        init=lambda fp: fp.const(0.0),
        step=lambda fp, acc, tile: fp.op(MMA_OP, acc, tile),
        identity_note=IDENTITY_NOTE,
    )
    return Reduction(f"{class_name(False)}[K={K}]", (kstep,), kslice_leaf(m, n))


def reduction_chunked(K: int, C: int, m: int, n: int) -> Reduction:
    """The fixed-chunk class ``gemm.bf16.kchunk[C]``: K is cut into ``C`` equal chunks; each chunk is an MMA fold from
    +0; the chunk partials are folded in chunk order with ``add.rn`` from +0."""
    assert K % C == 0 and (K // C) % MMA_K == 0, (K, C)
    Kc = K // C
    kstep = Digit(
        "kstep",
        Kc // MMA_K,
        "fold",
        init=lambda fp: fp.const(0.0),
        step=lambda fp, acc, tile: fp.op(MMA_OP, acc, tile),
        identity_note=IDENTITY_NOTE,
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
        return slice_leaf(fp, {"kstep": idx["chunk"] * (Kc // MMA_K) + idx["kstep"]})

    return Reduction(f"{class_name(True)}[K={K},C={C}]", (kstep, chunk), leaf)


def output_elem(fp, acc, out_fmt: str = "bf16"):
    """The epilogue chain of the class: exactly one ``cvt.rn <out>`` after the fold."""
    return fp.cvt_out(acc, out_fmt)


class GemmSpec:
    def __init__(
        self,
        K: int,
        out_fmt: str = "bf16",
        alg: TermAlgebra | None = None,
        chunks: int = 1,
    ):
        self.K, self.out_fmt, self.chunks = K, out_fmt, chunks
        self.alg = alg or TermAlgebra()

    def acc(self, m: int, n: int) -> Term:
        if self.chunks == 1:
            return reduction(self.K, m, n).canonical(self.alg)
        return reduction_chunked(self.K, self.chunks, m, n).canonical(self.alg)

    def out(self, m: int, n: int) -> Term:
        return output_elem(self.alg, self.acc(m, n), self.out_fmt)


def spec_fingerprint(K: int, out_fmt: str = "bf16", chunks: int = 1) -> str:
    """The identifier of the class for one weight: the term encodes ceil(K / 16) opaque steps; the raw K (hence the
    zero-filled tail of the last step), the chunk count and the output format are part of the specification too, so
    they enter the identifier."""
    s = GemmSpec(K, out_fmt, chunks=chunks)
    tag = "gemm-bf16-" + ("kfold" if chunks == 1 else f"kchunk{chunks}") + "-"
    parts = [
        f"K={K}",
        f"chunks={chunks}",
        f"out={out_fmt}",
        fingerprint(s.out(0, 0)),
        fingerprint(s.out(1, 1)),
    ]
    return tag + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
