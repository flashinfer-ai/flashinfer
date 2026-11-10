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

Each class has a biased variant (``BIAS_NOTE``): the bias of the output column, widened from bf16 to binary32 exactly, is
added once with ``add.rn`` to the fully reduced accumulator before the single ``cvt.rn``; the fold is untouched.

The block-scale fp8 class ``gemm.e4m3.kblock128`` (format key ``e4m3_bs``; :func:`flashinfer.gemm_fp8_nt_groupwise` with
``scale_granularity_mnk = (1, 128, 128)``): the same e4m3 ``mma32`` step, but K is folded in blocks of ``BLOCK_K`` elements
(one fp8 smem stage: 4 steps), each block an ``mma32`` fold from +0, and the block partials are folded in block order from +0
with ``fma.rn(mul.rn(s_a[m, kb], s_b[n // SCALE_BLOCK_N, kb]), P_kb, acc)``: A's fp32 scale of the row's 1x128 block and B's
fp32 scale of the 128x128 weight block, leaves of the fold (:func:`reduction_blockscale`). No chain scalar (the conversion
alone), no split variant. A different term from the per-tensor fp8 fold at every K (Lean ``kblock128_ne_kfold32``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

from ...batch_invariant.canon import Digit, Reduction
from ...batch_invariant.fp_interface.term import Term, TermAlgebra, fingerprint

#: The operand formats of the dense classes and their opaque K-step: bf16 on ``kind::f16`` (16 elements per step),
#: the two fp8 formats on ``kind::f8f6f4`` (32 elements per step).
IN_FMTS = (
    "bf16",
    "e4m3",
    "e5m2",
    "mxfp8",
    "mxfp8_k64",
    "nvfp4",
    "nvfp4_k128",
    "mxfp4",
    "mxfp4_k128",
    "e4m3_bs",
)
FP8_FMTS = ("e4m3", "e5m2")
#: The block-scale fp8 class (module docstring): e4m3 elements, fp32 scales per 1x128 block of A and per 128x128 block of B,
#: the scales applied by the kernel's promotion pass once per K block (no instruction operand).
FP8_BLOCKSCALE_FMTS = ("e4m3_bs",)
BLOCK_K = 128  # elements of K per scale block: the class constant (scale_granularity_mnk = (1, 128, 128))
SCALE_BLOCK_N = 128  # output columns per weight scale block
#: The MX format: e4m3 elements with one UE8M0 (power-of-two) block scale per 32 elements in the 128x4 swizzled layout, the
#: scale consumed by the block-scaled instruction itself (no software step touches it). Two classes, one per instruction:
#: ``mxfp8`` on ``kind::mxf8f6f4`` with 32 elements per step (one scale per row per step; the sm_100 / sm_103 parts) and
#: ``mxfp8_k64`` on the sm_107 instruction with 64 elements per step (two scales per row per step). A wider step folds more
#: products per accumulator update: a different per-element fold, hence a different class (``gemm.mxfp8.kfold64``), distinct
#: from ``gemm.mxfp8.kfold`` at every K. The API runs the class of the device's architecture family (`schedule.mx_class_for`).
#: The 4-bit formats (e2m1 elements, two per byte): NVFP4 with one e4m3 block scale per 16 elements (``nvfp4`` on the sm_100 /
#: sm_103 ``kind::mxf4nvf4`` instruction, 64 elements per step; ``nvfp4_k128`` on the sm_107 one, 128 per step) and MXFP4 with
#: one UE8M0 scale per 32 elements (``mxfp4`` on ``kind::mxf4``, 64 per step; ``mxfp4_k128`` on the sm_107 instruction, 128
#: per step). Their chain multiplies by the caller's scalar ``alpha`` (``mm_fp4``'s 1-element tensor) before the conversion.
NVFP4_FMTS = ("nvfp4", "nvfp4_k128")
MXFP4_FMTS = ("mxfp4", "mxfp4_k128")
MXFP8_FMTS = ("mxfp8", "mxfp8_k64")
MX_FMTS = MXFP8_FMTS + MXFP4_FMTS  # the UE8M0 formats
FP4_FMTS = NVFP4_FMTS + MXFP4_FMTS
BLOCK_SCALED = frozenset(
    MX_FMTS + NVFP4_FMTS
)  # formats whose step takes scale-factor operands (the leaf carries them)
ALPHA_GIVEN = frozenset(
    FP4_FMTS
)  # formats whose chain scalar is the caller's alpha, not the product of two per-tensor scales
MMA_K_OF = {
    "bf16": 16,
    "e4m3": 32,
    "e5m2": 32,
    "mxfp8": 32,
    "mxfp8_k64": 64,
    "nvfp4": 64,
    "nvfp4_k128": 128,
    "mxfp4": 64,
    "mxfp4_k128": 128,
    "e4m3_bs": 32,
}
MMA_OP_OF = {
    "bf16": "mma_bf16_k16",
    "e4m3": "mma_fp8_k32",
    "e5m2": "mma_fp8_k32",
    "mxfp8": "mma_mxfp8_k32",
    "mxfp8_k64": "mma_mxfp8_k64",
    "nvfp4": "mma_nvfp4_k64",
    "nvfp4_k128": "mma_nvfp4_k128",
    "mxfp4": "mma_mxfp4_k64",
    "mxfp4_k128": "mma_mxfp4_k128",
    "e4m3_bs": "mma_fp8_k32",
}
ELEM_BITS_OF = {
    "bf16": 16,
    "e4m3": 8,
    "e5m2": 8,
    "mxfp8": 8,
    "mxfp8_k64": 8,
    "nvfp4": 4,
    "nvfp4_k128": 4,
    "mxfp4": 4,
    "mxfp4_k128": 4,
    "e4m3_bs": 8,
}
#: Dense MMA throughput relative to bf16 (what the dispatcher's estimate uses): the 8-bit kinds run at twice the bf16 rate,
#: the 4-bit ones at four times; a block-scaled step at the rate of the dense step of its width.
MMA_RATE_OF = {
    "bf16": 1.0,
    "e4m3": 2.0,
    "e5m2": 2.0,
    "mxfp8": 2.0,
    "mxfp8_k64": 2.0,
    "nvfp4": 4.0,
    "nvfp4_k128": 4.0,
    "mxfp4": 4.0,
    "mxfp4_k128": 4.0,
    "e4m3_bs": 2.0,
}
SF_VEC = {
    "mxfp8": 32,
    "mxfp8_k64": 32,
    "nvfp4": 16,
    "nvfp4_k128": 16,
    "mxfp4": 32,
    "mxfp4_k128": 32,
}  # elements per block scale
SF_FMT = {
    "mxfp8": "e8m0",
    "mxfp8_k64": "e8m0",
    "nvfp4": "e4m3",
    "nvfp4_k128": "e4m3",
    "mxfp4": "e8m0",
    "mxfp4_k128": "e8m0",
}  # the block scale's format
#: The 128x4 swizzled scale-factor layout (what ``mxfp8_quantize(..., is_sf_swizzled_layout=True)`` writes): atoms of 128
#: rows x 4 blocks of ``SF_VEC`` elements (512 bytes), K-tiles consecutive, row-tiles strided by the K-tiles of the whole K.
SF_ATOM_ROWS, SF_ATOM_KBLOCKS = 128, 4


def sf_bytes(rows: int, k: int, in_fmt: str = "mxfp8") -> int:
    """Bytes of the 128x4 swizzled scale-factor buffer of a ``(rows, k)`` operand of the format: whole atoms."""
    return (
        -(-rows // SF_ATOM_ROWS)
        * SF_ATOM_ROWS
        * -(-k // (SF_VEC[in_fmt] * SF_ATOM_KBLOCKS))
        * SF_ATOM_KBLOCKS
    )


CLASS_TAG = {
    "mxfp8_k64": ("mxfp8.kfold64", "mxfp8.kchunk64"),
    "nvfp4_k128": ("nvfp4.kfold128", "nvfp4.kchunk128"),
    "mxfp4_k128": ("mxfp4.kfold128", "mxfp4.kchunk128"),
    "e4m3_bs": (
        "e4m3.kblock128",
        "e4m3.kblock128",
    ),  # the block-scale class has no split variant
}  # class names where the format key is not the class name
MMA_K = MMA_K_OF["bf16"]  # the bf16 K step (the bf16 class's constant)
MMA_OP = MMA_OP_OF["bf16"]
BIAS_OP = "bias_col"  # the bias leaf: the bias of one output column as a binary32 value
IN_FMT = "bf16"
ELEM_BITS = ELEM_BITS_OF["bf16"]


def is_scaled(in_fmt: str) -> bool:
    """The fp8 classes carry the per-tensor scale chain and the 4-bit classes the caller's alpha; the bf16 and the MXFP8
    classes do not (``mm_mxfp8`` has no alpha; the block scales are the instruction's operands, not a chain step)."""
    return in_fmt in FP8_FMTS or in_fmt in ALPHA_GIVEN


def chunk_unit(in_fmt: str) -> int:
    """The K granule of the fixed-chunk rule (``schedule.chunks_for``): a chunk is a whole number of MMA steps and, for the
    block-scaled formats, a whole number of scale-factor atom columns (the 128x4 layout stores 4 consecutive K-blocks of
    ``SF_VEC`` elements together, so a chunk boundary must fall on a ``4 * SF_VEC`` element boundary): the lcm of the two.
    16 / 32 for bf16 / fp8, 128 for the MX classes (4 x 32 elements: 4 steps of the sm_100 instruction, 2 of the sm_107 one)."""
    import math

    mk = MMA_K_OF[in_fmt]
    if in_fmt not in BLOCK_SCALED:
        return mk
    atom = 4 * SF_VEC[in_fmt]
    return mk * atom // math.gcd(mk, atom)


def class_name(in_fmt: str, chunked: bool) -> str:
    """``gemm.<fmt>.kfold`` / ``gemm.<fmt>.kchunk`` (with ``.scaled`` for the fp8 formats), or the declared class name of a
    format whose key differs from it (``mxfp8_k64``: ``gemm.mxfp8.kfold64`` / ``gemm.mxfp8.kchunk64``)."""
    tag = CLASS_TAG.get(in_fmt, (f"{in_fmt}.kfold", f"{in_fmt}.kchunk"))
    return "gemm." + tag[1 if chunked else 0] + (".scaled" if is_scaled(in_fmt) else "")


def k_steps(K: int, in_fmt: str = "bf16") -> int:
    return -(-K // MMA_K_OF[in_fmt])


def identity_note(in_fmt: str) -> str:
    mk = MMA_K_OF[in_fmt]
    if in_fmt in NVFP4_FMTS:
        # the zero-filled e2m1 values and e4m3 scales are +0: every padded product is +0 and the step adds nothing
        return (
            f"MMA{mk}(D, A=0, SFA=s, B=0, SFB=s) == D for finite s: the K remainder's values are TMA zero fill (+0), its scales "
            "the scale buffers' padding bytes (the quantizers write zeros: e4m3 +0; the contract requires finite bytes), so "
            "every padded product is +0 and the step adds nothing; every configuration issues the same padded steps (the inner "
            "radix is fixed), so this corner cannot separate two configurations (R3); the fingerprint hashes the raw K so a "
            "padded and an unpadded K are distinct specifications"
        )
    if in_fmt in BLOCK_SCALED:
        # UE8M0 has no zero: the zero-filled scale byte 0x00 is 2^-127, a finite positive power of two; the zero-filled values
        # are +0, so every padded product is (+0 * 2^-127) * (+0 * 2^-127) = +0 and the step adds nothing
        return (
            f"MMA{mk}(D, A=0, SFA=s, B=0, SFB=s) == D for finite s: the K remainder's values are TMA zero fill (+0), its scales "
            "the scale buffers' padding bytes (the quantizers write zeros: UE8M0 0x00 is the finite 2^-127; the contract "
            "requires finite bytes), so every padded product is +0 and the step adds nothing; every configuration issues the "
            "same padded steps (the inner radix is fixed), so this corner cannot separate two configurations (R3); the "
            "fingerprint hashes the raw K so a padded and an unpadded K are distinct specifications"
        )
    return (
        f"MMA{mk}(D, 0, 0) == D: the zero-filled K remainder (whole padded steps of the last K-block, TMA zero fill of "
        "both operands) adds products that are all +0; every configuration issues the same padded steps (the inner radix "
        "is fixed), so this corner cannot separate two configurations (R3); the fingerprint hashes the raw K so a padded "
        "and an unpadded K are distinct specifications. The kernel's first K step runs the instruction without an accumulator "
        "input (D = the step's products) where the specification writes MMA(+0, ...): the two can differ only in the sign of "
        "an all-zero result (every configuration does the same, so R3 holds; the chunked class's add(+0, P) normalises it)"
    )


IDENTITY_NOTE = identity_note("bf16")


BIAS_NOTE = (
    "The biased chain is cvt.rn(add.rn(acc, bias[n])): the bias of the output column is added once, in fp32, to the fully "
    "reduced accumulator, then the single rounding to the output format. The bf16 bias is widened to binary32 exactly "
    "(every bf16 value is a binary32 value), so the widening is a bit movement, not a rounding step. The add is "
    "applied in exactly one place per class: the epilogue of the unsplit fold and of the in-CTA chunk fold, the fixup "
    "of the workspace split (its fp32 partials stay bias-free), so no configuration, chunk count or part can move it, and "
    "a row's bits never depend on M. A biased and an unbiased call are different specifications: the term carries the "
    "bias leaf and the fingerprint the bias flag; bias=None is the unbiased chain, bit for bit"
)


def bias_leaf(fp, n: int):
    """The bias of output column ``n`` as a binary32 value: a leaf of the term (an input, like the operand slices)."""
    return fp.op(BIAS_OP, n)


def kslice_leaf(m: int, n: int, in_fmt: str = "bf16"):
    """The K-slice leaf of step ``t``: the (A, B) slices of row ``m`` and column ``n``, plus the (SFA, SFB) scale-factor
    slices for the block-scaled classes (one UE8M0 scale per ``SF_VEC`` elements, consumed by the instruction itself)."""
    if in_fmt in BLOCK_SCALED:

        def leaf_bs(fp, idx):
            t = idx["kstep"]
            return fp.op(
                "absf_kslice",
                fp.op("a_rows", m, t),
                fp.op("sfa_rows", m, t),
                fp.op("b_cols", t, n),
                fp.op("sfb_cols", t, n),
            )

        return leaf_bs

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
    return Reduction(
        f"{class_name(in_fmt, False)}[K={K}]", (kstep,), kslice_leaf(m, n, in_fmt)
    )


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
    slice_leaf = kslice_leaf(m, n, in_fmt)

    def leaf(fp, idx):
        return slice_leaf(fp, {"kstep": idx["chunk"] * (Kc // mk) + idx["kstep"]})

    return Reduction(f"{class_name(in_fmt, True)}[K={K},C={C}]", (kstep, chunk), leaf)


def block_scale_term(fp, m: int, n: int, kb: int):
    """The scale of block ``kb`` for output element (m, n): ``mul.rn(s_a[m, kb], s_b[n // SCALE_BLOCK_N, kb])``, the two fp32
    block scales (leaves: inputs, like the operand slices) multiplied once with one correctly rounded multiply on the device."""
    return fp.mul(fp.op("sfa_blk", m, kb), fp.op("sfb_blk", n // SCALE_BLOCK_N, kb))


def reduction_blockscale(K: int, m: int, n: int, in_fmt: str = "e4m3_bs") -> Reduction:
    """The block-scale class ``gemm.e4m3.kblock128``: K is cut into ``K / BLOCK_K`` blocks of ``BLOCK_K`` elements (``BLOCK_K / 32``
    opaque ``mma32`` steps each; K a whole number of blocks); each block is an ``mma32`` fold from +0 (its partial ``P_kb``), and
    the partials are folded in block order from +0 with ``acc = fma.rn(block_scale_term(kb), P_kb, acc)`` (one correctly rounded
    fused multiply-add per block; the first block's ``fma(s, P_0, +0)`` is the same, so an exact ``-0`` product gives ``+0``). The
    block fold is an indexed ``fold`` digit (the scale is a per-block coefficient); the kernel's promotion pass performs exactly it."""
    assert in_fmt in FP8_BLOCKSCALE_FMTS and K % BLOCK_K == 0, (in_fmt, K)
    op, mk = MMA_OP_OF[in_fmt], MMA_K_OF[in_fmt]
    spb = BLOCK_K // mk
    kstep = Digit(
        "kstep",
        spb,
        "fold",
        init=lambda fp: fp.const(0.0),
        step=lambda fp, acc, tile: fp.op(op, acc, tile),
        identity_note="K is a whole number of blocks: no padded step",
    )
    block = Digit(
        "block",
        K // BLOCK_K,
        "fold",
        init=lambda fp: fp.const(0.0),
        step=lambda fp, acc, partial, kb: fp.fma(
            block_scale_term(fp, m, n, kb), partial, acc
        ),
        identity_note="fma(s, +0, acc) == acc for finite s: a block of zero operands adds nothing (the scales are finite inputs)",
        indexed=True,
    )
    slice_leaf = kslice_leaf(m, n, in_fmt)

    def leaf(fp, idx):
        return slice_leaf(fp, {"kstep": idx["block"] * spb + idx["kstep"]})

    return Reduction(f"{class_name(in_fmt, False)}[K={K}]", (kstep, block), leaf)


def scale_term(fp, in_fmt: str = "e4m3"):
    """``alpha = mul.rn(a_scale, b_scale)``: the two per-tensor fp8 scales are runtime scalars (``sym``) and their product
    is formed on the device with one correctly rounded multiply (never on the host, whose double product would round
    differently). For the 4-bit classes ``alpha`` is itself the input (``mm_fp4``'s 1-element tensor, by convention
    ``1 / (a_global * b_global)`` formed by the caller: how it is formed is part of the input, like the block scales), so
    the term is the bare symbol. How the scales are formed is part of the input, like the operands themselves."""
    if in_fmt in ALPHA_GIVEN:
        return fp.sym("alpha")
    return fp.mul(fp.sym("a_scale"), fp.sym("b_scale"))


@dataclass(frozen=True)
class EpilogueChain:
    """The elementwise ops after the fold, in order: ``mul.rn alpha`` for the scaled (fp8) classes, ``add.rn bias`` for
    the biased chain (the bias of the output column, widened to binary32 exactly: ``BIAS_NOTE``), then exactly one
    ``cvt.rn <out>``. One object, one ``apply``: the spec term (:func:`output_elem`, ``fp`` = the term algebra), the
    kernel (``blocks/epilogue.py``: ``fp`` = ``CuteF32`` for the ops before the conversion, the vector ``.to(out dtype)``
    for the conversion itself) and the fixup all run this code."""

    ops: Tuple[str, ...]

    @staticmethod
    def for_class(out_fmt: str, scaled: bool, bias: bool = False) -> "EpilogueChain":
        return EpilogueChain(
            (("mul.rn alpha",) if scaled else ())
            + (("add.rn bias",) if bias else ())
            + (f"cvt.rn {out_fmt}",)
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

    @property
    def biased(self) -> bool:
        return "add.rn bias" in self.pre

    def apply_pre(self, fp, x, alpha=None, bias=None):
        """``alpha`` is the scaled chain's runtime scalar, ``bias`` the biased chain's bias of this element's column (a
        binary32 value: the bf16 bias widened exactly)."""
        for op in self.pre:
            if op == "mul.rn alpha":
                assert alpha is not None, "the scaled chain needs alpha"
                x = fp.mul(x, alpha)
            elif op == "add.rn bias":
                assert bias is not None, "the biased chain needs the column's bias"
                x = fp.add(x, bias)
            else:  # a new chain op must be added here, nowhere else
                raise ValueError(f"unknown epilogue op {op!r}")
        return x

    def apply(self, fp, x, alpha=None, bias=None):
        return fp.cvt_out(self.apply_pre(fp, x, alpha, bias), self.out_fmt)

    def describe(self) -> str:
        return " -> ".join(self.ops)


def output_elem(
    fp, acc, out_fmt: str = "bf16", alpha=None, bias_n: Optional[int] = None
):
    """The epilogue chain of the class: ``cvt_rn(acc)``; with ``alpha``, ``cvt_rn(mul.rn(acc, alpha))``; with a bias
    (``bias_n`` the output column), one ``add.rn`` of the bias leaf to the folded accumulator before the conversion
    (``BIAS_NOTE``)."""
    chain = EpilogueChain.for_class(out_fmt, alpha is not None, bias_n is not None)
    bias = bias_leaf(fp, bias_n) if bias_n is not None else None
    return chain.apply(fp, acc, alpha, bias)


class GemmSpec:
    def __init__(
        self,
        K: int,
        out_fmt: str = "bf16",
        alg: Optional[TermAlgebra] = None,
        chunks: int = 1,
        in_fmt: str = "bf16",
        bias: bool = False,
    ):
        self.K, self.out_fmt, self.chunks, self.in_fmt = K, out_fmt, chunks, in_fmt
        self.scaled = is_scaled(in_fmt)
        if bias and in_fmt != "bf16":
            raise ValueError(
                f"the biased chain exists for the bf16 class only (mm_bf16's bias); no kernel implements it for {in_fmt!r}"
            )
        self.bias = bias
        if in_fmt in FP8_BLOCKSCALE_FMTS and chunks != 1:
            raise ValueError(
                f"the block-scale class {class_name(in_fmt, False)} has no split variant (chunks = {chunks})"
            )
        self.alg = alg or TermAlgebra()

    def acc(self, m: int, n: int) -> Term:
        if self.in_fmt in FP8_BLOCKSCALE_FMTS:
            return reduction_blockscale(self.K, m, n, self.in_fmt).canonical(self.alg)
        if self.chunks == 1:
            return reduction(self.K, m, n, self.in_fmt).canonical(self.alg)
        return reduction_chunked(self.K, self.chunks, m, n, self.in_fmt).canonical(
            self.alg
        )

    def out(self, m: int, n: int) -> Term:
        alpha = scale_term(self.alg, self.in_fmt) if self.scaled else None
        return output_elem(
            self.alg, self.acc(m, n), self.out_fmt, alpha, n if self.bias else None
        )


def spec_fingerprint(
    K: int,
    out_fmt: str = "bf16",
    chunks: int = 1,
    in_fmt: str = "bf16",
    bias: bool = False,
) -> str:
    """The identifier of the class for one weight: the term encodes ceil(K / step) opaque steps; the raw K (hence the
    zero-filled tail of the last step), the chunk count, the formats, the chain and the bias flag are part of the
    specification too, so they enter the identifier (``-bias`` in the tag; without a bias the identifier is the
    unbiased class's, unchanged). The biased variant exists for the bf16 class only; ``bias=True`` with another format
    raises."""
    s = GemmSpec(K, out_fmt, chunks=chunks, in_fmt=in_fmt, bias=bias)
    if in_fmt in FP8_BLOCKSCALE_FMTS:
        tag = f"gemm-{in_fmt.split('_')[0]}-kblock{BLOCK_K}-"  # gemm-e4m3-kblock128-<hash>
    else:
        tag = f"gemm-{in_fmt.replace('_', '-')}-" + (
            "kfold" if chunks == 1 else f"kchunk{chunks}"
        )  # the sm_107 MX class tags as gemm-mxfp8-k64-...
        tag += ("-scaled" if s.scaled else "") + ("-bias" if bias else "") + "-"
    parts = [f"K={K}", f"chunks={chunks}", f"out={out_fmt}"]
    if in_fmt != "bf16":
        parts += [f"in={in_fmt}", f"scaled={int(s.scaled)}"]
    parts += [fingerprint(s.out(0, 0)), fingerprint(s.out(1, 1))]
    if in_fmt in FP8_BLOCKSCALE_FMTS:
        parts.append(
            fingerprint(s.out(1, SCALE_BLOCK_N))
        )  # a column of the second weight scale block
    if bias:
        parts.append("bias=1")
    return tag + hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
