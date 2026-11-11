# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Generate ``lean/BIKernels/Cases.lean`` from the Python side.

For the GEMM classes (bf16 at the ``mma16`` step; e4m3 / e5m2 at the ``mma32`` step with the scaled chain; mxfp8 at the
``mma32mx`` step and mxfp8_k64 at the ``mma64mx8`` step, the block-scaled steps with the unscaled chain; nvfp4 / nvfp4_k128
at ``mma64bs`` / ``mma128bs`` and mxfp4 / mxfp4_k128 at ``mma64mx`` / ``mma128mx4``, the 4-bit block-scaled steps with the
alpha chain) the file carries

* the spec terms of both classes of each format at sampled (K, chunks): the *Python spec* (``gemm.spec.GemmSpec``)
  S-expression, with a Lean theorem that the Lean model (``BIK.GEMM.kfold`` / ``kchunk`` and their ``32`` / ``MX8`` /
  ``MX8K64`` instances through ``outputElem``, an independent implementation) prints the same canonical S-expression
  by kernel ``decide``;
* the mainloop's blocked lowering (``canon.BlockedFold`` as ``MainloopBlock.lowering()`` emits it and
  ``gemm/blocks/mainloop.py`` / ``blocks/blockscaled.py`` execute it): its inner radix per format and the *Python
  engine's* trace at sampled block counts, reproduced by the Lean loop model ``kblockLoop*`` node for node, plus
  ``kblockLoop*_eq_kfold*`` instantiated on the emitted radix;
* the epilogue chain per output format, unscaled (bf16, the MXFP8 classes) and scaled (fp8: ``mul.rn alpha`` then the
  conversion), and the biased chain of the bf16 classes;
* the dispatcher's chunk rules ``chunks_for(N, K, fmt)`` as tables that Lean's ``chunksFor`` / ``chunksForFp8`` /
  ``chunksForMx`` re-derive;
* the generator's configuration spaces at the reference shape and the split classes' spaces at a chunked shape (for the
  block-scaled classes also a decode shape, whose space holds the swapped token-tile records, and a second chunked
  shape with the in-cluster split's factor 2), every record checked against ``ConfigOK`` / ``ConfigOKFp8`` /
  ``ConfigOKMx8`` / ``ConfigOKMx8K64`` (the Lean mirrors of ``Knobs.well_formed``), counted, and
  ``space_invariant*`` instantiated on them; and every record the dispatchers return over a shape grid, each shown to
  be a member of one of the emitted spaces;
* for the block-scale fp8 class ``gemm.e4m3.kblock128`` (``e4m3_bs``: the ``mma32`` step folded per 128-element block, the
  block partials folded by ``fma`` with the product of the two fp32 block scales): the spec's output at (0, 0) and at
  (3, 129) per (block count, output format) reproduced by ``BIK.GEMM.outputKblock``, the kernel's promotion loop
  (``GEMMConfig.promoteLoop``) against the engine's per-block traces, its space against ``ConfigOKFp8BS``, and the
  dispatcher's picks over the DeepSeek grid as members of that space.

Consequently a change to a spec, a lowering, a knob constraint or a chunk rule that is not reflected in the
Lean model makes ``lake build`` fail; and a regenerated file that differs from the committed one fails the
device-free conformance test (``tests/batch_invariant/test_lean.py``), which needs no Lean toolchain.

    python -m flashinfer.batch_invariant.lean_cases          # rewrite lean/BIKernels/Cases.lean
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Optional, Tuple

from .fp_interface import term
from ..gemm.batch_invariant.heuristic import PARTS
from ..gemm.batch_invariant.generator import (
    REFERENCE_SHAPE,
    Knobs,
    MainloopBlock,
    enumerate_space,
)
from ..gemm.batch_invariant.schedule import chunks_for, default_config, fp4_class_for
from ..gemm.batch_invariant.spec import (
    BLOCK_K,
    MMA_K_OF,
    GemmSpec,
    reduction,
    reduction_blockscale,
)

LEAN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lean")
CASES_PATH = os.path.join(LEAN_DIR, "BIKernels", "Cases.lean")

# attention toy cases: (name, D, T, C, p, window_left, row id, d) -- T = 16 (one PV step per tile) and C = 32 keep the printed
# terms small; D = 64 / 128 / 256 are the three classes (4 / 8 / 16 MMA steps per score)
ATTN_CASES = [
    ("a0", 64, 16, 32, 40, -1, 0, 0),
    ("a1", 64, 16, 32, 40, 20, 0, 1),
    ("a2", 64, 16, 32, 7, -1, 3, 0),
    ("a3", 64, 16, 32, 33, 5, 1, 2),
    ("a4", 128, 16, 32, 40, -1, 0, 0),
    ("a5", 128, 16, 32, 33, 5, 1, 2),
    ("a6", 256, 16, 32, 40, -1, 0, 0),
    ("a7", 256, 16, 32, 33, 5, 1, 2),
    (
        "a8",
        64,
        16,
        32,
        100,
        0,
        0,
        0,
    ),  # the window starts past the first chunks: chunks 0-2 absent, key 100 alone visible
    ("a9", 64, 32, 64, 40, -1, 0, 0),  # two PV slices per tile (T = 2 x MMA_K)
]
ATTN_HEAD_DIMS = (64, 128, 256)

GEMM_CASES = [
    (64, 1),
    (128, 1),
    (256, 2),
    (256, 4),
    (512, 8),
]  # (K, chunks); 32 steps is what kernel `decide` carries
GEMM_LOOP_CASES = [
    1,
    2,
    3,
    5,
]  # block counts at which the mainloop lowering's trace is conformed node for node
GEMM_OUT_CASES = [
    (64, "bf16"),
    (128, "bf16"),
    (64, "f32"),
]  # (K, out_fmt): f32 is the split class's partial store
GEMM_BIAS_CASES = [
    (64, 1),
    (256, 2),
]  # (K, chunks): the biased chain (the bias added once before the conversion) on both classes
CHUNK_TABLE_SHAPES = [
    (0, 4096),  # an empty weight: no split (natural-number division in Lean)
    (64, 64),
    (64, 8192),
    (64, 262144),
    (256, 131072),
    (512, 65536),
    (1024, 8192),
    (4096, 4096),
    (8192, 8192),
    (8192, 262144),
    (2048, 2048),
    (128, 4096),
]
C4_SHAPE = (
    16,
    128,
    8192,
    4,
)  # (M, N, K, chunks): a split-class shape whose space has the in-CTA chunk-fold twins
DISPATCH_MS = (1, 8, 64, 512, 4096, 16384)
DISPATCH_NS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
DISPATCH_BIG = (16384, 65536, 262144)
#: The fp8 classes (one Lean step, `mma32`, for both formats: e5m2 has the same spaces, rules and chain as e4m3).
FP8_FMT = "e4m3"
GEMM_FP8_CASES = [
    (128, 1),
    (256, 1),
    (512, 2),
    (1024, 4),
]  # (K, chunks) at the 32-wide step: 4 .. 32 steps
GEMM_FP8_LOOP_CASES = [1, 2, 3, 5]
GEMM_FP8_OUT_CASES = [(128, "bf16"), (256, "bf16"), (128, "f32")]
FP8_CHUNK_EXTRA = [
    (64, 4096 + 32 * 64),
    (64, 1040 * 16),
    (8, 1040),
    (64, 8208),
]  # the 32-alignment of the fp8 chunk rule matters here
#: The MXFP8 classes: the sm_100 / sm_103 instruction (`mma32mx`, format key ``mxfp8``) and the sm_107 one (`mma64mx8`,
#: ``mxfp8_k64``); one chunk rule (128-element granules) for both.
MX_FMT, MX_K64_FMT = "mxfp8", "mxfp8_k64"
GEMM_MX_CASES = [
    (128, 1),
    (256, 1),
    (512, 2),
    (512, 4),
]  # (K, chunks): 4 .. 16 steps at the 32-wide step (the block-scaled leaf is long: 32 steps overflow kernel `decide`)
GEMM_MX_LOOP_CASES = [1, 2, 3, 5]
GEMM_MX_OUT_CASES = [(128, "bf16"), (256, "f16"), (128, "f32")]
MX_CHUNK_EXTRA = [
    (64, 16640),  # 130 granules: C = 2 where the fp8 rule gives 8
    (64, 8256),  # 64.5 granules: no split
    (64, 4224),  # 33 granules: no split
    (8, 1040),
]  # the 128-alignment of the MX chunk rule matters here
#: A decode shape: its block-scaled space holds every swapped token-tile record (M at most the narrowest token tile).
DECODE_SHAPE = (8, 4096, 4096, 1)
#: A split-class shape with two chunks: its block-scaled space holds the in-cluster split records of factor 2.
C2_SHAPE = (16, 128, 4096, 2)
#: The 4-bit classes: NVFP4 (``nvfp4``: the sm_100 / sm_103 instruction, 64 elements per step; ``nvfp4_k128``: the sm_107 one)
#: and MXFP4 (``mxfp4`` / ``mxfp4_k128``), the caller's alpha as the chain's scalar; the K = 64 cases at 4 .. 16 steps.
GEMM_FP4_CASES = [(256, 1), (512, 1), (1024, 2), (1024, 4)]
GEMM_FP4_K128_CASES = [(512, 1), (1024, 1), (2048, 2), (2048, 4)]
GEMM_FP4_OUT_CASES = [(256, "bf16"), (512, "f16"), (256, "f32")]
GEMM_FP4_K128_OUT_CASES = [(512, "bf16"), (1024, "f16"), (512, "f32")]
#: The block-scale fp8 class (``e4m3_bs``, ``gemm.e4m3.kblock128``): (K, out_fmt) at 1 .. 4 blocks of ``BLOCK_K`` elements; the
#: block counts at which the kernel's promotion loop (Lean ``promoteLoop``) is conformed to the engine's per-block traces; the
#: DeepSeek (N, K) grid and the M values over which the dispatcher's picks are shown to be members of the space.
FP8BS_FMT = "e4m3_bs"
GEMM_FP8BS_OUT_CASES = [(128, "bf16"), (256, "bf16"), (384, "f16"), (512, "f32")]
GEMM_FP8BS_LOOP_CASES = [1, 2, 3]
GEMM_FP8BS_PAIRS = (
    (7168, 7168),
    (7168, 2048),
    (2048, 7168),
    (4096, 7168),
    (18432, 7168),
    (7168, 18432),
    (576, 7168),
    (7168, 16384),
)
GEMM_FP8BS_MS = (1, 16, 128, 1024, 4096, 16384)


@dataclasses.dataclass(frozen=True)
class LeanFormat:
    """The Lean side of one input format: the name infix ``p`` of its step's definitions (``kfold{p}``, ``kchunk{p}``,
    ``kblockLoop{p}``, ``tileResult{p}``, ``space_invariant{p}``, ``outputKfold{p}``), the prefixes of its generated
    cases (``tag``: also the conformance line tag) and theorems (``th``), its knob-constraint mirror (``ok``), its chunk
    rule and table (``table`` is None when the rule is rendered under another format), the weight layouts the kernel
    takes, the parts that run the class (None: every part) and the extra spaces of the block-scaled classes
    (``(suffix, shape, count theorem, count predicate, what the shape's space holds)``)."""

    in_fmt: str
    p: str
    tag: str
    th: str
    sp: str
    dn: str
    ok: str
    inner_name: str
    rule: str
    table: Optional[str]
    chunk_tag: str
    scaled: bool
    layouts: Tuple[str, ...]
    families: Optional[Tuple[str, ...]]
    cases: Tuple[Tuple[int, int], ...]
    loop_cases: Tuple[int, ...]
    out_cases: Tuple[Tuple[int, str], ...]
    chunk_extra: Tuple[Tuple[int, int], ...]
    extra_spaces: Tuple[Tuple[str, Tuple[int, int, int, int], str, str, str], ...] = ()
    class_rule: Optional[str] = (
        None  # the Lean class rule the format's shape table checks (the sm_107 NVFP4 class)
    )

    @property
    def mk(self) -> int:
        return MMA_K_OF[self.in_fmt]

    @property
    def step(self) -> str:
        return STEP_OF_P[self.p]

    @property
    def elem(self) -> str:
        """The Lean chain applied to a fold: ``outputElem false`` (bf16, the MXFP8 classes), ``outputElem true`` (the fp8
        classes) or ``outputElemAlpha`` (the 4-bit classes)."""
        if self.p in ("BS", "BS128", "MX4", "MX4K128"):
            return "outputElemAlpha"
        return f"outputElem {'true' if self.scaled else 'false'}"

    def chain_head(self, fmt: str) -> str:
        """``outputKfold{p} "<fmt>"``: the class's chain (the bf16 definition takes the ``scaled`` flag)."""
        if self.p == "":
            return f'outputKfold false "{fmt}"'
        return f'outputKfold{self.p} "{fmt}"'

    def fold_term(self, steps: int, C: int) -> str:
        if C == 1:
            return f"kfold{self.p} ⟨0⟩ ⟨0⟩ {steps}"
        return f"kchunk{self.p} ⟨0⟩ ⟨0⟩ {C} {steps // C}"

    @property
    def decided(self) -> Tuple[str, ...]:
        """The format's theorems closed by ``native_decide``."""
        names = [
            f"{self.th}_space_ok",
            f"{self.th}_space_c4_ok",
            f"{self.th}_space_c4_cta_twins",
            f"{self.th}_dispatch_in_space",
        ]
        names += [f"{self.th}_space_{sfx.lower()}_ok" for sfx, *_ in self.extra_spaces]
        return tuple(names)


STEP_OF_P = {
    "": "mma16",
    "32": "mma32",
    "MX8": "mma32mx",
    "MX8K64": "mma64mx8",
    "BS": "mma64bs",
    "BS128": "mma128bs",
    "MX4": "mma64mx",
    "MX4K128": "mma128mx4",
}
CHUNK_SUFFIX = {"": "", "32": "_fp8", "MX8": "_mx", "BS": "_fp4", "BS128": "_fp4k128"}

_MX_EXTRA = (
    (
        "Decode",
        DECODE_SHAPE,
        "swapped",
        "c.swapAb",
        "the swapped token-tile records (`swapAb`: the decode specialisation, one token tile for all tokens)",
    ),
    (
        "C2",
        C2_SHAPE,
        "cluster",
        "c.clusterK == 2",
        "the in-cluster split records of factor 2 (`clusterK`) next to the dense records and the in-CTA chunk-fold twins",
    ),
)

FORMATS: Tuple[LeanFormat, ...] = (
    LeanFormat(
        "bf16",
        "",
        "gemm",
        "gemm",
        "gemmSpace",
        "gemmDispatch",
        "ConfigOK",
        "gemmMainloopInner",
        "chunksFor",
        "chunkTable",
        "chunks",
        False,
        ("n", "k"),
        None,
        tuple(GEMM_CASES),
        tuple(GEMM_LOOP_CASES),
        tuple(GEMM_OUT_CASES),
        (),
    ),
    LeanFormat(
        FP8_FMT,
        "32",
        "gemm32",
        "gemm_fp8",
        "gemmFp8Space",
        "gemmFp8Dispatch",
        "ConfigOKFp8",
        "gemmMainloopInnerFp8",
        "chunksForFp8",
        "chunkTableFp8",
        "chunks32",
        True,
        ("n", "k"),
        None,
        tuple(GEMM_FP8_CASES),
        tuple(GEMM_FP8_LOOP_CASES),
        tuple(GEMM_FP8_OUT_CASES),
        tuple(FP8_CHUNK_EXTRA),
    ),
    LeanFormat(
        MX_FMT,
        "MX8",
        "gemmmx8",
        "gemm_mx8",
        "gemmMx8Space",
        "gemmMx8Dispatch",
        "ConfigOKMx8",
        "gemmMainloopInnerMx8",
        "chunksForMx",
        "chunkTableMx",
        "chunksmx",
        False,
        ("k",),
        ("sm_100", "sm_103"),
        tuple(GEMM_MX_CASES),
        tuple(GEMM_MX_LOOP_CASES),
        tuple(GEMM_MX_OUT_CASES),
        tuple(MX_CHUNK_EXTRA),
        _MX_EXTRA,
    ),
    LeanFormat(
        MX_K64_FMT,
        "MX8K64",
        "gemmmx8k64",
        "gemm_mx8k64",
        "gemmMx8K64Space",
        "gemmMx8K64Dispatch",
        "ConfigOKMx8K64",
        "gemmMainloopInnerMx8K64",
        "chunksForMx",
        None,
        "chunksmx",
        False,
        ("k",),
        ("sm_107",),
        tuple(GEMM_MX_CASES),
        tuple(GEMM_MX_LOOP_CASES),
        tuple(GEMM_MX_OUT_CASES),
        tuple(MX_CHUNK_EXTRA),
        _MX_EXTRA,
    ),
    LeanFormat(
        "nvfp4",
        "BS",
        "gemmbs",
        "gemm_nvfp4",
        "gemmNvfp4Space",
        "gemmNvfp4Dispatch",
        "ConfigOKNvfp4",
        "gemmMainloopInnerNvfp4",
        "chunksForFp4",
        "chunkTableFp4",
        "chunksfp4",
        True,
        ("k",),
        None,
        tuple(GEMM_FP4_CASES),
        tuple(GEMM_MX_LOOP_CASES),
        tuple(GEMM_FP4_OUT_CASES),
        tuple(MX_CHUNK_EXTRA),
        _MX_EXTRA,
    ),
    LeanFormat(
        "nvfp4_k128",
        "BS128",
        "gemmbs128",
        "gemm_nvfp4k128",
        "gemmNvfp4K128Space",
        "gemmNvfp4K128Dispatch",
        "ConfigOKNvfp4K128",
        "gemmMainloopInnerNvfp4K128",
        "chunksForFp4K128",
        "chunkTableFp4K128",
        "chunksfp4k128",
        True,
        ("k",),
        ("sm_107",),
        tuple(GEMM_FP4_K128_CASES),
        tuple(GEMM_MX_LOOP_CASES),
        tuple(GEMM_FP4_K128_OUT_CASES),
        tuple(MX_CHUNK_EXTRA),
        _MX_EXTRA,
        "fp4ClassK128",
    ),
    LeanFormat(
        "mxfp4",
        "MX4",
        "gemmmx4",
        "gemm_mx4",
        "gemmMx4Space",
        "gemmMx4Dispatch",
        "ConfigOKMx4",
        "gemmMainloopInnerMx4",
        "chunksForMx",
        None,
        "chunksmx",
        True,
        ("k",),
        ("sm_100", "sm_103"),
        tuple(GEMM_FP4_CASES),
        tuple(GEMM_MX_LOOP_CASES),
        tuple(GEMM_FP4_OUT_CASES),
        tuple(MX_CHUNK_EXTRA),
        _MX_EXTRA,
    ),
    LeanFormat(
        "mxfp4_k128",
        "MX4K128",
        "gemmmx4k128",
        "gemm_mx4k128",
        "gemmMx4K128Space",
        "gemmMx4K128Dispatch",
        "ConfigOKMx4K128",
        "gemmMainloopInnerMx4K128",
        "chunksForMx",
        None,
        "chunksmx",
        True,
        ("k",),
        ("sm_107",),
        tuple(GEMM_FP4_K128_CASES),
        tuple(GEMM_MX_LOOP_CASES),
        tuple(GEMM_FP4_K128_OUT_CASES),
        tuple(MX_CHUNK_EXTRA),
        _MX_EXTRA,
    ),
)
BY_FMT: Dict[str, LeanFormat] = {F.in_fmt: F for F in FORMATS}
#: The generated theorems closed by ``native_decide`` (the exhaustive checks over the configuration spaces); every
#: other case is kernel ``decide``. ``lean/Axioms.lean`` pins exactly these names.
DECIDED_THEOREMS = (
    tuple(n for F in FORMATS for n in F.decided)
    + ("gemm_fp8bs_space_ok", "gemm_fp8bs_dispatch_in_space")
    + tuple(f"attn_{n}_spec" for n, *_ in ATTN_CASES)
    + tuple(
        f"attn_space_d{D}_{what}" for D in ATTN_HEAD_DIMS for what in ("size", "ok")
    )
)
#: `native_decide` theorems of the hand-written modules (not of the generated `Cases.lean`): the attention class-separation
#: statements on toy rows; listed in `lean/Axioms.lean` with the generated ones.
MODULE_DECIDED_THEOREMS = (
    "Attn.output_depends_on_C",
    "Attn.output_depends_on_D",
    "Attn.output_depends_on_D_64",
)


def lean_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def spec_cases(in_fmt: str = "bf16") -> List[Tuple[int, int, str]]:
    """``(K, chunks, sexpr)``: the Python spec's output element (0, 0) of each class of the format."""
    return [
        (K, C, term.sexpr(GemmSpec(K, chunks=C, in_fmt=in_fmt).out(0, 0)))
        for K, C in BY_FMT[in_fmt].cases
    ]


def loop_cases(in_fmt: str = "bf16") -> Tuple[int, List[Tuple[int, str]]]:
    """The lowering's inner radix and ``(nblk, sexpr)``: the engine's trace of the blocked fold at ``nblk`` blocks."""
    low = MainloopBlock(in_fmt).lowering()
    alg = term.TermAlgebra()
    return low.inner, [
        (
            nblk,
            term.sexpr(
                low.trace(
                    reduction(nblk * low.inner * MMA_K_OF[in_fmt], 0, 0, in_fmt),
                    alg,
                    nblk,
                )
            ),
        )
        for nblk in BY_FMT[in_fmt].loop_cases
    ]


def out_cases(in_fmt: str = "bf16") -> List[Tuple[int, str, str]]:
    return [
        (K, fmt, term.sexpr(GemmSpec(K, fmt, in_fmt=in_fmt).out(0, 0)))
        for K, fmt in BY_FMT[in_fmt].out_cases
    ]


def bias_cases() -> List[Tuple[int, int, str]]:
    """``(K, chunks, sexpr)``: the Python spec's biased output element (0, 0) of each class."""
    return [
        (K, C, term.sexpr(GemmSpec(K, chunks=C, bias=True).out(0, 0)))
        for K, C in GEMM_BIAS_CASES
    ]


def attn_cases() -> List[Tuple[str, int, int, int, int, int, int, int, str]]:
    """The attention toy cases with the Python spec's S-expression of the output element."""
    from ..attention.batch_invariant.spec import AttentionSpec, RowSpec
    from .fp_interface.term import sexpr

    out = []
    for name, D, T, C, p, win, rid, d in ATTN_CASES:
        sx = sexpr(
            AttentionSpec(D, T=T, C=C).output(RowSpec(p=p, window_left=win, row=rid), d)
        )
        out.append((name, D, T, C, p, win, rid, d, sx))
    return out


def attn_spaces() -> Dict[int, list]:
    """The attention kernel's configuration space per head dim (`engine.all_valid_configs`)."""
    from ..attention.batch_invariant.engine import all_valid_configs

    spaces = {D: all_valid_configs(D) for D in ATTN_HEAD_DIMS}
    for D, cfgs in spaces.items():
        assert all(c.well_formed(D) for c in cfgs), D
    return spaces


def chunk_table(in_fmt: str = "bf16") -> List[Tuple[int, int, int]]:
    shapes = CHUNK_TABLE_SHAPES + list(BY_FMT[in_fmt].chunk_extra)
    return [(N, K, chunks_for(N, K, in_fmt)) for N, K in shapes]


def class_table() -> List[Tuple[int, int, bool]]:
    """``(N, K, k128)``: whether the sm_107 parts run the NVFP4 weight shape at the sm_107 instruction's class
    (`schedule.fp4_class_for`), over the chunk-table shapes; Lean's ``fp4ClassK128`` re-derives it."""
    shapes = CHUNK_TABLE_SHAPES + list(MX_CHUNK_EXTRA)
    return [(N, K, fp4_class_for(N, K, "sm_107") == "nvfp4_k128") for N, K in shapes]


def spaces(in_fmt: str = "bf16") -> Tuple[List[Knobs], List[Knobs]]:
    """The generator's space at the reference shape and the split class's space at ``C4_SHAPE``: the bf16 spaces in the
    ``n`` layout (its space is layout-free), the fp8 spaces in the ``k`` layout (the whole knob product: the ``n`` layout
    drops the 2-CTA 16-wide records, `operands.can_implement`), the block-scaled spaces in the ``k`` layout (the only one
    the instruction takes)."""
    M, N, K, C = C4_SHAPE
    lay = "n" if in_fmt == "bf16" else "k"
    return (
        enumerate_space(*REFERENCE_SHAPE, b_major=lay, in_fmt=in_fmt),
        enumerate_space(M, N, K, chunks=C, b_major=lay, in_fmt=in_fmt),
    )


def extra_spaces(
    in_fmt: str,
) -> List[Tuple[str, Tuple[int, int, int, int], str, str, str, List[Knobs]]]:
    """The block-scaled classes' further spaces (``LeanFormat.extra_spaces``) with their records."""
    F = BY_FMT[in_fmt]
    return [
        (
            sfx,
            shape,
            cname,
            pred,
            doc,
            enumerate_space(
                shape[0],
                shape[1],
                shape[2],
                chunks=shape[3],
                b_major="k",
                in_fmt=in_fmt,
            ),
        )
        for sfx, shape, cname, pred, doc in F.extra_spaces
    ]


def dispatch_cases(in_fmt: str = "bf16") -> Dict[Knobs, Tuple[int, int, int]]:
    """Every record the dispatcher returns over the grid (the format's weight layouts, every part of `PARTS` that runs the
    class) at the parts' nominal numbers (``nominal=True``: a device's queried capacity or L2 size never enters the
    rendering), with the first shape that selects it."""
    F = BY_FMT[in_fmt]
    disp: Dict[Knobs, Tuple[int, int, int]] = {}
    pairs = [(N, K) for N in DISPATCH_NS for K in DISPATCH_NS]
    pairs += [(N, K) for K in DISPATCH_BIG for N in DISPATCH_NS[::3]]
    pairs += [(N, K) for N in DISPATCH_BIG for K in DISPATCH_NS[::3]]
    for N, K in pairs:
        C = chunks_for(N, K, in_fmt)
        for M in DISPATCH_MS:
            for b_major in F.layouts:
                for family, sm in PARTS:
                    if F.families is not None and family not in F.families:
                        continue
                    k = dataclasses.replace(
                        default_config(
                            M, N, K, C, b_major, sm, family, in_fmt, nominal=True
                        ),
                        grid_clusters=0,
                    )
                    disp.setdefault(k, (M, N, K))
    return disp


def fp8bs_cases() -> List[Tuple[int, str, str, str]]:
    """``(KB, out_fmt, sexpr at (0, 0), sexpr at (3, 129))``: the block-scale class's output terms
    (`GemmSpec(K, fmt, in_fmt="e4m3_bs")`); the element (3, 129) reads the second weight scale block."""
    out = []
    for K, fmt in GEMM_FP8BS_OUT_CASES:
        s = GemmSpec(K, fmt, in_fmt=FP8BS_FMT)
        out.append(
            (K // BLOCK_K, fmt, term.sexpr(s.out(0, 0)), term.sexpr(s.out(3, 129)))
        )
    return out


def fp8bs_loop_cases() -> List[Tuple[int, str]]:
    """``(KB, sexpr)``: the kernel model of the block-scale class on the Python side -- the engine's one-stage trace of the
    dense lowering per block (`MainloopBlock("e4m3_bs").lowering()`, what the MMA warp issues) folded by the block digit's
    indexed step in block order (what the epilogue warps' promotion performs); Lean's `promoteLoop` prints it."""
    low = MainloopBlock(FP8BS_FMT).lowering()
    alg = term.TermAlgebra()
    out = []
    for KB in GEMM_FP8BS_LOOP_CASES:
        red = reduction_blockscale(KB * BLOCK_K, 0, 0)
        partials = [low.trace(red, alg, 1, fixed={"block": kb}) for kb in range(KB)]
        out.append((KB, term.sexpr(red.digits[1].reduce(alg, partials))))
    return out


def fp8bs_space() -> List[Knobs]:
    """The block-scale class's space at the reference shape (the knob product the body accepts in the K-major layout, the
    only one the class takes)."""
    return enumerate_space(*REFERENCE_SHAPE, b_major="k", in_fmt=FP8BS_FMT)


def fp8bs_dispatch_cases() -> Dict[Knobs, Tuple[int, int, int]]:
    """Every record the block-scale dispatcher returns over the DeepSeek grid at every part of `PARTS` at its nominal
    numbers, with the first shape that selects it."""
    disp: Dict[Knobs, Tuple[int, int, int]] = {}
    for N, K in GEMM_FP8BS_PAIRS:
        for M in GEMM_FP8BS_MS:
            for family, sm in PARTS:
                k = dataclasses.replace(
                    default_config(
                        M, N, K, 1, "k", sm, family, FP8BS_FMT, nominal=True
                    ),
                    grid_clusters=0,
                )
                disp.setdefault(k, (M, N, K))
    return disp


def _deep(L: List[str], n: int) -> None:
    """A list longer than the file's default recursion depth needs the deeper one for kernel ``decide`` on it."""
    if n > 500:
        L.append("set_option maxRecDepth 16384 in")


def _render_format(L: List[str], F: LeanFormat) -> None:
    """The cases of one format."""
    in_fmt, p, tag, th, sp, ok = F.in_fmt, F.p, F.tag, F.th, F.sp, F.ok
    kfold, loop = f"kfold{p}", f"kblockLoop{p}"
    mk = F.mk
    step = F.step
    chain_doc = (
        ", the alpha chain"
        if F.elem == "outputElemAlpha"
        else (", the scaled chain" if F.scaled else "")
    )
    L.append(
        f"/-! The spec terms of the two {in_fmt} classes (`gemm/spec.py` vs `GEMM.lean`, the `{step}` step"
        + chain_doc
        + "): the Python spec's output element, reproduced by the Lean model node for node. -/"
    )
    for i, (K, C, sx) in enumerate(spec_cases(in_fmt)):
        L.append(
            f'theorem {tag}{i}_spec : ({F.elem} "bf16" ({F.fold_term(K // mk, C)})).sexpr = {lean_str(sx)} := by decide  -- K={K} C={C}'
        )
    L.append(
        f"def {tag}Cases : List (Nat × Nat) := ["
        + ", ".join(f"({K // mk}, {C})" for K, C in F.cases)
        + "]"
    )
    L.append("")
    inner, loops = loop_cases(in_fmt)
    inner_name = F.inner_name
    L.append(
        f'/-! The {in_fmt} mainloop\'s blocked lowering (`canon.BlockedFold`, `MainloopBlock("{in_fmt}").lowering()`, executed by '
        f"`gemm/blocks/mainloop.py`): inner radix {inner}. `{loop}` on it equals `{kfold}` for every block count and "
        f"stage count, and the Python trace of the lowering is reproduced node for node at sampled block counts. -/"
    )
    L.append(f"def {inner_name} : Nat := {inner}")
    L.append(
        f"theorem {tag}_mainloop_eq_kfold : ∀ (S nblk : Nat) (a : RowData) (b : ColData), "
        f"{loop} a b {inner_name} S .zero nblk = {kfold} a b (nblk * {inner_name}) := "
        f"fun S nblk a b => {loop}_eq_{kfold} a b {inner_name} S nblk"
    )
    for i, (nblk, sx) in enumerate(loops):
        L.append(
            f"theorem {tag}loop{i} : ({loop} ⟨0⟩ ⟨0⟩ {inner_name} 3 .zero {nblk}).sexpr = {lean_str(sx)} := by decide"
        )
    L.append(
        f"def {tag}LoopCases : List Nat := ["
        + ", ".join(str(n) for n in F.loop_cases)
        + "]"
    )
    L.append("")
    L.append(
        f"/-! The {in_fmt} epilogue chain (`spec.output_elem`"
        + (
            ", `mul.rn alpha` (the given scalar) then the conversion"
            if F.elem == "outputElemAlpha"
            else (", `mul.rn alpha` then the conversion" if F.scaled else "")
        )
        + f"): `outputKfold{p}` against the Python spec's output term per output format. -/"
    )
    for i, (K, fmt, sx) in enumerate(out_cases(in_fmt)):
        L.append(
            f"theorem {tag}out{i} : ({F.chain_head(fmt)} ⟨0⟩ ⟨0⟩ {K // mk}).sexpr = {lean_str(sx)} := by decide"
        )
    L.append(
        f"def {tag}OutCases : List (Nat × String) := ["
        + ", ".join(f'({K // mk}, "{f}")' for K, f in F.out_cases)
        + "]"
    )
    L.append("")
    if F.table is not None:
        L.append(
            f'/-- `chunks_for(N, K, "{in_fmt}")` as the Python dispatcher computes it; Lean re-derives it. -/'
        )
        L.append(
            f"def {F.table} : List (Nat × Nat × Nat) := ["
            + ", ".join(f"({N}, {K}, {C})" for N, K, C in chunk_table(in_fmt))
            + "]"
        )
        suffix = CHUNK_SUFFIX[p]
        L.append(
            f"theorem chunk_rule_matches{suffix} : {F.table}.all (fun ⟨N, K, C⟩ => {F.rule} N K == C) = true := by decide"
        )
        L.append("")
    if F.class_rule is not None:
        L.append(
            f"/-- The sm_107 parts' NVFP4 class per weight shape (`schedule.fp4_class_for`: true = the sm_107 instruction's class); "
            f"Lean's `{F.class_rule}` re-derives it. -/"
        )
        L.append(
            "def fp4ClassTable : List (Nat × Nat × Bool) := ["
            + ", ".join(
                f"({N}, {K}, {'true' if b else 'false'})" for N, K, b in class_table()
            )
            + "]"
        )
        L.append(
            f"theorem fp4_class_rule_matches : fp4ClassTable.all (fun ⟨N, K, b⟩ => {F.class_rule} N K == b) = true := by decide"
        )
        L.append("")
    space, spacec = spaces(in_fmt)
    inv = f"space_invariant{p}"
    tile = f"tileResult{p}"
    lay = "n" if in_fmt == "bf16" else "k"
    L.append(
        f"/-! The generator's {in_fmt} configuration space at the reference shape {REFERENCE_SHAPE} "
        f"(`gemm.generator.enumerate_space`, the `{lay}` weight layout): every knob record, checked against the kernel's "
        f"static constraints (`{ok}`), counted, and the `config_invariance` theorem instantiated on it. -/"
    )
    L.append(f"def {sp} : List Config := [" + ", ".join(k.lean() for k in space) + "]")
    _deep(L, len(space))
    L.append(f"theorem {th}_space_size : {sp}.length = {len(space)} := by decide")
    L.append(f"theorem {th}_space_ok : {sp}.all {ok} = true := by native_decide")
    L.append(
        f"theorem {th}_space_invariant : ∀ c₁ ∈ {sp}, ∀ c₂ ∈ {sp}, ∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat), "
        f"{tile} c₁ S₁ p a b kb nblk = {tile} c₂ S₂ q a b kb nblk := {inv} {sp}"
    )
    M4, N4, K4, C4 = C4_SHAPE
    n_cta = sum(k.cta_chunks for k in spacec)
    twins = [
        t
        for t in (dataclasses.replace(k, cta_chunks=True) for k in space)
        if t.well_formed(in_fmt)
    ]
    assert [k for k in spacec if k.cta_chunks] == twins, (
        "the in-CTA chunk-fold records are not the well-formed twins of the dense space"
    )
    spc = f"{sp}C4"
    L.append(
        f"/-! The {in_fmt} split class's space at ({M4}, {N4}, {K4}) with {C4} chunks: the dense records the shape admits plus the "
        f"in-CTA chunk-fold twin (`ctaChunks`) of every record of the reference space that satisfies `{ok}` with the fold on"
        + (
            " (the block-scaled class: the TMEM budget of the running sum, no B-reuse tile), and the in-cluster split records of factor 4 (`clusterK`)"
            if F.extra_spaces
            else ""
        )
        + ". -/"
    )
    L.append(
        f"def {spc} : List Config := [" + ", ".join(k.lean() for k in spacec) + "]"
    )
    _deep(L, len(spacec))
    L.append(f"theorem {th}_space_c4_size : {spc}.length = {len(spacec)} := by decide")
    L.append(f"theorem {th}_space_c4_ok : {spc}.all {ok} = true := by native_decide")
    _deep(L, len(spacec))
    L.append(
        f"theorem {th}_space_c4_cta : ({spc}.filter (fun c => c.ctaChunks)).length = {n_cta} := by decide"
    )
    L.append(
        f"theorem {th}_space_c4_cta_twins : {spc}.filter (fun c => c.ctaChunks) = ({sp}.map (fun c => {{ c with ctaChunks := true }})).filter {ok} := by native_decide"
    )
    if F.extra_spaces:
        n_ck = sum(k.cluster_k == C4 for k in spacec)
        _deep(L, len(spacec))
        L.append(
            f"theorem {th}_space_c4_cluster : ({spc}.filter (fun c => c.clusterK == {C4})).length = {n_ck} := by decide"
        )
    L.append(
        f"theorem {th}_space_c4_invariant : ∀ c₁ ∈ {spc}, ∀ c₂ ∈ {spc}, ∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat), "
        f"{tile} c₁ S₁ p a b kb nblk = {tile} c₂ S₂ q a b kb nblk := {inv} {spc}"
    )
    L.append("")
    members = [sp, spc]
    for sfx, shape, cname, pred, doc, recs in extra_spaces(in_fmt):
        name = f"{sp}{sfx}"
        low = sfx.lower()
        n_pred = sum(
            (k.swap_ab if cname == "swapped" else k.cluster_k == shape[3]) for k in recs
        )
        L.append(
            f"/-! The {in_fmt} space at {shape[:3]}"
            + (f" with {shape[3]} chunks" if shape[3] > 1 else "")
            + f": {doc}. -/"
        )
        L.append(
            f"def {name} : List Config := [" + ", ".join(k.lean() for k in recs) + "]"
        )
        _deep(L, len(recs))
        L.append(
            f"theorem {th}_space_{low}_size : {name}.length = {len(recs)} := by decide"
        )
        L.append(
            f"theorem {th}_space_{low}_ok : {name}.all {ok} = true := by native_decide"
        )
        _deep(L, len(recs))
        L.append(
            f"theorem {th}_space_{low}_{cname} : ({name}.filter (fun c => {pred})).length = {n_pred} := by decide"
        )
        L.append(
            f"theorem {th}_space_{low}_invariant : ∀ c₁ ∈ {name}, ∀ c₂ ∈ {name}, ∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (kb nblk : Nat), "
            f"{tile} c₁ S₁ p a b kb nblk = {tile} c₂ S₂ q a b kb nblk := {inv} {name}"
        )
        L.append("")
        members.append(name)
    disp = dispatch_cases(in_fmt)
    parts = (
        "every part"
        if F.families is None
        else "the " + " / ".join(F.families) + " parts"
    )
    L.append(
        f"/-! Every knob record the {in_fmt} dispatcher returns over the shape grid ({'both weight layouts' if len(F.layouts) == 2 else 'the K-major weight layout'}; "
        f"{parts} of the nominal table Rubin NVL8 / NVL72, B200 / GB200, B300: the table's co-resident capacities and L2 sizes, "
        "never a device's queried numbers), with the first shape that selects it; each is a member of one of the enumerated "
        "spaces above (R3 membership). -/"
    )
    L.append(
        f"def {F.dn} : List (Nat × Nat × Nat × Config) := ["
        + ", ".join(f"({M}, {N}, {K}, {k.lean()})" for k, (M, N, K) in disp.items())
        + "]"
    )
    L.append(
        f"theorem {th}_dispatch_in_space : {F.dn}.all (fun ⟨_, _, _, c⟩ => "
        + " || ".join(f"{m}.any (fun s => s == c)" for m in members)
        + ") = true := by native_decide"
    )
    L.append("")


def _render_bias(L: List[str]) -> None:
    """The biased chain of the bf16 classes: `outputElemBias` against the Python spec's biased output term."""
    mk = MMA_K_OF["bf16"]
    L.append(
        "/-! The biased chain (`spec.output_elem` with a bias column): `outputElemBias` against the Python spec's biased "
        "output term, on both bf16 classes: the bias leaf added once to the whole fold, then the conversion. -/"
    )
    for i, (K, C, sx) in enumerate(bias_cases()):
        steps = K // mk
        t = f"kfold ⟨0⟩ ⟨0⟩ {steps}" if C == 1 else f"kchunk ⟨0⟩ ⟨0⟩ {C} {steps // C}"
        L.append(
            f'theorem gemmbias{i}_spec : (outputElemBias "bf16" ({t}) ⟨0⟩).sexpr = {lean_str(sx)} := by decide  -- K={K} C={C}'
        )
    L.append(
        "def gemmBiasCases : List (Nat × Nat) := ["
        + ", ".join(f"({K // mk}, {C})" for K, C in GEMM_BIAS_CASES)
        + "]"
    )
    L.append("")


def _render_fp8_blockscale(L: List[str]) -> None:
    """The block-scale fp8 class: `outputKblock` against the Python spec at (0, 0) and at (3, 129), `promoteLoop` against the
    engine's assembled per-block traces, the space against `ConfigOKFp8BS` with `space_invariantKB` instantiated on it, and
    the dispatcher's picks over the DeepSeek grid as members of the space."""
    L.append(
        "/-! The block-scale fp8 class `gemm.e4m3.kblock128` (`gemm/spec.py: reduction_blockscale` vs `GEMM.lean: kblock128`: the "
        "`mma32` step folded per 128-element block, the block partials folded by `fma` with the block scales' product, the "
        "conversion alone): the Python spec's output element at (0, 0) and at (3, 129) -- the latter reads the second weight "
        "scale block -- reproduced by the Lean model node for node. -/"
    )
    for i, (KB, fmt, sx0, sx1) in enumerate(fp8bs_cases()):
        L.append(
            f'theorem gemmfp8bs{i}_spec : (outputKblock "{fmt}" ⟨0⟩ ⟨0⟩ {KB}).sexpr = {lean_str(sx0)} := by decide  -- K={KB * BLOCK_K}'
        )
        L.append(
            f'theorem gemmfp8bs{i}_spec2 : (outputKblock "{fmt}" ⟨3⟩ ⟨129⟩ {KB}).sexpr = {lean_str(sx1)} := by decide'
        )
    L.append(
        "def gemmFp8BsOutCases : List (Nat × String) := ["
        + ", ".join(f'({K // BLOCK_K}, "{f}")' for K, f in GEMM_FP8BS_OUT_CASES)
        + "]"
    )
    L.append("")
    L.append(
        "/-! The kernel's promotion loop (`GEMMConfig.promoteLoop`: the dense lowering's one-stage fold per block, the partials "
        "folded by `fma` in block order) equals `kblock128` for every stage count and block count, and the engine's per-block "
        "traces assembled by the block digit are reproduced node for node at sampled block counts. -/"
    )
    L.append(
        "theorem gemm_fp8bs_loop_eq_class : ∀ (S KB : Nat) (a : RowData) (b : ColData), promoteLoop a b S KB = kblock128 a b KB := "
        "fun S KB a b => promoteLoop_eq_kblock128 a b S KB"
    )
    for i, (KB, sx) in enumerate(fp8bs_loop_cases()):
        L.append(
            f"theorem gemmfp8bsloop{i} : (promoteLoop ⟨0⟩ ⟨0⟩ 3 {KB}).sexpr = {lean_str(sx)} := by decide"
        )
    L.append(
        "def gemmFp8BsLoopCases : List Nat := ["
        + ", ".join(str(n) for n in GEMM_FP8BS_LOOP_CASES)
        + "]"
    )
    L.append("")
    space = fp8bs_space()
    L.append(
        f"/-! The block-scale class's configuration space at the reference shape {REFERENCE_SHAPE} "
        f'(`gemm.generator.enumerate_space(..., in_fmt="{FP8BS_FMT}")`, the `k` weight layout: 128-row CTA tiles, tile N 16 .. 128 '
        "dividing the weight scale block, no in-cluster split, swapped or in-CTA chunk-fold record): every knob record, checked "
        "against the kernel's static constraints (`ConfigOKFp8BS`), counted, and the `config_invarianceKB` theorem instantiated on it. -/"
    )
    L.append(
        "def gemmFp8BsSpace : List Config := ["
        + ", ".join(k.lean() for k in space)
        + "]"
    )
    _deep(L, len(space))
    L.append(
        f"theorem gemm_fp8bs_space_size : gemmFp8BsSpace.length = {len(space)} := by decide"
    )
    L.append(
        "theorem gemm_fp8bs_space_ok : gemmFp8BsSpace.all ConfigOKFp8BS = true := by native_decide"
    )
    L.append(
        "theorem gemm_fp8bs_space_invariant : ∀ c₁ ∈ gemmFp8BsSpace, ∀ c₂ ∈ gemmFp8BsSpace, ∀ (S₁ S₂ : Nat) (p q : GTerm) (a : RowData) (b : ColData) (KB : Nat), "
        "tileResultKB c₁ S₁ p a b KB = tileResultKB c₂ S₂ q a b KB := space_invariantKB gemmFp8BsSpace"
    )
    disp = fp8bs_dispatch_cases()
    L.append(
        "/-! Every knob record the block-scale dispatcher returns over the DeepSeek (N, K) grid (the K-major weight layout; every "
        "part of the nominal table Rubin NVL8 / NVL72, B200 / GB200, B300, never a device's queried numbers), with the first shape "
        "that selects it: a member of the space above (R3 membership). -/"
    )
    L.append(
        "def gemmFp8BsDispatch : List (Nat × Nat × Nat × Config) := ["
        + ", ".join(f"({M}, {N}, {K}, {k.lean()})" for k, (M, N, K) in disp.items())
        + "]"
    )
    L.append(
        "theorem gemm_fp8bs_dispatch_in_space : gemmFp8BsDispatch.all (fun ⟨_, _, _, c⟩ => gemmFp8BsSpace.any (fun s => s == c)) = true := by native_decide"
    )
    L.append("")


def _render_attention(L: List[str]) -> None:
    """The attention class: the toy-case spec terms and the per-head-dim configuration spaces."""
    mk = MMA_K_OF["bf16"]
    L.append(
        "/-! The attention class on toy sizes (T = 16: one PV step per tile; C = 32; D = 64 / 128 / 256 = 4 / 8 / 16 MMA steps per "
        "score): the Python spec's output element (`attention/batch_invariant/spec.py`) reproduced by `Attention.lean` node for node, "
        "and the split kernel's equality to it instantiated on every case. -/"
    )
    for name, D, T, C, p, win, rid, d, sx in attn_cases():
        tb = T.bit_length() - 1
        L.append(
            f"def attnP_{name} : Attn.Params := ⟨{D // mk}, {tb}, {C}, {p}, {win}, {rid}, {d}⟩"
        )
        L.append(
            f"theorem attn_{name}_spec : (Attn.output attnP_{name}).sexpr = {lean_str(sx)} := by native_decide"
        )
        L.append(
            f"theorem attn_{name}_split : ∀ σ : Nat → Nat, Attn.splitKernel attnP_{name} σ = Attn.output attnP_{name} := "
            f"fun σ => Attn.splitKernel_eq_fused _ σ"
        )
    L.append(
        "def attnCases : List (String × Attn.Params) := ["
        + ", ".join(f'("{n}", attnP_{n})' for n, *_ in ATTN_CASES)
        + "]"
    )
    L.append("")
    L.append(
        "/-! The attention kernel's configuration space per head dim (`engine.all_valid_configs(D)`): every knob record checked "
        "against the static constraints (`ConfigOKD`), counted, and `space_invariant` instantiated on it (R3). -/"
    )
    for D, cfgs in attn_spaces().items():
        L.append(
            f"def attnSpaceD{D} : List Attn.Config := ["
            + ", ".join(c.lean() for c in cfgs)
            + "]"
        )
        L.append(
            f"theorem attn_space_d{D}_size : attnSpaceD{D}.length = {len(cfgs)} := by native_decide"
        )
        L.append(
            f"theorem attn_space_d{D}_ok : attnSpaceD{D}.all (Attn.ConfigOKD {D}) = true := by native_decide"
        )
        L.append(
            f"theorem attn_space_d{D}_invariant : ∀ b₁ ∈ attnSpaceD{D}, ∀ b₂ ∈ attnSpaceD{D}, "
            "∀ (f₁ f₂ : Bool) (P : Attn.Params) (σ τ : Nat → Nat), Attn.kernel b₁ f₁ P σ = Attn.kernel b₂ f₂ P τ := "
            f"Attn.space_invariant attnSpaceD{D}"
        )
    L.append("")


def render() -> str:
    """The content of ``Cases.lean``."""
    L = [
        "-- Copyright (c) 2026 by FlashInfer team.",
        "-- SPDX-License-Identifier: Apache-2.0",
        "import BIKernels.GEMM",
        "import BIKernels.GEMMConfig",
        "import BIKernels.Export",
        "import BIKernels.Attention",
        "namespace BIK",
        "-- GENERATED by flashinfer/batch_invariant/lean_cases.py from the Python spec / engine; do not edit.",
        "set_option maxRecDepth 4096",
        "open GEMM",
        "",
    ]
    _render_format(L, BY_FMT["bf16"])
    _render_bias(L)
    for F in FORMATS[1:]:
        _render_format(L, F)
    _render_fp8_blockscale(L)
    _render_attention(L)
    L.append("end BIK")
    return "\n".join(L) + "\n"


def summary() -> str:
    parts = []
    for F in FORMATS:
        space, spacec = spaces(F.in_fmt)
        extra = "+".join(str(len(r[-1])) for r in extra_spaces(F.in_fmt))
        parts.append(
            f"{F.in_fmt}: specs={len(F.cases)} loops={len(F.loop_cases)} outs={len(F.out_cases)} "
            f"chunk-shapes={len(CHUNK_TABLE_SHAPES) + len(F.chunk_extra)} space={len(space)} split={len(spacec)}"
            + (f" extra={extra}" if extra else "")
            + f" dispatch={len(dispatch_cases(F.in_fmt))}"
        )
    return (
        "lean_cases coverage: "
        + "; ".join(parts)
        + f"; bias cases={len(GEMM_BIAS_CASES)}"
        + f"; {FP8BS_FMT}: specs={2 * len(GEMM_FP8BS_OUT_CASES)} loops={len(GEMM_FP8BS_LOOP_CASES)} "
        f"space={len(fp8bs_space())} dispatch={len(fp8bs_dispatch_cases())}"
        + f"; native_decide theorems={len(DECIDED_THEOREMS)}"
        + f"; attention cases={len(ATTN_CASES)} attention spaces={[len(v) for v in attn_spaces().values()]}"
    )


def main(out_path: str = CASES_PATH) -> str:
    content = render()
    with open(out_path, "w") as f:
        f.write(content)
    s = summary()
    print(s)
    return s


if __name__ == "__main__":
    main()
