# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The dense bf16 GEMM as a kernel generator on the engine.

A GEMM kernel is generated from three typed parts, and nothing else:

* the **reduction** (``spec.py``): a ``fold`` digit over the K steps of one opaque MMA instruction, and for the
  split class a second ``fold`` over chunks -- a ``canon.Plan`` whose register-level folds are the only source of
  per-element order;
* the **mainloop block** (:class:`MainloopBlock`): the tcgen05 K pipeline (``blocks/mainloop.py``), opaque to the
  engine below one instruction; its instruction variant is a function of the knobs and is the single hardware
  assumption (A2) the GPU configuration-agreement tests discharge;
* the **knobs** (:class:`Knobs`): tile M/N, 1-/2-CTA MMA, cluster shape, raster order, swizzle, grid size and the
  in-CTA chunk fold. They choose which CTA owns which output tile, the pipeline depth and the launch geometry;
  they cannot reach the reduction or the epilogue arithmetic. The enumerated knob product filtered by the kernel's
  own acceptance is the configuration space the guarantee is claimed over.

``lean_cases.py`` emits the knob records of the whole space and of the dispatcher's choices from these objects,
so ``GEMMConfig.lean`` validates the actual space (``gemm_space_ok``, ``gemm_space_size``), the actual dispatch
(``gemm_dispatch_in_space``) and instantiates ``config_invariance`` on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ...batch_invariant import canon
from .spec import ELEM_BITS, MMA_K, MMA_OP, GemmSpec, k_steps

MMA_K_BLOCKS = 4  # instruction K-steps per smem stage: the static inner radix of the mainloop's blocked fold
K_TILE_BYTES = (
    128  # bytes of K per operand row in one smem stage (the 128-byte-swizzled TMA box)
)


def inner_radix() -> int:
    """Instruction K-steps per smem stage: what one 128-byte stage holds at the 16-element step (4). The inner radix
    of ``MainloopBlock.lowering()``; Lean's ``kblockLoop_eq_kfold`` holds for every value of it."""
    return K_TILE_BYTES * 8 // (MMA_K * ELEM_BITS)


# ----------------------------------------------------------------------------- knobs

TILES_N = (32, 64, 128, 192, 256)
#: The decode tiles: tile N 8 / 16 on the 1-CTA instruction (kind::f16 admits N >= 8 step 8) and 16 on the 2-CTA one.
#: More CTAs on the weight stream at small M (bytes in flight = CTAs x stages x B-stage bytes).
TILES_N_DENSE_NARROW = (8, 16)
#: Pruned from the ENUMERATED space (still well-formed, still compilable; never within 3 % of a grid point's best record):
#: (tile, 2-CTA) pairs. (128, 8) 1-CTA: a 128-row A tile for eight output columns; (256, 16) 2-CTA: the pair's per-k-tile
#: hand-shake for 16 columns.
PRUNED_DENSE = frozenset({((128, 8), False), ((256, 16), True)})
TILES_M = {
    False: (64, 128),
    True: (128, 256),
}  # 1-CTA / 2-CTA MMA (mma tiler M; the CTA holds half for 2-CTA)
CLUSTERS = ((1, 1), (2, 1), (1, 2), (2, 2), (4, 1), (1, 4))
RASTERS = ("m", "n")
SWIZZLES = (1, 2, 4, 8)
#: The in-CTA chunk fold (``Knobs.cta_chunks``): a record of the split class ``kchunk[C]`` in which ONE CTA (pair) owns an
#: output tile for all C chunks: the mainloop folds chunk c into a fresh TMEM accumulator (``D_0 = +0``, the chunk's K steps
#: ascending), and the epilogue warps fold the completed chunk partial into a TMEM running sum in chunk order from +0 with
#: ``add.rn``, then apply the chain once after the last chunk -- the workspace fixup's fold, with the partials in TMEM instead
#: of an fp32 round trip through HBM and no second launch. The running sum takes one accumulator's worth of TMEM columns,
#: so the record exists where ``(stages + 1) x tile N`` fit the TMEM: tile N <= CTA_CHUNKS_MAX_TN.
CTA_CHUNKS_MAX_TN = 256


@dataclass(frozen=True)
class Knobs:
    """One point of the generator's configuration space. Field layout, equality and ``describe()`` are stable:
    they key the compiled-kernel cache."""

    mma_tiler_mn: Tuple[int, int] = (256, 256)
    cluster_shape_mn: Tuple[int, int] = (2, 1)
    use_2cta_instrs: bool = True
    use_tma_store: bool = True  # the only store path that compiles with dynamic shapes
    raster_along: str = "m"
    swizzle_size: int = 1
    grid_clusters: int = 0  # persistent grid size in clusters; 0 = one per SM
    cta_chunks: bool = False  # the in-CTA chunk fold (split class): one CTA folds every chunk of its tile

    # named views
    @property
    def tile_m(self) -> int:
        return self.mma_tiler_mn[0]

    @property
    def tile_n(self) -> int:
        return self.mma_tiler_mn[1]

    @property
    def cluster_m(self) -> int:
        return self.cluster_shape_mn[0]

    @property
    def cluster_n(self) -> int:
        return self.cluster_shape_mn[1]

    @property
    def cta_tile_mn(self) -> Tuple[int, int]:
        return (self.tile_m // (2 if self.use_2cta_instrs else 1), self.tile_n)

    def well_formed(self) -> bool:
        """The body's static knob constraints (mirror of ``BIK.GEMM.ConfigOK``)."""
        tm_ok = self.tile_m in TILES_M[self.use_2cta_instrs]
        # the decode tiles: N 16 on both CTA groups; N 8 on the 1-CTA kind::f16 instruction only
        tn_ok = (32 <= self.tile_n <= 256 and self.tile_n % 32 == 0) or (
            self.tile_n in TILES_N_DENSE_NARROW
            and (self.tile_n >= 16 or not self.use_2cta_instrs)
        )
        cm, cn = self.cluster_shape_mn
        cl_ok = (
            cm in (1, 2, 4, 8, 16)
            and cn in (1, 2, 4, 8, 16)
            and cm * cn <= 16
            and (not self.use_2cta_instrs or cm % 2 == 0)
        )
        ok = (
            tm_ok
            and tn_ok
            and cl_ok
            and self.raster_along in RASTERS
            and self.swizzle_size in SWIZZLES
            and self.use_tma_store
        )
        if self.cta_chunks:
            ok = (
                ok and self.tile_n <= CTA_CHUNKS_MAX_TN
            )  # the running sum fits next to the accumulator stages
        return ok

    def describe(self) -> str:
        return (
            f"mma{self.tile_m}x{self.tile_n}_cl{self.cluster_m}x{self.cluster_n}"
            f"_{'2cta' if self.use_2cta_instrs else '1cta'}_{'tma' if self.use_tma_store else 'st'}_r{self.raster_along}_sw{self.swizzle_size}"
            + (f"_g{self.grid_clusters}" if self.grid_clusters else "")
            + ("_cc" if self.cta_chunks else "")
        )

    def instruction(self) -> str:
        """The MMA instruction variant these knobs select: the A2 assumption surface."""
        return f"tcgen05.mma.cta_group::{2 if self.use_2cta_instrs else 1}.kind::f16 {self.tile_m}x{self.tile_n}x{MMA_K}"

    def lean(self) -> str:
        """The record ``BIK.GEMM.Config`` (tm, tn, clusterM, clusterN, twoCta, rasterM, swizzle, ctaChunks)."""
        return (
            f"⟨{self.tile_m}, {self.tile_n}, {self.cluster_m}, {self.cluster_n}, "
            f"{'true' if self.use_2cta_instrs else 'false'}, {'true' if self.raster_along == 'm' else 'false'}, {self.swizzle_size}, "
            f"{'true' if self.cta_chunks else 'false'}⟩"
        )


GemmConfig = Knobs  # the API's name for a record


def knob_product(
    swizzles=SWIZZLES, clusters=CLUSTERS, cta_chunks: bool = False
) -> List[Knobs]:
    """Every well-formed knob combination, before the kernel's shape-dependent acceptance: 888 records (720 at the
    tiles N 32..256 plus 168 decode-tile records). ``cta_chunks=True`` adds the in-CTA chunk-fold twin of every
    record under the TMEM budget, which only the split classes accept."""
    out = []
    for two in (False, True):
        for tm in TILES_M[two]:
            for tn in TILES_N + TILES_N_DENSE_NARROW:
                for cl in clusters:
                    if two and cl[0] % 2:
                        continue
                    for raster in RASTERS:
                        for sw in swizzles:
                            for cc in (False, True) if cta_chunks else (False,):
                                k = Knobs((tm, tn), cl, two, True, raster, sw, 0, cc)
                                if (
                                    k.well_formed()
                                    and (k.mma_tiler_mn, two) not in PRUNED_DENSE
                                ):
                                    out.append(k)
    return out


def enumerate_space(
    M: int,
    N: int,
    K: int,
    swizzles=SWIZZLES,
    clusters=CLUSTERS,
    chunks: int = 1,
    b_major: str = "n",
) -> List[Knobs]:
    """The configuration space for a shape and class: the knob product the kernel body accepts (``can_implement``)
    for ``chunks`` (the class's chunk count). This is the set R3 is claimed over; the GPU configuration-agreement tests
    check the dispatcher's candidates and sampled records of it against each other."""
    from .blocks.operands import can_implement

    return [
        k
        for k in knob_product(swizzles, clusters, cta_chunks=chunks > 1)
        if can_implement(k, M, N, K, b_major, chunks)
    ]


REFERENCE_SHAPE = (1000, 4096, 4096)  # the shape whose space is emitted to Lean


# ----------------------------------------------------------------------------- blocks


@dataclass(frozen=True)
class MainloopBlock:
    """The K pipeline: TMA -> smem ring -> tcgen05.mma -> TMEM accumulator (``blocks/producer.py``,
    ``blocks/mainloop.py``). The engine sees it as the ``kstep`` fold digit; the block's step width and K-block are
    fixed; its instruction variant by the knobs."""

    @property
    def step(self) -> int:
        return MMA_K

    @property
    def inner(self) -> int:
        """Instruction K-steps per smem stage (the lowering's inner radix)."""
        return inner_radix()

    @property
    def k_block(self) -> int:
        return self.inner * self.step  # K elements per smem stage (128 bytes of K)

    @property
    def op(self) -> str:
        return MMA_OP

    def steps(self, K: int) -> int:
        return k_steps(K)

    def lowering(self) -> canon.BlockedFold:
        """How the kernel issues the ``kstep`` fold: blocks of ``inner`` instructions (one smem stage), blocks
        ascending, instructions ascending, the first one with the accumulator reset. ``blocks/mainloop.py`` executes
        exactly this object."""
        return canon.BlockedFold("kstep", self.inner)

    def assumption(self, knobs: Knobs) -> str:
        return (
            f"A2: {knobs.instruction()} advances each accumulator element by the same per-element function as every "
            "other variant in the space (tested, not proved)"
        )


# ----------------------------------------------------------------------------- the generated kernel


@dataclass(frozen=True)
class GemmKernelPlan:
    """What the generator emits for one call site: the engine plan (numerics), the mainloop block, the knobs
    (schedule) and the output format. ``blocks/compile.py`` compiles exactly this."""

    knobs: Knobs
    out_fmt: str
    chunks: int
    K: int

    @property
    def block(self) -> MainloopBlock:
        return MainloopBlock()

    def spec(self, alg=None) -> GemmSpec:
        return GemmSpec(self.K, self.out_fmt, alg=alg, chunks=self.chunks)

    def placement(self) -> canon.Placement:
        """Both folds are register-level: the K fold is the accumulator of one output element (TMEM lane); the chunk
        fold is one fixup thread (or the owning CTA's epilogue) folding the partials in index order."""
        f = {
            "kstep": [
                (
                    self.block.steps(self.K) // self.chunks
                    if self.chunks > 1
                    else self.block.steps(self.K),
                    "reg",
                )
            ]
        }
        if self.chunks > 1:
            f["chunk"] = [(self.chunks, "reg")]
        return canon.Placement.make(f)

    def plan(self):
        from .spec import reduction, reduction_chunked

        red = (
            reduction(self.K, 0, 0)
            if self.chunks == 1
            else reduction_chunked(self.K, self.chunks, 0, 0)
        )
        return canon.plan(red, self.placement())

    def verify(self, alg=None):
        """Every simulated unit of the plan ends with the spec's canonical accumulator term, and the blocked lowering
        the kernel executes (``MainloopBlock.lowering``) builds, over the block count the kernel runs for this K, the
        canonical fold of the padded K (the zero-filled tail steps are identities: Lean ``fold_pad``), per chunk for
        the split class."""
        from ...batch_invariant.fp_interface.term import TermAlgebra
        from .spec import reduction, reduction_chunked

        alg = alg or TermAlgebra()
        p = self.plan()
        got = canon.verify(p, alg)
        assert got is p.red.canonical(alg), "generated plan is not canonical"
        low = self.block.lowering()
        steps_per_chunk = self.block.steps(self.K // self.chunks)
        nb = low.blocks_for(steps_per_chunk)
        K_pad = nb * low.inner * self.block.step * self.chunks
        if self.chunks == 1:
            red_pad = reduction(K_pad, 0, 0)
            assert low.trace(red_pad, alg, nb) is red_pad.canonical(alg), (
                "mainloop lowering is not the canonical fold"
            )
        else:
            red_pad = reduction_chunked(K_pad, self.chunks, 0, 0)
            kstep = red_pad.digits[0]
            for c in range(self.chunks):
                partial = kstep.reduce(
                    alg,
                    [
                        red_pad.leaf(alg, {"chunk": c, "kstep": t})
                        for t in range(kstep.radix)
                    ],
                )
                assert low.trace(red_pad, alg, nb, fixed={"chunk": c}) is partial, (
                    f"mainloop lowering differs on chunk {c}"
                )
        return got

    def describe(self) -> str:
        return (
            f"{self.knobs.describe()} | {self.block.op} x{self.block.steps(self.K)}"
            + (f" in {self.chunks} chunks" if self.chunks > 1 else "")
            + f" | cvt.rn {self.out_fmt}"
        )
