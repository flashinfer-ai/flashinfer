# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The dense GEMM classes as a kernel generator on the engine.

A GEMM kernel is generated from four typed parts, and nothing else:

* the **reduction** (``spec.py``): a ``fold`` digit over the K steps of one opaque MMA instruction, and for the
  split class a second ``fold`` over chunks -- a ``canon.Plan`` whose register-level folds are the only source of
  per-element order;
* the **mainloop block** (:class:`MainloopBlock`): the tcgen05 K pipeline (``blocks/mainloop.py``), opaque to the
  engine below one instruction; its step width is the format's, its instruction variant is a function of the knobs
  and is the single hardware assumption (A2) the GPU configuration-agreement tests discharge;
* the **knobs** (:class:`Knobs`): tile M/N, 1-/2-CTA MMA, cluster shape, raster order, swizzle, grid size and the
  in-CTA chunk fold. They choose which CTA owns which output tile, the pipeline depth and the launch geometry;
  they cannot reach the reduction or the epilogue arithmetic. The enumerated knob product filtered by the kernel's
  own acceptance is the configuration space the guarantee is claimed over;
* the **epilogue chain** (``spec.EpilogueChain``): the elementwise ops the specification applies after the fold
  (the fp8 classes' per-tensor scale, the conversion).

``lean_cases.py`` emits the knob records of the whole space and of the dispatcher's choices from these objects,
so ``GEMMConfig.lean`` validates the actual spaces (``gemm_space_ok``, ``gemm_fp8_space_ok``), the actual dispatch
(``gemm_dispatch_in_space``) and instantiates ``config_invariance`` on them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

from ...batch_invariant import canon
from .spec import (
    ALPHA_GIVEN,
    BLOCK_K,
    BLOCK_SCALED,
    ELEM_BITS_OF,
    FP8_BLOCKSCALE_FMTS,
    MMA_K_OF,
    MMA_OP_OF,
    SF_FMT,
    SF_VEC,
    EpilogueChain,
    GemmSpec,
    is_scaled,
    k_steps,
)

MMA_K_BLOCKS = 4  # instruction K-steps per smem stage: the static inner radix of the mainloop's blocked fold
K_TILE_BYTES = (
    128  # bytes of K per operand row in one smem stage (the 128-byte-swizzled TMA box)
)


def inner_radix(in_fmt: str = "bf16") -> int:
    """Instruction K-steps per smem stage: what one 128-byte stage holds at the format's step (4 for the 32-byte steps
    of both bf16 and fp8). The inner radix of ``MainloopBlock.lowering()``; Lean's ``kblockLoopG_eq_kfoldG`` holds for
    every value of it."""
    return K_TILE_BYTES * 8 // (MMA_K_OF[in_fmt] * ELEM_BITS_OF[in_fmt])


# ----------------------------------------------------------------------------- knobs

TILES_N = (32, 64, 128, 192, 256)
#: The decode tiles: tile N 8 / 16 on the 1-CTA instruction (kind::f16 admits N >= 8 step 8; kind::f8f6f4 N >= 16) and 16
#: on the 2-CTA one. More CTAs on the weight stream at small M (bytes in flight = CTAs x stages x B-stage bytes).
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
#: so the record exists where ``(stages + 1) x tile N`` (plus the scale-factor columns of a block-scaled class) fit the
#: TMEM: tile N <= CTA_CHUNKS_MAX_TN[format] (192 under the 512-column budget of the sm_100 block-scaled class, 256 otherwise).
CTA_CHUNKS_MAX_TN = {
    "bf16": 256,
    "e4m3": 256,
    "e5m2": 256,
    "mxfp8": 192,
    "mxfp8_k64": 256,
    "nvfp4": 192,
    "nvfp4_k128": 256,
    "mxfp4": 192,
    "mxfp4_k128": 256,
    "e4m3_bs": 128,
}
#: The block-scaled instruction fixes MMA M at 128 (1-CTA) / 256 (2-CTA): its scale-factor atom spans 128 rows, so no
#: 64-row CTA tile exists; tile N below 64 would need sub-tile scale-factor addressing; a scale-factor tile multicasts to
#: at most 4 CTAs.
TILES_N_BLOCKSCALED = (64, 128, 192, 256)
TILES_M_BLOCKSCALED = {False: (128,), True: (256,)}
#: The block-scale fp8 class (``e4m3_bs``: the dense ``kind::f8f6f4`` body with a per-block promotion by the epilogue warps): the
#: 128-row CTA tile of the one-row-per-thread promotion (MMA M 128 1-CTA / 256 2-CTA), tile N dividing the 128-column weight
#: scale block (the register running sum bounds it at 128), the 16-wide tile on the 1-CTA instruction only; no split variant,
#: no in-CTA fold, no swapped record.
TILES_N_FP8BS = (16, 32, 64, 128, 256)
TILES_M_FP8BS = TILES_M_BLOCKSCALED
MAX_CLUSTER_DIM_BLOCKSCALED = 4
#: In-cluster split-K (block-scaled body): the ``cluster_k`` CTAs of a (cluster_k, 1, 1) cluster each fold one K chunk and
#: the partials are folded in chunk order through distributed shared memory in the epilogue; the record computes the class
#: ``kchunk[cluster_k]`` exactly as the workspace split does. 1-CTA MMA, no M / N cluster, tile N bounded by the mailbox.
CLUSTER_K = (1, 2, 4)
MAX_TILE_N_CLUSTER_K = {2: 128, 4: 64}
#: The decode specialisation (block-scaled body): operands swapped (the kernel computes C^T with the weights as the MMA's
#: A operand on the 128-row axis and the tokens as B) with one token tile holding every token: the narrow tiles 8 / 16 / 32
#: on the 1-CTA instruction, or the 64-wide tile on either CTA group. The block-scaled step is symmetric in its operands
#: (checked bit for bit), so the records compute the same class. No N cluster (the narrow scale-factor tile is not
#: multicast), no in-cluster split.
TILES_N_NARROW = (8, 16, 32)
SWAP_WIDE_TILE_N = 64
#: The 64-wide swapped token tile exists for the UE8M0 classes (their scale-factor atom covers the tile's K extent of one stage);
#: the NVFP4 classes keep the narrow token tiles only.
SWAP_WIDE_FMTS = ("mxfp8", "mxfp8_k64", "mxfp4", "mxfp4_k128")
#: B-reuse (the sm_107 instructions only): a 512-row tiler on the 2-CTA 256-row instruction; each CTA pair issues two
#: instructions per K step on the two halves of the tile, the B fragment loaded into the collector buffer by the first and
#: consumed by the second: half the B smem traffic per MMA. One accumulator per half, the same K steps in order from +0:
#: a placement of the class. Dense records only (no swap, no in-cluster split, no in-CTA chunk fold).
TILE_M_BREUSE = 512
BREUSE_FMTS = ("nvfp4_k128", "mxfp8_k64", "mxfp4_k128")


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
    grid_clusters: int = 0  # persistent grid size in clusters; 0 = the launch device's co-resident cluster count
    cta_chunks: bool = False  # the in-CTA chunk fold (split class): one CTA folds every chunk of its tile
    cluster_k: int = 1  # in-cluster split-K factor (block-scaled body): the record computes the class kchunk[cluster_k]
    swap_ab: bool = False  # the decode specialisation (block-scaled body): operands swapped, one token tile for all tokens

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

    @property
    def b_reuse(self) -> bool:
        """The B-reuse record: tile M = 2 x the instruction's M (512 on the 2-CTA instruction; the sm_107 block-scaled
        classes only)."""
        return self.tile_m == TILE_M_BREUSE

    @property
    def inst_m(self) -> int:
        """The MMA instruction's M: the tile M, or half of it for the B-reuse record (two instructions per K step)."""
        return self.tile_m // 2 if self.b_reuse else self.tile_m

    def well_formed(self, in_fmt: str = "bf16") -> bool:
        """The body's static knob constraints (mirror of ``BIK.GEMM.ConfigOK`` / ``ConfigOKFp8`` / ``ConfigOKBlockScaled*``):
        for the dense classes the 8-wide decode tile exists on the 1-CTA ``kind::f16`` instruction only (``kind::f8f6f4``
        admits N >= 16) and the block-scaled knobs are off; for the block-scaled classes the instruction's own
        restrictions (MMA M 128 / 256, tile N 64 .. 256 or a swapped token tile, cluster dims at most 4, the in-cluster
        split and B-reuse records' shapes)."""
        two = self.use_2cta_instrs
        cm, cn = self.cluster_shape_mn
        cl_ok = (
            cm in (1, 2, 4, 8, 16)
            and cn in (1, 2, 4, 8, 16)
            and cm * cn <= 16
            and (not two or cm % 2 == 0)
        )
        ok = (
            cl_ok
            and self.raster_along in RASTERS
            and self.swizzle_size in SWIZZLES
            and self.use_tma_store
        )
        if in_fmt in FP8_BLOCKSCALE_FMTS:
            return (
                ok
                and self.tile_m in TILES_M_FP8BS[two]
                and self.tile_n in TILES_N_FP8BS
                and (self.tile_n >= 32 or not two)
                and self.cluster_k == 1
                and not self.swap_ab
                and not self.cta_chunks
            )
        if in_fmt in BLOCK_SCALED:
            tm_ok = self.tile_m in TILES_M_BLOCKSCALED[two] or (
                self.b_reuse
                and in_fmt in BREUSE_FMTS
                and two
                and not self.swap_ab
                and self.cluster_k == 1
            )
            ok = (
                ok
                and tm_ok
                and cm <= MAX_CLUSTER_DIM_BLOCKSCALED
                and cn <= MAX_CLUSTER_DIM_BLOCKSCALED
                and self.cluster_k in CLUSTER_K
            )
            if self.swap_ab:
                ok = (
                    ok
                    and cn == 1
                    and self.cluster_k == 1
                    and (
                        (self.tile_n in TILES_N_NARROW and not two)
                        or (
                            self.tile_n == SWAP_WIDE_TILE_N and in_fmt in SWAP_WIDE_FMTS
                        )
                    )
                )
            else:
                ok = ok and self.tile_n in TILES_N_BLOCKSCALED
            if self.cluster_k > 1:
                ok = (
                    ok
                    and cm == 1
                    and cn == 1
                    and not two
                    and self.tile_n <= MAX_TILE_N_CLUSTER_K[self.cluster_k]
                )
            if self.cta_chunks:
                ok = (
                    ok
                    and self.cluster_k == 1
                    and not self.swap_ab
                    and not self.b_reuse
                    and self.tile_n <= CTA_CHUNKS_MAX_TN[in_fmt]
                )
            return ok
        tm_ok = self.tile_m in TILES_M[two]
        # the decode tiles: N 16 on both CTA groups; N 8 on the 1-CTA kind::f16 instruction only
        tn_ok = (32 <= self.tile_n <= 256 and self.tile_n % 32 == 0) or (
            self.tile_n in TILES_N_DENSE_NARROW
            and (self.tile_n >= 16 or (not two and in_fmt == "bf16"))
        )
        ok = ok and tm_ok and tn_ok and self.cluster_k == 1 and not self.swap_ab
        if self.cta_chunks:
            ok = (
                ok and self.tile_n <= CTA_CHUNKS_MAX_TN[in_fmt]
            )  # the running sum fits next to the accumulator stages
        return ok

    def describe(self) -> str:
        return (
            f"mma{self.tile_m}x{self.tile_n}_cl{self.cluster_m}x{self.cluster_n}"
            f"_{'2cta' if self.use_2cta_instrs else '1cta'}_{'tma' if self.use_tma_store else 'st'}_r{self.raster_along}_sw{self.swizzle_size}"
            + (f"_g{self.grid_clusters}" if self.grid_clusters else "")
            + (f"_ck{self.cluster_k}" if self.cluster_k > 1 else "")
            + ("_swap" if self.swap_ab else "")
            + ("_cc" if self.cta_chunks else "")
        )

    def instruction(self, in_fmt: str = "bf16") -> str:
        """The MMA instruction variant these knobs select for the format: the A2 assumption surface."""
        if in_fmt in BLOCK_SCALED:
            kind = {
                "mxfp8": "kind::mxf8f6f4.block_scale",
                "mxfp8_k64": "kind::mxf8f6f4.block_scale",
                "nvfp4": "kind::mxf4nvf4.block_scale",
                "nvfp4_k128": "kind::mxf4nvf4.block_scale",
                "mxfp4": "kind::mxf4.block_scale",
                "mxfp4_k128": "kind::mxf4nvf4.block_scale",
            }[in_fmt]
            sf = f" sf={SF_FMT[in_fmt]}/{SF_VEC[in_fmt]}"
            arch = " (sm_107a)" if in_fmt.endswith(("_k64", "_k128")) else ""
            breuse = (
                " x2 b-reuse (FILL/LASTUSE)"
                if self.b_reuse and in_fmt in BREUSE_FMTS
                else ""
            )
        else:
            kind = "kind::f16" if in_fmt == "bf16" else "kind::f8f6f4"
            sf = arch = breuse = ""
        return (
            f"tcgen05.mma.cta_group::{2 if self.use_2cta_instrs else 1}.{kind} "
            f"{self.inst_m}x{self.tile_n}x{MMA_K_OF[in_fmt]}{sf}{arch}{breuse}"
        )

    def lean(self) -> str:
        """The record ``BIK.GEMM.Config`` (tm, tn, clusterM, clusterN, twoCta, rasterM, swizzle, ctaChunks, clusterK,
        swapAb)."""
        return (
            f"⟨{self.tile_m}, {self.tile_n}, {self.cluster_m}, {self.cluster_n}, "
            f"{'true' if self.use_2cta_instrs else 'false'}, {'true' if self.raster_along == 'm' else 'false'}, {self.swizzle_size}, "
            f"{'true' if self.cta_chunks else 'false'}, {self.cluster_k}, {'true' if self.swap_ab else 'false'}⟩"
        )


GemmConfig = Knobs  # the API's name for a record


def knob_product(
    swizzles=SWIZZLES,
    clusters=CLUSTERS,
    cta_chunks: bool = False,
    in_fmt: str = "bf16",
) -> List[Knobs]:
    """Every well-formed knob combination, before the kernel's shape-dependent acceptance: 888 records for bf16 (720 at the
    tiles N 32..256 plus 168 decode-tile records), 840 for the fp8 formats (no 8-wide tile on ``kind::f8f6f4``).
    ``cta_chunks=True`` adds the in-CTA chunk-fold twin of every record under the TMEM budget, which only the split
    classes accept."""
    out = []
    bs = in_fmt in BLOCK_SCALED
    for two in (False, True):
        tms = (
            TILES_M_BLOCKSCALED[two]
            + ((TILE_M_BREUSE,) if two and in_fmt in BREUSE_FMTS else ())
            if bs
            else TILES_M[two]
        )
        tns = (
            TILES_N_BLOCKSCALED + TILES_N_NARROW
            if bs
            else TILES_N + TILES_N_DENSE_NARROW
        )
        for tm in tms:
            for tn in tns:
                for cl in clusters:
                    if two and cl[0] % 2:
                        continue
                    for raster in RASTERS:
                        for sw in swizzles:
                            for ck in CLUSTER_K if bs else (1,):
                                for swap in (False, True) if bs else (False,):
                                    for cc in (False, True) if cta_chunks else (False,):
                                        k = Knobs(
                                            (tm, tn),
                                            cl,
                                            two,
                                            True,
                                            raster,
                                            sw,
                                            0,
                                            cc,
                                            ck,
                                            swap,
                                        )
                                        if k.well_formed(in_fmt) and (
                                            bs
                                            or (k.mma_tiler_mn, two) not in PRUNED_DENSE
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
    in_fmt: str = "bf16",
) -> List[Knobs]:
    """The configuration space for a shape, layout and class: the knob product the kernel body accepts
    (``can_implement``) for ``chunks`` (the class's chunk count). This is the set R3 is claimed over; the GPU
    configuration-agreement tests check the dispatcher's candidates and sampled records of it against each other."""
    from .blocks.operands import can_implement

    return [
        k
        for k in knob_product(swizzles, clusters, cta_chunks=chunks > 1, in_fmt=in_fmt)
        if can_implement(k, M, N, K, b_major, chunks, in_fmt)
    ]


REFERENCE_SHAPE = (1000, 4096, 4096)  # the shape whose space is emitted to Lean


# ----------------------------------------------------------------------------- blocks


@dataclass(frozen=True)
class MainloopBlock:
    """The K pipeline: TMA -> smem ring -> tcgen05.mma -> TMEM accumulator (``blocks/producer.py``,
    ``blocks/mainloop.py``). The engine sees it as the ``kstep`` fold digit; the block's step width and K-block are
    fixed by the format; its instruction variant by the knobs."""

    in_fmt: str = "bf16"
    #: an override of the K-steps per smem stage (``StaticPlan(k_blocks=)``); None: the format's ``inner_radix``
    inner_steps: Optional[int] = None

    @property
    def step(self) -> int:
        return MMA_K_OF[self.in_fmt]

    @property
    def inner(self) -> int:
        """Instruction K-steps per smem stage (the lowering's inner radix)."""
        return self.inner_steps or inner_radix(self.in_fmt)

    @property
    def k_block(self) -> int:
        return self.inner * self.step  # K elements per smem stage (128 bytes of K)

    @property
    def op(self) -> str:
        return MMA_OP_OF[self.in_fmt]

    def steps(self, K: int) -> int:
        return k_steps(K, self.in_fmt)

    def lowering(self) -> canon.BlockedFold:
        """How the kernel issues the ``kstep`` fold: blocks of ``inner`` instructions (one smem stage), blocks
        ascending, instructions ascending, the first one with the accumulator reset. ``blocks/mainloop.py`` executes
        exactly this object."""
        return canon.BlockedFold("kstep", self.inner)

    def assumption(self, knobs: Knobs) -> str:
        return (
            f"A2: {knobs.instruction(self.in_fmt)} advances each accumulator element by the same per-element function as "
            "every other variant in the space (tested, not proved)"
        )


# ----------------------------------------------------------------------------- the generated kernel


@dataclass(frozen=True)
class GemmKernelPlan:
    """What the generator emits for one call site: the engine plan (numerics), the mainloop block, the knobs
    (schedule), the formats and the epilogue chain. ``blocks/compile.py`` compiles exactly this."""

    knobs: Knobs
    out_fmt: str
    chunks: int
    K: int
    in_fmt: str = "bf16"

    @property
    def block(self) -> MainloopBlock:
        return MainloopBlock(self.in_fmt)

    @property
    def block_scale(self) -> bool:
        """The block-scale fp8 class: the K fold per 128-element block, the block fold with the scales
        (`spec.reduction_blockscale`)."""
        return self.in_fmt in FP8_BLOCKSCALE_FMTS

    @property
    def epilogue(self) -> EpilogueChain:
        return EpilogueChain.for_class(self.out_fmt, is_scaled(self.in_fmt))

    def spec(self, alg=None) -> GemmSpec:
        return GemmSpec(
            self.K, self.out_fmt, alg=alg, chunks=self.chunks, in_fmt=self.in_fmt
        )

    def placement(self) -> canon.Placement:
        """Both folds are register-level: the K fold is the accumulator of one output element (TMEM lane); the chunk
        fold is one fixup thread (or the owning CTA's epilogue) folding the partials in index order; the block-scale
        class's block fold is one epilogue thread's running sum over the blocks in index order."""
        if self.block_scale:
            return canon.Placement.make(
                {
                    "kstep": [(BLOCK_K // self.block.step, "reg")],
                    "block": [(self.K // BLOCK_K, "reg")],
                }
            )
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
        from .spec import reduction, reduction_blockscale, reduction_chunked

        if self.block_scale:
            return canon.plan(
                reduction_blockscale(self.K, 0, 0, self.in_fmt), self.placement()
            )
        red = (
            reduction(self.K, 0, 0, self.in_fmt)
            if self.chunks == 1
            else reduction_chunked(self.K, self.chunks, 0, 0, self.in_fmt)
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
        if self.block_scale:
            # every block is one stage of the lowering (one block of `inner` steps from the reset accumulator: the kernel issues
            # `mma_fold` once per k-tile), and its trace is the block's partial of the spec
            from .spec import reduction_blockscale

            red = reduction_blockscale(self.K, 0, 0, self.in_fmt)
            kstep = red.digits[0]
            assert low.inner == kstep.radix == BLOCK_K // self.block.step, (
                low.inner,
                kstep.radix,
            )
            for kb in range(self.K // BLOCK_K):
                partial = kstep.reduce(
                    alg,
                    [
                        red.leaf(alg, {"block": kb, "kstep": t})
                        for t in range(kstep.radix)
                    ],
                )
                assert low.trace(red, alg, 1, fixed={"block": kb}) is partial, (
                    f"mainloop lowering differs on block {kb}"
                )
            return got
        steps_per_chunk = self.block.steps(self.K // self.chunks)
        nb = low.blocks_for(steps_per_chunk)
        K_pad = nb * low.inner * self.block.step * self.chunks
        if self.chunks == 1:
            red_pad = reduction(K_pad, 0, 0, self.in_fmt)
            assert low.trace(red_pad, alg, nb) is red_pad.canonical(alg), (
                "mainloop lowering is not the canonical fold"
            )
        else:
            red_pad = reduction_chunked(K_pad, self.chunks, 0, 0, self.in_fmt)
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
            + f" | {self.epilogue.describe()}"
            + (
                (
                    " | alpha = the given scalar"
                    if self.in_fmt in ALPHA_GIVEN
                    else " | alpha = mul.rn(a_scale, b_scale)"
                )
                if is_scaled(self.in_fmt)
                else ""
            )
        )
