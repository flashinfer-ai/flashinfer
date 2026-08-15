# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SM120 block-scaled GEMM builder for the warp-specialized kernel family: composable per-phase configs + top builder."""
import abc
import enum

import cutlass
import cutlass.utils
import cutlass.cute as cute
import cutlass.cute.nvgpu.warp.mma as warp_mma
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync

from .scheduler import MoeSchedStages, MoeWorkTile



class MmaConfig:
    """MMA phase: warp-MMA op + the warp layout its tile implies + permutation flag."""
    def __init__(self, op, mma_tile_mn, num_math_warps, swap_ab=False):
        self.op = op
        self.mma_tile_mn = tuple(mma_tile_mn)
        self.num_math_warps = num_math_warps
        self.swap_ab = swap_ab
        self.use_mxf8f6f4 = not isinstance(op, warp_mma.MmaMXF4NVF4Op)
        self.plain_fp8 = isinstance(op, warp_mma.MmaFP8Op)
        assert self.mma_tile_mn[1] // self.num_warp_n >= op.shape_mnk[1], (
            f"tile_n {self.mma_tile_mn[1]} over {self.num_warp_n} N-warps leaves each warp less "
            f"than the atom's N extent {op.shape_mnk[1]}")

    @property
    def num_warp_m(self):
        """M-major by owner decision, not upstream's bn-derived rule; swap-AB uses upstream's (builder.cuh:52-55)."""
        if self.swap_ab:
            bn, atom_n = self.mma_tile_mn[1], self.op.shape_mnk[1]
            return self.num_math_warps // (4 if bn >= 32 else bn // atom_n)
        return 4 if self.mma_tile_mn[0] >= 64 else 2

    @property
    def num_warp_n(self):
        return self.num_math_warps // self.num_warp_m

    def mma_warp_layout(self):
        """The MNK warp layout (call inside @cute.jit)."""
        warp_m = self.num_warp_m
        return cute.make_ordered_layout((warp_m, self.num_math_warps // warp_m, 1), order=(0, 1, 2))

    @property
    def sf_vec(self):
        """Block-scale vector length the op itself declares; fp8 has none, so K rides the atom."""
        return self.op.shape_mnk[2] if self.plain_fp8 else self.op.sf_vec_size

    def make_tiled_mma(self, tile):
        """Build the TiledMMA (call inside @cute.jit)."""
        # get_permutation_mnk pins perm_n at 32 regardless of bn, which a swapped bn cannot hold
        if self.plain_fp8 or self.swap_ab:
            perm = (tile[0], tile[1], self.sf_vec)
        else:
            perm = sm120_utils.get_permutation_mnk(tile, self.sf_vec, self.use_mxf8f6f4)
        return cute.make_tiled_mma(self.op, self.mma_warp_layout(), permutation_mnk=perm)

    def make_s2r_a(self, tiledmma, smem_dtype, transpose, unpack_bits=None):
        if unpack_bits is None:
            ld = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=transpose, num_matrices=4), smem_dtype)
        else:
            ld = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x16x8bOp(transpose=False, num_matrices=4,
                                                  unpack_bits=unpack_bits), smem_dtype)
        return cute.make_tiled_copy_A(ld, tiledmma)

    @property
    def ldsm_matrices_b(self):
        """How many matrices one B ldmatrix moves, from kPerWarpN (flashinfer ab_tma_load.cuh:42-48)."""
        per_warp_n = self.mma_tile_mn[1] // self.num_warp_n
        return 4 if per_warp_n >= 16 else (2 if per_warp_n >= 8 else 1)

    def make_s2r_b(self, tiledmma, smem_dtype, transpose, unpack_bits=None):
        num_matrices = self.ldsm_matrices_b
        if unpack_bits is None:
            ld = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=transpose, num_matrices=num_matrices), smem_dtype)
        else:
            ld = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x16x8bOp(transpose=False, num_matrices=num_matrices,
                                                  unpack_bits=unpack_bits), smem_dtype)
        return cute.make_tiled_copy_B(ld, tiledmma)

    def make_s2r_sf(self, sf_dtype, tv_layout, tiler):
        """SF S2R tiled copy (consumer side)."""
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), sf_dtype)
        return cute.make_tiled_copy(atom, tv_layout, tiler)


class LoadABConfig:
    """A/B G2S phase: per-operand dtypes and extents in, k-major smem layouts + TMA atoms out."""
    def __init__(self, tile, ab_stage, a_dtype, b_dtype,
                 a_smem_dtype=None, b_smem_dtype=None,
                 a_tma_internal=None, b_tma_internal=None,
                 a_unpack_bits=None, b_unpack_bits=None):
        self.tile, self.ab_stage = tuple(tile), ab_stage
        self.a_dtype, self.b_dtype = a_dtype, b_dtype
        self.a_smem_dtype = a_smem_dtype or a_dtype
        self.b_smem_dtype = b_smem_dtype or b_dtype
        self.a_tma_internal, self.b_tma_internal = a_tma_internal, b_tma_internal
        self.a_unpack_bits, self.b_unpack_bits = a_unpack_bits, b_unpack_bits

        bm, bn, bk = self.tile
        # smem always holds one value per element, natively or because the TMA unpacked it on the way in
        self.tma_box_a, self.tma_box_b = (bm, bk), (bn, bk)
        # the transaction counts gmem bytes, which the source dtype over the K tile gives directly
        self.tma_bytes_a = bm * bk * a_dtype.width // 8
        self.tma_bytes_b = bn * bk * b_dtype.width // 8
        self.tma_bytes_ab = self.tma_bytes_a + self.tma_bytes_b

    def _smem_layout(self, box, smem_dtype):
        atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(cutlass.utils.LayoutEnum.ROW_MAJOR, smem_dtype, box[1]), smem_dtype)
        return cute.tile_to_shape(atom, (*box, self.ab_stage), order=(0, 1, 2))

    def smem_bytes_a(self):
        """A staging cost. tile_to_shape over a dense box, so cosize is box * ab_stage."""
        bm, _, bk = self.tile
        return bm * bk * self.a_smem_dtype.width * self.ab_stage // 8

    def smem_bytes_b(self):
        _, bn, bk = self.tile
        return bn * bk * self.b_smem_dtype.width * self.ab_stage // 8

    def smem_bytes(self):
        return self.smem_bytes_a() + self.smem_bytes_b()

    def make_smem_layout_a(self):
        return self._smem_layout(self.tma_box_a, self.a_smem_dtype)

    def make_smem_layout_b(self):
        return self._smem_layout(self.tma_box_b, self.b_smem_dtype)

    def _tma_atom(self, g, smem_layout, box, internal):
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), g, cute.slice_(smem_layout, (None, None, 0)), box,
            num_multicast=1, internal_type=internal)

    def make_tma_atom_a(self, gA, smem_layout_a):
        return self._tma_atom(gA, smem_layout_a, self.tma_box_a, self.a_tma_internal)

    def make_tma_atom_b(self, gB, smem_layout_b):
        return self._tma_atom(gB, smem_layout_b, self.tma_box_b, self.b_tma_internal)


class StoreMethod(enum.Enum):
    """Identity of an epilogue store path. Each value has exactly one StoreConfig subclass."""
    R2G_WG = enum.auto()
    STAGED_R2G = enum.auto()
    DIRECT_STG = enum.auto()


class StoreConfig(abc.ABC):
    """Epilogue store phase. One subclass per store path; the base holds what they share."""
    STG_BYTES = 16    # the widest vectorized global store, i.e. the copy atom's width
    METHOD = None           # StoreMethod this subclass implements
    HAS_STORE_WARP = None   # declared per path, not derived: it is a property of the store method
    # Bytes one swizzle period covers, per atom kind (get_smem_layout_atom's own thresholds).
    _PERIOD_BYTES = {
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128: 128,
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64: 64,
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32: 32,
        cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_INTER: STG_BYTES,
    }

    def __init__(self, out_dtype, mma_threads, epi_stage=1):
        self.out_dtype = out_dtype
        self.mma_threads = mma_threads
        # Buffered epilogue tiles; the store_full/store_empty barriers count the same quantity.
        self.epi_stage = epi_stage

    @property
    def s2g_vec(self):
        """Elements per thread per pass: the store atom's 16 B, counted in the output dtype."""
        vec = self.STG_BYTES * 8 // self.out_dtype.width
        assert vec >= 1 and vec & (vec - 1) == 0, (
            f"{self.out_dtype} gives {vec} elements per {self.STG_BYTES} B store; the swizzle's M "
            f"mode counts elements and needs a power of two")
        return vec

    def smem_layout_atom_kind(self, tile):
        """The atom whose swizzle suits sC's contiguous extent, by the same rule the A/B path uses."""
        return sm90_utils.get_smem_layout_atom(
            cutlass.utils.LayoutEnum.ROW_MAJOR, self.out_dtype, self.epi_tile(tile)[1])

    def epi_tile(self, tile):
        """Epilogue subtile. Upstream stages a smaller one on the TMA (32 x bn) and staged (64 x 32)"""
        return tile[0], tile[1]

    def num_epi(self, tile):
        epi_m, epi_n = self.epi_tile(tile)
        return tile[0] // epi_m, tile[1] // epi_n

    @property
    @abc.abstractmethod
    def num_store_threads(self):
        """Threads that consume sC, i.e. that arrive on store_empty."""

    def smem_bytes(self, tile):
        epi_m, epi_n = self.epi_tile(tile)
        return epi_m * epi_n * self.epi_stage * self.out_dtype.width // 8

    def make_smem_layout(self, tile):
        """sC's layout (call inside @cute.jit)."""
        epi_m, epi_n = self.epi_tile(tile)
        atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            self.smem_layout_atom_kind(tile), self.out_dtype)
        return cute.tile_to_shape(atom, (epi_m, epi_n, self.epi_stage), order=(0, 1, 2))

    def s2g_thr_layout(self, tile):
        """Thread layout covering exactly the threads that drain sC (call inside @cute.jit)."""
        threads_n = self._PERIOD_BYTES[self.smem_layout_atom_kind(tile)] // self.STG_BYTES
        threads_m, rem = divmod(self.num_store_threads, threads_n)
        assert rem == 0, (
            f"{type(self).__name__}: {self.num_store_threads} draining threads do not split into "
            f"rows of {threads_n}; the S2G thread layout must cover the participating threads "
            f"exactly, or the threads left out write past their slice of gD")
        return cute.make_layout((threads_m, threads_n), stride=(threads_n, 1))

    def make_tiled_s2g(self, atom, tile):
        """Build the S2G TiledCopy (call inside @cute.jit)."""
        return cute.make_tiled_copy_tv(atom, self.s2g_thr_layout(tile),
                                       cute.make_layout((1, self.s2g_vec)))


class R2GWgStoreConfig(StoreConfig):
    """The whole math warpgroup drains sC to gmem itself, so there is no store warp."""
    METHOD = StoreMethod.R2G_WG
    HAS_STORE_WARP = False

    @property
    def num_store_threads(self):
        return self.mma_threads


class StagedR2GStoreConfig(StoreConfig):
    """One dedicated store warp drains sC while the math warpgroup moves on."""
    METHOD = StoreMethod.STAGED_R2G
    HAS_STORE_WARP = True
    STORE_THREADS = 32

    @property
    def num_store_threads(self):
        return self.STORE_THREADS


class DirectStgStoreConfig(StoreConfig):
    """Each math thread stores its own accumulator fragment, so there is no sC and no handoff."""
    METHOD = StoreMethod.DIRECT_STG
    HAS_STORE_WARP = False

    @property
    def num_store_threads(self):
        return self.mma_threads

    def smem_bytes(self, tile):
        return 0


STORE_CONFIGS = {StoreMethod.STAGED_R2G: StagedR2GStoreConfig, StoreMethod.R2G_WG: R2GWgStoreConfig,
                 StoreMethod.DIRECT_STG: DirectStgStoreConfig}


class Sm120GemmBuilder:
    """Top WS builder: composes the phase configs + derives the warp-role layout (3 warpgroups, 384 threads)."""
    ACC = cutlass.Float32
    I64, I32, I16, I8 = cutlass.Int64, cutlass.Int32, cutlass.Int16, cutlass.Int8

    def __init__(self, mma, load_ab, load_sf, store, ab_stage, tile, epi_bar_id=2, union_smem=False):
        self.mma, self.load_ab, self.load_sf, self.store = mma, load_ab, load_sf, store
        # union_smem: sC views the A/B buffers, so store_empty gates the producer's refill.
        self.union_smem = union_smem
        assert not union_smem or store.METHOD is StoreMethod.R2G_WG, (
            f"union_smem overlays sC on A/B, which only {StoreMethod.R2G_WG} drains itself; "
            f"{store.METHOD} would leave store_empty without a counterpart")
        assert not union_smem or store.smem_bytes(tile) <= load_ab.smem_bytes(), (
            f"sC is {store.smem_bytes(tile)} B over {load_ab.smem_bytes()} B of A/B staging; past "
            f"that it overlays whatever follows, which no size model reports")
        self.TILE = tuple(tile)
        self.ab_stage = ab_stage
        self.epi_bar_id = epi_bar_id
        num_math_warps = mma.num_math_warps
        self.num_math_warps = num_math_warps
        self.mma_threads = num_math_warps * 32
        assert store.mma_threads == self.mma_threads, (
            f"StoreConfig was built for {store.mma_threads} math threads but MmaConfig gives "
            f"{self.mma_threads}; the S2G thread layout is derived from that count")
        self.store_threads = store.num_store_threads
        # producer warps right after the math warps: sched, ab, sf, then store (never two in the last slot).
        self.sched_warp = num_math_warps
        self.ab_warp = num_math_warps + 1
        self.sf_warp = num_math_warps + 2
        self.store_warp = num_math_warps + 3
        num_producer_warps = 2 + self.load_sf.sf_load_warp + (1 if store.HAS_STORE_WARP else 0)
        assert num_producer_warps <= 4, (
            f"sched, ab, {self.load_sf.sf_load_warp} SF and "
            f"{'a store' if store.HAS_STORE_WARP else 'no store'} warp need "
            f"{num_producer_warps} producer slots, but the layout has four")
        self.threads = (num_math_warps + num_producer_warps) * 32
        self.epi_threads = self.mma_threads + (32 if store.HAS_STORE_WARP else 0)
        self.reg_prod = self.REG_PROD
        self.reg_math = self._reg_math(num_math_warps, num_producer_warps)
        self.sched_stages = MoeSchedStages
        self.store_stages = store.epi_stage
        self.fields = MoeWorkTile.FIELDS
        self.num_sched_consumers = num_math_warps + num_producer_warps - 1

    # barriers and the work tile are dozens of bytes against tens of KB, so the search reserves a flat slab.
    MBAR_RESERVE = 1024
    MAX_AB_STAGE = 4

    # setmaxregister budget: the grid is persistent, so the whole per-SM register file is this CTA's.
    REGS_PER_SM = 65536
    REG_GRANULARITY = 8     # setmaxregister operates in steps of 8
    REG_MAX = 256
    REG_PROD = 40

    @classmethod
    def _reg_math(cls, num_math_warps, num_producer_warps):
        """Registers per math thread: whatever the producers leave, rounded down to the granularity."""
        left = cls.REGS_PER_SM - num_producer_warps * 32 * cls.REG_PROD
        per_thread = left // (num_math_warps * 32)
        reg_math = min(cls.REG_MAX, per_thread - per_thread % cls.REG_GRANULARITY)
        assert reg_math >= cls.REG_PROD, (
            f"{num_math_warps} math and {num_producer_warps} producer warps leave {reg_math} registers "
            f"per math thread, below the {cls.REG_PROD} the producers get")
        return reg_math

    @property
    def smem_bytes(self):
        """Tensor-buffer smem for this configuration. Excludes barriers (see MBAR_RESERVE)."""
        total = self.load_ab.smem_bytes() + self.load_sf.smem_bytes(self.TILE, self.ab_stage)
        if not self.union_smem:
            total += self.store.smem_bytes(self.TILE)
        return total

    @classmethod
    def max_ab_stage(cls, make_cfg, tile):
        """Deepest A/B ring the tile affords in smem and the arm's SF ring can index."""
        capacity = cutlass.utils.SmemAllocator.capacity_in_bytes() - cls.MBAR_RESERVE
        for stage in range(cls.MAX_AB_STAGE, 0, -1):
            cfg = make_cfg(tile, stage)
            if cfg.load_sf.ab_stages_contract(stage) and cfg.smem_bytes <= capacity:
                return stage
        raise ValueError(f"no ab_stage fits {capacity} B of smem for tile {tile}")

    def is_prod_wg(self, warp_idx):
        return warp_idx >= self.num_math_warps
