# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SM120 per-variant scale-factor configs (fp8 float-scale, mxfp8 x mxfp4 UE8M0), each self-contained."""
import cutlass
import cutlass.cute as cute

from ._common import SF_M_ALIGN, TMA_ALIGN_BYTES
from ._common import align, ceil_div


class Sm120SfConfigMxfp8Mxfp4:
    """mxfp8 (pure + mix) self-contained ue8m0 SF config for the SM120 block-scaled MMA. K-independent (sized by stages, never total K)."""
    KTILE_SF = 1
    PACK_NSF = 4
    SF_VEC = 32

    def __init__(self, grank_a, grank_b):
        assert grank_a % grank_b == 0 or grank_b % grank_a == 0, (
            f"SF granularities must nest: neither {grank_a} nor {grank_b} divides the other, so the "
            f"coarser side's pack does not hold a whole number of the finer side's")
        self.sf_dtype = cutlass.Int32
        self.use_ue8m0 = True
        self.grank_a = grank_a
        self.grank_b = grank_b
        self.sf_load_warp = 1

    def _k_tiles_per_pack(self, grank, tile_k):
        """tile-Ks covered by one int32 SF pack (pack_nk = grank*PACK_NSF)."""
        pack_nk = grank * self.PACK_NSF
        assert pack_nk % tile_k == 0, "grank*PACK_NSF must be a multiple of tile_k (int32 SF pack aligns to tile-K)"
        return pack_nk // tile_k

    def k_tiles_per_pack_a(self, tile_k):
        return self._k_tiles_per_pack(self.grank_a, tile_k)

    def k_tiles_per_pack_b(self, tile_k):
        return self._k_tiles_per_pack(self.grank_b, tile_k)

    def _sf_stages(self, ab_stage, k_tiles_per_pack):
        """SF stages backing ab_stage k-tiles; never below 1, or a lone outliving pack gets no buffer."""
        return max(1, ceil_div(ab_stage, k_tiles_per_pack))

    def ab_stages_contract(self, ab_stage):
        """Powers of two only: the stage index is a bit mask (sf_mxfp8_tma_load.cuh:80)."""
        return ab_stage & (ab_stage - 1) == 0

    def sfa_stages(self, ab_stage, tile_k):
        return self._sf_stages(ab_stage, self.k_tiles_per_pack_a(tile_k))

    def sfb_stages(self, ab_stage, tile_k):
        return self._sf_stages(ab_stage, self.k_tiles_per_pack_b(tile_k))

    def _stage_stride(self, mn):
        """Elements between SF stages: the mn extent floored at 128 B (sf_mxfp8_tma_load.cuh:256-263)."""
        return max(mn, 128 * 8 // self.sf_dtype.width)

    def _smem_layout(self, mn, sf_stages):
        return cute.make_layout((mn, self.KTILE_SF, sf_stages), stride=(1, mn, self._stage_stride(mn)))

    def _cosize(self, mn, sf_stages):
        return (sf_stages - 1) * self._stage_stride(mn) + mn

    def smem_bytes(self, tile, ab_stage):
        """SFA+SFB smem cost; mirrors _smem_layout's cosize so the stage search budgets what is allocated."""
        bm, bn, bk = tile
        packs = (self._cosize(bm, self.sfa_stages(ab_stage, bk))
                 + self._cosize(bn, self.sfb_stages(ab_stage, bk)))
        return packs * (self.sf_dtype.width // 8)

    def make_smem_layout_sfa(self, tile_m, sf_stages):
        return self._smem_layout(tile_m, sf_stages)

    def make_smem_layout_sfb(self, tile_n, sf_stages):
        return self._smem_layout(tile_n, sf_stages)

    def make_s2r_sf(self, tv_layout, tiler):
        """int32 SF S2R tiled copy; tv_layout from get_layoutSF{A,B}_TV, tiler = (perm_mn, KTILE_SF=1)."""
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.sf_dtype)
        return cute.make_tiled_copy(atom, tv_layout, tiler)

    def _deduce_layout(self, mn, K, L, grank):
        """MN-major SF gmem layout; mn rides m_align so the TMA globalStride stays 16B-aligned."""
        pack_nk = grank * self.PACK_NSF
        scale_k = ceil_div(K, pack_nk)
        mn = align(mn, SF_M_ALIGN)
        return cute.make_layout((mn, scale_k, L), stride=(1, mn, mn * scale_k))

    def deduce_sfa_layout(self, mn, K, L):
        return self._deduce_layout(mn, K, L, self.grank_a)

    def deduce_sfb_layout(self, mn, K, L):
        return self._deduce_layout(mn, K, L, self.grank_b)

    def thrfrg_SFA(self, sf_layout, tiled_mma):
        """SFA thread fragment; AtomLayoutSFA_TV = ((2,2,8),1):((8,0,1),16)."""
        atom_shape_mnk = tiled_mma.shape_mnk
        perm = tiled_mma.permutation_mnk
        thr_vmnk = tiled_mma.thr_layout_vmnk
        atom_sf_layout = cute.make_layout(((2, 2, 8), self.KTILE_SF), stride=((8, 0, 1), 16))
        t_tile = (perm[0], cute.make_layout(self.KTILE_SF))
        t_tensor = cute.logical_divide(sf_layout, t_tile)
        a_tile = (cute.make_layout(atom_shape_mnk[0]), cute.make_layout(self.KTILE_SF))
        a_tensor = cute.zipped_divide(t_tensor, a_tile)
        tv_tensor = cute.composition(a_tensor, (atom_sf_layout, None))
        thr_tile = (None, (cute.make_layout(cute.size(thr_vmnk[1])),
                           cute.make_layout(cute.size(thr_vmnk[3]))))
        return cute.zipped_divide(tv_tensor, thr_tile)

    @staticmethod
    def _thr_partition(thr_tensor, thr_coord):
        """One slice per value mode, counted off the layout so the rank follows the warp layout."""
        rest = (None, tuple([None] * cute.rank(thr_tensor.layout, mode=[1, 1])))
        return thr_tensor[thr_coord, rest]

    def partition_fragment_SFA(self, sf_tensor, thr_mma, tidx):
        thr_tensor = cute.make_tensor(sf_tensor.iterator, self.thrfrg_SFA(sf_tensor.layout, thr_mma))
        thr_vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)
        thr_vmk = (thr_vmnk[0], (thr_vmnk[1], thr_vmnk[3]))
        return cute.make_fragment_like(self._thr_partition(thr_tensor, thr_vmk))

    def get_layoutSFA_TV(self, tiled_mma):
        """int32 (thr,val) layout for the SFA S2R copy."""
        perm = tiled_mma.permutation_mnk
        thr_vmnk = tiled_mma.thr_layout_vmnk
        ref = cute.make_layout((cute.size(perm[0]), self.KTILE_SF))
        atile = (None, (cute.make_layout(shape=(cute.size(thr_vmnk[1]), cute.size(thr_vmnk[2])),
                                         stride=(1, 0)), None))
        thrfrg = self.thrfrg_SFA(ref, tiled_mma)
        tv = cute.composition(thrfrg, (atile, None))
        return cute.composition(tv, (cute.right_inverse(thr_vmnk), None))

    def make_sf_ue8m0_view(self, frg_int32, grank, tile_k):
        """Restore the packed int32 fragment into the per-mma-atom ue8m0 view fed into the MMA (shared by SFA/SFB)."""
        ue8m0_ptr = cute.recast_ptr(frg_int32.iterator, dtype=cutlass.Float8E8M0FNU)
        num_mn = cute.size(frg_int32.layout, mode=[1])
        pack_nk = grank * self.PACK_NSF
        atoms_per_sf = min(grank, tile_k) // self.SF_VEC
        num_sf_per_tk = tile_k // grank if tile_k >= grank else 1
        num_tk_per_sf = grank // tile_k if grank > tile_k else 1
        num_tk_groups = (pack_nk // tile_k) // num_tk_per_sf
        view = cute.make_layout(
            (self.SF_VEC, num_mn, (atoms_per_sf, num_sf_per_tk), (num_tk_per_sf, num_tk_groups)),
            stride=(0, self.PACK_NSF, (0, 1), (0, num_sf_per_tk)))
        return cute.make_tensor(ue8m0_ptr, view)

    def make_sfa_ue8m0_view(self, frg_int32, tile_k):
        """SFA (act) ue8m0 view with kGranK = grank_a. See make_sf_ue8m0_view."""
        return self.make_sf_ue8m0_view(frg_int32, self.grank_a, tile_k)

    def thrfrg_SFB(self, sf_layout, tiled_mma):
        """SFB thread fragment; AtomLayoutSFB_TV = ((4,8),1):((0,1),8)."""
        atom_shape_mnk = tiled_mma.shape_mnk
        perm = tiled_mma.permutation_mnk
        thr_vmnk = tiled_mma.thr_layout_vmnk
        atom_sf_layout = cute.make_layout(((4, 8), self.KTILE_SF), stride=((0, 1), 8))
        t_tile = (perm[1], cute.make_layout(self.KTILE_SF))
        t_tensor = cute.logical_divide(sf_layout, t_tile)
        a_tile = (cute.make_layout(atom_shape_mnk[1]), cute.make_layout(self.KTILE_SF))
        a_tensor = cute.zipped_divide(t_tensor, a_tile)
        tv_tensor = cute.composition(a_tensor, (atom_sf_layout, None))
        thr_tile = (None, (cute.make_layout(cute.size(thr_vmnk[2])),
                           cute.make_layout(cute.size(thr_vmnk[3]))))
        return cute.zipped_divide(tv_tensor, thr_tile)

    def partition_fragment_SFB(self, sf_tensor, thr_mma, tidx):
        thr_tensor = cute.make_tensor(sf_tensor.iterator, self.thrfrg_SFB(sf_tensor.layout, thr_mma))
        thr_vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)
        thr_vnk = (thr_vmnk[0], (thr_vmnk[2], thr_vmnk[3]))
        return cute.make_fragment_like(self._thr_partition(thr_tensor, thr_vnk))

    def get_layoutSFB_TV(self, tiled_mma):
        """int32 (thr,val) layout for the SFB S2R copy."""
        perm = tiled_mma.permutation_mnk
        thr_vmnk = tiled_mma.thr_layout_vmnk
        ref = cute.make_layout((cute.size(perm[1]), self.KTILE_SF))
        btile = (None, (cute.make_layout(shape=(cute.size(thr_vmnk[1]), cute.size(thr_vmnk[2])),
                                         stride=(0, 1)), None))
        thrfrg = self.thrfrg_SFB(ref, tiled_mma)
        tv = cute.composition(thrfrg, (btile, None))
        return cute.composition(tv, (cute.right_inverse(thr_vmnk), None))

    def make_sfb_ue8m0_view(self, frg_int32, tile_k):
        """SFB (weight) ue8m0 view with kGranK = grank_b. See make_sf_ue8m0_view."""
        return self.make_sf_ue8m0_view(frg_int32, self.grank_b, tile_k)


class Sm120SfConfigFp8:
    """fp8 block_scaling (DSv3) self-contained SF config for the SM120 plain-fp8 MMA (fp32 scale applied outside the MMA)."""
    NUM_SCALE_COPY_THREADS = 32

    def __init__(self, gran_m, gran_n, gran_k):
        self.sf_dtype = cutlass.Float32
        self.use_ue8m0 = False
        self.gran_m = gran_m
        self.gran_n = gran_n
        self.gran_k = gran_k
        self.sf_load_warp = 1
        # rows each expert's scales are padded to, the same quantity the host pads by
        SF_M_ALIGN = TMA_ALIGN_BYTES // (self.sf_dtype.width // 8)

    def k_tiles_per_pack(self, tile_k):
        """k-tiles one SF entry covers, i.e. the SF load cadence against the A/B loop."""
        assert self.gran_k % tile_k == 0, (
            f"gran_k ({self.gran_k}) must be a multiple of tile_k ({tile_k}): an SF entry has to "
            f"cover a whole number of k-tiles")
        return self.gran_k // tile_k

    def sf_stages(self, ab_stage, tile_k):
        """SF ring depth backing ab_stage k-tiles; equals ab_stage while the cadence is 1."""
        return max(1, ceil_div(ab_stage, self.k_tiles_per_pack(tile_k)))

    def ab_stages_contract(self, ab_stage):
        """Any depth: the SF ring shares the A/B stage counter, and both wrap on a runtime compare."""
        return True

    def assert_k_invariant(self, tile_k):
        """Require tile_k == gran_k (the 2D staged SF smem layout has no in-tile K dim)."""
        assert tile_k == self.gran_k, (
            f"tile_k ({tile_k}) must equal gran_k ({self.gran_k}): the 2D staged SF layout has no in-tile "
            f"K dim (kTileScaleK must be 1); use tile_k == gran_k or extend to a 3D K-scale layout.")

    def assert_mn_invariant(self, tile_m, tile_n):
        """Require whole granules on both axes; which one is coarse is what swap-AB exchanges."""
        assert tile_m % self.gran_m == 0 and tile_n % self.gran_n == 0, (
            f"tile ({tile_m}, {tile_n}) must cover whole scale granules "
            f"({self.gran_m}, {self.gran_n}); a partial granule indexes past the SF tensor.")

    def deduce_sfa_layout(self, M, K, L):
        """MN-major SFA gmem layout [ceil(M/gran_m), ceil(K/gran_k), L]."""
        scale_m = ceil_div(M, self.gran_m)
        scale_k = ceil_div(K, self.gran_k)
        return cute.make_layout((scale_m, scale_k, L), stride=(1, scale_m, scale_m * scale_k))

    def deduce_sfb_layout(self, N, K, L):
        """MN-major SFB gmem layout [ceil(N/gran_n), ceil(K/gran_k), L]."""
        scale_n = ceil_div(N, self.gran_n)
        scale_k = ceil_div(K, self.gran_k)
        return cute.make_layout((scale_n, scale_k, L), stride=(1, scale_n, scale_n * scale_k))

    def make_smem_layout_sfa(self, tile_m, stages):
        """Staged SFA consume view [kTileScaleM, SF_Stages]."""
        tscale_m = ceil_div(tile_m, self.gran_m)
        return cute.make_layout((tscale_m, stages), stride=(1, tscale_m))

    def make_smem_layout_sfb(self, tile_n, stages):
        """Staged SFB consume view [kTileScaleN, SF_Stages]."""
        tscale_n = ceil_div(tile_n, self.gran_n)
        return cute.make_layout((tscale_n, stages), stride=(1, tscale_n))

    def smem_bytes(self, tile, ab_stage):
        """SFA+SFB smem cost. SFA is allocated over its TMA view, which adds no elements."""
        bm, bn, _ = tile
        entries = (ceil_div(bm, self.gran_m) + ceil_div(bn, self.gran_n)) * ab_stage
        return entries * (self.sf_dtype.width // 8)

    def make_tma_smem_layout_sfa(self, tile_m, stages):
        """SFA TMA-target smem [kTileScaleM, 1, SF_Stages], overlaying the SmemLayoutSFA buffer."""
        tscale_m = ceil_div(tile_m, self.gran_m)
        return cute.make_layout((tscale_m, 1, stages), stride=(1, tscale_m, tscale_m))

    def partition_scale_as_c(self, sSFA, sSFB, thr_mma):
        """ViewAsC: re-view the staged SF smem as the C tile. Returns (tCsSFA, tCsSFB, tCrSFA, tCrSFB)."""
        tscale_m, stages = cute.size(sSFA, mode=[0]), cute.size(sSFA, mode=[1])
        tscale_n = cute.size(sSFB, mode=[0])
        tile_m, tile_n = self.gran_m * tscale_m, self.gran_n * tscale_n
        sSFA_vc = cute.make_tensor(sSFA.iterator, cute.make_layout(
            ((self.gran_m, tscale_m), tile_n, stages), stride=((0, 1), 0, tscale_m)))
        sSFB_vc = cute.make_tensor(sSFB.iterator, cute.make_layout(
            (tile_m, (self.gran_n, tscale_n), stages), stride=(0, (0, 1), tscale_n)))
        tCsSFA = thr_mma.partition_C(sSFA_vc)
        tCsSFB = thr_mma.partition_C(sSFB_vc)
        tCrSFA = cute.make_fragment_like(tCsSFA[None, None, None, 0])
        tCrSFB = cute.make_fragment_like(tCsSFB[None, None, None, 0])
        return tCsSFA, tCsSFB, tCrSFA, tCrSFB
